from typing import Optional
import tempfile

import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic
import redis

from common.config import settings
from common.rabbitmq import PikaSession
from app.utils import crop_image, image_to_bytes
from app.utils import compute_iou


redis = redis.Redis(
    host=settings.redis.host,
    port=settings.redis.port,
)


class Detection(BaseModel):
    """
    Object detected in a frame.

    Attributes:
    - xyxy (list[int]): xyxy coordinates of the bounding box, size (4,)
    - score (float): confidence score of the detection
    - label (int): label of the detection
    - ftl (int): frame to live, number of frames to consider the detection
                 actual. This is used to fight 'flickering' detections.
    """
    box: list[int]
    score: float
    label: int
    ftl: int = settings.detector.detection_ftl


def get_detections(source_id: int) -> Optional[list[Detection]]:
    """
    Get actual detections for a source from Redis.
    Actual detections are detections from previous frames whuch ftl and
    ttl are not expired.
    - ftl - frames-to-live, fight 'flickering' detections
    - ttl - time-to-live, remove detections from inactive sources

    Parameters:
    - source_id (int): id of the source

    Returns:
    - detections (list[Detection]): list of detections for the source,
        or None if redis record does not exist
    """
    detections = redis.json().get(f'detector:detections:{source_id}', '$')
    if detections:
        return [Detection(**d) for d in detections[0]]


def set_detections(source_id: int, detections: list[Detection]):
    """
    Set detections for a source in Redis.

    Parameters:
    - source_id (int): id of the source
    - detections (list[Detection]): list of detections for the source
    """
    detections = [d.dict() for d in detections]
    key = f'detector:detections:{source_id}'
    redis.json().set(key, '$', detections)
    redis.expire(key, settings.detector.detection_ttl)


def get_new_detections(prev_detections: list[Detection],
                       detections: list[Detection]) -> list[Detection]:
    """
    Filter out detections that were already present in the previous frames.
    Detections are considered the same if:
    a. Their IoU is greater than `iou_identity_threshold_hard
    b. They have the same label and their IoU is greater
       than `iou_identity_threshold_soft`

    Parameters:
    - prev_detections (list[Detection]): list of detections from the previous
        frames
    - detections (list[Detection]): list of detections from the current frame

    Returns:
    - new_detections (list[Detection]): list of detections from the current
        frame that are not in the previous frames
    - old_detections (list[Detection]): list of detections from the previous
        frames that are still in the current frame
    """
    if not prev_detections:
        return detections, []

    new_detections = []
    old_detections = []
    th_soft = settings.detector.iou_identity_threshold_soft
    th_hard = settings.detector.iou_identity_threshold_hard
    for detecion in detections:
        ious = compute_iou(np.array(detecion.box),
                           np.array([d.box for d in prev_detections]))
        max_iou = ious.max()
        best_match = prev_detections[ious.argmax()]
        if max_iou > th_hard or max_iou > th_soft and \
                detecion.label == best_match.label:
            old_detections.append(best_match)
        else:
            new_detections.append(detecion)
    return new_detections, old_detections


session = PikaSession()

model = YOLO(settings.detector.model)
model.to('cuda')


@session.on_message
def on_message(channel: BlockingChannel, method: Basic.Deliver,
               properties: pika.BasicProperties, body: bytes):
    sm_name = properties.headers['sm_name']
    source_id = int(properties.headers['source_id'])
    start_time = float(properties.headers['start_time'])
    end_time = float(properties.headers['end_time'])
    n_frames = int(properties.headers['n_frames'])

    # Get previous detections from Redis, redis acts as unified state storage
    # for all detector instances
    prev_detections = get_detections(source_id) or []
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        f.write(body)
        f.flush()
        results = model(
            f.name,
            conf=settings.detector.conf_threshold,
            iou=settings.detector.iou_threshold,
            max_det=settings.detector.max_detections,
            classes=settings.detector.classes,
            verbose=False,
            stream=True
        )

        for i, result in enumerate(results):
            timestamp = start_time + (end_time - start_time) * i / n_frames

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()

            detections = [
                Detection(box=box.tolist(), score=score, label=label)
                for box, score, label in zip(boxes, scores, labels)
            ]

            # new detections - detections that are not in the previous frames
            # old detections - detections from pevious frames that are still
            # in the current frame
            new_detections, old_detections = get_new_detections(
                prev_detections, detections
            )
            # Remove detections that are still in the frame, because they
            # are included in the `new detections` with actual bounding boxes
            prev_detections = [
                detection for detection in prev_detections
                if detection not in old_detections and detection.ftl > 0
            ]
            # Add all detections from the current frame, because they are
            # either new or their older versions was removed from the list
            prev_detections.extend(detections)
            # Decrease the frame-to-live counter for all detections in the
            # list
            for detection in prev_detections:
                detection.ftl -= 1

            for detection in new_detections:
                frame_crop = crop_image(result.orig_img, detection.box)
                session.publish(
                    exchange=settings.rabbitmq.frame_crops_exchange,
                    routing_key='',
                    body=image_to_bytes(frame_crop),
                    properties=pika.BasicProperties(
                        content_type='image/jpeg',
                        headers={
                            'sm_name': sm_name,
                            'source_id': str(source_id),
                            'timestamp': str(timestamp),
                        }
                    )
                )
    # Set actual detections for the source in Redis
    set_detections(source_id, prev_detections)
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    session.startup()
    session.consume(settings.rabbitmq.video_chunks_queue)
