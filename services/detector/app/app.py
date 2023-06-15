import tempfile

import numpy as np
import torch
from ultralytics import YOLO
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic

from common.config import settings
from common.clients.amqp import Session
from app.detections import (
    Detection,
    get_detections,
    set_detections,
    get_new_detections,
    is_child_detection,
)
from app.utils import (
    crop_image,
    image_to_bytes,
    outer_box,
)
import app.monitoring as monitoring


session = Session()
session.set_connection_params(
    host=str(settings.rabbitmq.host),
    port=settings.rabbitmq.port,
    virtual_host=settings.rabbitmq.virtual_host,
    username=settings.rabbitmq.ml_processing_username,
    password=settings.rabbitmq.ml_processing_password,
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = YOLO(settings.detector.model)
model.to(device)


@session.on_message
def on_message(channel: BlockingChannel, method: Basic.Deliver,
               properties: pika.BasicProperties, body: bytes):
    source_manager_id = properties.headers['source_manager_id']
    source_id = properties.headers['source_id']
    chunk_id = properties.headers['chunk_id']
    frame_count = int(properties.headers['frame_count'])
    start_time = float(properties.headers['start_time'])
    end_time = float(properties.headers['end_time'])

    monitoring.chunk_fps.observe(frame_count / (end_time - start_time))

    # Get previous detections from Redis, redis acts as unified state storage
    # for all detector instances
    prev_detections = get_detections(source_manager_id, source_id) or []
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        f.write(body)
        f.flush()
        results = model(
            f.name,
            conf=settings.detector.conf_threshold,
            iou=settings.detector.iou_threshold,
            max_det=settings.detector.max_detections,
            classes=settings.detector.classes,
            agnostic_nms=settings.detector.agnostic_nms,
            verbose=False,
            stream=True,
        )

        monitoring.timer.start('inference')
        for i, result in enumerate(results):
            monitoring.processing_duration_seconds.labels('inference')\
                .observe(monitoring.timer.get('inference'))
            monitoring.timer.start('preprocessing')

            timestamp = start_time + (end_time - start_time) * i / frame_count

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()

            detections = []
            for box, score, label in zip(boxes, scores, labels):
                shape = box[2:] - box[:2]
                if any(shape < settings.detector.min_box_size):
                    continue
                detections.append(
                    Detection(box=box.tolist(), score=score, label=label)
                )

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

            # Increase size of the bounding box to capture it's `child`
            # detections. You can read more about this
            # in the `is_child_detection` docstring.
            # Example: bag worn by a person will be included in the person's
            # bounding box
            for d_parent in new_detections:
                child_detections = [d_parent]
                for d_child in new_detections:
                    if is_child_detection(d_parent, d_child):
                        child_detections.append(d_child)
                if len(child_detections) > 1:
                    child_boxes = np.array([d.box for d in child_detections])
                    d_parent.box = outer_box(child_boxes).tolist()

            monitoring.processing_duration_seconds.labels('preprocessing')\
                .observe(monitoring.timer.get('preprocessing'))
            monitoring.timer.start('publishing')

            for detection in new_detections:
                frame_crop = crop_image(result.orig_img, detection.box)
                frame_crop = frame_crop[..., ::-1]  # BGR to RGB
                session.publish(
                    exchange=settings.rabbitmq.frame_crops_exchange,
                    routing_key='',
                    body=image_to_bytes(frame_crop),
                    properties=pika.BasicProperties(
                        content_type='image/png',
                        headers={
                            'source_manager_id': source_manager_id,
                            'source_id': str(source_id),
                            'chunk_id': str(chunk_id),
                            'position': str(i),
                            'timestamp': str(timestamp),
                            'box': ','.join(map(str, detection.box)),
                        }
                    )
                )
                monitoring.detection_labels.observe(detection.label)

            monitoring.processing_duration_seconds.labels('publishing')\
                .observe(monitoring.timer.get('publishing'))
            monitoring.detections_count.observe(len(new_detections))
            monitoring.timer.start('inference')
    # Set actual detections for the source in Redis
    set_detections(source_manager_id, source_id, prev_detections)
    channel.basic_ack(delivery_tag=method.delivery_tag)

    monitoring.messages_processed_total.inc()
    monitoring.push_metrics()


def main():
    session.start_consuming(
        settings.rabbitmq.video_chunks_queue,
        prefetch_count=settings.rabbitmq.prefetch_count
    )
