from typing import Optional
import tempfile
from io import BytesIO
from PIL import Image

from pydantic import BaseModel
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic
from ultralytics import YOLO
import redis

from common.config import settings
from common.rabbitmq import PikaSession
from app.utils import intersection_over_union, crop_image


redis = redis.Redis(
    host=settings.redis.host,
    port=settings.redis.port,
)


class Detection(BaseModel):
    xyxy: list[int]
    conf: float
    cls: int
    is_new: bool = True
    ftl: int = settings.detector.detection_ftl


def get_last_detections(source_id: int) -> Optional[list[Detection]]:
    detections = redis.json().get(f'detector:detections:{source_id}', '$')
    if detections:
        return [Detection(**d) for d in detections[0]]


def set_last_detections(source_id: int, detections: list[Detection]):
    detections = [d.dict() for d in detections]
    redis.json().set(f'detector:detections:{source_id}', '$', detections)


session = PikaSession()
model = YOLO(settings.detector.model)


@session.on_message
def on_message(channel: BlockingChannel, method: Basic.Deliver,
               properties: pika.BasicProperties, body: bytes):
    source_id = int(properties.headers['source_id'])
    start_time = float(properties.headers['start_time'])
    end_time = float(properties.headers['end_time'])
    n_frames = int(properties.headers['n_frames'])
    with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
        f.write(body)
        f.flush()
        results = model(
            f.name,
            device=settings.detector.device,
            conf=settings.detector.conf_threshold,
            iou=settings.detector.iou_threshold,
            max_det=settings.detector.max_detections,
            classes=settings.detector.classes,
            verbose=False,
            stream=True
        )

        prev_detections = get_last_detections(source_id) or []
        for i, result in enumerate(results):
            detections = [
                Detection(xyxy=box.xyxy[0].tolist(), conf=box.conf[0], cls=box.cls[0])
                for box in result.boxes.numpy()
            ]
            timestamp = start_time + (end_time - start_time) * i / n_frames

            th_soft = settings.detector.identity_threshold_soft
            th_hard = settings.detector.identity_threshold_hard
            for di in detections:
                for dj in prev_detections:
                    iou = intersection_over_union(di.xyxy, dj.xyxy)
                    if di.cls == dj.cls and iou > th_soft or iou > th_hard:
                        dj.ftl = settings.detector.detection_ftl
                        di.is_new = False
                        break

            detections = [d for d in detections if d.is_new]
            for d in prev_detections:
                d.ftl -= 1
            prev_detections = [d for d in prev_detections if d.ftl > 0]
            prev_detections.extend(detections)

            for detection in detections:
                frame_crop = crop_image(result.orig_img, detection.xyxy)
                frame_crop = Image.fromarray(frame_crop)
                body = BytesIO()
                frame_crop.save(body, format='JPEG')
                body = body.getvalue()
                session.publish(
                    exchange=settings.rabbitmq.frame_crops_exchange,
                    routing_key='',
                    body=body,
                    properties=pika.BasicProperties(
                        content_type='image/jpeg',
                        headers={
                            'source_id': str(source_id),
                            'timestamp': str(timestamp),
                        }
                    )
                )
    set_last_detections(source_id, prev_detections)
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    session.startup()
    session.consume(settings.rabbitmq.video_chunks_queue)
