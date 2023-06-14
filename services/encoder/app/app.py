from io import BytesIO
from PIL import Image

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic
from transformers import CLIPModel, CLIPProcessor

from common.config import settings
from common.clients.amqp import Session
from common.database import redis
import app.monitoring as monitoring


session = Session()
session.set_connection_params(
    host=str(settings.rabbitmq.host),
    port=settings.rabbitmq.port,
    virtual_host=settings.rabbitmq.virtual_host,
    username=settings.rabbitmq.ml_processing_username,
    password=settings.rabbitmq.ml_processing_password,
)


model = CLIPModel.from_pretrained(settings.encoder.model)
processor = CLIPProcessor.from_pretrained(settings.encoder.model)


@session.on_message
def on_message(channel: BlockingChannel, method: Basic.Deliver,
               properties: pika.BasicProperties, body: bytes):
    monitoring.timer.start('preprocessing')

    source_manager_id = properties.headers['source_manager_id']
    source_id = properties.headers['source_id']
    chunk_id = properties.headers['chunk_id']
    position = properties.headers['position']
    timestamp = properties.headers['timestamp']
    box = properties.headers['box']

    image = Image.open(BytesIO(body))
    inputs = processor(images=image, return_tensors='pt', padding=True)

    monitoring.processing_duration_seconds.labels('preprocessing')\
        .observe(monitoring.timer.get('preprocessing'))
    monitoring.timer.start('inference')

    image_features = model.get_image_features(**inputs)
    image_features = image_features.detach().cpu().numpy()
    encoded = image_features[0].astype('float32').tobytes()

    monitoring.processing_duration_seconds.labels('inference')\
        .observe(monitoring.timer.get('inference'))
    monitoring.timer.start('publishing')

    frame = {
        'source_manager_id': source_manager_id,
        'source_id': source_id,
        'chunk_id': chunk_id,
        'position': position,
        'timestamp': timestamp,
        'embedding': encoded,
        'box': box,
    }
    key = f'frame:{redis.incr("frame_index")}'
    redis.hset(key, mapping=frame)
    redis.expire(key, settings.encoder.embedding_ttl)

    channel.basic_ack(delivery_tag=method.delivery_tag)

    monitoring.processing_duration_seconds.labels('publishing')\
        .observe(monitoring.timer.get('publishing'))
    monitoring.messages_processed_total.inc()
    monitoring.push_metrics()


def main():
    session.start_consuming(
        settings.rabbitmq.frame_crops_queue,
        prefetch_count=settings.rabbitmq.prefetch_count
    )
