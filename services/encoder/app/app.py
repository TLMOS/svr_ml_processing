from io import BytesIO
from PIL import Image

import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic
from redis_om import Migrator
import torch
import clip

from common.config import settings
from common.rabbitmq import PikaSession
from app.models import Embedding


session = PikaSession()

model, preprocess = clip.load(settings.encoder.model, device='cuda')


@session.on_message
def on_message(channel: BlockingChannel, method: Basic.Deliver,
               properties: pika.BasicProperties, body: bytes):
    sm_name = properties.headers['sm_name']
    source_id = int(properties.headers['source_id'])
    timestamp = float(properties.headers['timestamp'])

    image = Image.open(BytesIO(body))
    with torch.no_grad():
        image = preprocess(image).unsqueeze(0).to('cuda')
        image_features = model.encode_image(image).cpu().numpy()[0]

    embedding = Embedding(
        sm_name=sm_name,
        source_id=source_id,
        timestamp=timestamp,
        features=image_features.tobytes().hex()
    )
    embedding.save()
    Embedding.db().expire(embedding.key(), settings.encoder.embedding_ttl)
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main():
    Migrator().run()
    session.startup()
    session.consume(settings.rabbitmq.frame_crops_queue)
