from typing import Optional, Callable

import pika
from pika import BlockingConnection
from pika.adapters.blocking_connection import BlockingChannel

from common.config import settings


class PikaSession:
    """
    Wrapper for pika connection to RabbitMQ.

    Attributes:
    - is_opened (bool): True if the connection is opened
    """

    def __init__(self):
        self._is_opened: bool = False
        self._connection: Optional[BlockingConnection] = None
        self._input_channel: Optional[BlockingChannel] = None
        self._output_channel: Optional[BlockingChannel] = None
        self._on_message: Optional[Callable] = None

    @property
    def is_opened(self):
        return self._is_opened

    def startup(self):
        self._connection = BlockingConnection(
            pika.ConnectionParameters(
                host=settings.rabbitmq.host,
                port=settings.rabbitmq.port,
                virtual_host=settings.rabbitmq.vhost,
                credentials=pika.PlainCredentials(
                    username=settings.rabbitmq.username,
                    password=settings.rabbitmq.password
                )
            )
        )
        self._is_opened = True

    def shutdown(self):
        self._connection.close()
        self._connection = None
        self._input_channel = None
        self._output_channel = None
        self._is_opened = False

    def on_message(self, func: Callable):
        """Decorator for setting message callback."""
        self._on_message = func
        return func

    def publish(self, exchange: str, routing_key: str, body: bytes,
                properties: pika.BasicProperties = None):
        """Publish message to RabbitMQ."""
        if not self._output_channel:
            self._output_channel = self._connection.channel()
        self._output_channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=properties
        )

    def consume(self, queue: str, auto_ack: bool = False):
        """
        Consume messages from RabbitMQ.

        Creates a temporary queue and binds it to the specified exchange.

        Message callback should be set via `on_message` decorator.
        """
        if not self._input_channel:
            self._input_channel = self._connection.channel()

        self._input_channel.basic_qos(prefetch_count=1)
        self._input_channel.basic_consume(
            queue=queue,
            on_message_callback=self._on_message,
            auto_ack=auto_ack
        )
        self._input_channel.start_consuming()
