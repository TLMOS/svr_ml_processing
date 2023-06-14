from typing import Optional, Callable

import pika
from pika import BlockingConnection, BasicProperties, ConnectionParameters
from pika.adapters.blocking_connection import BlockingChannel


class Session:
    """
    Simple wrapper for pika blocking connection.

    Attributes:
    - is_opened (bool): True if the connection is opened
    """

    def __init__(self):
        self._connection_params: Optional[ConnectionParameters] = None
        self._connection: Optional[BlockingConnection] = None
        self._output_channels: dict[str, BlockingChannel] = {}
        self._on_message_callback: Optional[Callable] = None

    @property
    def is_opened(self) -> bool:
        """Check if connection is opened."""
        if self._connection and self._connection.is_open:
            return True
        return False

    def set_connection_params(self, host: str, port: int,
                              virtual_host: str, username: str,
                              password: str):
        """Set connection parameters."""
        self._connection_params = ConnectionParameters(
            host=host,
            port=port,
            virtual_host=virtual_host,
            credentials=pika.PlainCredentials(username, password),
            heartbeat=600,
            blocked_connection_timeout=300,
        )

    def open(self):
        """Open connection to RabbitMQ."""
        self._connection = BlockingConnection(self._connection_params)

    def close(self):
        """Safely close connection to RabbitMQ."""
        try:
            self._connection.close()
        except Exception:
            pass  # Connection is already closed
        self._connection = None
        self._output_channels = {}

    def ensure_connection(self):
        """Ensure connection to RabbitMQ."""
        if not self.is_opened:
            self.close()
            self.open()

    def on_message(self, func: Callable):
        """Decorator for setting message callback."""
        self._on_message_callback = func
        return func

    def publish(self, exchange: str, routing_key: str, body: bytes,
                properties: BasicProperties = None):
        """Publish message to RabbitMQ."""
        self.ensure_connection()
        if exchange not in self._output_channels:
            self._output_channels[exchange] = self._connection.channel()
        self._output_channels[exchange].basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=properties
        )

    def start_consuming(
        self,
        queue: str,
        auto_ack: bool = False,
        prefetch_count: int = 1
    ):
        """
        Consume messages from RabbitMQ.

        Message callback should be set via `on_message` decorator.
        """
        self.ensure_connection()
        input_channel = self._connection.channel()
        input_channel.basic_qos(prefetch_count=prefetch_count)
        input_channel.basic_consume(
            queue=queue,
            on_message_callback=self._on_message_callback,
            auto_ack=auto_ack
        )
        input_channel.start_consuming()
