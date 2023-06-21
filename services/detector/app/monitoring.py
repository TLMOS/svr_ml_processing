import os
import time

from prometheus_client import (
    CollectorRegistry,
    push_to_gateway,
    Summary,
    Counter,
    Histogram,
)

from common.config import settings


HOSTNAME = os.environ['HOSTNAME']
NAMESPACE = 'ml_processing_detector'


class Timer:
    def __init__(self):
        self.start_times = {}

    def start(self, name):
        self.start_times[name] = time.perf_counter()

    def get(self, name):
        return time.perf_counter() - self.start_times[name]


timer = Timer()


def push_metrics():
    if settings.monitoring.send_metrics:
        push_to_gateway(
            settings.monitoring.url,
            job=HOSTNAME,
            registry=registry,
        )


registry = CollectorRegistry()


processing_duration_seconds = Summary(
    name='processing_duration_seconds',
    documentation='Duration of processing stages',
    namespace=NAMESPACE,
    labelnames=['stage'],
    registry=registry,
)

messages_processed_total = Counter(
    name='messages_processed_total',
    documentation='Total number of processed messages',
    namespace=NAMESPACE,
    labelnames=[],
    registry=registry,
)

detections_count = Summary(
    name='detections_count',
    documentation='Number of detections per frame',
    namespace=NAMESPACE,
    labelnames=[],
    registry=registry,
)

detection_labels = Histogram(
    name='detection_labels',
    documentation='Labels of detections',
    namespace=NAMESPACE,
    labelnames=[],
    buckets=settings.detector.classes,
    registry=registry,
)

chunk_fps = Summary(
    name='chunk_fps',
    documentation='Frames per second in a chunk',
    namespace=NAMESPACE,
    labelnames=[],
    registry=registry,
)
