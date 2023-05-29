from pathlib import Path

from pydantic import BaseModel, BaseSettings


basedir = Path(__file__).parent.parent.absolute()


class DetectorSettings(BaseModel):
    model: str = 'models/yolov8n.pt'
    device: str = 'cpu'
    conf_threshold: float = 0.6
    iou_threshold: float = 0.45
    max_detections: int = 300
    classes: list[int] = [0, 1, 2, 3, 5, 7, 15, 16, 24, 25, 26, 28, 30, 31,
                          32, 36, 39, 41, 43, 63, 66, 67, 73, 77]
    identity_threshold_soft: float = 0.3
    identity_threshold_hard: float = 0.9
    detection_ftl: int = 7


class EncoderSettings(BaseModel):
    model: str = 'RN50'
    device: str = 'cpu'


class RedisSettings(BaseModel):
    host: str = 'redis'
    port: int = 6379


class RabbitMQSettings(BaseModel):
    host: str = 'rabbitmq'
    port: int = 5672
    vhost: str = '/'
    username: str = 'guest'
    password: str = 'guest'

    video_chunks_queue: str = 'video_chunks'
    frame_crops_exchange: str = 'frame_crops'
    frame_crops_queue: str = 'frame_crops'


class PathsSettings(BaseModel):
    pass


class VideoSettings(BaseModel):
    frame_width: int = 640
    frame_height: int = 480
    frame_size: tuple[int, int] = (frame_width, frame_height)


class Settings(BaseSettings):
    detector: DetectorSettings = DetectorSettings()
    encoder: EncoderSettings = EncoderSettings()
    redis: RedisSettings = RedisSettings()
    rabbitmq: RabbitMQSettings = RabbitMQSettings()
    paths: PathsSettings = PathsSettings()
    video: VideoSettings = VideoSettings()

    class Config:
        env_nested_delimiter = '__'


settings = Settings()


# Resolve paths
# settings.paths.chunks_dir = settings.paths.chunks_dir.resolve()
# settings.paths.chunks_dir.mkdir(parents=True, exist_ok=True)
