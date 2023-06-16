from typing import Optional, Literal
from pathlib import Path

from pydantic import (
    BaseModel,
    BaseSettings,
    Field,
    PositiveInt,
    FilePath
)


basedir = Path(__file__).parent.parent.absolute()


class DetectorSettings(BaseModel):
    model: str = FilePath | Literal['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt,',
                                    'yolov8l.pt', 'yolov8x.pt']
    input_width: int = Field(640, ge=28, le=1920)
    input_height: int = Field(480, ge=28, le=1080)
    conf_threshold: float = Field(0.4, ge=0, le=1)
    iou_threshold: float = Field(0.7, ge=0, le=1)
    max_detections: PositiveInt = 100
    min_box_size: PositiveInt = 48
    agnostic_nms: bool = True
    classes: list[int] = [0, 1, 2, 3, 5, 7, 15, 16, 24, 25, 26, 28, 30, 31,
                          32, 36, 39, 41, 43, 63, 66, 67, 73, 77]

    class_hierarchy = {
        0: [1, 3, 24, 25, 26, 28, 30, 31, 32, 36,
            39, 41, 43, 63, 66, 67, 73, 77]
    }
    ioa_class_hierarchy_threshold: float = Field(0.3, ge=0, le=1)

    iou_identity_threshold_soft: float = Field(0.45, ge=0, le=1)
    iou_identity_threshold_hard: float = Field(0.85, ge=0, le=1)

    detection_ftl: PositiveInt = 2
    detection_ttl: PositiveInt = 60


class EncoderSettings(BaseModel):
    model: str = 'openai/clip-vit-base-patch32'
    embedding_ttl: int = 30 * 24 * 60 * 60


class RedisSettings(BaseModel):
    host: str = '0.0.0.0'
    port: PositiveInt = 6379
    db: PositiveInt = 0
    username: Optional[str] = None
    password: Optional[str] = None


class RabbitMQSettings(BaseModel):
    host: str = '0.0.0.0'
    port: PositiveInt = 5672
    virtual_host: str = '/'
    ml_processing_username: str = 'ml_processing'
    ml_processing_password: str = 'ml_processing'

    video_chunks_queue: str = 'video_chunks'
    frame_crops_exchange: str = 'frame_crops'
    frame_crops_queue: str = 'frame_crops'

    prefetch_count: PositiveInt = 3


class Settings(BaseSettings):
    detector: DetectorSettings = DetectorSettings()
    encoder: EncoderSettings = EncoderSettings()
    redis: RedisSettings = RedisSettings()
    rabbitmq: RabbitMQSettings = RabbitMQSettings()

    class Config:
        env_nested_delimiter = '__'


settings = Settings()
