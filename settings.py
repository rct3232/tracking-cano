from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Thresholds:
    speed_slow: float = 20.0
    speed_fast: float = 40.0
    overlap: float = 0.3
    distance: float = 50.0
    dash_threshold: float = 15.0
    rotation_threshold: float = 45.0
    hysteresis: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> 'Thresholds':
        valid = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class YOLOConfig:
    model_size: str = "s"
    model_path: Optional[str] = None
    quantize: bool = False
    frame_skip: int = 0
    conf_threshold: float = 0.25
    iou_threshold: float = 0.70
    yolo_classes: Optional[List[str]] = None
    tile_grid_x: int = 2
    tile_grid_y: int = 2
    tile_overlap: int = 20
    tile_enabled: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'YOLOConfig':
        valid = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class LLMConfig:
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    vision_enabled: bool = True
    vision_quality: int = 60
    vision_max_width: int = 1024
    snapshot_count: int = 5
    snapshot_interval: float = 30.0
    collect_interval: float = 0.5
    collect_count: int = 5
    max_stale_threshold: float = 10.0
    cooldown_seconds: float = 30.0
    early_trigger: float = 5.0
    json_response_format: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> 'LLMConfig':
        valid = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class LogConfig:
    db_url: str = "sqlite:///logs/tracking.db"
    log_dir: str = "logs"

    @classmethod
    def from_dict(cls, d: dict) -> 'LogConfig':
        valid = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class MinIOConfig:
    endpoint: str = ""
    access_key: str = ""
    secret_key: str = ""
    bucket: str = "snapshots"

    @classmethod
    def from_env(cls) -> 'MinIOConfig':
        import os
        return cls(
            endpoint=os.environ.get("MINIO_ENDPOINT", ""),
            access_key=os.environ.get("MINIO_ACCESS_KEY", ""),
            secret_key=os.environ.get("MINIO_SECRET_KEY", ""),
            bucket=os.environ.get("MINIO_BUCKET", "snapshots"),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.access_key and self.secret_key)


@dataclass
class ReconnectConfig:
    max_failures: int = 5
    base_delay: float = 1.0
    max_delay: float = 300.0
    read_backoff: float = 0.5
    reconnect_backoff: float = 2.0
    pts_lag_threshold: float = 60.0

    def __eq__(self, other):
        if not isinstance(other, ReconnectConfig):
            return NotImplemented
        return (
            self.max_failures == other.max_failures
            and self.base_delay == other.base_delay
            and self.max_delay == other.max_delay
            and self.read_backoff == other.read_backoff
            and self.reconnect_backoff == other.reconnect_backoff
            and self.pts_lag_threshold == other.pts_lag_threshold
        )

    @classmethod
    def from_dict(cls, d: dict) -> 'ReconnectConfig':
        valid = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class PipelineConfig:
    target_classes: List[str] = field(default_factory=lambda: ["cat"])
    interaction_classes: Optional[List[str]] = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_system_prompt: Optional[str] = None
