from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    api_base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_name: str = "gpt-4o-mini"
    vision_enabled: bool = True
    vision_quality: int = 60
    vision_max_width: int = 1024
    collect_interval: float = 0.5
    collect_count: int = 5
    max_stale_threshold: float = 10.0
    json_response_format: bool = True
    log_language: str = "en"

    @classmethod
    def from_dict(cls, d: dict) -> 'LLMConfig':
        valid = {k: v for k, v in d.items() if k in cls.__annotations__}
        return cls(**valid)


@dataclass
class LogConfig:
    db_url: str = "sqlite:///logs/tracking.db"
    log_dir: str = "logs"
    retention_hours: int = 24
    cleanup_enabled: bool = True

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
    retention_hours: int = 24
    cleanup_enabled: bool = True

    @classmethod
    def from_env(cls) -> 'MinIOConfig':
        import os
        return cls(
            endpoint=os.environ.get("MINIO_ENDPOINT", ""),
            access_key=os.environ.get("MINIO_ACCESS_KEY", ""),
            secret_key=os.environ.get("MINIO_SECRET_KEY", ""),
            bucket=os.environ.get("MINIO_BUCKET", "snapshots"),
            retention_hours=int(os.environ.get("MINIO_RETENTION_HOURS", "24")),
            cleanup_enabled=os.environ.get("MINIO_CLEANUP_ENABLED", "true").lower() != "false",
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



