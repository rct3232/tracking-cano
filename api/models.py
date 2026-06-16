"""Pydantic models for REST API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# ── Camera ────────────────────────────────────────────────────────

class CameraCreate(BaseModel):
    id: str
    source: str
    target_classes: List[str]
    status: str = "active"
    interaction_classes: Optional[List[str]] = None
    model_size: str = "s"
    model_path: Optional[str] = None
    quantize: bool = False
    frame_skip: int = 0
    llm_system_prompt: Optional[str] = None


class CameraUpdate(BaseModel):
    source: Optional[str] = None
    status: Optional[str] = None
    target_classes: Optional[List[str]] = None
    interaction_classes: Optional[List[str]] = None
    model_size: Optional[str] = None
    model_path: Optional[str] = None
    quantize: Optional[bool] = None
    frame_skip: Optional[int] = None
    llm_system_prompt: Optional[str] = None


class CameraResponse(BaseModel):
    id: str
    source: str
    status: str
    target_classes: List[str]
    interaction_classes: Optional[List[str]] = None
    model_size: str
    frame_skip: int
    worker_state: Optional[str] = None  # "running" | "stopped" | "collector"


# ── Space ─────────────────────────────────────────────────────────

class SpaceCreate(BaseModel):
    id: str
    name: str
    cameras: List[str]
    llm_system_prompt: Optional[str] = None


class SpaceUpdate(BaseModel):
    name: Optional[str] = None
    cameras: Optional[List[str]] = None
    llm_system_prompt: Optional[str] = None


class SpaceResponse(BaseModel):
    id: str
    name: str
    cameras: List[str]
    state: Optional[str] = None  # "detecting" | "logging" | "cooling"


# ── Logs ──────────────────────────────────────────────────────────

class LogQuery(BaseModel):
    log_type: Optional[str] = None
    subject_id: Optional[str] = None
    limit: int = 100
    offset: int = 0


class LogEntryResponse(BaseModel):
    id: int
    timestamp: datetime
    log_type: str
    subject_id: Optional[str] = None
    target_present: Optional[bool] = None
    description: Optional[str] = None
    target_coordinate: Optional[str] = None


# ── Config ────────────────────────────────────────────────────────

class ThresholdsUpdate(BaseModel):
    speed_slow: Optional[float] = None
    speed_fast: Optional[float] = None
    overlap: Optional[float] = None
    distance: Optional[float] = None
    dash_threshold: Optional[float] = None
    rotation_threshold: Optional[float] = None
    hysteresis: Optional[float] = None


class YOLOUpdate(BaseModel):
    conf_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    tile_enabled: Optional[bool] = None
    tile_grid_x: Optional[int] = None
    tile_grid_y: Optional[int] = None
    tile_overlap: Optional[int] = None
    frame_skip: Optional[int] = None


class LLMUpdate(BaseModel):
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    vision_enabled: Optional[bool] = None
    vision_quality: Optional[int] = None
    vision_max_width: Optional[int] = None
    snapshot_count: Optional[int] = None
    snapshot_interval: Optional[float] = None
    cooldown_seconds: Optional[float] = None
    early_trigger: Optional[float] = None
    json_response_format: Optional[bool] = None


# ── Status ────────────────────────────────────────────────────────

class SystemStatus(BaseModel):
    mode: str
    cameras: List[CameraResponse]
    spaces: List[SpaceResponse]
    uptime_seconds: float
