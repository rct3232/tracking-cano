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
    llm_system_prompt: Optional[str] = None


class CameraUpdate(BaseModel):
    source: Optional[str] = None
    status: Optional[str] = None
    target_classes: Optional[List[str]] = None
    llm_system_prompt: Optional[str] = None


class CameraResponse(BaseModel):
    id: str
    source: str
    status: str
    target_classes: List[str]
    worker_state: Optional[str] = None  # "collector" | "stopped"


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


class LogLabelUpdate(BaseModel):
    is_false_positive: Optional[bool] = None
    is_false_negative: Optional[bool] = None


class LogEntryResponse(BaseModel):
    id: int
    timestamp: datetime
    log_type: str
    subject_id: Optional[str] = None
    target_present: Optional[bool] = None
    description: Optional[str] = None
    target_coordinate: Optional[str] = None
    visual_evidence: Optional[str] = None
    image_url: Optional[str] = None
    is_false_positive: bool = False
    is_false_negative: bool = False


# ── Config ────────────────────────────────────────────────────────

class LLMUpdate(BaseModel):
    api_base_url: Optional[str] = None
    model_name: Optional[str] = None
    vision_enabled: Optional[bool] = None
    vision_quality: Optional[int] = None
    vision_max_width: Optional[int] = None
    cooldown_seconds: Optional[float] = None
    json_response_format: Optional[bool] = None


# ── Status ────────────────────────────────────────────────────────

class SystemStatus(BaseModel):
    cameras: List[CameraResponse]
    spaces: List[SpaceResponse]
    uptime_seconds: float
