from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Thresholds:
    speed_slow: float = field(default_factory=lambda: float(os.getenv("SPEED_SLOW", "20.0")))
    speed_fast: float = field(default_factory=lambda: float(os.getenv("SPEED_FAST", "40.0")))
    overlap: float = field(default_factory=lambda: float(os.getenv("OVERLAP", "0.3")))
    distance: float = field(default_factory=lambda: float(os.getenv("DISTANCE", "50.0")))
    dash_threshold: float = field(default_factory=lambda: float(os.getenv("DASH_THRESHOLD", "15.0")))
    rotation_threshold: float = field(default_factory=lambda: float(os.getenv("ROTATION_THRESHOLD", "45.0")))
    hysteresis: float = field(default_factory=lambda: float(os.getenv("HYSTERESIS", "5.0")))
    min_frames: int = field(default_factory=lambda: int(os.getenv("MIN_FRAMES", "3")))


@dataclass
class YOLOConfig:
    model_size: str = "s"
    model_path: Optional[str] = None
    quantize: bool = False
    frame_skip: int = field(default_factory=lambda: int(os.getenv("FRAME_SKIP", "0")))
    conf_threshold: float = field(default_factory=lambda: float(os.getenv("CONF_THRESHOLD", "0.25")))
    iou_threshold: float = field(default_factory=lambda: float(os.getenv("IOU_THRESHOLD", "0.70")))
    yolo_classes: Optional[List[str]] = None
    tile_grid_x: int = field(default_factory=lambda: int(os.getenv("TILE_GRID_X", "2")))
    tile_grid_y: int = field(default_factory=lambda: int(os.getenv("TILE_GRID_Y", "2")))
    tile_overlap: int = field(default_factory=lambda: int(os.getenv("TILE_OVERLAP", "20")))
    tile_enabled: bool = field(default_factory=lambda: os.getenv("TILE_ENABLED", "0") == "1")


@dataclass
class LLMConfig:
    api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    cooldown_seconds: float = 3.0
    vision_enabled: bool = field(default_factory=lambda: os.getenv("VISION_ENABLED", "1") == "1")
    vision_quality: int = field(default_factory=lambda: int(os.getenv("VISION_QUALITY", "60")))
    vision_max_width: int = field(default_factory=lambda: int(os.getenv("VISION_MAX_WIDTH", "1024")))
    snapshot_count: int = field(default_factory=lambda: int(os.getenv("VISION_SNAPSHOT_COUNT", "5")))
    snapshot_interval: float = field(default_factory=lambda: float(os.getenv("VISION_INTERVAL_SECONDS", "30")))


@dataclass
class PipelineConfig:
    target_classes: List[str] = field(default_factory=lambda: ["cat"])
    interaction_classes: Optional[List[str]] = None
    thresholds: Thresholds = field(default_factory=Thresholds)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_system_prompt: Optional[str] = None
