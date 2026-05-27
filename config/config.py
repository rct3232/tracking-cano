from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv
import os

load_dotenv()


@dataclass
class Thresholds:
    speed_slow: float = 20.0
    speed_fast: float = 40.0
    overlap: float = 0.3
    distance: float = 50.0
    dash_threshold: float = 15.0
    rotation_threshold: float = 45.0
    hysteresis: float = 5.0
    min_frames: int = 3
    surge_min_frames: int = 2


@dataclass
class YOLOConfig:
    model_path: str = "yolo26s.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.70


@dataclass
class LLMConfig:
    api_base_url: str = field(default_factory=lambda: os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", "gpt-4o-mini"))
    cooldown_seconds: float = 3.0
    language: str = "ko"


@dataclass
class PipelineConfig:
    target_classes: List[str] = field(default_factory=lambda: ["cat"])
    interaction_classes: List[str] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
