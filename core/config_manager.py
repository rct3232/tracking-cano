"""Configuration manager — YAML loading, go2rtc URL resolution, hot-reload."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

from utils.video import resolve_source

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {
    "overlap": 0.3,
    "distance": 50,
    "speed_slow": 20,
    "speed_fast": 40,
}

DEFAULT_LLM = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "api_endpoint": "https://api.openai.com/v1",
    "temperature": 0.7,
}

# ── Data structures ────────────────────────────────────────────────

class CameraConfig:
    """Single camera configuration with resolved source URL."""

    def __init__(self, cfg: Dict[str, Any], go2rtc_url: Optional[str]):
        raw_source = cfg.get("source", "")
        self.id: str = cfg["id"]
        self.source: str = resolve_source(raw_source, go2rtc_url)
        self.status: str = cfg.get("status", "active")
        self.target_classes: List[str] = cfg.get("target_classes", [])

    def __repr__(self) -> str:
        return f"CameraConfig(id={self.id!r}, source={self.source!r}, status={self.status!r})"


class SpaceConfig:
    """A space (room/area) with associated cameras."""

    def __init__(self, cfg: Dict[str, Any]):
        self.id: str = cfg["id"]
        self.name: str = cfg.get("name", self.id)
        self.camera_ids: List[str] = cfg.get("cameras", [])

    def __repr__(self) -> str:
        return f"SpaceConfig(id={self.id!r}, name={self.name!r})"


class AppConfig:
    """Full application configuration."""

    def __init__(
        self,
        cameras: List[CameraConfig],
        spaces: List[SpaceConfig],
        thresholds: Dict[str, Any],
        llm: Dict[str, Any],
    ):
        self.cameras = cameras
        self.spaces = spaces
        self.thresholds = thresholds
        self.llm = llm

# ── Loading ────────────────────────────────────────────────────────

def load_config(
    config_path: str = "config/spaces.yaml",
    env_path: str = ".env",
) -> AppConfig:
    """Load and validate the configuration file.

    Args:
        config_path: Path to the YAML config file.
        env_path: Path to the .env file.

    Returns:
        Parsed AppConfig with resolved camera source URLs.
    """
    load_dotenv(dotenv_path=env_path, override=True)
    go2rtc_url = _get_go2rtc_url()

    raw = _read_yaml(config_path)
    cameras = _parse_cameras(raw, go2rtc_url)
    spaces = _parse_spaces(raw)
    thresholds = _parse_thresholds(raw)
    llm = _parse_llm(raw)

    logger.info(
        "Config loaded: %d cameras, %d spaces",
        len(cameras),
        len(spaces),
    )
    return AppConfig(cameras, spaces, thresholds, llm)

def _get_go2rtc_url() -> Optional[str]:
    """Read GO2RTC_URL from environment (loaded via .env)."""
    import os
    url = os.environ.get("GO2RTC_URL", "").strip()
    return url if url else None

def _read_yaml(path: str) -> Dict[str, Any]:
    """Read and parse a YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(data).__name__}")
    return data

def _parse_cameras(raw: Dict[str, Any], go2rtc_url: Optional[str]) -> List[CameraConfig]:
    """Parse the cameras list from raw config."""
    camera_list = raw.get("cameras", [])
    if not isinstance(camera_list, list):
        raise ValueError("'cameras' must be a list")

    cameras: List[CameraConfig] = []
    seen_ids: set = set()
    for item in camera_list:
        if not isinstance(item, dict) or "id" not in item or "source" not in item:
            raise ValueError(f"Each camera must have 'id' and 'source': {item}")
        cam_id = item["id"]
        if cam_id in seen_ids:
            raise ValueError(f"Duplicate camera id: {cam_id}")
        seen_ids.add(cam_id)
        cameras.append(CameraConfig(item, go2rtc_url))
    return cameras

def _parse_spaces(raw: Dict[str, Any]) -> List[SpaceConfig]:
    """Parse the spaces list from raw config."""
    space_list = raw.get("spaces", [])
    if not isinstance(space_list, list):
        raise ValueError("'spaces' must be a list")
    return [SpaceConfig(item) for item in space_list]

def _parse_thresholds(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse thresholds, falling back to defaults."""
    raw_thresholds = raw.get("thresholds", {})
    if not isinstance(raw_thresholds, dict):
        raise ValueError("'thresholds' must be a mapping")
    result = dict(DEFAULT_THRESHOLDS)
    result.update(raw_thresholds)
    return result

def _parse_llm(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse LLM config, falling back to defaults."""
    raw_llm = raw.get("llm", {})
    if not isinstance(raw_llm, dict):
        raise ValueError("'llm' must be a mapping")
    result = dict(DEFAULT_LLM)
    result.update(raw_llm)
    return result
