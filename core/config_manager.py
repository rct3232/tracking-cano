"""Configuration manager — YAML loading, hot-reload."""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler, FileModifiedEvent

from settings import LLMConfig, LogConfig, Thresholds, YOLOConfig

logger = logging.getLogger(__name__)

# ── Data structures ────────────────────────────────────────────────

class CameraConfig:
    """Single camera configuration with resolved source URL."""

    def __init__(self, cfg: Dict[str, Any]):
        self.id: str = cfg["id"]
        self.source: str = cfg.get("source", "")
        self.status: str = cfg.get("status", "active")
        target = cfg.get("target_classes")
        if not target:
            raise ValueError(f"Camera '{cfg['id']}': 'target_classes' is required")
        self.target_classes: List[str] = target
        self.interaction_classes: Optional[List[str]] = cfg.get("interaction_classes", None)
        self.model_size: str = cfg.get("model_size", "s")
        self.model_path: Optional[str] = cfg.get("model_path", None)
        self.quantize: bool = cfg.get("quantize", False)
        self.frame_skip: int = cfg.get("frame_skip", 0)
        self.llm_system_prompt: Optional[str] = cfg.get("llm_system_prompt", None)

    def __repr__(self) -> str:
        return f"CameraConfig(id={self.id!r}, source={self.source!r}, status={self.status!r})"


class SpaceConfig:
    """A space (room/area) with associated cameras."""

    def __init__(self, cfg: Dict[str, Any]):
        self.id: str = cfg["id"]
        self.name: str = cfg.get("name", self.id)
        self.camera_ids: List[str] = cfg.get("cameras", [])
        self.llm_system_prompt: Optional[str] = cfg.get("llm_system_prompt", None)

    def __repr__(self) -> str:
        return f"SpaceConfig(id={self.id!r}, name={self.name!r})"


class AppConfig:
    """Full application configuration loaded from YAML."""

    def __init__(
        self,
        cameras: List[CameraConfig],
        spaces: List[SpaceConfig],
        thresholds: Thresholds,
        yolo: YOLOConfig,
        llm: LLMConfig,
        log: LogConfig,
        mode: str = "cv_pipeline",
    ):
        self.cameras = cameras
        self.spaces = spaces
        self.thresholds = thresholds
        self.yolo = yolo
        self.llm = llm
        self.log = log
        self.mode = mode

# ── Loading ────────────────────────────────────────────────────────

def load_config(
    config_path: str = "configuration.yaml",
) -> AppConfig:
    """Load and validate the configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed AppConfig with all settings resolved.
    """
    raw = _read_yaml(config_path)

    thresholds = Thresholds.from_dict(raw.get("thresholds", {}))
    yolo = YOLOConfig.from_dict(raw.get("yolo", {}))
    llm = LLMConfig.from_dict(raw.get("llm", {}))
    log = LogConfig.from_dict(raw.get("logging", {}))
    mode = raw.get("mode", "cv_pipeline")

    # 12-factor: env overrides for secrets / deployment-specific values
    llm.api_key = os.environ.get("API_KEY", llm.api_key)
    if "DATABASE_URL" in os.environ:
        log.db_url = os.environ["DATABASE_URL"]

    cameras = _parse_cameras(raw)
    spaces = _parse_spaces(raw)

    logger.info(
        "Config loaded: %d cameras, %d spaces, mode=%s",
        len(cameras),
        len(spaces),
        mode,
    )
    return AppConfig(cameras, spaces, thresholds, yolo, llm, log, mode)


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


def _parse_cameras(raw: Dict[str, Any]) -> List[CameraConfig]:
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
        cameras.append(CameraConfig(item))
    return cameras


def _parse_spaces(raw: Dict[str, Any]) -> List[SpaceConfig]:
    """Parse the spaces list from raw config."""
    space_list = raw.get("spaces", [])
    if not isinstance(space_list, list):
        raise ValueError("'spaces' must be a list")
    return [SpaceConfig(item) for item in space_list]


# ── Diff ────────────────────────────────────────────────────────────

class ConfigDiff:
    added_cameras: Set[str]
    removed_cameras: Set[str]
    added_spaces: Set[str]
    removed_spaces: Set[str]
    reassigned_cameras: Dict[str, tuple[str, str]]

    def __init__(self):
        self.added_cameras = set()
        self.removed_cameras = set()
        self.added_spaces = set()
        self.removed_spaces = set()
        self.reassigned_cameras = {}

    @property
    def is_empty(self) -> bool:
        return not (self.added_cameras or self.removed_cameras or self.added_spaces or self.removed_spaces)

    def __repr__(self) -> str:
        return (
            f"ConfigDiff(added_cameras={self.added_cameras}, "
            f"removed_cameras={self.removed_cameras}, "
            f"added_spaces={self.added_spaces}, "
            f"removed_spaces={self.removed_spaces})"
        )


def _build_cam_to_space(app_config: AppConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for space in app_config.spaces:
        for cam_id in space.camera_ids:
            mapping[cam_id] = space.id
    return mapping


def diff_configs(old: AppConfig, new: AppConfig) -> ConfigDiff:
    old_cam_ids = {c.id for c in old.cameras}
    new_cam_ids = {c.id for c in new.cameras}
    old_space_ids = {s.id for s in old.spaces}
    new_space_ids = {s.id for s in new.spaces}
    diff = ConfigDiff()
    diff.added_cameras = new_cam_ids - old_cam_ids
    diff.removed_cameras = old_cam_ids - new_cam_ids
    diff.added_spaces = new_space_ids - old_space_ids
    diff.removed_spaces = old_space_ids - new_space_ids

    # Detect camera reassignments (same ID, different space)
    old_cam_to_space = _build_cam_to_space(old)
    new_cam_to_space = _build_cam_to_space(new)
    for cam_id in old_cam_ids & new_cam_ids:
        old_space = old_cam_to_space.get(cam_id)
        new_space = new_cam_to_space.get(cam_id)
        if old_space != new_space:
            diff.reassigned_cameras[cam_id] = (old_space, new_space)

    return diff


# ── Hot-reload ──────────────────────────────────────────────────────

class ConfigWatcher:
    def __init__(self, config_path: str, on_change: Callable[[AppConfig, ConfigDiff], None]):
        self.config_path = Path(config_path)
        self.on_change = on_change
        self.observer = Observer()
        self._last_config: Optional[AppConfig] = None

    def start(self):
        self._last_config = load_config(str(self.config_path))
        self.observer.schedule(
            _ConfigEventHandler(self.config_path, self._on_modified),
            str(self.config_path.parent),
            recursive=False,
        )
        self.observer.start()
        logger.info("ConfigWatcher started for %s", self.config_path)

    def stop(self):
        self.observer.stop()
        self.observer.join(timeout=3)
        logger.info("ConfigWatcher stopped")

    def get_current(self) -> AppConfig:
        return self._last_config  # type: ignore[return-value]

    def _on_modified(self):
        try:
            new_config = load_config(str(self.config_path))
        except Exception as e:
            logger.error("Failed to reload config: %s", e)
            return
        if self._last_config is None:
            self._last_config = new_config
            return
        diff = diff_configs(self._last_config, new_config)
        if diff.is_empty:
            return
        logger.info("Config change detected: %s", diff)
        self._last_config = new_config
        self.on_change(new_config, diff)


class _ConfigEventHandler(FileSystemEventHandler):
    def __init__(self, config_path: Path, callback: Callable[[], None]):
        self.config_path = config_path
        self._callback = callback

    def on_modified(self, event: FileSystemEvent):
        if isinstance(event, FileModifiedEvent) and Path(event.src_path) == self.config_path:
            self._callback()
