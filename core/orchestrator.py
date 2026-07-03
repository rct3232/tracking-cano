import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from settings import LLMConfig, ReconnectConfig
from core.config_manager import AppConfig, CameraConfig, SpaceConfig

from core.vision_worker import _BatchCollector
from nlp.logger import CameraSnapshot, SpaceLogger

logger = logging.getLogger(__name__)

_STREAM_PREFIXES = ("rtsp://", "http://", "https://")


def _is_stream_source(source: str) -> bool:
    return source.startswith(_STREAM_PREFIXES)


class _SimpleVisionDetector:
    """Per-space independent detection threads. Each space runs its own round-robin loop concurrently."""

    def __init__(
        self,
        spaces: List[SpaceConfig],
        collectors: Dict[str, _BatchCollector],
        space_logger: SpaceLogger,
        orchestrator: "Orchestrator",
        config: LLMConfig,
        cam_to_space: Dict[str, str],
        all_finished_check=None,
    ):
        self._stop_event = threading.Event()
        self._spaces_lock = threading.Lock()
        self._spaces: Dict[str, SpaceConfig] = {s.id: s for s in spaces}
        self._collectors = collectors
        self._space_logger = space_logger
        self._orchestrator = orchestrator
        self._config = config
        self._cam_to_space = cam_to_space
        self._is_all_finished = all_finished_check or (lambda: False)
        self._detect_index: Dict[str, int] = {}  # space_id → next camera index
        for sid in self._spaces:
            self._detect_index[sid] = 0
        self._space_threads: Dict[str, threading.Thread] = {}

    def add_space(self, space: SpaceConfig):
        with self._spaces_lock:
            self._spaces[space.id] = space
        self._detect_index[space.id] = 0
        if not self._stop_event.is_set():
            t = threading.Thread(target=self._run_space, args=(space.id,), daemon=True, name=f"detect-{space.id}")
            t.start()
            self._space_threads[space.id] = t

    def remove_space(self, space_id: str):
        with self._spaces_lock:
            self._spaces.pop(space_id, None)
        self._detect_index.pop(space_id, None)

    def remove_camera_from_space(self, space_id: str, cam_id: str):
        space = self._spaces.get(space_id)
        if space and cam_id in space.camera_ids:
            space.camera_ids = [c for c in space.camera_ids if c != cam_id]

    def add_camera_to_space(self, space_id: str, cam_id: str):
        space = self._spaces.get(space_id)
        if space and cam_id not in space.camera_ids:
            space.camera_ids.append(cam_id)

    def start(self):
        for space_id in list(self._spaces.keys()):
            t = threading.Thread(target=self._run_space, args=(space_id,), daemon=True, name=f"detect-{space_id}")
            t.start()
            self._space_threads[space_id] = t
        logger.debug("[vision-detector] started with %d space threads", len(self._space_threads))

    def stop(self):
        self._stop_event.set()
        for t in self._space_threads.values():
            t.join(timeout=5)
        self._space_threads.clear()

    def _run_space(self, space_id: str):
        while not self._stop_event.is_set():
            try:
                if self._is_all_finished():
                    self._stop_event.wait(1.0)
                    continue

                with self._spaces_lock:
                    space = self._spaces.get(space_id)
                if not space or not space.camera_ids:
                    self._stop_event.wait(1.0)
                    continue

                idx = self._detect_index.get(space_id, 0)
                if idx >= len(space.camera_ids):
                    self._detect_index[space_id] = 0
                    self._stop_event.wait(0.1)
                    continue

                cam_id = space.camera_ids[idx]
                collector = self._collectors.get(cam_id)
                if not collector or not collector.buffer:
                    self._detect_index[space_id] = idx + 1
                    self._stop_event.wait(0.5)
                    continue

                entry = collector.buffer[-1]
                age = time.monotonic() - entry.captured_at
                health = "healthy"
                if age > self._config.max_stale_threshold:
                    health = "degraded"
                if health != "healthy":
                    self._detect_index[space_id] = idx + 1
                    self._stop_event.wait(0.5)
                    continue

                # freeze: T0 시점 모든 카메라 buffer 복사
                frozen_buffers: Dict[str, List] = {}
                for cid in space.camera_ids:
                    coll = self._collectors.get(cid)
                    if coll and coll.buffer:
                        frozen_buffers[cid] = list(coll.buffer)

                logger.debug("[vision-detector:%s] cam=%s calling vision_detect", space_id, cam_id)
                result = self._space_logger.vision_detect(
                    camera_id=cam_id,
                    image_b64=entry.image_b64,
                    llm_system_prompt=space.llm_system_prompt or None,
                    target_classes=self._get_target_classes(space_id) or None,
                )
                if result is None:
                    logger.debug("[vision-detector:%s] cam=%s vision_detect returned None → skip", space_id, cam_id)
                    self._detect_index[space_id] = idx + 1
                    continue

                if result.get("target_present", False):
                    logger.info("[space:%s] cam=%s target_present=True → snapshot", space_id, cam_id)
                    self._update_all_snapshots(space_id, frozen_buffers, detect_cam_id=cam_id, detect_entry=entry)
                    detect_context = {
                        "camera": cam_id,
                        "summarize": result.get("summarize", ""),
                    }
                    self._orchestrator.request_space_snapshot(space_id, detect_context=detect_context)

                self._detect_index[space_id] = idx + 1
            except Exception:
                logger.exception("[vision-detector:%s] _run_space crashed", space_id)
                return

    def _get_target_classes(self, space_id: str) -> List[str]:
        """Collect unique target_classes from all cameras in a space."""
        seen: List[str] = []
        for cam in self._orchestrator.app_config.cameras:
            if self._cam_to_space.get(cam.id) == space_id and cam.target_classes:
                for cls in cam.target_classes:
                    if cls not in seen:
                        seen.append(cls)
        return seen

    def _update_all_snapshots(self, space_id: str, frozen_buffers: Dict[str, List],
                               detect_cam_id: str | None = None, detect_entry=None):
        space = self._spaces.get(space_id)
        if not space:
            return
        for cam_id in space.camera_ids:
            images: List[str] = []
            if cam_id == detect_cam_id and detect_entry is not None:
                images = [detect_entry.image_b64]
                other = [e.image_b64 for e in frozen_buffers.get(cam_id, []) if e is not detect_entry]
                images.extend(other)
            else:
                frozen = frozen_buffers.get(cam_id)
                if not frozen:
                    continue
                images = [e.image_b64 for e in frozen]
            if not images:
                continue
            snap = CameraSnapshot(
                camera_id=cam_id,
                target_present=True,
                timestamp=time.monotonic(),
                image_b64=images[0],
                images=images,
            )
            self._orchestrator.update_snapshot(cam_id, snap)





class Orchestrator:
    def __init__(self, app_config: AppConfig, space_logger: Optional[SpaceLogger] = None,
                 repo=None, event_bus=None):
        self.app_config = app_config
        self.space_logger = space_logger
        self._repo = repo
        self._event_bus = event_bus
        self._lock = threading.Lock()
        self._cam_to_space: Dict[str, str] = _build_cam_to_space(app_config)
        self._collectors: Dict[str, _BatchCollector] = {}
        self._vision_detector: Optional[_SimpleVisionDetector] = None
        self._snapshots: Dict[str, CameraSnapshot] = {}
        self._snapshot_debounce: Dict[str, float] = {}
        self._snapshot_lock = threading.Lock()

    def update_snapshot(self, camera_id: str, snapshot: CameraSnapshot):
        with self._snapshot_lock:
            self._snapshots[camera_id] = snapshot

    def request_space_snapshot(self, space_id: str, detect_context=None):
        if not self.space_logger:
            return

        now = time.monotonic()
        last = self._snapshot_debounce.get(space_id, 0)
        if now - last < 5.0:
            logger.debug("[space:%s] snapshot debounce", space_id)
            return
        self._snapshot_debounce[space_id] = now

        with self._lock:
            space_cameras = [cam_id for cam_id, sid in self._cam_to_space.items() if sid == space_id]
        if not space_cameras:
            return

        snapshots: Dict[str, CameraSnapshot] = {}
        with self._snapshot_lock:
            for cam_id in space_cameras:
                snap = self._snapshots.get(cam_id)
                if snap is None or not snap.target_present:
                    continue
                snapshots[cam_id] = snap
        if not snapshots:
            return

        space_obj = next((s for s in self.app_config.spaces if s.id == space_id), None)
        space_name = space_obj.name if space_obj else space_id

        target_classes: List[str] = []
        for cam in self.app_config.cameras:
            if self._cam_to_space.get(cam.id) == space_id and cam.target_classes:
                for cls in cam.target_classes:
                    if cls not in target_classes:
                        target_classes.append(cls)

        self.space_logger.space_snapshot(
            space_id=space_id,
            space_name=space_name,
            snapshots=snapshots,
            vision_enabled=self.app_config.llm.vision_enabled,
            target_classes=target_classes or None,
            llm_system_prompt=space_obj.llm_system_prompt if space_obj else None,
            detect_context=detect_context,
        )

    def update_config(self, new_config: AppConfig):
        self.app_config = new_config
        self._cam_to_space = _build_cam_to_space(new_config)
        if self.space_logger:
            self.space_logger.config = new_config.llm
            self.space_logger.client = None
        if self._vision_detector:
            self._vision_detector._config = new_config.llm
            self._vision_detector._cam_to_space = self._cam_to_space
            with self._vision_detector._spaces_lock:
                for s in new_config.spaces:
                    self._vision_detector._spaces[s.id] = s
        self._snapshot_debounce.clear()

    @property
    def spaces(self) -> list[SpaceConfig]:
        return self.app_config.spaces

    @property
    def all_finished(self) -> bool:
        with self._lock:
            return len(self._collectors) == 0

    def start(self):
        for cam in self.app_config.cameras:
            if cam.status != "active":
                logger.info("Skipping inactive camera: %s", cam.id)
                continue
            self.add_camera(cam)

        logger.debug("[init] space_logger=%r all_spaces=%d", self.space_logger is not None, len(self.app_config.spaces))
        if self.space_logger:
            self._vision_detector = _SimpleVisionDetector(
                spaces=self.app_config.spaces,
                collectors=self._collectors,
                space_logger=self.space_logger,
                orchestrator=self,
                config=self.app_config.llm,
                cam_to_space=self._cam_to_space,
                all_finished_check=lambda: self.all_finished,
            )
            self._vision_detector.start()

    def add_camera(self, camera: CameraConfig, start_event: threading.Event | None = None,
                    capture_start: float | None = None):
        space_id = self._cam_to_space.get(camera.id)
        stop_event = threading.Event()
        rc = self.app_config.reconnect
        config = self.app_config.llm

        def _on_collector_finished(cid: str):
            with self._lock:
                self._collectors.pop(cid, None)
            logger.info("Collector %s removed from orchestrator", cid)

        collector = _BatchCollector(
            camera_id=camera.id,
            source=camera.source,
            stop_event=stop_event,
            collect_interval=config.collect_interval,
            collect_count=config.collect_count,
            vision_quality=config.vision_quality,
            vision_max_width=config.vision_max_width,
            on_finished=_on_collector_finished,
            start_event=start_event,
            capture_start=capture_start,
            reconnect=rc,
        )
        with self._lock:
            self._collectors[camera.id] = collector
        collector.start()
        logger.info("Batch collector %s started (space=%s)", camera.id, space_id)

    def remove_camera(self, camera_id: str):
        with self._lock:
            collector = self._collectors.pop(camera_id, None)
        if collector:
            collector.stop()
        self._cam_to_space.pop(camera_id, None)
        with self._snapshot_lock:
            self._snapshots.pop(camera_id, None)
        if self._event_bus:
            self._event_bus.publish({"type": "camera.removed", "data": {"camera_id": camera_id}})

    def reassign_camera(self, camera_id: str, old_space_id: str, new_space_id: str):
        self._cam_to_space[camera_id] = new_space_id
        with self._snapshot_lock:
            self._snapshots.pop(camera_id, None)
        with self._lock:
            collector = self._collectors.pop(camera_id, None)
        if collector:
            collector.stop()
        cam = next((c for c in self.app_config.cameras if c.id == camera_id), None)
        if cam:
            self.add_camera(cam)

    def get_space_cameras(self, space_id: str) -> list[str]:
        for space in self.spaces:
            if space.id == space_id:
                return list(space.camera_ids)
        return []

    def stop(self):
        if self._vision_detector:
            self._vision_detector.stop()
        with self._lock:
            collectors = list(self._collectors.values())
        for c in collectors:
            c.stop()
        self._collectors.clear()
        logger.info("All cameras stopped")


def _build_cam_to_space(app_config: AppConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for space in app_config.spaces:
        for cam_id in space.camera_ids:
            mapping[cam_id] = space.id
    return mapping
