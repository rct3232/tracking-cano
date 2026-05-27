import logging
import threading
import time
from typing import Dict, List, Optional

from config.config import LLMConfig, PipelineConfig, Thresholds, YOLOConfig
from core.config_manager import AppConfig, CameraConfig, SpaceConfig
from core.pipeline import Pipeline
from nlp.logger import SpaceLogger
from utils.video import create_capture

logger = logging.getLogger(__name__)


class _CameraWorker:
    def __init__(self, camera_id: str, pipeline: Pipeline, cap, source: str, stop_event: threading.Event):
        self.camera_id = camera_id
        self.pipeline = pipeline
        self.cap = cap
        self.source = source
        self.stop_event = stop_event
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{camera_id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.cap.release()
        logger.info("Camera %s stopped", self.camera_id)

    def _run(self):
        frame_id = 0
        consecutive_failures = 0
        max_failures = 5
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    logger.warning("Camera %s read failure (1/%d), source=%s", self.camera_id, max_failures, self.source)
                if consecutive_failures >= max_failures:
                    logger.warning("Camera %s reconnecting... (%d consecutive failures)", self.camera_id, consecutive_failures)
                    self.cap.release()
                    new_cap = create_capture(self.source)
                    if new_cap is None:
                        logger.error("Camera %s reconnect failed for %s", self.camera_id, self.source)
                        time.sleep(2)
                        continue
                    self.cap = new_cap
                    consecutive_failures = 0
                else:
                    time.sleep(0.5)
                continue
            consecutive_failures = 0
            result = self.pipeline.process_frame(frame, frame_id)
            if result:
                logger.info("[%s] %s", self.camera_id, result)
            frame_id += 1


def _make_pipeline_config(camera: CameraConfig) -> PipelineConfig:
    thresholds = Thresholds()
    llm = LLMConfig()
    yolo = YOLOConfig()
    return PipelineConfig(
        target_classes=camera.target_classes,
        interaction_classes=camera.interaction_classes,
        thresholds=thresholds,
        yolo=yolo,
        llm=llm,
    )


class Orchestrator:
    def __init__(self, app_config: AppConfig, space_logger: Optional[SpaceLogger] = None):
        self.app_config = app_config
        self.space_logger = space_logger
        self._workers: Dict[str, _CameraWorker] = {}
        self._lock = threading.Lock()
        self._cam_to_space: Dict[str, str] = _build_cam_to_space(app_config)
        if self.space_logger:
            for space in app_config.spaces:
                self.space_logger.set_camera_count(space.id, len(space.camera_ids))

    @property
    def spaces(self) -> list[SpaceConfig]:
        return self.app_config.spaces

    def start(self):
        for cam in self.app_config.cameras:
            if cam.status != "active":
                logger.info("Skipping inactive camera: %s", cam.id)
                continue
            self.add_camera(cam)

    def add_camera(self, camera: CameraConfig):
        space_id = self._cam_to_space.get(camera.id)
        cap = create_capture(camera.source)
        if cap is None:
            logger.error("Cannot open camera %s from %s", camera.id, camera.source)
            return
        config = _make_pipeline_config(camera)
        pipeline = Pipeline(config, camera.id, self.space_logger, space_id)
        stop_event = threading.Event()
        worker = _CameraWorker(camera.id, pipeline, cap, camera.source, stop_event)
        with self._lock:
            self._workers[camera.id] = worker
        worker.start()
        logger.info("Camera %s started (%s, space=%s)", camera.id, camera.source, space_id)

    def remove_camera(self, camera_id: str):
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
        self._cam_to_space.pop(camera_id, None)

    def reassign_camera(self, camera_id: str, old_space_id: str, new_space_id: str):
        self._cam_to_space[camera_id] = new_space_id
        if self.space_logger:
            self.space_logger.flush(old_space_id, old_space_id)
            old_space = next((s for s in self.spaces if s.id == old_space_id), None)
            new_space = next((s for s in self.spaces if s.id == new_space_id), None)
            if old_space:
                self.space_logger.set_camera_count(old_space_id, len(old_space.camera_ids))
            if new_space:
                self.space_logger.set_camera_count(new_space_id, len(new_space.camera_ids))
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker:
            worker.stop()
        cam = next((c for c in self.app_config.cameras if c.id == camera_id), None)
        if cam:
            self.add_camera(cam)

    def get_space_cameras(self, space_id: str) -> list[str]:
        for space in self.spaces:
            if space.id == space_id:
                return space.camera_ids
        return []

    def flush_spaces(self):
        if not self.space_logger:
            return
        for space in self.spaces:
            text = self.space_logger.flush(space.id, space.name)
            if text:
                logger.info("[%s] %s", space.id, text)

    def stop(self):
        with self._lock:
            workers = list(self._workers.values())
        for w in workers:
            w.stop()
        self._workers.clear()
        logger.info("All cameras stopped")


def _build_cam_to_space(app_config: AppConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for space in app_config.spaces:
        for cam_id in space.camera_ids:
            mapping[cam_id] = space.id
    return mapping
