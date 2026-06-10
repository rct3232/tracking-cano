import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2

from config.config import LLMConfig, PipelineConfig, Thresholds, YOLOConfig
from core.config_manager import AppConfig, CameraConfig, SpaceConfig

from core.pipeline import DetectResult, LogEvent, Pipeline
from core.vision_worker import _BatchCollector
from nlp.logger import NLPLogger, SpaceLogger
from utils.video import create_capture

logger = logging.getLogger(__name__)

_STREAM_PREFIXES = ("rtsp://", "http://", "https://")


def _is_stream_source(source: str) -> bool:
    return source.startswith(_STREAM_PREFIXES)


@dataclass
class _SpaceState:
    id: str
    name: str
    camera_ids: List[str]
    state: str = "detecting"       # "detecting" | "logging" | "cooling"
    detect_idx: int = 0
    cooldown_until: float = 0.0
    llm_system_prompt: Optional[str] = None
    target_classes: List[str] = field(default_factory=list)


class _VisionScheduler:
    """Per-space state machine manager (Layer 2).

    Single thread, 100ms polling. Processes one detection step per space
    per iteration. Spaces are independent — one space's LOGGING does not
    affect another's DETECTING.
    """
    def __init__(
        self,
        spaces: List[SpaceConfig],
        collectors: Dict[str, _BatchCollector],
        nlp_logger: NLPLogger,
        space_logger: SpaceLogger,
        config: LLMConfig,
        cam_to_space: Dict[str, str],
        app_config: AppConfig,
        all_finished_check=None,
    ):
        self._stop_event = threading.Event()
        self._states: Dict[str, _SpaceState] = {}
        self._collectors = collectors
        self._nlp_logger = nlp_logger
        self._space_logger = space_logger
        self._config = config
        self._cam_to_space = cam_to_space
        self._is_all_finished = all_finished_check or (lambda: False)
        self._thread = threading.Thread(target=self._run, daemon=True, name="vision-scheduler")

        for space in spaces:
            target_classes = list(dict.fromkeys(
                cls for cam in app_config.cameras
                if cam.id in space.camera_ids and cam.target_classes
                for cls in cam.target_classes
            ))
            self._states[space.id] = _SpaceState(
                id=space.id,
                name=space.name,
                camera_ids=list(space.camera_ids),
                llm_system_prompt=space.llm_system_prompt or None,
                target_classes=target_classes,
            )

    def start(self):
        self._thread.start()
        logger.debug("[vision-scheduler] started with %d spaces", len(self._states))

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop_event.is_set():
            if self._is_all_finished():
                self._stop_event.wait(1.0)
                continue
            now = time.monotonic()
            for state in self._states.values():
                if state.state == "cooling":
                    if now >= state.cooldown_until:
                        logger.debug("[space:%s] cooldown expired → detecting", state.id)
                        state.state = "detecting"
                        state.detect_idx = 0
                if state.state == "detecting":
                    self._process_detection_step(state)
                elif state.state == "logging":
                    pass
            self._stop_event.wait(0.1)

    def _get_camera_health(self, cam_id: str) -> tuple[str, str | None]:
        collector = self._collectors.get(cam_id)
        if collector is None or not collector.buffer:
            return "dead", None
        entry = collector.buffer[-1]
        age = time.monotonic() - entry.captured_at
        if age > self._config.max_stale_threshold:
            return "degraded", None
        return "healthy", entry.image_b64

    def _process_detection_step(self, state: _SpaceState):
        while state.detect_idx < len(state.camera_ids):
            cam_id = state.camera_ids[state.detect_idx]
            health, image_b64 = self._get_camera_health(cam_id)
            if health != "healthy" or image_b64 is None:
                logger.debug("[space:%s] cam=%s %s skip", state.id, cam_id, health)
                state.detect_idx += 1
                continue

            result = self._nlp_logger.vision_detect(
                camera_id=cam_id,
                image_b64=image_b64,
                llm_system_prompt=state.llm_system_prompt,
                target_classes=state.target_classes,
            )
            if result is None:
                logger.warning("[space:%s] cam=%s detect failed, skip", state.id, cam_id)
                state.detect_idx += 1
                continue

            if result.get("target_present", False):
                logger.info("[space:%s] cam=%s target_present=True → immediate logging", state.id, cam_id)
                self._transition_to_logging(state, detect_cam_id=cam_id, detect_image_b64=image_b64)
                return

            state.detect_idx += 1
            return

        logger.debug("[space:%s] all cameras done, no target → immediate restart", state.id)
        state.detect_idx = 0

    def _transition_to_logging(self, state: _SpaceState,
                               detect_cam_id: str | None = None,
                               detect_image_b64: str | None = None):
        logger.info("[space:%s] target found → logging + cooling", state.id)
        state.state = "logging"

        now = time.monotonic()
        cooldown_duration = self._config.cooldown_seconds - self._config.early_trigger
        state.cooldown_until = now + max(cooldown_duration, 0)

        images: List[tuple[str, str]] = []
        camera_health: Dict[str, str] = {}
        for cam_id in state.camera_ids:
            health, image_b64 = self._get_camera_health(cam_id)
            camera_health[cam_id] = health
            if cam_id == detect_cam_id and detect_image_b64 is not None:
                image_b64 = detect_image_b64
            if health == "healthy" and image_b64 is not None:
                images.append((cam_id, image_b64))

        self._space_logger.flush_vision(
            space_id=state.id,
            space_name=state.name,
            nlp_logger=self._nlp_logger,
            llm_system_prompt=state.llm_system_prompt,
            target_classes=state.target_classes,
            valid_camera_ids=state.camera_ids,
            camera_health=camera_health,
            override_images=images,
        )

        state.state = "cooling"


class _CameraWorker:
    def __init__(
        self,
        camera_id: str,
        pipeline: Pipeline,
        cap,
        source: str,
        stop_event: threading.Event,
        frame_skip: int = 0,
        on_finished: Optional[Callable[[str], None]] = None,
    ):
        self.camera_id = camera_id
        self.pipeline = pipeline
        self.cap = cap
        self.source = source
        self.stop_event = stop_event
        self.frame_skip = frame_skip
        self.on_finished = on_finished
        self._is_stream = _is_stream_source(source)
        self._finished = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{camera_id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if not self._finished:
            self.pipeline.stop()
            self.cap.release()
            self._finished = True
        logger.info("Camera %s stopped", self.camera_id)

    def _run(self):
        try:
            frame_id = 0
            consecutive_failures = 0
            max_failures = 5
            skip_interval = self.frame_skip + 1 if self.frame_skip > 0 else 1
            fps_log_interval = 5.0
            fps_frame_count = 0
            last_fps_log = time.perf_counter()
            connect_wall = time.monotonic()
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    if self._is_stream:
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
                            connect_wall = time.monotonic()
                        else:
                            time.sleep(0.5)
                        continue
                    else:
                        logger.info("Camera %s video ended (%s)", self.camera_id, self.source)
                        break
                consecutive_failures = 0
                pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if pts_ms > 0:
                    lag_ms = pts_ms - (time.monotonic() - connect_wall) * 1000
                    if lag_ms > 60_000:
                        logger.warning("[%s] PTS lag=%.0fms > 60s, forcing reconnect", self.camera_id, lag_ms)
                        consecutive_failures = max_failures
                        continue
                if frame_id % skip_interval == 0:
                    t0 = time.perf_counter()
                    detect, log_event = self.pipeline.process_frame(frame, frame_id)
                    dt = time.perf_counter() - t0
                    logger.debug("[%s] frame=%d infer=%.0fms", self.camera_id, frame_id, dt * 1000)
                    if detect.target_present:
                        logger.info("[%s] target_present=true class=%s bbox=%s", self.camera_id, detect.class_name, detect.target_coordinate)
                    else:
                        logger.debug("[%s] target_present=false", self.camera_id)
                else:
                    logger.debug("[%s] frame=%d skipped (interval=%d)", self.camera_id, frame_id, skip_interval)
                frame_id += 1
                fps_frame_count += 1
                now = time.perf_counter()
                if now - last_fps_log >= fps_log_interval:
                    fps = fps_frame_count / (now - last_fps_log)
                    logger.debug("[CAM FPS] %s frame=%d fps=%.1f", self.camera_id, frame_id, fps)
                    fps_frame_count = 0
                    last_fps_log = now

            self.pipeline.stop()
            self.cap.release()
            self._finished = True
            logger.info("Camera %s finished", self.camera_id)
            if self.on_finished:
                self.on_finished(self.camera_id)
        except Exception:
            logger.exception("Camera %s worker crashed", self.camera_id)
            self.pipeline.stop()
            self.cap.release()
            self._finished = True
            if self.on_finished:
                self.on_finished(self.camera_id)


def _make_pipeline_config(camera: CameraConfig, app_config: Optional[AppConfig] = None, default_model_path: Optional[str] = None) -> PipelineConfig:
    thresholds = Thresholds()
    llm = LLMConfig()
    if app_config:
        for k, v in app_config.thresholds.items():
            if hasattr(thresholds, k):
                setattr(thresholds, k, v)
    model_path = camera.model_path or default_model_path or f"yolo26{camera.model_size}.pt"
    yolo = YOLOConfig(
        model_size=camera.model_size,
        model_path=model_path,
        quantize=camera.quantize,
        frame_skip=camera.frame_skip,
    )
    if camera.interaction_classes is None:
        yolo.yolo_classes = None
    else:
        all_classes = list(dict.fromkeys(camera.target_classes + camera.interaction_classes))
        yolo.yolo_classes = all_classes if all_classes else None
    return PipelineConfig(
        target_classes=camera.target_classes,
        interaction_classes=camera.interaction_classes,
        thresholds=thresholds,
        yolo=yolo,
        llm=llm,
        llm_system_prompt=camera.llm_system_prompt,
    )


class Orchestrator:
    def __init__(self, app_config: AppConfig, space_logger: Optional[SpaceLogger] = None, default_model_path: Optional[str] = None, repo=None):
        self.app_config = app_config
        self.space_logger = space_logger
        self._default_model_path = default_model_path
        self._repo = repo
        self._workers: Dict[str, _CameraWorker] = {}
        self._lock = threading.Lock()
        self._cam_to_space: Dict[str, str] = _build_cam_to_space(app_config)
        if self.space_logger:
            for space in app_config.spaces:
                self.space_logger.set_camera_count(space.id, len(space.camera_ids))
        self._vision_nlp_logger: Optional[NLPLogger] = None
        self._vision_scheduler: Optional[_VisionScheduler] = None
        self._collectors: Dict[str, _BatchCollector] = {}

    @property
    def spaces(self) -> list[SpaceConfig]:
        return self.app_config.spaces

    def _ensure_vision_nlp(self, llm_config):
        if self._vision_nlp_logger is None:
            from nlp.logger import NLPLogger as NLPLoggerCls
            self._vision_nlp_logger = NLPLoggerCls(llm_config, repo=self._repo)

    @property
    def all_finished(self) -> bool:
        with self._lock:
            return len(self._workers) == 0 and len(self._collectors) == 0

    def start(self):
        video_cam_ids = [
            cam.id for cam in self.app_config.cameras
            if cam.status == "active" and not _is_stream_source(cam.source)
        ]
        video_count = len(video_cam_ids)
        barrier = threading.Barrier(video_count) if video_count > 0 else None

        for cam in self.app_config.cameras:
            if cam.status != "active":
                logger.info("Skipping inactive camera: %s", cam.id)
                continue
            self.add_camera(cam, barrier=barrier, loop_count=1)

        from os import environ
        mode = environ.get("MODE", "cv_pipeline")
        logger.debug("[init] MODE=%s space_logger=%r all_spaces=%d", mode, self.space_logger is not None, len(self.app_config.spaces))
        if mode == "llm_vision" and self.space_logger and self._vision_nlp_logger:
            self._vision_scheduler = _VisionScheduler(
                spaces=self.app_config.spaces,
                collectors=self._collectors,
                nlp_logger=self._vision_nlp_logger,
                space_logger=self.space_logger,
                config=self._vision_nlp_logger.config,
                cam_to_space=self._cam_to_space,
                app_config=self.app_config,
                all_finished_check=lambda: self.all_finished,
            )
            self._vision_scheduler.start()

    def add_camera(self, camera: CameraConfig, start_event: threading.Event | None = None,
                   capture_start: float | None = None,
                   barrier: threading.Barrier | None = None, loop_count: int = 1):
        space_id = self._cam_to_space.get(camera.id)
        cap = create_capture(camera.source)
        if cap is None:
            logger.error("Cannot open camera %s from %s", camera.id, camera.source)
            return
        config = _make_pipeline_config(camera, self.app_config, self._default_model_path)
        pipeline = Pipeline(config, camera.id, self.space_logger, space_id, repo=self._repo)
        stop_event = threading.Event()
        from os import environ
        mode = environ.get("MODE", "cv_pipeline")
        if mode == "llm_vision":
            from core.vision_worker import _BatchCollector, _VisionOnlyWorker
            self._ensure_vision_nlp(config.llm)

            space_id = self._cam_to_space.get(camera.id)
            if space_id and self.space_logger:
                def _on_collector_finished(cid: str):
                    with self._lock:
                        self._collectors.pop(cid, None)
                    logger.info("Collector %s removed from orchestrator", cid)
                collector = _BatchCollector(
                    camera_id=camera.id,
                    source=camera.source,
                    stop_event=stop_event,
                    collect_interval=config.llm.collect_interval,
                    collect_count=config.llm.collect_count,
                    vision_quality=config.llm.vision_quality,
                    vision_max_width=config.llm.vision_max_width,
                    on_finished=_on_collector_finished,
                    start_event=start_event,
                    capture_start=capture_start,
                    loop_count=loop_count,
                    barrier=barrier,
                )
                with self._lock:
                    self._collectors[camera.id] = collector
                collector.start()
                logger.info("Batch collector %s started (space=%s)", camera.id, space_id)
                return
            else:
                # Non-space standalone path (legacy)
                worker = _VisionOnlyWorker(
                    camera_id=camera.id,
                    source=camera.source,
                    stop_event=stop_event,
                    frame_skip=camera.frame_skip,
                    snapshot_count=config.llm.snapshot_count,
                    vision_quality=config.llm.vision_quality,
                    vision_max_width=config.llm.vision_max_width,
                    on_batch_ready=lambda **kw: self._vision_nlp_logger.vision_log(  # type: ignore[union-attr]
                        kw["images"], kw["camera_id"],
                        camera.llm_system_prompt, camera.target_classes,
                    ),
                    llm_system_prompt=camera.llm_system_prompt,
                    target_classes=camera.target_classes,
                )
        else:
            worker = _CameraWorker(
                camera.id, pipeline, cap, camera.source, stop_event, camera.frame_skip,
                on_finished=self.worker_finished,
            )
        with self._lock:
            self._workers[camera.id] = worker
        worker.start()
        logger.info("Camera %s started (%s, space=%s, mode=%s)", camera.id, camera.source, space_id, mode)

    def worker_finished(self, camera_id: str):
        with self._lock:
            self._workers.pop(camera_id, None)
        logger.info("Worker %s removed from orchestrator (%d remaining)", camera_id, len(self._workers))

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
        if self._vision_scheduler:
            self._vision_scheduler.stop()
        with self._lock:
            collectors = list(self._collectors.values())
            workers = list(self._workers.values())
        for c in collectors:
            c.stop()
        for w in workers:
            w.stop()
        self._collectors.clear()
        self._workers.clear()
        logger.info("All cameras stopped")


def _build_cam_to_space(app_config: AppConfig) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for space in app_config.spaces:
        for cam_id in space.camera_ids:
            mapping[cam_id] = space.id
    return mapping
