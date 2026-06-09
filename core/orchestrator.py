import logging
import threading
import time
from typing import Callable, Dict, Optional

from config.config import LLMConfig, PipelineConfig, Thresholds, YOLOConfig
from core.config_manager import AppConfig, CameraConfig, SpaceConfig

from core.pipeline import Pipeline
from nlp.logger import NLPLogger, SpaceLogger
from utils.video import create_capture

logger = logging.getLogger(__name__)

_STREAM_PREFIXES = ("rtsp://", "http://", "https://")


def _is_stream_source(source: str) -> bool:
    return source.startswith(_STREAM_PREFIXES)


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
                        else:
                            time.sleep(0.5)
                        continue
                    else:
                        logger.info("Camera %s video ended (%s)", self.camera_id, self.source)
                        break
                consecutive_failures = 0
                if frame_id % skip_interval == 0:
                    t0 = time.perf_counter()
                    result = self.pipeline.process_frame(frame, frame_id)
                    dt = time.perf_counter() - t0
                    logger.debug("[%s] frame=%d infer=%.0fms", self.camera_id, frame_id, dt * 1000)
                    if result:
                        logger.info("[%s] %s", self.camera_id, result)
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
    def __init__(self, app_config: AppConfig, space_logger: Optional[SpaceLogger] = None, default_model_path: Optional[str] = None):
        self.app_config = app_config
        self.space_logger = space_logger
        self._default_model_path = default_model_path
        self._workers: Dict[str, _CameraWorker] = {}
        self._lock = threading.Lock()
        self._cam_to_space: Dict[str, str] = _build_cam_to_space(app_config)
        if self.space_logger:
            for space in app_config.spaces:
                self.space_logger.set_camera_count(space.id, len(space.camera_ids))
        self._vision_nlp_logger: Optional[NLPLogger] = None
        self._flush_stop_event: threading.Event | None = None
        self._flush_thread: Optional[threading.Thread] = None

    @property
    def spaces(self) -> list[SpaceConfig]:
        return self.app_config.spaces

    def _ensure_vision_nlp(self, llm_config):
        if self._vision_nlp_logger is None:
            from nlp.logger import NLPLogger as NLPLoggerCls
            self._vision_nlp_logger = NLPLoggerCls(llm_config)

    @property
    def all_finished(self) -> bool:
        with self._lock:
            return len(self._workers) == 0

    def start(self):
        for cam in self.app_config.cameras:
            if cam.status != "active":
                logger.info("Skipping inactive camera: %s", cam.id)
                continue
            self.add_camera(cam)

    # Start periodic vision flush thread
        from os import environ
        mode = environ.get("MODE", "cv_pipeline")
        logger.debug("[init] MODE=%s space_logger=%r all_spaces=%d", mode, self.space_logger is not None, len(self.app_config.spaces))
        if mode == "llm_vision" and self.space_logger:
            self._flush_stop_event = threading.Event()
            interval = float(environ.get("VISION_INTERVAL_SECONDS", "30"))
            self._flush_thread = threading.Thread(
                target=self._vision_flush_loop, args=(self._flush_stop_event, interval), daemon=True, name="vision-flush"
            )
            self._flush_thread.start()

    def _vision_flush_loop(self, stop_event: threading.Event, interval: float):
        logger.debug("[flush-loop] started with interval=%ds", int(interval))
        while not stop_event.is_set():
            if stop_event.wait(timeout=interval):
                break
            logger.debug("[flush-loop] cycle start")
            for space in self.app_config.spaces:
                llm_prompt = space.llm_system_prompt or None
                all_target_classes = list(dict.fromkeys(
                    cls for c in self.app_config.cameras
                    if self._cam_to_space.get(c.id) == space.id and c.target_classes
                    for cls in c.target_classes
                ))
                text = self.space_logger.flush_vision(
                    space.id, space.name, self._vision_nlp_logger,  # type: ignore[arg-type]
                    llm_prompt, all_target_classes or None,
                    space.camera_ids,
                )
                if text and '"target_present": true' in text:
                    logger.info("[space:%s][vision] %s", space.id, text)

    def add_camera(self, camera: CameraConfig):
        space_id = self._cam_to_space.get(camera.id)
        cap = create_capture(camera.source)
        if cap is None:
            logger.error("Cannot open camera %s from %s", camera.id, camera.source)
            return
        config = _make_pipeline_config(camera, self.app_config, self._default_model_path)
        pipeline = Pipeline(config, camera.id, self.space_logger, space_id)
        stop_event = threading.Event()
        from os import environ
        mode = environ.get("MODE", "cv_pipeline")
        if mode == "llm_vision":
            from core.vision_worker import _VisionOnlyWorker
            self._ensure_vision_nlp(config.llm)

            space_id = self._cam_to_space.get(camera.id)
            if space_id and self.space_logger:
                # Space-aware aggregator 경로 — 버퍼 업데이트만 (LLM 호출 없음)
                worker = _VisionOnlyWorker(
                    camera_id=camera.id,
                    source=camera.source,
                    stop_event=stop_event,
                    frame_skip=camera.frame_skip,
                    snapshot_count=config.llm.snapshot_count,
                    vision_quality=config.llm.vision_quality,
                    vision_max_width=config.llm.vision_max_width,
                    on_batch_ready=lambda **kw: self.space_logger.vision_collect(
                        space_id, kw["camera_id"], kw["images"]
                    ),
                )
            else:
                # 기존 직접 LLM 호출 경로 (space 없는 카메라)
                worker = _VisionOnlyWorker(
                    camera_id=camera.id,
                    source=camera.source,
                    stop_event=stop_event,
                    frame_skip=camera.frame_skip,
                    snapshot_count=config.llm.snapshot_count,
                    snapshot_interval=config.llm.snapshot_interval,
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
        if self._flush_stop_event:
            self._flush_stop_event.set()
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
