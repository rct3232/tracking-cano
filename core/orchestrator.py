import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import cv2

from settings import LLMConfig, PipelineConfig, ReconnectConfig, YOLOConfig
from core.config_manager import AppConfig, CameraConfig, SpaceConfig

from core.pipeline import DetectResult, LogEvent, Pipeline
from core.vision_worker import _BatchCollector
from nlp.logger import CameraSnapshot, SpaceLogger
from utils.video import create_capture

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
                        "reasoning": result.get("reasoning", ""),
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
        orchestrator: Optional["Orchestrator"] = None,
        space_id: Optional[str] = None,
        vision_enabled: bool = False,
        reconnect: ReconnectConfig | None = None,
    ):
        self.camera_id = camera_id
        self.pipeline = pipeline
        self.cap = cap
        self.source = source
        self.stop_event = stop_event
        self.frame_skip = frame_skip
        self.on_finished = on_finished
        self.orchestrator = orchestrator
        self.space_id = space_id
        self._vision_enabled = vision_enabled
        self._is_stream = _is_stream_source(source)
        self._finished = False
        self.reconnect = reconnect or ReconnectConfig()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{camera_id}")
        self._vision_quality = pipeline.config.llm.vision_quality if hasattr(pipeline.config, 'llm') else 60
        self._vision_max_width = pipeline.config.llm.vision_max_width if hasattr(pipeline.config, 'llm') else 1024

    def _encode_frame(self, frame, target_coordinate=None, label=None):
        h, w = frame.shape[:2]
        if self._vision_max_width > 0 and w > self._vision_max_width:
            scale = self._vision_max_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        import base64 as _b64
        _, buf = cv2.imencode(".png", frame)
        raw_b64 = _b64.b64encode(buf).decode("utf-8")
        if target_coordinate:
            from utils.image import draw_normalized_bbox
            return draw_normalized_bbox(raw_b64, target_coordinate, label=label)
        return raw_b64

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
            rc = self.reconnect
            frame_id = 0
            consecutive_failures = 0
            max_failures = rc.max_failures
            skip_interval = self.frame_skip + 1 if self.frame_skip > 0 else 1
            fps_log_interval = 5.0
            fps_frame_count = 0
            last_fps_log = time.perf_counter()
            connect_wall = time.monotonic()
            backoff_delay = rc.base_delay

            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    if self._is_stream:
                        consecutive_failures += 1
                        if consecutive_failures == 1:
                            logger.warning("Camera %s read failure (1/%d), source=%s", self.camera_id, max_failures, self.source)
                        if consecutive_failures >= max_failures:
                            logger.warning("Camera %s reconnecting... (%d failures, backoff=%.1fs)", self.camera_id, consecutive_failures, backoff_delay)
                            self.cap.release()
                            time.sleep(backoff_delay)
                            try:
                                new_cap = create_capture(self.source)
                            except Exception:
                                logger.exception("[%s] create_capture failed during reconnect", self.camera_id)
                                new_cap = None
                            if new_cap is None:
                                logger.error("Camera %s reconnect failed for %s", self.camera_id, self.source)
                                time.sleep(rc.reconnect_backoff)
                                backoff_delay = min(backoff_delay * 2, rc.max_delay)
                                continue
                            self.cap = new_cap
                            consecutive_failures = 0
                            connect_wall = time.monotonic()
                            backoff_delay = rc.base_delay
                        else:
                            time.sleep(rc.read_backoff)
                        continue
                    else:
                        logger.info("Camera %s video ended (%s)", self.camera_id, self.source)
                        break
                consecutive_failures = 0

                pts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                if pts_ms > 0:
                    lag_ms = pts_ms - (time.monotonic() - connect_wall) * 1000
                    if lag_ms > rc.pts_lag_threshold * 1000:
                        logger.warning("[%s] PTS lag=%.0fms > %.0fs, forcing reconnect", self.camera_id, lag_ms, rc.pts_lag_threshold)
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

                    # Build snapshot and update registry
                    image_b64 = None
                    tracked_list = []
                    interactions = []
                    coord = None
                    if detect.target_present:
                        if log_event is not None:
                            tracked_list = log_event.tracked_list
                            interactions = log_event.interactions or []
                            coord = log_event.target_coordinate
                        else:
                            coord = detect.target_coordinate
                    if frame is not None:
                        image_b64 = self._encode_frame(frame)  # raw frame only; bbox drawn at save time

                    snap = CameraSnapshot(
                        camera_id=self.camera_id,
                        target_present=detect.target_present,
                        timestamp=time.monotonic(),
                        tracked_list=tracked_list,
                        interactions=interactions,
                        image_b64=image_b64,
                        target_coordinate=coord,
                    )
                    if self.orchestrator and detect.target_present:
                        self.orchestrator.update_snapshot(self.camera_id, snap)

                    # Interaction change → space snapshot
                    if log_event is not None and self.space_id and self.orchestrator:
                        self.orchestrator.request_space_snapshot(self.space_id)
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


def _make_pipeline_config(camera: CameraConfig, app_config: AppConfig, default_model_path: str | None = None) -> PipelineConfig:
    model_path = camera.model_path or default_model_path or f"yolo26{camera.model_size}.pt"
    yolo = YOLOConfig(
        conf_threshold=app_config.yolo.conf_threshold,
        iou_threshold=app_config.yolo.iou_threshold,
        tile_enabled=app_config.yolo.tile_enabled,
        tile_grid_x=app_config.yolo.tile_grid_x,
        tile_grid_y=app_config.yolo.tile_grid_y,
        tile_overlap=app_config.yolo.tile_overlap,
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
        thresholds=app_config.thresholds,
        yolo=yolo,
        llm=app_config.llm,
        llm_system_prompt=camera.llm_system_prompt,
    )


class Orchestrator:
    def __init__(self, app_config: AppConfig, space_logger: Optional[SpaceLogger] = None,
                 default_model_path: Optional[str] = None, repo=None, event_bus=None):
        self.app_config = app_config
        self.space_logger = space_logger
        self._default_model_path = default_model_path
        self._repo = repo
        self._event_bus = event_bus
        self._workers: Dict[str, _CameraWorker] = {}
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

        mode = self.app_config.mode
        logger.debug("[init] MODE=%s space_logger=%r all_spaces=%d", mode, self.space_logger is not None, len(self.app_config.spaces))
        if mode == "llm_vision" and self.space_logger:
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
                    capture_start: float | None = None,
                    barrier: threading.Barrier | None = None, loop_count: int = 1):
        space_id = self._cam_to_space.get(camera.id)
        stop_event = threading.Event()
        mode = self.app_config.mode
        rc = self.app_config.reconnect

        if mode == "llm_vision":
            from core.vision_worker import _BatchCollector
            config = _make_pipeline_config(camera, self.app_config, self._default_model_path)

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
                reconnect=rc,
            )
            with self._lock:
                self._collectors[camera.id] = collector
            collector.start()
            logger.info("Batch collector %s started (space=%s, mode=%s)", camera.id, space_id, mode)
            return
        else:
            cap = create_capture(camera.source)
            backoff_delay = rc.base_delay

            while cap is None and _is_stream_source(camera.source):
                logger.warning("[%s] Cannot open camera, retrying in %.1fs", camera.id, backoff_delay)
                time.sleep(backoff_delay)
                try:
                    cap = create_capture(camera.source)
                except Exception:
                    logger.exception("[%s] create_capture failed during initial connect", camera.id)
                    cap = None
                if cap is None:
                    backoff_delay = min(backoff_delay * 2, rc.max_delay)

            if cap is None:
                logger.error("Cannot open camera %s from %s", camera.id, camera.source)
                return
            config = _make_pipeline_config(camera, self.app_config, self._default_model_path)
            pipeline = Pipeline(config, camera.id)
            worker = _CameraWorker(
                camera.id, pipeline, cap, camera.source, stop_event, camera.frame_skip,
                on_finished=self.worker_finished,
                orchestrator=self,
                space_id=space_id,
                vision_enabled=self.app_config.llm.vision_enabled,
                reconnect=rc,
            )
            with self._lock:
                self._workers[camera.id] = worker
            worker.start()
            logger.info("Camera %s started (%s, space=%s, mode=%s)", camera.id, camera.source, space_id, mode)
            if self._event_bus:
                self._event_bus.publish({"type": "camera.added", "data": {"camera_id": camera.id}})

    def worker_finished(self, camera_id: str):
        with self._lock:
            self._workers.pop(camera_id, None)
        logger.info("Worker %s removed from orchestrator (%d remaining)", camera_id, len(self._workers))

    def remove_camera(self, camera_id: str):
        with self._lock:
            worker = self._workers.pop(camera_id, None)
            collector = self._collectors.pop(camera_id, None)
        if worker:
            worker.stop()
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
            worker = self._workers.pop(camera_id, None)
            collector = self._collectors.pop(camera_id, None)
        if worker:
            worker.stop()
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
