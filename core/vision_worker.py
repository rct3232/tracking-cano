import base64
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2

from settings import ReconnectConfig
from utils.video import create_capture

logger = logging.getLogger(__name__)

_STREAM_PREFIXES = ("rtsp://", "http://", "https://")


def _is_stream_source(source: str) -> bool:
    return any(source.startswith(p) for p in _STREAM_PREFIXES)


@dataclass
class _FrameEntry:
    image_b64: str
    captured_at: float  # time.monotonic()


class _BatchCollector:
    """Timer-based frame capture with sliding-window buffer (Layer 1).

    Reads every frame (RTSP decoder buffer management), but only encodes
    and appends to buffer on a collect_interval timer.

    Space-aware mode: buffer is read externally by _SimpleVisionDetector.
    Standalone mode: pass on_capture callback for per-capture notification.

    For video file sources, uses seek-based capture aligned to a shared
    wall-clock (capture_start) so all collectors capture the same
    video timestamp across cameras.
    """
    def __init__(
        self,
        camera_id: str,
        source: str,
        stop_event: threading.Event,
        collect_interval: float = 0.5,
        collect_count: int = 5,
        vision_quality: int = 60,
        vision_max_width: int = 1024,
        on_capture: Callable | None = None,
        on_finished: Callable[[str], None] | None = None,
        start_event: threading.Event | None = None,
        capture_start: float | None = None,
        loop_count: int = 1,
        barrier: threading.Barrier | None = None,
        reconnect: ReconnectConfig | None = None,
    ):
        self.camera_id = camera_id
        self.source = source
        self.stop_event = stop_event
        self.collect_interval = collect_interval
        self.collect_count = collect_count
        self.vision_quality = vision_quality
        self.vision_max_width = vision_max_width
        self.on_capture = on_capture
        self.on_finished = on_finished
        self._start_event = start_event
        self._capture_start = capture_start or time.monotonic()
        self.loop_count = loop_count
        self._barrier = barrier
        self._is_stream = _is_stream_source(source)
        self._finished = False
        self.reconnect = reconnect or ReconnectConfig()
        self.buffer: deque[_FrameEntry] = deque(maxlen=collect_count)
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"collect-{camera_id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self._finished = True
        logger.info("Batch collector %s stopped", self.camera_id)

    def _encode_frame(self, frame, captured_at: float):
        _, buf = cv2.imencode(".png", frame)
        image_b64 = base64.b64encode(buf).decode("utf-8")
        entry = _FrameEntry(image_b64, captured_at)
        self.buffer.append(entry)
        logger.debug("[collect:%s] captured frame, buffer=%d", self.camera_id, len(self.buffer))
        if self.on_capture:
            self.on_capture(self.camera_id, image_b64)

    _run_video_id = 0

    def _run_video(self, cap):
        _BatchCollector._run_video_id += 1
        run_id = _BatchCollector._run_video_id
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        steps_per_loop = int(duration / self.collect_interval) if duration > 0 else 0
        total_steps = steps_per_loop * self.loop_count
        logger.debug(
            "[%s] video #%d: fps=%.1f frames=%d duration=%.1fs interval=%.1f steps_per_loop=%d total_steps=%d loop_count=%d",
            self.camera_id, run_id, fps, total_frames, duration,
            self.collect_interval, steps_per_loop, total_steps, self.loop_count,
        )

        if self._barrier is not None:
            logger.debug("[%s] waiting on barrier (run #%d)", self.camera_id, run_id)
            self._barrier.wait()

        first_step_time = math.ceil(time.monotonic() / self.collect_interval) * self.collect_interval
        step = 0
        log_interval = max(1, total_steps // 5) if total_steps > 0 else 1

        while not self.stop_event.is_set() and step < total_steps:
            target_wall = first_step_time + step * self.collect_interval
            nap = target_wall - time.monotonic()
            if nap > 0:
                time.sleep(nap)

            video_pos = (step * self.collect_interval) % duration
            cap.set(cv2.CAP_PROP_POS_MSEC, video_pos * 1000)
            ret, frame = cap.read()
            if ret:
                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                logger.debug("[collect:%s] video pts=%.0fms (expected=%.0fms)", self.camera_id, pts_ms, video_pos * 1000)
                self._encode_frame(frame, time.monotonic())
            else:
                logger.warning("[%s] read failed at step %d, pos=%.1fs", self.camera_id, step, video_pos)

            step += 1
            if step % log_interval == 0:
                logger.debug("[%s] progress: step=%d/%d", self.camera_id, step, total_steps)

        logger.info("[%s] _run_video #%d done: completed %d/%d steps (stop=%s)", self.camera_id, run_id, step, total_steps, self.stop_event.is_set())

    def _run(self):
        if self._finished:
            logger.warning("[%s] _run called but already finished", self.camera_id)
            return
        if self._start_event is not None:
            self._start_event.wait()

        cap = create_capture(self.source)
        rc = self.reconnect
        backoff_delay = rc.base_delay

        while cap is None and self._is_stream:
            logger.warning("[%s] Cannot open camera, retrying in %.1fs", self.camera_id, backoff_delay)
            time.sleep(backoff_delay)
            try:
                cap = create_capture(self.source)
            except Exception:
                logger.exception("[%s] create_capture failed during initial connect", self.camera_id)
                cap = None
            if cap is None:
                backoff_delay = min(backoff_delay * 2, rc.max_delay)

        if cap is None:
            logger.error("Cannot open camera %s from %s", self.camera_id, self.source)
            self._finished = True
            if self.on_finished:
                self.on_finished(self.camera_id)
            return

        connect_wall = time.monotonic()
        logger.debug("[%s] _run started (is_stream=%s)", self.camera_id, self._is_stream)
        if not self._is_stream:
            self._run_video(cap)
            cap.release()
            self._finished = True
            logger.info("Batch collector %s finished", self.camera_id)
            if self.on_finished:
                self.on_finished(self.camera_id)
            return

        next_capture = math.ceil(time.monotonic() / self.collect_interval) * self.collect_interval

        try:
            consecutive_failures = 0
            max_failures = rc.max_failures
            backoff_delay = rc.base_delay

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.warning(
                            "Batch collector %s reconnecting... (%d failures, backoff=%.1fs)",
                            self.camera_id,
                            consecutive_failures,
                            backoff_delay,
                        )
                        cap.release()
                        time.sleep(backoff_delay)
                        try:
                            new_cap = create_capture(self.source)
                        except Exception:
                            logger.exception("[%s] create_capture failed during reconnect", self.camera_id)
                            new_cap = None
                        if new_cap is None:
                            logger.error("Batch collector %s reconnect failed for %s", self.camera_id, self.source)
                            time.sleep(rc.reconnect_backoff)
                            backoff_delay = min(backoff_delay * 2, rc.max_delay)
                            continue
                        cap = new_cap
                        self.buffer.clear()
                        consecutive_failures = 0
                        connect_wall = time.monotonic()
                        backoff_delay = rc.base_delay
                    else:
                        time.sleep(rc.read_backoff)
                    continue

                consecutive_failures = 0

                pts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pts_ms > 0:
                    lag_ms = pts_ms - (time.monotonic() - connect_wall) * 1000
                    if lag_ms > rc.pts_lag_threshold * 1000:
                        logger.warning("[collect:%s] PTS lag=%.0fms > %.0fs, forcing reconnect", self.camera_id, lag_ms, rc.pts_lag_threshold)
                        consecutive_failures = max_failures
                        continue
                now = time.monotonic()
                if now >= next_capture:
                    self._encode_frame(frame, now)
                    next_capture += self.collect_interval
                    while next_capture <= now:
                        next_capture += self.collect_interval

        except Exception:
            logger.exception("Batch collector %s crashed", self.camera_id)
        finally:
            cap.release()
            self._finished = True
            logger.info("Batch collector %s finished", self.camera_id)
            if self.on_finished:
                self.on_finished(self.camera_id)


