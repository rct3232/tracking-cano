import base64
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2

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

    Space-aware mode: buffer is read externally by _VisionScheduler.
    Standalone mode: pass on_capture callback for per-capture notification.
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
        self._is_stream = _is_stream_source(source)
        self._finished = False
        self.buffer: deque[_FrameEntry] = deque(maxlen=collect_count)
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"collect-{camera_id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if not self._finished:
            self.cap.release()
            self._finished = True
        logger.info("Batch collector %s stopped", self.camera_id)

    def _run(self):
        cap = create_capture(self.source)
        if cap is None:
            logger.error("Cannot open camera %s from %s", self.camera_id, self.source)
            self._finished = True
            return

        try:
            consecutive_failures = 0
            max_failures = 5
            last_capture = 0.0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self._is_stream:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            logger.warning(
                                "Batch collector %s reconnecting... (%d failures)",
                                self.camera_id,
                                consecutive_failures,
                            )
                            cap.release()
                            time.sleep(1)
                            cap = create_capture(self.source)
                            if cap is None:
                                logger.error("Batch collector %s reconnect failed for %s", self.camera_id, self.source)
                                time.sleep(2)
                                continue
                            self.buffer.clear()
                            consecutive_failures = 0
                        else:
                            time.sleep(0.5)
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.debug("Batch collector %s rewinding video", self.camera_id)
                    continue

                consecutive_failures = 0

                now = time.monotonic()
                if now - last_capture >= self.collect_interval:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.vision_quality])
                    image_b64 = base64.b64encode(buf).decode("utf-8")
                    entry = _FrameEntry(image_b64, now)
                    self.buffer.append(entry)
                    logger.debug(
                        "[collect:%s] captured frame, buffer=%d",
                        self.camera_id,
                        len(self.buffer),
                    )
                    if self.on_capture:
                        self.on_capture(self.camera_id, image_b64)
                    last_capture = now

        except Exception:
            logger.exception("Batch collector %s crashed", self.camera_id)
        finally:
            cap.release()
            self._finished = True
            logger.info("Batch collector %s finished", self.camera_id)
            if self.on_finished:
                self.on_finished(self.camera_id)


class _VisionOnlyWorker:
    """Legacy batch collector for standalone (non-space) mode.

    Accumulates frames until snapshot_count is reached, then fires
    on_batch_ready callback. Used only for non-space LLM vision path.
    """
    def __init__(
        self,
        camera_id: str,
        source: str,
        stop_event: threading.Event,
        frame_skip: int = 0,
        snapshot_count: int = 5,
        vision_quality: int = 60,
        vision_max_width: int = 1024,
        on_batch_ready: Callable | None = None,
        llm_system_prompt: str | None = None,
        target_classes: list | None = None,
    ):
        self.camera_id = camera_id
        self.source = source
        self.stop_event = stop_event
        self.frame_skip = frame_skip
        self.snapshot_count = snapshot_count
        self.vision_quality = vision_quality
        self.vision_max_width = vision_max_width
        self.on_batch_ready = on_batch_ready
        self.llm_system_prompt = llm_system_prompt
        self.target_classes = target_classes or []
        self._is_stream = _is_stream_source(source)
        self._finished = False
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"vis-{camera_id}")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        if not self._finished:
            self.cap.release()
            self._finished = True
        logger.info("Vision worker %s stopped", self.camera_id)

    def _run(self):
        cap = create_capture(self.source)
        if cap is None:
            logger.error("Cannot open camera %s from %s", self.camera_id, self.source)
            self._finished = True
            return

        try:
            buffer: deque = deque()
            frame_id = 0
            consecutive_failures = 0
            max_failures = 5
            skip_interval = 1

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self._is_stream:
                        consecutive_failures += 1
                        if consecutive_failures >= max_failures:
                            logger.warning(
                                "Vision worker %s reconnecting... (%d failures)",
                                self.camera_id,
                                consecutive_failures,
                            )
                            cap.release()
                            time.sleep(1)
                            cap = create_capture(self.source)
                            if cap is None:
                                logger.error("Vision worker %s reconnect failed for %s", self.camera_id, self.source)
                                time.sleep(2)
                                continue
                            consecutive_failures = 0
                        else:
                            time.sleep(0.5)
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_id = 0
                        logger.debug("Vision worker %s rewinding video", self.camera_id)
                    continue

                consecutive_failures = 0

                if frame_id % skip_interval != 0:
                    frame_id += 1
                    continue

                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.vision_quality])
                image_b64 = base64.b64encode(buf).decode("utf-8")

                buffer.append(image_b64)
                logger.debug(
                    "[vision:%s] frame=%d buffer_size=%d",
                    self.camera_id,
                    frame_id,
                    len(buffer),
                )

                if len(buffer) >= self.snapshot_count:
                    batch = list(buffer)
                    logger.info(
                        "[vision:%s] submitting batch of %d images (frame=%d)",
                        self.camera_id,
                        len(batch),
                        frame_id,
                    )
                    if self.on_batch_ready:
                        self.on_batch_ready(
                            camera_id=self.camera_id,
                            images=batch,
                            llm_system_prompt=self.llm_system_prompt,
                            target_classes=self.target_classes,
                        )

                    buffer.popleft()

                frame_id += 1

        except Exception:
            logger.exception("Vision worker %s crashed", self.camera_id)
        finally:
            cap.release()
            self._finished = True
            logger.info("Vision worker %s finished", self.camera_id)
