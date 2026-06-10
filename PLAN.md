# PLAN.md — Vision LLM Pipeline Upgrade

## SPEC

### Objective
Replace the current periodic-flush vision pipeline with an event-driven, two-layer architecture: a timer-based batch collector (Layer 1) and a per-space state machine (Layer 2) that decides when to call the LLM for target detection and space logging.

### Scope
- `config/config.py` — new config variables
- `core/vision_worker.py` — `_BatchCollector` replaces `_VisionOnlyWorker`
- `nlp/logger.py` — `SpaceLogger` / new state machine additions
- `core/orchestrator.py` — space scheduler thread replaces `_vision_flush_loop`
- `.env.example` — new env vars

### Key Components

```
Layer 1 [Batch Collector]        Layer 2 [Space Scheduler]
  per camera, daemon thread        per space state machine
  ┌──────────────────────┐        ┌──────────────────────────────┐
  │ camera-A collector   │        │  Space-X                     │
  │  · 0.5s timer        │        │  DETECTING → LOGGING+COOLING │
  │  · buffer(maxlen=5)  │        │       ↑              ↓       │
  │  · always running    │        │       └────── ← ─────┘       │
  ├──────────────────────┤        │  Space-Y (완전 독립)         │
  │ camera-B collector   │        └──────────────────────────────┘
  │  · 동일              │
  └──────────────────────┘
```

### Success Criteria
1. Batch collector captures at exactly `collect_interval` (0.5s) regardless of FPS
2. Space state machine transitions correctly: DETECTING → (target_found) LOGGING+COOLING vs (all false) immediate DETECTING restart
3. Camera health tracking prevents stale images from being sent to LLM
4. Spaces operate independently — one space's LOGGING does not block another's DETECTING
5. Reconnect clears stale buffer; degraded/dead cameras excluded from detection and space logging

---

## Progress

### Phase 1: Config Variables — COMPLETED
- Added `collect_interval`, `collect_count`, `max_stale_threshold`, `cooldown_seconds`, `early_trigger` to `LLMConfig` in `config/config.py:47-51`

### Phase 2: Layer 1 — _BatchCollector — COMPLETED
- Added `_FrameEntry` dataclass with `image_b64` + `captured_at`
- Added `_BatchCollector` class with timer-based capture (`collect_interval`), sliding-window `buffer: deque[_FrameEntry](maxlen=collect_count)`, and `buffer.clear()` on reconnect
- Kept legacy `_VisionOnlyWorker` for non-space standalone path
- File: `core/vision_worker.py:25-192`

### Phase 3: Layer 2 — Space Scheduler — COMPLETED
- Added `_SpaceState` dataclass and `_VisionScheduler` class in `core/orchestrator.py`
- Scheduler loop: 100ms polling, 1 detection step per space per iteration
- State machine: DETECTING → (target_found) LOGGING+COOLING / (all false) immediate restart
- Space independence: LOGGING blocks only its own DETECTING, not other spaces
- Camera health: healthy → LLM detect / degraded → skip / dead → skip
- Removed old `_vision_flush_loop` and `_flush_thread`

### Phase 4: SpaceLogger Integration — COMPLETED
- Added `NLPLogger.vision_detect()` — synchronous single-image detection with `DETECT_SYSTEM_PROMPT`
- Modified `_process_vision_batch_space()` to accept `camera_health: Dict[str, str]` for degraded/dead camera text annotations
- Modified `SpaceLogger.flush_vision()` to accept `override_images` + `camera_health` for scheduler direct call path
- File: `nlp/logger.py:144-200, 618-645, 656-677`

### Phase 5: .env.example Update — COMPLETED
- Added: `VISION_COLLECT_INTERVAL`, `VISION_COLLECT_COUNT`, `VISION_MAX_STALE`, `VISION_COOLDOWN_SECONDS`, `VISION_EARLY_TRIGGER`

### Phase 6: LLM Bbox + Drawing — COMPLETED
- Added `target_coordinate` field to `VISION_SPACE_SYSTEM_PROMPT` (normalized 0~1 xywh)
- Added `draw_normalized_bbox()` in `utils/image.py`
- Parsed LLM bbox response and drew green rectangles on saved output images in `_process_vision_batch_space()`
- Improved bbox prompt precision: "tight bounding box, 2-3 decimal places"

### Phase 7: Detect Frame Synchronization — COMPLETED
- `_process_detection_step()` now returns `detect_cam_id` + `detect_image_b64`
- `_transition_to_logging()` overwrites trigger camera's image with the detect frame (not buffered frame)
- Ensures the saved bbox image matches the exact frame sent to LLM

### Phase 8: Capture Timing Sync — COMPLETED
- `_BatchCollector` accepts optional `start_event: threading.Event` — waits before capturing
- `orchestrator.start()` creates a shared `start_signal` for video file sources (RTSP passes `None`)
- Timing alignment: `next_capture = math.ceil(start_mono / interval) * interval` ensures all collectors capture at same time boundaries
- `_get_camera_health()` reverted to 2-tuple `(health, image_b64)` — `captured_wall` chain removed
- `_transition_to_logging()` images type reverted to `List[tuple[str, str]]`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Hybrid read/encode | Read every frame (RTSP buffer mgmt), encode only on 0.5s timer |
| Reconnect → buffer.clear() | Prevent stale image false positives after stream recovery |
| max_stale_threshold=10s | buffer refresh cycle 2.5s × 4 — ample margin |
| Cooldown timer = cooldown - early_trigger | Simplifies "5s before end restart" into single timer |
| False → immediate DETECTING restart | No reason to wait when no target detected |
| Detection continues after true | Full scan of all cameras before deciding |
| Degraded/dead camera exclusion | Prevents LLM confusion from stale/absent data |
| Space-independent states | One space's logging doesn't block others |

---

## Testing Strategy
1. **Unit:** State machine transitions (DETECT→LOG→COOL, false→immediate restart)
2. **Unit:** Camera health classification (healthy/degraded/dead) with mocked timestamps
3. **Integration:** RTSP disconnect/reconnect cycle → verify buffer.clear() and health transition
4. **Integration:** Two spaces running independently → verify no cross-blocking
5. **E2E:** Full pipeline with mock LLM → verify correct sequence of detect/log calls
