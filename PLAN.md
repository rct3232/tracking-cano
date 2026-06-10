# PLAN.md — Pipeline Output Unification (v2)

## SPEC

### Objective
Unify the output format (JSON structure, log file naming, console output) and narrow behavioral gaps between `cv_pipeline` and `llm_vision` modes, while preserving each pipeline's architectural strengths (YOLO low-cost detection vs LLM vision stateless analysis).

### Scope
- `core/pipeline.py` — `process_frame()` return type change, DetectResult/LogEvent separation
- `nlp/logger.py` — `SYSTEM_PROMPT` / `_process_task()` JSON output, log file naming, `vision_detect()` log level, `SpaceLogger.flush()` JSON output
- `core/orchestrator.py` — `_CameraWorker` detect/log console output split
- `config/config.py` — `cooldown_seconds` duplicate removal

### Key Components

**Unified JSON Output Structure (both modes):**
```json
{
  "target_present": true,
  "cameras": {
    "{cam_id}": {
      "description": "One short sentence describing target behavior.",
      "target_coordinate": [x1, y1, x2, y2] | null
    }
  },
  "reasoning": "One sentence summarizing the overall situation."
}
```

- `target_coordinate`: cv_pipeline → YOLO bbox normalized, llm_vision → LLM response
- `description`: cv_pipeline → LLM ON: LLM / OFF: template, llm_vision → LLM
- `reasoning`: SpaceLogger → LLM ON: LLM synthesis / OFF: description join

**Phase separation (cv_pipeline):**
```
process_frame(frame, frame_id)
  │
  ├─ YOLO + ByteTrack (always runs)
  │
  ├─ return (DetectResult, LogEvent | None)
  │     DetectResult: target_present, class_name, target_coordinate (always)
  │     LogEvent:     state/interaction change detected → NLPLogger.log()
  │
  ├─ _CameraWorker detect phase (console only):
  │     target_present=true  → logger.info()
  │     target_present=false → logger.debug()
  │
  └─ _CameraWorker logging phase (file + console):
        LogEvent exists → NLPLogger.log() → JSON file + console.info()
```

**Console output rule (both modes):**

| State | `-v` absent (INFO) | `-v` present (DEBUG) |
|-------|---------------------|----------------------|
| target_present=true | ✅ INFO | ✅ INFO |
| target_present=false | ❌ hidden | ✅ DEBUG |
| Logging event | ✅ INFO | ✅ INFO |

**Log file naming:**

| Before | After | Mode |
|--------|-------|------|
| `logs/vision_{cam_id}_{date}.log` | `logs/{cam_id}_{date}.log` | llm_vision per-camera |
| `logs/vision_space_{space}_{date}.log` | `logs/{space}_{date}.log` | llm_vision space |
| `logs/{cam_id}_{date}.log` | unchanged | cv_pipeline per-camera |
| `logs/{space_id}_{date}.log` | `logs/{space_id}_{date}.log` (JSON content) | cv_pipeline space |

### Success Criteria
1. Both modes produce identical JSON structure in per-camera log files
2. Log file naming is identical across modes
3. Console output format and verbosity rules are identical across modes
4. `Process_frame()` returns structured DetectResult for all frames
5. SpaceLogger output is JSON in both modes
6. `cooldown_seconds` duplicate in `LLMConfig` is removed
7. Existing behavior preserved when LLM is disabled (template-based descriptions)

---

## Progress

### Phase 1: `Pipeline.process_frame()` return type change — COMPLETED
- Added `DetectResult` and `LogEvent` dataclasses in `core/pipeline.py`
- Changed `process_frame()` return from `Optional[str]` to `tuple[DetectResult, Optional[LogEvent]]`
- Updated callers: `_CameraWorker._run()` in `core/orchestrator.py`, `main.py` standalone/video paths
- **Files:** `core/pipeline.py`, `core/orchestrator.py`, `main.py`

### Phase 2: `_CameraWorker` console output split — COMPLETED
- `target_present=true` → `logger.info()`, `target_present=false` → `logger.debug()`
- Removed old `logger.info("[%s] %s", self.camera_id, result)` line
- **File:** `core/orchestrator.py`

### Phase 3: `NLPLogger._process_task()` JSON output — COMPLETED
- `SYSTEM_PROMPT` rewritten to request JSON with `description`/`reasoning` fields
- `_LogTask` dataclass: added `target_coordinate` and `target_classes` fields
- `log()` method: extracts YOLO bbox → normalized `target_coordinate`, passes to task
- `_process_task()`: uses `response_format=json_object`, fallback for unsupported models, parses JSON, fills `target_coordinate` from YOLO, saves unified JSON
- Added `_log_fallback()`: LLM disabled path builds template JSON from YOLO state
- `_build_prompt()`: accepts `target_coordinate`/`target_classes`, instructs JSON output
- **File:** `nlp/logger.py`

### Phase 4: llm_vision log file naming unification — COMPLETED
- `_process_vision_task()`: `vision_{task.camera_id}_` → `{task.camera_id}_`
- `_process_vision_batch_space()`: `vision_{clean_cam_id}` → `{clean_cam_id}`
- `_process_vision_batch_space()`: `vision_space_{space_name}_` → `{space_name}_`
- **File:** `nlp/logger.py`

### Phase 5: `SpaceLogger.flush()` JSON output — COMPLETED
- `SPACE_SYSTEM_PROMPT` rewritten to request JSON
- `flush()`: parses per-camera JSON entries, merges `cameras` object directly
- `reasoning` generation: LLM ON → LLM synthesis with `response_format=json_object` / LLM OFF → description join
- **File:** `nlp/logger.py`

### Phase 6: `vision_detect()` console output level change — COMPLETED
- `target_present=True` → `logger.info()`, `target_present=False` → `logger.debug()`
- **File:** `nlp/logger.py`

### Phase 7: cv_pipeline image save with bbox annotation — COMPLETED
- Replaced `annotate_image()` with raw encode + `draw_normalized_bbox()` using YOLO `target_coordinate`
- Removed `annotate_image` import, added `cv2` import
- Saved images from both modes use same `draw_normalized_bbox()` function
- **File:** `nlp/logger.py`

### Phase 8: `cooldown_seconds` duplicate removal — COMPLETED
- Removed line 41: `cooldown_seconds: float = 3.0` (hardcoded, overridden by line 50)
- Kept line 50: env-based `VISION_COOLDOWN_SECONDS` (default 30.0)
- Added comment explaining usage by both pipelines
- **File:** `config/config.py`

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `target_coordinate` source differs per mode | cv_pipeline: YOLO bbox (more precise), llm_vision: LLM response (only source available) |
| Detect phase has no file log (both modes) | Detection is transient; only state changes deserve persistent log |
| `SpaceLogger.flush()` JSON by merging, not regenerating | Per-camera descriptions already exist from previous LLM/template calls |
| Keep debounce/cooldown separate | Different purposes: low-cost state monitoring vs high-cost stateless call rate limiting |
| `cooldown_seconds` config shared but semantics differ | Both are "minimum interval between LLM calls" — same knob, different mechanisms |
| Detect phase `target_present=false` → DEBUG level | Reduces noise in normal operation; still available for debugging with `-v` |

## Not In Scope (preserved differences)

| Aspect | cv_pipeline | llm_vision |
|--------|-------------|------------|
| Detection | YOLO + ByteTrack (local, every frame) | LLM vision_detect() (remote, 100ms polling) |
| Frame capture | Immediate processing | Timer-based buffer (0.5s, maxlen=5) |
| Debounce/cooldown | LLMCallDebouncer | State machine DETECTING→LOGGING+COOLING |
| Camera health | Not tracked | healthy/degraded/dead based on buffer age |
| Space collection | Text observation accumulation | Multi-image batch analysis |
| Logging trigger | State/interaction change | target_present → LOGGING state |
