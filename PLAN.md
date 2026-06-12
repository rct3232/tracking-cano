# PLAN.md — Snapshot 기반 Space Logging 리팩토링

## SPEC

### Objective
CV pipeline과 LLM vision pipeline의 space logging을 통합한다. Pipeline 유형에 관계없이 `detect → snapshot → space logging` 순서로 통일한다. 기존의 per-camera buffer accumulation + periodic flush를 제거하고, interaction change 또는 vision detect success 발생 시 즉시 공간 내 모든 카메라의 현재 상태를 수집하여 한 번에 space logging한다.

### Scope
- 기존 `_CameraWorker` + `Pipeline`의 tracking/interaction detection 유지
- `NLPLogger`의 queue-based per-camera LLM 처리 제거
- `SpaceLogger`의 buffer/flush 메커니즘 → `space_snapshot()` 기반 즉시 처리로 대체
- `_VisionScheduler` state machine 제거, `_BatchCollector`는 image buffer로만 유지
- CV와 LLM vision이 동일한 `SpaceLogger.space_snapshot()` 메서드로 수렴
- `SpaceLogger`가 LLM client + prompt + debounce + persistence를 모두 담당

### Key Components
| 컴포넌트 | 역할 |
|----------|------|
| `CameraSnapshot` | 카메라 1대의 현재 상태 스냅샷 (tracked_list, interactions, image_b64 등) |
| `Orchestrator._snapshots` | 모든 카메라의 최신 snapshot registry |
| `Orchestrator.request_space_snapshot()` | interaction change trigger → 모든 camera snapshot 수집 → space_snapshot 호출 |
| `SpaceLogger.space_snapshot()` | 통합 스냅샷 처리: LLM prompt 구성 → 호출 → per-camera log + image + space log 저장 |
| `Pipeline.process_frame()` | tracking/interaction detection만 수행, LLM 호출 제거 |

### Success Criteria
1. CV pipeline에서 interaction change 발생 시 즉시 space snapshot logging (10s 주기 불필요)
2. LLM vision pipeline에서 target detect 시 동일한 `space_snapshot()` 호출
3. DB에 per-camera detect log + space-level aggregated log가 정상 저장됨
4. `vision_enabled=true` 시 LLM이 이미지 기반 description 생성
5. `vision_enabled=false` 시 LLM이 tracking data 기반 description 생성
6. 모든 기존 기능 (hot-reload, 카메라 추가/제거, DB logging, event_bus) 정상 동작

---

## Architecture

```
[CV Pipeline]                              [LLM Vision Pipeline]
     │                                             │
     │ tracking + interaction                      │ image buffer + external detect
     │ detection                                   │
     ▼                                             ▼
process_frame() → LogEvent               _BatchCollector → trigger
     │                                             │
     └─────────────┐              ┌────────────────┘
                   ▼              ▼
          Orchestrator.request_space_snapshot(space_id)
                   │
                   ▼
          Collect all CameraSnapshots in space
                   │
                   ▼
          SpaceLogger.space_snapshot(space_id, space_name, snapshots)
                   │
          ┌────────┴────────────┐
          ▼                     ▼
   vision_enabled=true     vision_enabled=false
   (images + tracking)     (tracking data only)
          │                     │
          └────────┬────────────┘
                   ▼
          LLM call → unified JSON
          {target_present, cameras: {id: {description, coord?}}, reasoning}
                   │
                   ▼
          for each camera:
            ├─ _save_image(image_b64, camera_id)  [if vision_enabled]
            ├─ _db_insert(log_type="detect", camera_id, desc, coord, reasoning)
            └─ buffer for space log
                   │
                   ▼
          _db_insert(log_type="space", space_id, reasoning, raw_json)
```

---

## Phase 1: Foundation — CameraSnapshot + Snapshot Registry

### Step 1-1: `CameraSnapshot` dataclass

**파일:** `nlp/logger.py` (또는 공용 모듈)

```python
@dataclass
class CameraSnapshot:
    camera_id: str
    target_present: bool
    timestamp: float                         # time.monotonic()
    tracked_list: List[TrackedBBox] = field(default_factory=list)
    interactions: List[InteractionResult] = field(default_factory=list)
    image_b64: Optional[str] = None
    target_coordinate: Optional[List[float]] = None
```

### Step 1-2: `Orchestrator`에 snapshot registry 추가

**파일:** `core/orchestrator.py`

```python
class Orchestrator:
    def __init__(self, ...):
        ...
        self._snapshots: Dict[str, CameraSnapshot] = {}   # camera_id → latest
        self._snapshot_debounce: Dict[str, float] = {}     # space_id → last snapshot
        self._snapshot_lock = threading.Lock()
```

**메서드:**

```python
def update_snapshot(self, camera_id: str, snapshot: CameraSnapshot):
    """CameraWorker가 매 frame 종료마다 최신 snapshot 갱신"""
    with self._snapshot_lock:
        self._snapshots[camera_id] = snapshot

def request_space_snapshot(self, space_id: str):
    """interaction change 또는 vision detect success 시 호출
       → space 내 모든 camera snapshot 수집
       → debounce 체크
       → SpaceLogger.space_snapshot() 호출
    """
    if not self.space_logger:
        return

    # debounce (space당 5s cooldown)
    now = time.monotonic()
    last = self._snapshot_debounce.get(space_id, 0)
    if now - last < 5.0:
        logger.debug("[space:%s] snapshot debounce", space_id)
        return
    self._snapshot_debounce[space_id] = now

    # space의 모든 camera ID 조회
    with self._lock:
        space_cameras = [cam_id for cam_id, sid in self._cam_to_space.items() if sid == space_id]
    if not space_cameras:
        return

    # 각 camera의 최신 snapshot 수집
    snapshots: Dict[str, CameraSnapshot] = {}
    with self._snapshot_lock:
        for cam_id in space_cameras:
            snap = self._snapshots.get(cam_id)
            if snap is None:
                continue
            snapshots[cam_id] = snap

    if not snapshots:
        return

    space_name = next((s.name for s in self.spaces if s.id == space_id), space_id)
    self.space_logger.space_snapshot(
        space_id=space_id,
        space_name=space_name,
        snapshots=snapshots,
        vision_enabled=self.app_config.llm.vision_enabled,
    )
```

---

## Phase 2: Pipeline — tracking only, no LLM

### Step 2-1: `Pipeline.process_frame()`에서 `nlp_logger.log()` 제거

**파일:** `core/pipeline.py`

**변경:**
- Line 107-108: `self.nlp_logger.log([t], frame, ...)` 제거
- Line 127: `self.nlp_logger.log([t], frame, ...)` 제거
- `LogEvent` 리턴은 유지 (`LogEvent`에 `image_b64` 필드 필요 시 추가)
- `self.nlp_logger` 의존성 제거 → 생성자에서 제거

**결과:**
```python
# pipeline.py: state change 감지 시 (기존 line 107-108)
if hold >= self.config.thresholds.min_frames:
    # nlp_logger.log() 호출 제거
    log_event = LogEvent(...)
    self._prev_states[t.track_id] = state
    ...

# pipeline.py: interaction change 감지 시 (기존 line 127)
if self._interactions_changed(prev_interactions, interactions):
    # nlp_logger.log() 호출 제거
    log_event = LogEvent(...)
    self._prev_interactions[t.track_id] = interactions
```

### Step 2-2: `Pipeline.__init__()`에서 `nlp_logger` 제거

더 이상 queue-based LLM 호출을 하지 않으므로 `NLPLogger` 인스턴스가 필요 없음.
- `self.nlp_logger` 필드와 생성자 파라미터 제거
- 필요한 경우 `_save_image`는 `Orchestrator`나 `CameraWorker`가 처리

### Step 2-3: `CameraWorker` 변경 — snapshot 갱신 + trigger

**파일:** `core/orchestrator.py` ( `_CameraWorker` class )

**변경:**
```python
def _run(self):
    ...
    while not self.stop_event.is_set():
        ret, frame = self.cap.read()
        ...
        if frame_id % skip_interval == 0:
            detect, log_event = self.pipeline.process_frame(frame, frame_id)

            # 현재 상태 snapshot 생성 및 갱신
            image_b64 = None
            tracked_list = []
            interactions = []
            coord = None
            if detect.target_present:
                # vision_enabled 시 image encode
                if self._vision_enabled:
                    image_b64 = self._encode_frame(frame, detect.target_coordinate)
                # LogEvent에서 tracked_list, interactions 추출
                if log_event:
                    tracked_list = log_event.tracked_list
                    interactions = log_event.interactions
                    coord = log_event.target_coordinate

            snap = CameraSnapshot(
                camera_id=self.camera_id,
                target_present=detect.target_present,
                timestamp=time.monotonic(),
                tracked_list=tracked_list,
                interactions=interactions,
                image_b64=image_b64,
                target_coordinate=coord,
            )
            self.orchestrator.update_snapshot(self.camera_id, snap)

            # interaction change 발생 시 즉시 space snapshot trigger
            if log_event is not None and self.space_id:
                self.orchestrator.request_space_snapshot(self.space_id)
```

---

## Phase 3: SpaceLogger — `space_snapshot()` 구현

### Step 3-1: Prompt 구성 (두 모드)

**파일:** `nlp/logger.py`

**System prompt — vision_enabled=true (image 기반):**
```python
SNAPSHOT_VISION_PROMPT = (
    "You are an object behavior observation specialist analyzing a space from multiple camera angles. "
    "Each camera's image is labeled with its camera ID in brackets, e.g. '[livingroom]'.\n\n"
    "RULES:\n"
    "1) CAMERA IDs: Use ONLY the bare camera IDs from the labels.\n"
    "2) For each camera that shows a target object, provide:\n"
    '   - "description": one sentence describing the target\'s behavior and position\n'
    '   - "target_coordinate": normalized bounding box [x1,y1,x2,y2] if visible\n'
    "3) If a camera has no target, set description to 'no target detected'.\n"
    "4) Then synthesize all camera observations into one overall sentence in 'reasoning'.\n\n"
    'OUTPUT: Valid JSON: {"target_present": bool, "cameras": {cam_id: {description, target_coordinate?}}, "reasoning": "one sentence"}\n'
    "No markdown, no code fences.\n"
)
```

**System prompt — vision_enabled=false (tracking data 기반):**
```python
SNAPSHOT_TRACKING_PROMPT = (
    "You are an object behavior observation specialist analyzing tracking data from multiple cameras. "
    "Each camera's data is labeled with its camera ID.\n\n"
    "RULES:\n"
    "1) Camera IDs: Use ONLY the bare camera IDs from the labels.\n"
    "2) For each camera with tracking data, describe the target's behavior "
    "(movement direction, speed, interactions with nearby objects).\n"
    "3) If a camera has no target, set description to 'no target detected'.\n"
    "4) Synthesize all camera observations into one overall sentence in 'reasoning'.\n\n"
    'OUTPUT: Valid JSON: {"target_present": bool, "cameras": {cam_id: {description, target_coordinate?}}, "reasoning": "one sentence"}\n'
    "No markdown, no code fences.\n"
)
```

### Step 3-2: `SpaceLogger.space_snapshot()` 구현

```python
def space_snapshot(self, space_id: str, space_name: str,
                   snapshots: Dict[str, "CameraSnapshot"],
                   vision_enabled: bool) -> Optional[str]:
    """통합 space snapshot 처리. CV/LLM vision 모두 이 메서드로 수렴."""

    timestamp = datetime.now(timezone.utc).isoformat()
    self._ensure_client()
    if self.client is None:
        return None

    # LLM prompt 구성
    prompt_lines = [f"Timestamp: {timestamp}", f"Space: {space_name}", ""]

    user_messages = []
    if vision_enabled:
        # 이미지 기반 prompt
        user_messages.append({"type": "text", "text": f"Timestamp: {timestamp}\nSpace: {space_name}"})
        for cam_id in sorted(snapshots.keys()):
            snap = snapshots[cam_id]
            user_messages.append({"type": "text", "text": f"\n--- [{cam_id}] ---"})
            if snap.image_b64:
                user_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{snap.image_b64}"}
                })
            # tracking data도 함께 제공 (bbox 등)
            coord_str = f" | bbox {snap.target_coordinate}" if snap.target_coordinate else ""
            tracking_str = self._build_tracking_summary(snap.tracked_list, snap.interactions)
            if tracking_str:
                user_messages.append({"type": "text", "text": tracking_str + coord_str})
        system_prompt = SNAPSHOT_VISION_PROMPT
    else:
        # tracking data 기반 prompt
        for cam_id in sorted(snapshots.keys()):
            snap = snapshots[cam_id]
            if not snap.target_present:
                prompt_lines.append(f"- {cam_id}: no target detected")
                continue
            tracking_str = self._build_tracking_summary(snap.tracked_list, snap.interactions)
            coord_str = f" | bbox {snap.target_coordinate}" if snap.target_coordinate else ""
            prompt_lines.append(f"- {cam_id}: {tracking_str}{coord_str}")
        system_prompt = SNAPSHOT_TRACKING_PROMPT
        user_messages = [{"type": "text", "text": "\n".join(prompt_lines)}]

    # debounce
    if not self.debouncer.should_call(f"{space_id}_snapshot"):
        logger.debug("[space:%s] snapshot debounce suppress", space_id)
        return None

    # LLM 호출
    try:
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_messages},
            ],
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        return self._snapshot_fallback(space_id, snapshots, timestamp)

    parsed = self._parse_json_response(text, f"snapshot_{space_id}")
    if not parsed:
        return self._snapshot_fallback(space_id, snapshots, timestamp)

    # 결과 저장
    cameras_resp = parsed.get("cameras", {})
    reasoning = parsed.get("reasoning", "")
    target_present_all = any(s.target_present for s in snapshots.values())

    # per-camera 저장
    for cam_id, snap in snapshots.items():
        cam_resp = cameras_resp.get(cam_id, {})
        if isinstance(cam_resp, str):
            desc = cam_resp
            coord = None
        elif isinstance(cam_resp, dict):
            desc = cam_resp.get("description", "") or cam_resp.get("reasoning", "")
            coord = cam_resp.get("target_coordinate") or snap.target_coordinate
        else:
            desc = f"target={snap.target_present}"
            coord = snap.target_coordinate

        # image 저장 (vision_enabled 시)
        if vision_enabled and snap.image_b64:
            self._save_snapshot_image(snap.image_b64, space_name, cam_id, timestamp, coord)

        # per-camera detect log
        self._db_insert(
            log_type="detect",
            timestamp=timestamp,
            camera_id=cam_id,
            space_id=space_id,
            target_present=snap.target_present,
            description=desc,
            target_coordinate=coord,
            reasoning=reasoning,
        )

    # space-level log
    log_entry = {
        "target_present": target_present_all,
        "cameras": {cam_id: {"description": desc, "target_coordinate": snap.target_coordinate}
                    for cam_id, snap in snapshots.items()},
        "reasoning": reasoning,
    }
    log_text = _json.dumps(log_entry)
    self._db_insert(
        log_type="space",
        timestamp=timestamp,
        space_id=space_id,
        target_present=target_present_all,
        reasoning=reasoning,
        raw_json=log_text,
    )

    logger.info("[snapshot:%s] cameras=%d reasoning=%s", space_id, len(snapshots), reasoning)
    return log_text
```

### Step 3-3: Helper 메서드들

```python
def _build_tracking_summary(self, tracked_list: List[TrackedBBox],
                            interactions: List[InteractionResult]) -> str:
    """tracking data를 LLM-friendly text로 변환"""
    parts = []
    for t in tracked_list:
        movement = self._state_to_movement_str(t)
        parts.append(f"{t.class_name}: {movement}")
    if interactions:
        for ir in interactions:
            rel = {"interacting": "touching", "contact": "touching", "nearby": "near"}.get(ir.relation_type, ir.relation_type)
            parts.append(f"nearby: {ir.class_name} ({rel})")
    return " | ".join(parts)

def _save_snapshot_image(self, image_b64: str, space_name: str,
                         cam_id: str, timestamp: str, coord: Optional[List[float]] = None):
    """snapshot 이미지를 output에 저장 (bbox 옵션)"""
    try:
        capture_ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H:%M:%S")
        filename = f"{space_name}_{cam_id}_{capture_ts}.jpg"
        img_data = image_b64
        if coord:
            img_data = draw_normalized_bbox(image_b64, coord, label=space_name)
        (self.log_dir / filename).write_bytes(base64.b64decode(img_data))
    except Exception as e:
        logger.error("Failed to save snapshot image %s: %s", filename, e)

def _snapshot_fallback(self, space_id: str, snapshots: Dict[str, "CameraSnapshot"],
                       timestamp: str) -> str:
    """LLM 실패 시 structured fallback"""
    import json as _json
    cameras = {}
    for cam_id, snap in snapshots.items():
        desc = "no target detected"
        coord = None
        if snap.target_present and snap.tracked_list:
            desc = self._build_tracking_summary(snap.tracked_list, snap.interactions)
            coord = snap.target_coordinate
        cameras[cam_id] = {"description": desc}
        if coord:
            cameras[cam_id]["target_coordinate"] = coord

    log_entry = {
        "target_present": any(s.target_present for s in snapshots.values()),
        "cameras": cameras,
        "reasoning": " | ".join(f"{cid}: {c['description']}" for cid, c in cameras.items()),
    }
    log_text = _json.dumps(log_entry)
    self._db_insert(
        log_type="space",
        timestamp=timestamp,
        space_id=space_id,
        target_present=log_entry["target_present"],
        reasoning=log_entry["reasoning"],
        raw_json=log_text,
    )
    return log_text
```

### Step 3-4: `SpaceLogger.__init__()` 변경

```python
class SpaceLogger:
    def __init__(self, config, log_dir="logs", flush_threshold=0, repo=None, event_bus=None):
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._repo = repo
        self._event_bus = event_bus
        self.debouncer = LLMCallDebouncer(cooldown_seconds=5.0)
        self.client: Optional[OpenAI] = None
        self._lock = threading.Lock()
        # 제거: _buffer, _vision_buffer, _camera_counts (불필요)
```

---

## Phase 4: LLM Vision Pipeline 통합

### Step 4-1: `_VisionScheduler` 제거

**파일:** `core/orchestrator.py`

- `_VisionScheduler` 클래스 전체 제거
- `_SpaceState`, `_process_detection_step()`, `_transition_to_logging()` 제거
- `Orchestrator.__init__()`에서 `_vision_scheduler` 필드 제거
- `Orchestrator.start()`에서 scheduler 초기화/start 제거

### Step 4-2: `_BatchCollector` → snapshot trigger 연결

**파일:** `core/vision_worker.py` + `core/orchestrator.py`

`_BatchCollector`가 이미지를 수집하는 것은 유지. LLM vision의 "detect trigger" 방식:

**Option A:** `_BatchCollector`에 새로운 이미지가 들어올 때마다 `Orchestrator.update_snapshot()` 호출 (image 포함) + 주기적 detect check

**Option B:** `_VisionScheduler` 위치에 단순화된 detect loop (`_SimpleVisionDetector`) — 주기적으로 `_BatchCollector` buffer의 최신 이미지로 detect → 성공 시 `request_space_snapshot()`

```python
class _SimpleVisionDetector:
    """_VisionScheduler 대체: 주기적으로 각 camera의 최신 이미지를 detect하고
       target 발견 시 space snapshot trigger"""
    def __init__(self, ...):
        ...

    def _run(self):
        while not self._stop_event.is_set():
            for space in self._spaces:
                for cam_id in space.camera_ids:
                    collector = self._collectors.get(cam_id)
                    if not collector or not collector.buffer:
                        continue
                    image_b64 = collector.buffer[-1].image_b64
                    # 간단한 LLM detect or rule-based detect
                    # 성공 시 orchestrator.request_space_snapshot(space.id)
                    break  # 한 tick에 한 camera만
            self._stop_event.wait(0.1)  # 100ms polling 유지
```

### Step 4-3: `Orchestrator.start()` 정리

```python
def start(self):
    for cam in self.app_config.cameras:
        if cam.status != "active":
            continue
        self.add_camera(cam, ...)

    mode = self.app_config.mode
    if mode == "llm_vision" and self.space_logger:
        self._vision_detector = _SimpleVisionDetector(
            spaces=self.app_config.spaces,
            collectors=self._collectors,
            orchestrator=self,
            ...
        )
        self._vision_detector.start()
```

---

## Phase 5: 기존 코드 정리 (Dead Code 제거)

### Step 5-1: `NLPLogger` 정리

**파일:** `nlp/logger.py`

**제거할 것:**
- `NLPLogger.log()` (line 161-200) — pipeline이 직접 호출 중단
- `NLPLogger._log_fallback()` (line 202-229)
- `NLPLogger._queue`, `_LogTask` (line 53-62, 98)
- `NLPLogger._worker()`, `_process_task()` (line 510-591)
- `NLPLogger.vision_log()`, `_vision_queue`, `_VisionLogTask` (line 231-240)
- `NLPLogger._vision_worker()`, `_process_vision_task()` (line 242-285, 518-524)
- `NLPLogger.vision_detect()` (line 454-503)
- `NLPLogger._save_log()` (line 654-682) — `space_snapshot()`에서 직접 처리
- `NLPLogger._build_state_changes()`, `_build_prompt()` (line 593-652) — 불필요

**유지할 것:**
- `NLPLogger._db_insert()` (line 112-148) — or `SpaceLogger`로 이관
- `NLPLogger._save_image()` (line 150-159) — or `SpaceLogger`로 이관
- `NLPLogger._ensure_client()` (line 171-173 in old) — SpaceLogger에서 대체
- `NLPLogger.debouncer` — SpaceLogger에서 대체
- `NLPLogger._parse_json_response()` (line 420-452) — SpaceLogger로 이관

**결론:** `NLPLogger` 클래스는 제거하거나 `SpaceLogger`의 utility 메서드로 축소.

### Step 5-2: `SpaceLogger` 정리

**파일:** `nlp/logger.py`

**제거할 것:**
- `_buffer` (line 816), `_vision_buffer` (line 820), `_camera_counts` (line 818)
- `set_camera_count()` (line 869-870)
- `cleanup_space()` (line 872-876)
- `collect()` (line 878-885)
- `vision_collect()` (line 887-893)
- `flush_vision()` (line 895-935)
- `flush()` (line 937-1022)
- `try_flush()` (line 1024-1032)

**유지/변경:**
- `__init__()` 간소화
- `_ensure_client()` 유지
- `_db_insert()` 유지
- `_parse_json_response()` → `SpaceLogger`로 이동
- `space_snapshot()` 추가
- `_save_snapshot_image()` 추가
- `_build_tracking_summary()` 추가
- `_snapshot_fallback()` 추가

### Step 5-3: `main.py` 정리

```python
# flush timer 제거 (line 186-199 이전)
# orchestrator.flush_spaces() 호출 제거
# → try-finally에서도 불필요 (event-driven으로 즉시 처리)
```

### Step 5-4: `Orchestrator` 정리

- `flush_spaces()` 메서드 제거 (line 524-530)
- `_VisionScheduler` 관련 코드 제거 (line 56-206)
- `_vision_nlp_logger` 필드 제거 (line 362)
- `_ensure_vision_nlp()` 메서드 제거 (line 374-377)
- `reassign_camera()`에서 `flush()` 호출 제거 → `request_space_snapshot()`으로 대체

---

## Phase 6: Prompt System Prompts 정리

### Step 6-1: 새로운 Prompt 파일

**파일:** `nlp/prompts.py` (신규)

```python
# 기존 4개 prompt를 2개로 통합 정리

SNAPSHOT_VISION_PROMPT = """..."""  # vision_enabled=true (image 기반)
SNAPSHOT_TRACKING_PROMPT = """..."""  # vision_enabled=false (tracking data 기반)
```

### Step 6-2: `nlp/logger.py`에서 prompt import

```python
from nlp.prompts import SNAPSHOT_VISION_PROMPT, SNAPSHOT_TRACKING_PROMPT
```

기존 prompt (`SYSTEM_PROMPT`, `SPACE_SYSTEM_PROMPT`, `DETECT_SYSTEM_PROMPT`, `VISION_SPACE_SYSTEM_PROMPT`) 제거.

---

## Phase 7: Pipeline Config → `vision_enabled` 전달

### Step 7-1: `Pipeline.__init__()` 단순화

```python
class Pipeline:
    def __init__(self, config: PipelineConfig, camera_id: str, ...):
        # nlp_logger 파라미터 제거
        self.config = config
        self.camera_id = camera_id
        self.tracker = Tracker(config.yolo)
        self.interaction_detector = InteractionDetector(...)
        # _prev_states, _state_hold 등 state machine 유지
```

### Step 7-2: `_CameraWorker`에 `vision_enabled` 전달

```python
class _CameraWorker:
    def __init__(self, camera_id, pipeline, cap, source, stop_event,
                 frame_skip=0, on_finished=None, orchestrator=None,
                 space_id=None, vision_enabled=False):
        ...
        self._vision_enabled = vision_enabled
        self.orchestrator = orchestrator
        self.space_id = space_id
```

### Step 7-3: `Orchestrator.add_camera()`에서 전달

```python
def add_camera(self, camera, ...):
    ...
    mode = self.app_config.mode
    vision_enabled = self.app_config.llm.vision_enabled if hasattr(self.app_config.llm, 'vision_enabled') else False

    worker = _CameraWorker(
        camera_id=camera.id,
        pipeline=pipeline,
        ...
        orchestrator=self,
        space_id=space_id,
        vision_enabled=vision_enabled,
    )
```

---

## Phase 8: Integration & Verification

### Step 8-1: CV pipeline 동작 확인

```bash
# Mode: live (default) — CV pipeline
# video file: data/test.mp4 (target_classes: ["cat"])
docker compose up --build
# Console output에서 snapshot logging 확인
# [snapshot:livingroom] cameras=2 reasoning="..." (interaction change 시)
```

**Expected output:**
```
DEBUG [cam1] target 3 state WALKING->STOPPED hold=5/5
INFO  [snapshot:livingroom] cameras=2 reasoning="cat stopped near the sofa"
```

### Step 8-2: LLM vision pipeline 동작 확인

```bash
# Mode: llm_vision
# image source: rtsp or image sequence
docker compose up --build
```

**Expected output:**
```
INFO  [vision_detect] cam1 target_present=True reason="cat visible on sofa"
INFO  [snapshot:livingroom] cameras=2 reasoning="cat sitting on sofa, no interaction"
```

### Step 8-3: DB 검증

```bash
# SQLite
sqlite3 logs/tracking.db "SELECT log_type, camera_id, description FROM log_entries ORDER BY timestamp DESC LIMIT 10;"

# Expected: detect rows (per camera) + space row
```

### Step 8-4: Hot-reload + camera CRUD 검증

```bash
# Camera 추가 후 정상 동작
curl -X POST localhost:8000/api/cameras -H "Content-Type: application/json" \
  -d '{"id": "cam3", "source": "data/test2.mp4", "target_classes": ["dog"]}'
# → 새 camera의 snapshot이 space logging에 포함되는지 확인
```

---

## Progress

| Phase | Step | Status |
|-------|------|--------|
| Phase 1-1: CameraSnapshot dataclass | 생성 | ✅ |
| Phase 1-2: Orchestrator snapshot registry + `request_space_snapshot()` | 구현 | ✅ |
| Phase 2-1: Pipeline `nlp_logger.log()` 제거 | 수정 | ✅ |
| Phase 2-2: Pipeline `__init__()` 단순화 | 수정 | ✅ |
| Phase 2-3: CameraWorker snapshot 갱신 + trigger | 수정 | ✅ |
| Phase 3-1: Prompt 구성 (vision/tracking) | 작성 | ✅ |
| Phase 3-2: `SpaceLogger.space_snapshot()` | 구현 → **재수정 필요** | ⚠️ |
| Phase 3-3: Helper 메서드들 | 구현 | ✅ |
| Phase 3-4: `SpaceLogger.__init__()` 간소화 | 수정 | ✅ |
| Phase 4-1: `_VisionScheduler` 제거 | 삭제 | ✅ |
| Phase 4-2: `_SimpleVisionDetector` 구현 | 연결 → **재수정 필요** | ⚠️ |
| Phase 4-3: `Orchestrator.start()` 정리 | 수정 | ✅ |
| Phase 5-1: NLPLogger dead code 제거 | 정리 | ✅ |
| Phase 5-2: SpaceLogger dead code 제거 | 정리 | ✅ |
| Phase 5-3: main.py flush timer 제거 | 정리 | ✅ |
| Phase 5-4: Orchestrator dead code 제거 | 정리 | ✅ |
| Phase 6-1: `nlp/prompts.py` 신규 작성 | 작성 | ✅ |
| Phase 7-1~3: vision_enabled 전달 | 수정 | ✅ |
| Phase 8-1~4: Integration & Verification (Docker build + import test) | 테스트 | ✅ |
| **Phase 9-1: CameraSnapshot.images 필드 추가** | `nlp/logger.py` | ✅ |
| **Phase 9-2: _SimpleVisionDetector._run() buffer freeze** | `core/orchestrator.py` | ✅ |
| **Phase 9-3: _update_all_snapshots() frozen buffer 전달** | `core/orchestrator.py` | ✅ |
| **Phase 9-4: SpaceLogger.space_snapshot() multi-image 처리** | `nlp/logger.py` | ✅ |
| **Phase 9-5: SNAPSHOT_VISION_PROMPT multi-image 문구** | `nlp/prompts.py` | ✅ |
| **Phase 9-6: llm_vision 동기화 테스트** | `config_test_llm_vision.yaml` | ✅ |
| **회귀 테스트: cv_no_vision** | `config_test_cv_no_vision.yaml` | ✅ |
| **회귀 테스트: cv_vision** | `config_test_cv_vision.yaml` | ✅ |

---

## Phase 10: `min_frames` 히스테리시스 제거

### 배경
`min_frames`는 CV tracking state machine에서 상태 전환이 `min_frames` 프레임 이상 유지될 때만 `LogEvent`를 방출하도록 하는 플리커 방지 장치. 그러나:

1. **Snapshot 5s debounce**가 이미 중복 방지 역할을 함
2. **Interaction change**는 이미 `min_frames` 없이 즉시 LogEvent 방출 (pipeline.py:120)
3. YOLO `conf_threshold` + LLM `vision_detect`가 탐지 정밀도를 담당
4. 실제로 `min_frames` 때문에 1~2프레임만 탐지되는 케이스(cat 등)에서 LogEvent가 생성되지 않아 space snapshot이 아예 트리거되지 않음

### 변경 내용

### Step 10-1: `Thresholds` dataclass — `min_frames` 필드 제거

**파일:** `settings.py`

```python
@dataclass
class Thresholds:
    # min_frames: int = 3  ← 제거
    hysteresis: int = 5
    speed_slow: float = 10.0
    speed_fast: float = 40.0
    dash_threshold: float = 15.0
    rotation_threshold: float = 45.0
    distance: float = 50.0
    overlap: float = 0.3
```

### Step 10-2: `Pipeline.process_frame()` — `min_frames` 조건 제거

**파일:** `core/pipeline.py`

변경:
```python
# line 102-103 (state change): 항상 LogEvent 방출
if prev_state != state:
    log_event = LogEvent(
        tracked_list=[t],
        frame=frame,
        interactions=interactions,
        target_coordinate=target_coord,
        target_classes=self.config.target_classes,
    )
    self._prev_states[t.track_id] = state
    self._prev_frame_ids[t.track_id] = frame_id
    self._state_hold[t.track_id] = 0
    self._prev_interactions[t.track_id] = interactions

# line 115-117 (same state hold): state_hold 리셋 제거 (불필요)
# if hold >= self.config.thresholds.min_frames:  ← 제거
```

### Step 10-3: Config 파일 정리

- `configuration.yaml.example` — `min_frames: 3` 제거
- `api/models.py` — `ThresholdsConfig.min_frames` 필드 제거
- `docs/interface_contract.md` — `min_frames` 문서 제거
- 모든 `config_test_*.yaml`에서 `min_frames` 제거

---

## Phase 11: `space_snapshot()` — `target_classes` + `llm_system_prompt` 주입

### 배경
`vision_detect()`는 `target_classes`와 `llm_system_prompt`를 prompt에 주입하지만, `space_snapshot()`은 이 정보를 전혀 받지 못함. 때문에 snapshot prompt가 "무엇을 찾아야 하는지"를 LLM에게 알려주지 않아, LLM이 엉뚱한 사물(노란 매트 등)을 target으로 설명하는 false positive 발생.

과거 코드(commit e574e66)에서는 `_process_vision_batch_space()`가 `target_classes`와 `llm_system_prompt`를 정상적으로 주입하고 있었음.

### 설계

```
request_space_snapshot(space_id)
  │
  ├─ target_classes = _get_target_classes(space_id)
  ├─ llm_system_prompt = space.llm_system_prompt
  │
  └─ space_snapshot(space_id, space_name, snapshots,
                    vision_enabled, target_classes, llm_system_prompt)
       │
       ├─ vision_enabled=true:
       │   prompt에 target_classes + llm_system_prompt 주입
       ├─ vision_enabled=false:
       │   tracking prompt에 target_classes 주입
       └─ LLM 호출
```

### Step 11-1: `SpaceLogger.space_snapshot()` 시그니처 변경

**파일:** `nlp/logger.py`

```python
def space_snapshot(self, space_id: str, space_name: str,
                   snapshots: Dict[str, CameraSnapshot],
                   vision_enabled: bool,
                   target_classes: list[str] | None = None,
                   llm_system_prompt: str | None = None) -> Optional[str]:
```

### Step 11-2: vision_enabled 경로에 prompt 주입

`user_messages`에 `target_classes` 추가 (old `_process_vision_batch_space`와 동일 패턴):
```python
if target_classes:
    unique_classes = list(dict.fromkeys(target_classes))
    user_messages.append({"type": "text", "text": f"\nTarget objects: {', '.join(unique_classes)}"})

parts = [SNAPSHOT_VISION_PROMPT]
if llm_system_prompt:
    parts.append(f"Additional instructions:\n{llm_system_prompt}")
system_prompt = "\n".join(parts)
```

### Step 11-3: `Orchestrator.request_space_snapshot()` 변경

**파일:** `core/orchestrator.py`

`target_classes` 수집 + `llm_system_prompt` 조회 후 전달:
```python
target_classes = self._get_target_classes_for_space(space_id)
space_obj = next((s for s in self.app_config.spaces if s.id == space_id), None)
llm_system_prompt = space_obj.llm_system_prompt if space_obj else None

self.space_logger.space_snapshot(
    space_id=space_id,
    space_name=space_name,
    snapshots=snapshots,
    vision_enabled=self.app_config.llm.vision_enabled,
    target_classes=target_classes,
    llm_system_prompt=llm_system_prompt,
)
```

### Step 11-4: `_get_target_classes_for_space()` 헬퍼

`_SimpleVisionDetector._get_target_classes()`와 동일한 로직을 `Orchestrator`에서도 사용할 수 있도록 추출 (또는 재사용).

---

## Test Plan

### Phase 10 검증
- `cv_no_vision` 테스트: `min_frames` 제거 후 cat이 탐지된 프레임에서 즉시 LogEvent → `request_space_snapshot()` 호출 확인
- 기대: livingfront frame 392에서 target_present=true → snapshot trigger (5s debounce 통과 시)

### Phase 11 검증
- `llm_vision` 테스트: snapshot reasoning이 "yellow object" 대신 target_classes("cat")을 고려한 설명으로 변경되는지 확인
- `cv_vision` 테스트: vision_enabled=true CV pipeline에서도 동일하게 target_classes 주입 확인

---

## Progress

| Phase | Step | Status |
|-------|------|--------|
| Phase 1-1: CameraSnapshot dataclass | 생성 | ✅ |
| Phase 1-2: Orchestrator snapshot registry + `request_space_snapshot()` | 구현 | ✅ |
| Phase 2-1: Pipeline `nlp_logger.log()` 제거 | 수정 | ✅ |
| Phase 2-2: Pipeline `__init__()` 단순화 | 수정 | ✅ |
| Phase 2-3: CameraWorker snapshot 갱신 + trigger | 수정 | ✅ |
| Phase 3-1: Prompt 구성 (vision/tracking) | 작성 | ✅ |
| Phase 3-2: `SpaceLogger.space_snapshot()` | 구현 → **재수정 필요** | ⚠️ |
| Phase 3-3: Helper 메서드들 | 구현 | ✅ |
| Phase 3-4: `SpaceLogger.__init__()` 간소화 | 수정 | ✅ |
| Phase 4-1: `_VisionScheduler` 제거 | 삭제 | ✅ |
| Phase 4-2: `_SimpleVisionDetector` 구현 | 연결 → **재수정 필요** | ⚠️ |
| Phase 4-3: `Orchestrator.start()` 정리 | 수정 | ✅ |
| Phase 5-1: NLPLogger dead code 제거 | 정리 | ✅ |
| Phase 5-2: SpaceLogger dead code 제거 | 정리 | ✅ |
| Phase 5-3: main.py flush timer 제거 | 정리 | ✅ |
| Phase 5-4: Orchestrator dead code 제거 | 정리 | ✅ |
| Phase 6-1: `nlp/prompts.py` 신규 작성 | 작성 | ✅ |
| Phase 7-1~3: vision_enabled 전달 | 수정 | ✅ |
| Phase 8-1~4: Integration & Verification (Docker build + import test) | 테스트 | ✅ |
| Phase 9-1: CameraSnapshot.images 필드 추가 | `nlp/logger.py` | ✅ |
| Phase 9-2: _SimpleVisionDetector._run() buffer freeze | `core/orchestrator.py` | ✅ |
| Phase 9-3: _update_all_snapshots() frozen buffer 전달 | `core/orchestrator.py` | ✅ |
| Phase 9-4: SpaceLogger.space_snapshot() multi-image 처리 | `nlp/logger.py` | ✅ |
| Phase 9-5: SNAPSHOT_VISION_PROMPT multi-image 문구 | `nlp/prompts.py` | ✅ |
| Phase 9-6: llm_vision 동기화 테스트 | `config_test_llm_vision.yaml` | ✅ |
| 회귀 테스트: cv_no_vision | `config_test_cv_no_vision.yaml` | ✅ |
| 회귀 테스트: cv_vision | `config_test_cv_vision.yaml` | ✅ |
| **Phase 10-1: Thresholds dataclass min_frames 제거** | `settings.py` | ✅ |
| **Phase 10-2: Pipeline process_frame() 조건 제거** | `core/pipeline.py` | ✅ |
| **Phase 10-3: Config 파일 정리** | `.yaml`/`api/models.py`/`docs/` | ✅ |
| **Phase 11-1: space_snapshot() 시그니처 + prompt 주입** | `nlp/logger.py` | ✅ |
| **Phase 11-2: request_space_snapshot() prompt 전달** | `core/orchestrator.py` | ✅ |
| **Phase 10+11 통합 테스트** | Docker + cv_no_vision 25s | ✅ |

---

## Phase 9: LLM Vision Buffer Freeze — Snapshot 시점 동기화

### 배경
`_SimpleVisionDetector`가 `vision_detect()` 호출 후 LLM 응답을 기다리는 동안(1~수초) `_BatchCollector`는 계속 0.5s 간격으로 buffer를 갱신한다. `_update_all_snapshots()`에서 `collector.buffer[-1]`를 다시 읽으면 detection에 사용된 이미지와 다른 프레임이 snapshot에 포함되는 문제 발생.

또한 LLM vision 파이프라인은 **stateless**하므로, 단일 이미지가 아닌 시간순 이미지 시퀀스(2.5s × 5장)를 LLM에 전달하여 시공간 분석을 수행해야 함.

### 설계

```
_T0 iteration start:
  freeze = {cam_id: list(collector.buffer) for cam_id in space.camera_ids}  ← 모든 카메라 buffer 복사
  entry = collector.buffer[-1]                                               ← detect용 1장
  
  vision_detect(entry.image_b64) ──→ LLM (응답시간 무관)
  
_T0+N(1~30초):
  target_present=True:
    _update_all_snapshots(space_id, frozen_buffers, detect_cam_id, detect_entry=entry)
      detect_cam:   detect_entry + frozen[detect_cam_id][:-1]  → 총 5장
      other cams:   frozen[cam_id]                              → 5장
  
    space_snapshot() → LLM prompt:
      [hallway]     {detect장, t-2.0, t-1.5, t-1.0, t-0.5}    ← T0 기준 동일 시간축
      [livingroom]  {t-2.0, t-1.5, t-1.0, t-0.5, t}
      [livingfront] {t-2.0, t-1.5, t-1.0, t-0.5, t}
```

원본 `_BatchCollector.buffer`는 계속 0.5s마다 sliding. freeze 복사본만 snapshot에서 사용.

### CV pipeline 영향: 없음
- `_SimpleVisionDetector`: `mode == "llm_vision"`에서만 생성
- `_update_all_snapshots()`: `_SimpleVisionDetector` 메서드 (CV 경로와 무관)
- `CameraWorker` → `request_space_snapshot()`: CV는 `CameraSnapshot.image_b64`(단일)만 사용, `images=[]`
- `space_snapshot()`: `images`가 비어있으면 기존 `image_b64` fallback

### Phase 9-1: `CameraSnapshot.images` 필드 추가

**파일:** `nlp/logger.py`

```python
@dataclass
class CameraSnapshot:
    camera_id: str
    target_present: bool
    timestamp: float
    tracked_list: List = field(default_factory=list)
    interactions: List = field(default_factory=list)
    image_b64: Optional[str] = None        # 단일 이미지 (CV/fallback)
    images: List[str] = field(default_factory=list)  # 시계열 이미지 (LLM vision)
    target_coordinate: Optional[List[float]] = None
```

### Phase 9-2: `_SimpleVisionDetector._run()` buffer freeze

**파일:** `core/orchestrator.py`

```python
def _run(self):
    while not self._stop_event.is_set():
        try:
            if self._is_all_finished():
                self._stop_event.wait(1.0)
                continue

            for space_id, space in self._spaces.items():
                if not space.camera_ids:
                    continue
                idx = self._detect_index.get(space_id, 0)
                if idx >= len(space.camera_ids):
                    self._detect_index[space_id] = 0
                    continue
                cam_id = space.camera_ids[idx]
                collector = self._collectors.get(cam_id)
                if not collector or not collector.buffer:
                    self._detect_index[space_id] = idx + 1
                    break
                
                entry = collector.buffer[-1]
                age = time.monotonic() - entry.captured_at
                if age > self._config.max_stale_threshold:
                    self._detect_index[space_id] = idx + 1
                    break

                # freeze: T0 시점 모든 카메라 buffer 복사
                frozen_buffers: Dict[str, List] = {}
                for cid in space.camera_ids:
                    coll = self._collectors.get(cid)
                    if coll and coll.buffer:
                        frozen_buffers[cid] = list(coll.buffer)

                logger.debug("[vision-detector] cam=%s calling vision_detect", cam_id)
                result = self._space_logger.vision_detect(
                    camera_id=cam_id,
                    image_b64=entry.image_b64,
                    llm_system_prompt=space.llm_system_prompt or None,
                    target_classes=self._get_target_classes(space_id) or None,
                )
                if result is None:
                    self._detect_index[space_id] = idx + 1
                    break

                if result.get("target_present", False):
                    logger.info("[space:%s] cam=%s target_present=True → snapshot", space_id, cam_id)
                    self._update_all_snapshots(space_id, frozen_buffers, detect_cam_id=cam_id, detect_entry=entry)
                    self._orchestrator.request_space_snapshot(space_id)
                    return

                self._detect_index[space_id] = idx + 1
                break
            self._stop_event.wait(0.1)
        except Exception:
            logger.exception("[vision-detector] _run crashed")
            return
```

### Phase 9-3: `_update_all_snapshots()` frozen buffer 전달

**파일:** `core/orchestrator.py`

```python
def _update_all_snapshots(self, space_id: str, frozen_buffers: Dict[str, List],
                           detect_cam_id: str | None = None, detect_entry=None):
    space = self._spaces.get(space_id)
    if not space:
        return
    for cam_id in space.camera_ids:
        images: List[str] = []
        if cam_id == detect_cam_id and detect_entry is not None:
            # detect에 사용된 이미지가 첫장, 나머지는 frozen buffer에서
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
            images=images,
        )
        self._orchestrator.update_snapshot(cam_id, snap)
```

### Phase 9-4: `SpaceLogger.space_snapshot()` multi-image 처리

**파일:** `nlp/logger.py`

`space_snapshot()`의 vision_enabled 경로:
```python
# vision_enabled=True
if snap.images:
    for img_b64 in snap.images:
        user_messages.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })
elif snap.image_b64:
    # 단일 이미지 fallback (CV pipeline 등)
    user_messages.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{snap.image_b64}"}
    })
```

카메라 간 구분을 위한 `--- [cam_id] ---` 마커는 유지.

### Phase 9-5: `SNAPSHOT_VISION_PROMPT` 문구 수정

**파일:** `nlp/prompts.py`

```python
SNAPSHOT_VISION_PROMPT = (
    "You are an object behavior observation specialist analyzing a space from multiple camera angles. "
    "Each group of images is labeled with its camera ID in brackets, e.g. '[livingroom]'. "
    "Images within a group are in chronological order (earliest first).\n\n"
    ...
)
```

### Phase 9-6: 동기화 테스트

```bash
docker compose run --rm \
  -v $(pwd)/config_test_llm_vision.yaml:/app/config_test.yaml:ro \
  tracking-cano \
  python main.py --live "" --config /app/config_test.yaml --verbose
```

**검증 포인트:**
1. 모든 카메라의 frozen buffer가 동일 T0 시점인지 확인 (captured_at 비교)
2. detect camera: detect_entry가 첫번째 이미지인지 확인
3. LLM 응답에 여러 프레임에 걸친 시계열 설명이 포함되는지 확인 (예: "over 2.5 seconds the cat moved from ... to ...")

---

## Bugs Fixed

### Bug 1: SpaceLogger initialized with empty `api_key` (`main.py:149`)
- **Root cause**: `SpaceLogger(PipelineConfig().llm, ...)` — used a default-constructed `PipelineConfig` whose `llm.api_key=""` instead of the loaded config `app_config.llm` (which has `api_key` populated from `LLM_KEY` env var).
- **Fix**: Changed to `SpaceLogger(app_config.llm, ...)`.
- **Symptom**: `_ensure_client()` never created the OpenAI client; `vision_detect()` always returned `None` silently; no LLM calls were made.

### Bug 2: `_SimpleVisionDetector` `_run()` — silent crash on exception (pre-existing pattern)
- **Root cause**: Daemon threads silently die on unhandled exceptions. If the detector hit any error (e.g., empty `_spaces`, key error), the thread terminated with no log.
- **Fix**: Wrapped the main loop in `try/except` with `logger.exception()`.

## Test Results

### Test 1: `cv_no_vision` — CV pipeline, no LLM
- **Result**: ✅ Completed. All 3 cameras ran through frames. No `target_present=true` was logged because the test videos do not contain detectable cats at the frame positions captured. Camera hallway crashed due to concurrent YOLO model loading (pre-existing issue, not related to this refactoring).
- **Snapshot**: Not triggered — no interaction change detected (no targets).

### Test 2: `cv_vision` — CV pipeline, `vision_enabled=true`
- **Result**: ⚠️ Partially completed. Hallway camera crashed (`OSError: [Errno 22] Invalid argument` due to concurrent YOLO model loading). Livingroom and livingfront ran but detected no targets.
- **Known issue**: 3 `_CameraWorker` threads all call `_ensure_loaded()` simultaneously on first frame, causing file contention on `yolo26n.pt`.

### Test 3: `llm_vision` — LLM-only vision pipeline
- **Result**: ✅ Complete end-to-end working.
  - `_BatchCollector` captured 58 frames per camera (29s video at 0.5s intervals)
  - `_SimpleVisionDetector` polled buffers every 100ms, processed cameras round-robin
  - `vision_detect()` called LLM with camera image → received `200 OK` response
  - `[snapshot:room_livingroom] cameras=3 reasoning="..."` logged
- **Snapshot**: Successfully triggered. LLM detected "bag" in the scene (the test videos contain a bag/carrier, not a cat).

### Pre-existing Issues (not introduced by this refactoring)
1. **YOLO model concurrent loading crash**: Multiple `_CameraWorker` threads calling `_ensure_loaded()` simultaneously causes `OSError: [Errno 22] Invalid argument`. Solution: add file-level lock around YOLO loading in `tracker.py`.
2. **No cat detections**: Test videos seem to not contain detectable cats at visible frame positions. `conf_threshold: 0.05` yields zero detections across all cameras.

---

## Phase 12: Snapshot 이미지 저장 + detect log 누락 버그 수정

### 배경
3개 파이프라인(cv_no_vision, cv_vision, llm_vision) 통합 테스트 결과 다음 문제 발견:

| 문제 | 증상 | 심각도 |
|------|------|--------|
| **A**: `_snapshot_fallback()` detect log 누락 | space log 2건인데 per-camera detect가 3건뿐. 나머지 3건 증발. | 🔴 |
| **B**: 이미지 저장이 `vision_enabled`에 막힘 | vision off면 image_b64가 있어도 저장 안 함 | 🟠 |
| **C**: `image_b64` 인코딩 3중 조건 차단 | CV vision에서 LogEvent 없는 카메라는 image_b64=None → 저장 불가 | 🟠 |
| **D**: 이미지 저장 위치 `logs/` | 유저 의도는 `output/` 디렉토리 | 🟡 |
| **E**: `target_present`가 CV 결과만 반영 | LLM이 target=1이라도 DB에 target=0으로 기록 | 🟡 |

### 수정 상세

#### Step 12-1: `_snapshot_fallback()`에 detect log + image 저장 추가

**파일:** `nlp/logger.py` lines 396-423

**변경 전:**
```python
def _snapshot_fallback(self, space_id, snapshots, timestamp):
    cameras = {}
    for cam_id, snap in snapshots.items():
        desc = "no target detected"
        coord = None
        if snap.target_present and snap.tracked_list:
            desc = self._build_tracking_summary(...)
            coord = snap.target_coordinate
        cameras[cam_id] = {"description": desc}
        if coord:
            cameras[cam_id]["target_coordinate"] = coord

    log_entry = {"target_present": ..., "cameras": cameras, "reasoning": ...}
    self._db_insert(log_type="space", ...)   # ← space log만 insert
    return log_text
```

**변경 후:**
```python
def _snapshot_fallback(self, space_id, snapshots, timestamp, space_name=None):
    target_present_all = any(s.target_present for s in snapshots.values())
    cameras = {}
    for cam_id, snap in snapshots.items():
        desc = "no target detected"
        coord = None
        if snap.target_present and snap.tracked_list:
            desc = self._build_tracking_summary(...)
            coord = snap.target_coordinate
        cameras[cam_id] = {"description": desc}
        if coord:
            cameras[cam_id]["target_coordinate"] = coord

        if snap.image_b64 or snap.images:                      # ← image 저장 추가
            img_to_save = snap.image_b64 or snap.images[0]
            self._save_snapshot_image(img_to_save, space_name, cam_id, timestamp, coord)

        self._db_insert(                                        # ← detect log insert 추가
            log_type="detect", timestamp=timestamp,
            camera_id=cam_id, space_id=space_id,
            target_present=snap.target_present,
            description=desc, target_coordinate=coord,
        )

    log_entry = {"target_present": target_present_all, "cameras": cameras, "reasoning": ...}
    self._db_insert(log_type="space", ...)
    return log_text
```

또한 `space_snapshot()` 내부에서 `_snapshot_fallback()`을 호출하는 3개 지점(240, 315, 320)에 `space_name` 전달 추가 필요.

#### Step 12-2: 이미지 저장 조건에서 `vision_enabled` 가드 제거

**파일:** `nlp/logger.py` line 338

```python
# 변경 전
if vision_enabled and (snap.image_b64 or snap.images):

# 변경 후
if snap.image_b64 or snap.images:
```

효과: vision on/off 무관하게 snapshot 시점에 이미지가 있으면 항상 저장.

#### Step 12-3: 이미지 저장 경로 `logs/` → `output/`

**파일:** `nlp/logger.py` line 392

```python
# 변경 전
(self.log_dir / filename).write_bytes(base64.b64decode(img_data))

# 변경 후
(Path("output") / filename).write_bytes(base64.b64decode(img_data))
```

추가로 `_save_snapshot_image()` 시그니처에 `space_name`을 받지 않아도 filename에 포함되어 있으므로 유지.

#### Step 12-4: `CameraWorker`에서 항상 `image_b64` 인코딩

**파일:** `core/orchestrator.py` lines 285-291

**변경 전:**
```python
image_b64 = None
tracked_list = []
interactions = []
coord = None
if detect.target_present:
    if self._vision_enabled and log_event is not None and log_event.frame is not None:
        image_b64 = self._encode_frame(log_event.frame, detect.target_coordinate, label=detect.class_name)
    if log_event is not None:
        ...
```

**변경 후:**
```python
image_b64 = None
tracked_list = []
interactions = []
coord = None
if frame is not None:
    image_b64 = self._encode_frame(frame, None)
if detect.target_present:
    if log_event is not None:
        tracked_list = log_event.tracked_list
        interactions = log_event.interactions or []
        coord = log_event.target_coordinate
    else:
        coord = detect.target_coordinate

snap = CameraSnapshot(
    camera_id=self.camera_id,
    target_present=detect.target_present,
    timestamp=time.monotonic(),
    tracked_list=tracked_list,
    interactions=interactions,
    image_b64=image_b64,
    target_coordinate=coord,
)
```

주의: `_encode_frame()`은 `label` 매개변수가 없을 수 있으므로 시그니처 확인 필요. `draw_normalized_bbox`는 coord가 None이면 bbox를 그리지 않으므로 coord=None 전달 가능.

#### Step 12-5: `target_present`를 CV + LLM OR 연산으로 변경

**파일:** `nlp/logger.py` lines 322-367

**현재 로직:**
```python
cameras_resp = parsed.get("cameras", {})
reasoning = parsed.get("reasoning", "")
target_present_all = any(s.target_present for s in snapshots.values())  # CV only

for cam_id, snap in snapshots.items():
    cam_resp = cameras_resp.get(cam_id, {})
    if isinstance(cam_resp, dict):
        desc = cam_resp.get("description", "")
        coord = cam_resp.get("target_coordinate") or snap.target_coordinate
    ...
    self._db_insert(log_type="detect", target_present=snap.target_present, ...)  # CV only
```

**변경 후:**
```python
cameras_resp = parsed.get("cameras", {})
reasoning = parsed.get("reasoning", "")

per_camera_present: List[bool] = []
for cam_id, snap in snapshots.items():
    cam_resp = cameras_resp.get(cam_id, {})
    if isinstance(cam_resp, dict):
        desc = cam_resp.get("description", "") or cam_resp.get("reasoning", "")
        coord = cam_resp.get("target_coordinate") or snap.target_coordinate
        llm_present = cam_resp.get("target_present", False)
    elif isinstance(cam_resp, str):
        desc = cam_resp
        coord = None
        llm_present = False
    else:
        desc = f"target={snap.target_present}"
        coord = snap.target_coordinate
        llm_present = False

    merged_present = snap.target_present or llm_present
    per_camera_present.append(merged_present)

    if snap.image_b64 or snap.images:
        img_to_save = snap.image_b64 or snap.images[0]
        self._save_snapshot_image(img_to_save, space_name, cam_id, timestamp, coord)

    self._db_insert(
        log_type="detect", timestamp=timestamp,
        camera_id=cam_id, space_id=space_id,
        target_present=merged_present,  # ← CV || LLM
        description=desc, target_coordinate=coord, reasoning=reasoning,
    )

target_present_all = any(per_camera_present)  # ← space-level도 OR 반영
```

Prompt 수정도 필요 — per-camera JSON 응답에 `target_present` 필드 추가 요청:

**파일:** `nlp/prompts.py` line ~30

```python
# 현재 SNAPSHOT_TRACKING_PROMPT / SNAPSHOT_VISION_PROMPT에
# per-camera 응답 형식에 target_present 필드 명시
# Before: "{cam_id}: {{"description": "...", "target_coordinate": "..."}}"
# After:  "{cam_id}: {{"target_present": bool, "description": "...", "target_coordinate": "..."}}"
```

### 변경 요약

| Step | 파일 | 라인 | 변경 내용 |
|------|------|------|----------|
| 12-1 | `nlp/logger.py` | 396-423 | `_snapshot_fallback()`에 detect log + image 저장 루프 추가 |
| 12-1 | `nlp/logger.py` | 240, 315, 320 | `_snapshot_fallback()` 호출부에 `space_name` 전달 |
| 12-2 | `nlp/logger.py` | 338 | `if vision_enabled and (...)` → `if snap.image_b64 or snap.images:` |
| 12-3 | `nlp/logger.py` | 392 | `self.log_dir` → `Path("output")` |
| 12-4 | `core/orchestrator.py` | 285-291 | `image_b64` 항상 인코딩 (frame 사용) |
| 12-5 | `nlp/logger.py` | 322-367 | `target_present` CV \|\| LLM OR |
| 12-5 | `nlp/prompts.py` | ~30 | per-camera 응답에 `target_present` 필드 추가 |

### Test Plan

1. `cv_no_vision`: space log 1건당 detect log 3건 확인, `output/`에 이미지 3장 저장 확인
2. `cv_vision`: 동일 + LLM JSON의 `target_present`가 CV 결과와 OR되어 반영되는지 확인
3. `llm_vision`: 동일 + 모든 카메라 이미지 저장 + `target_present` 정확도 확인

---

## Progress

| Phase | Step | Status |
|-------|------|--------|
| Phase 1-1: CameraSnapshot dataclass | 생성 | ✅ |
| Phase 1-2: Orchestrator snapshot registry + `request_space_snapshot()` | 구현 | ✅ |
| Phase 2-1: Pipeline `nlp_logger.log()` 제거 | 수정 | ✅ |
| Phase 2-2: Pipeline `__init__()` 단순화 | 수정 | ✅ |
| Phase 2-3: CameraWorker snapshot 갱신 + trigger | 수정 | ✅ |
| Phase 3-1: Prompt 구성 (vision/tracking) | 작성 | ✅ |
| Phase 3-2: `SpaceLogger.space_snapshot()` | 구현 → **재수정 필요** | ⚠️ |
| Phase 3-3: Helper 메서드들 | 구현 | ✅ |
| Phase 3-4: `SpaceLogger.__init__()` 간소화 | 수정 | ✅ |
| Phase 4-1: `_VisionScheduler` 제거 | 삭제 | ✅ |
| Phase 4-2: `_SimpleVisionDetector` 구현 | 연결 → **재수정 필요** | ⚠️ |
| Phase 4-3: `Orchestrator.start()` 정리 | 수정 | ✅ |
| Phase 5-1: NLPLogger dead code 제거 | 정리 | ✅ |
| Phase 5-2: SpaceLogger dead code 제거 | 정리 | ✅ |
| Phase 5-3: main.py flush timer 제거 | 정리 | ✅ |
| Phase 5-4: Orchestrator dead code 제거 | 정리 | ✅ |
| Phase 6-1: `nlp/prompts.py` 신규 작성 | 작성 | ✅ |
| Phase 7-1~3: vision_enabled 전달 | 수정 | ✅ |
| Phase 8-1~4: Integration & Verification (Docker build + import test) | 테스트 | ✅ |
| Phase 9-1: CameraSnapshot.images 필드 추가 | `nlp/logger.py` | ✅ |
| Phase 9-2: _SimpleVisionDetector._run() buffer freeze | `core/orchestrator.py` | ✅ |
| Phase 9-3: _update_all_snapshots() frozen buffer 전달 | `core/orchestrator.py` | ✅ |
| Phase 9-4: SpaceLogger.space_snapshot() multi-image 처리 | `nlp/logger.py` | ✅ |
| Phase 9-5: SNAPSHOT_VISION_PROMPT multi-image 문구 | `nlp/prompts.py` | ✅ |
| Phase 9-6: llm_vision 동기화 테스트 | `config_test_llm_vision.yaml` | ✅ |
| 회귀 테스트: cv_no_vision | `config_test_cv_no_vision.yaml` | ✅ |
| 회귀 테스트: cv_vision | `config_test_cv_vision.yaml` | ✅ |
| Phase 10-1: Thresholds dataclass min_frames 제거 | `settings.py` | ✅ |
| Phase 10-2: Pipeline process_frame() 조건 제거 | `core/pipeline.py` | ✅ |
| Phase 10-3: Config 파일 정리 | `.yaml`/`api/models.py`/`docs/` | ✅ |
| Phase 11-1: space_snapshot() 시그니처 + prompt 주입 | `nlp/logger.py` | ✅ |
| Phase 11-2: request_space_snapshot() prompt 전달 | `core/orchestrator.py` | ✅ |
| Phase 10+11 통합 테스트 | Docker + cv_no_vision 25s | ✅ |
| **Phase 12-1: _snapshot_fallback() detect log + image** | `nlp/logger.py` | ✅ |
| **Phase 12-2: image 저장 vision_enabled 가드 제거** | `nlp/logger.py` | ✅ |
| **Phase 12-3: 저장 경로 output/** | `nlp/logger.py` | ✅ |
| **Phase 12-4: CameraWorker 항상 image_b64 인코딩** | `core/orchestrator.py` | ✅ |
| **Phase 12-5: target_present CV \|\| LLM OR** | `nlp/logger.py` + `nlp/prompts.py` | ✅ |
| **Phase 12 통합 테스트** | Docker + vision_enabled=true | ✅ |
| **Phase 13-1: Prompt hallucination guard** | `nlp/prompts.py` — "no target=omit", "uncertain=false" | ✅ |
| **Phase 13-2: CameraWorker raw frame only** | `core/orchestrator.py` — bbox at save time only | ✅ |
| **Phase 13 통합 테스트** | Docker + 3 configs | ✅ |

---

## Phase 14: DB Schema 리팩토링 — batch_id + subject_id + reasoning 통합

### 배경
Phase 12-5에서 OR 로직으로 target_present 개선, Phase 13에서 prompt 개선으로 LLM 응답 품질이 안정화됨. 이제 DB 스키마를 실제 사용 패턴에 맞게 정리: (1) snapshot 단위 그룹핑을 위한 batch_id, (2) entity 식별자 field 명칭 통일, (3) description/reasoning 중복 제거.

### 설계

**변경 전 스키마 (LogEntry):**

| column | type | 용도 |
|--------|------|------|
| id | Integer PK | auto-increment (앱에서 미사용) |
| camera_id | String | detect=카메라명, space=NULL |
| space_id | String | 공간 식별자 |
| reasoning | Text | detect: space reasoning 복사, space: reasoning |
| description | Text | detect: per-camera desc, space: NULL |

**변경 후 스키마:**

| column | type | 용도 |
|--------|------|------|
| id | Integer PK | auto-increment |
| batch_id | String UUID | snapshot 그룹 키, NOT NULL indexed |
| subject_id | String | detect=카메라명, space=space_id (was `camera_id`) |
| log_type | String | "detect" / "space" |
| timestamp | DateTime | 이벤트 시각 |
| target_present | Bool | |
| description | Text | detect: per-camera desc, space: space reasoning |
| target_coordinate | Text | JSON list or NULL |
| raw_json | Text | space log 전체 dump |
| created_at | DateTime | |

**변경 요약:**
1. `id` (PK) 유지
2. `camera_id` → `subject_id` (detect=카메라, space=공간)
3. `space_id` column **제거** (batch_id join으로 대체)
4. `reasoning` column **제거** → `description`에 통합
5. `batch_id` column **추가** (UUID4, snapshot 단위 그룹핑)

**Query 예시:**
```sql
-- 특정 snapshot의 모든 데이터 조회
SELECT * FROM log_entries WHERE batch_id = 'abc123...' ORDER BY log_type;

-- 최근 space 이벤트 N개
SELECT * FROM log_entries WHERE log_type='space' ORDER BY timestamp DESC LIMIT 10;

-- 특정 공간의 모든 detect
SELECT * FROM log_entries WHERE space_id='room_livingroom' AND log_type='detect';
```

### 변경 파일 및 단계

#### Phase 14-1: `storage/database.py` — 모델 컬럼 변경

```python
class LogEntry(Base):
    __tablename__ = "log_entries"

    row_id = Column(Integer, primary_key=True)          # was id
    batch_id = Column(String(36), nullable=False, index=True)  # NEW
    subject_id = Column(String(100), nullable=True, index=True)  # was camera_id
    space_id = Column(String(100), nullable=True, index=True)
    log_type = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    target_present = Column(Boolean, nullable=True)
    description = Column(Text, nullable=True)
    target_coordinate = Column(Text, nullable=True)
    raw_json = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False)
```

#### Phase 14-2: `nlp/logger.py:108` — `_db_insert()` 파라미터 변경

```python
def _db_insert(self, log_type: str, timestamp: str | None = None,
               batch_id: str = "", subject_id: str | None = None,
               space_id: str | None = None, target_present: bool | None = None,
               description: str | None = None,
               target_coordinate: list | None = None,
               raw_json: str | None = None):
    ...
    entry = LogEntry(
        batch_id=batch_id,
        subject_id=subject_id,
        ...
        description=description,
        ...
    )
```

- `camera_id` → `subject_id`
- `reasoning` 파라미터 제거
- `batch_id` 파라미터 추가

#### Phase 14-3: `nlp/logger.py:231` — `space_snapshot()` 수정

성공 경로:
```python
batch_id = uuid4().hex  # 한 번 생성, 전체 snapshot row가 공유

for cam_id, snap in snapshots.items():
    ...
    self._db_insert(
        log_type="detect",
        batch_id=batch_id,
        subject_id=cam_id,         # was camera_id
        description=desc,           # per-camera desc만
        # reasoning 제거
    )

self._db_insert(
    log_type="space",
    batch_id=batch_id,
    subject_id=space_id,           # space id를 subject_id에
    description=reasoning,          # space reasoning을 description에
    # raw_json: cameras dict + reasoning
)
```

#### Phase 14-4: `nlp/logger.py:405` — `_snapshot_fallback()` 수정

동일한 batch_id 패턴 적용. `reasoning` → `description` 통합.

#### Phase 14-5: 기타 `_db_insert()` 호출부 확인

- `vision_detect()` (line ~210): `_db_insert` 호출하는지 확인. 호출하면 batch_id 없이 subject_id로 변경.

### init_db() 처리

`init_db()`는 `Base.metadata.create_all()`을 호출하므로, 모델 변경 후 `logs/tracking.db` 삭제 후 재시작하면 새 스키마로 생성됨.

**주의:** 기존 DB와 호환 불가 — Phase 14 적용 시 반드시 `rm -f logs/tracking.db`로 초기화 필요.

### Test Plan

1. `cv_no_vision` 1회: batch_id 동일한지, detect 3 + space 1 모두 같은 batch_id
2. `cv_vision` 1회: 동일 검증 + description에 reasoning 통합 확인
3. `llm_vision` 1회: 동일 검증
4. `subject_id` 쿼리: `WHERE subject_id = 'livingroom'` 로 detect 로그 조회

---

## Progress

| Phase | Step | Status |
|-------|------|--------|
| ... (기존) | | |
| **Phase 14-1: LogEntry 모델 변경** | `storage/database.py` — id 유지, batch_id/subject_id 추가, space_id/reasoning 제거 | ✅ |
| **Phase 14-2: _db_insert() 파라미터 변경** | `nlp/logger.py` — camera_id→subject_id, reasoning 제거, batch_id 추가 | ✅ |
| **Phase 14-3: space_snapshot() batch_id + description 통합** | `nlp/logger.py` — uuid4().hex, subject_id=space_id, description=reasoning | ✅ |
| **Phase 14-4: _snapshot_fallback() 동일 적용** | `nlp/logger.py` — batch_id 파라미터로 전달받도록 변경 | ✅ |
| **Phase 14-5: API 레이어 업데이트** | `api/models.py`, `api/routes/logs.py`, `storage/repository.py` | ✅ |
| **Phase 14 통합 테스트** | Docker + 3 configs → 4+8+4=16 rows, 3+6+3=12 images | ✅ |
