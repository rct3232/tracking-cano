# LLM 프롬프트 템플릿 설계 — 설계 문서

## 1. 목적 및 범위

객체 행동 관찰 전문가 역할을 위한 시스템 프롬프트, snapshot 분석용 유저 프롬프트 구조, LLM 호출 조건을 정의한다.

---

## 2. 시스템 프롬프트 설계

### 2.1 역할 정의

```python
SYSTEM_PROMPT = (
    "You are an object behavior observation specialist. "
    "Describe the movement of tracked objects in one concise, objective sentence. "
    "Use natural expressions like 'moving left/right/up/down', 'rotating', "
    "'moving quickly', 'moving slowly', 'stopped' — never use pixel values or numerical measurements. "
    "If nearby objects are listed in the input, you MUST include them in your description. "
    "Do not omit any nearby objects that are provided. "
    "Never invent objects or relationships that are not in the input. "
    "No emotions, no speculation. Output exactly ONE sentence."
)
```

### 2.2 출력 형식 제약

- **한 문장 강제:** 상태 변화가 복합적일 경우에도 단일 문장으로 통합
- **금지 사항:** 추측, 감정어, 확인되지 않은 정보
- **길이 제한:** 최대 150 tokens (`max_tokens=150`)

### 2.2 Vision Detect 프롬프트 — 단일 프레임 LLM detect

`nlp/prompts.py`의 `DETECT_SYSTEM_PROMPT`:
```
You are a target detection assistant...
Return JSON: {target_present: bool, reasoning: str}
Return ONLY valid JSON.
```

출력 형식: `{"target_present": true, "reasoning": "..."}`

### 2.3 Snapshot 프롬프트 — space-level 종합 분석

`nlp/prompts.py`의 `SNAPSHOT_VISION_PROMPT` (vision_enabled=true) / `SNAPSHOT_TRACKING_PROMPT` (vision_enabled=false):

```
You are a multi-camera space monitoring assistant...
Analyze the camera feeds and determine target status.

Return JSON format:
{
  "target_present": true/false,
  "cameras": {
    "camera_id": {
      "target_present": true/false,
      "description": "...",
      "target_coordinate": [x1,y1,x2,y2]
    }
  },
  "reasoning": "..."
}
Rules:
- Do NOT confuse inanimate objects with the living target
- When uncertain, default to false
- Never guess
- Only include cameras where you are CONFIDENT. Omit no-target cameras entirely.
```

---

## 3. Snapshot 유저 프롬프트 템플릿

### 3.1 vision_enabled=true — 이미지 + tracking data

`space_snapshot()`은 각 카메라의 마지막 프레임(또는 버퍼 전체 5프레임)을 LLM에 전송:

```
Timestamp: {timestamp}
Space: {space_name}

--- [livingroom] ---
[image frames...]
cat: stopped | nearby: couch (touching) | bbox [0.75, 0.56, 0.79, 0.66]

--- [hallway] ---
[image frames...]
(no target detected)

Target objects: cat, person
```

### 3.2 vision_enabled=false — tracking data only

```
Timestamp: {timestamp}
Space: {space_name}

Target objects: cat

- livingroom: cat: rotating | nearby: couch (touching)
- hallway: no target detected
```

---

## 4. Snapshot 트리거 조건

### 4.1 cv_pipeline 모드

`Pipeline.process_frame()`에서 target_present 상태 변화 감지 시 snapshot trigger:

| 조건 | 설명 |
|------|------|
| target 등장 | `prev.target_present=false → current=true` |
| target 소실 | `prev.target_present=true → current=false` |
| 상호작용 변화 | interaction target 변경 |

### 4.2 llm_vision 모드

`_SimpleVisionDetector`가 round-robin으로 카메라별 `vision_detect()` 호출:
- `target_present=true` → 즉시 snapshot trigger
- 5초 디바운스 (space별)

### 4.3 Snapshot 디바운스

```python
# SpaceLogger: 5s cooldown for space_snapshot()
debouncer = LLMCallDebouncer(cooldown_seconds=5.0)

# Orchestrator: 2nd layer debounce
self._snapshot_debounce: Dict[str, float] = {}
```

---

## 6. 프롬프트 변수 스키마

### 6.1 ObjectState — 객체 상태

```yaml
object_state:
  track_id: int                    # ByteTrack 추적 ID
  class_name: str                  # COCO 클래스명 (예: "cat", "person")
  movement_state: enum             # STOPPED | SLOW_MOVE | FAST_MOVE | DASHING | ROTATING
  speed_px: float                 # 현재 속도 (px/frame)
  direction: str                   # 방향 (north, northeast, east, southeast, south, southwest, west, northwest)
```

### 6.2 Interaction — 상호작용

```yaml
interaction:
  interaction_type: enum           # NONE | NEARBY | CONTACT | INTERACTING
  target_class_name: str          # 상호작용 대상 클래스 (예: "couch", "chair")
  overlap_ratio: float             # IoU 비율 (프롬프트에는 포함 안 함, 디버깅용)
  distance_px: float              # 중심점 간 거리 (프롬프트에는 포함 안 함, 디버깅용)
```

### 6.3 Metadata — 메타데이터

```yaml
metadata:
  timestamp: str                   # ISO 8601 형식
  camera_id: str                  # 카메라 ID (예: "cam_01")
  space_name: str                 # 공간명 (예: "거실")
  frame_number: int               # 프레임 번호 (프롬프트에는 포함 안 함, 디버깅용)
```

### 6.4 변경 이력 — 이전/현재 상태 비교

```yaml
state_change:
  track_id: int
  previous:
    movement_state: enum
    interaction_type: enum
  current:
    movement_state: enum
    interaction_type: enum
```

프롬프트에는 "전체 상태"가 아닌 "변경된 필드만" 전달하여 토큰 절약을 도모한다.

---

## 7. 비용 최적화 전략

### 7.1 토큰 절약

| 전략 | 설명 |
|------|------|
| 시스템 프롬프트 최소화 | 역할 + 출력 형식만 포함, 예시는 제거 |
| No-target 카메라 생략 | 프롬프트에 "Omit no-target cameras entirely" 규칙으로 토큰 절약 |

### 7.2 호출 빈도 감소

| 전략 | 설명 |
|------|------|
| 디바운스 | snapshot 5s cooldown (SpaceLogger) + 5s (Orchestrator) 이중 디바운스 |
| 상태 변화 필터 | cv_pipeline: target_present 실제 변화 시에만 snapshot (매 프레임 아님) |

---

## 8. Snapshot 파이프라인

```
Detection Layer (cv_pipeline / llm_vision)
    │ target_present 변화 감지
    ▼
CameraSnapshot registry 업데이트 (buffer freeze at T0)
    │
    ▼
SpaceLogger.space_snapshot()
    │ vision_enabled=true: LLM (images + tracking) → JSON
    │ vision_enabled=false: LLM (tracking only) → JSON
    │ 실패: _snapshot_fallback() → tracking data fallback
    │
    ├─ per-camera _db_insert (detect, batch_id 동일)
    ├─ per-camera image save → output/
    └─ space _db_insert (space, reasoning → description, same batch_id)
```

### vision_detect() — 단일 이미지 detection

- **프롬프트:** `DETECT_SYSTEM_PROMPT`
- **입력:** base64 이미지 1장
- **출력:** `{"target_present": bool, "reasoning": str}` 또는 `None`
- **디바운스 없음** — 호출자(`_SimpleVisionDetector`)가 관리

### space_snapshot() — space-level 분석

- vision_enabled=true → `SNAPSHOT_VISION_PROMPT` (이미지 + tracking)
- vision_enabled=false → `SNAPSHOT_TRACKING_PROMPT` (tracking only)
- 동일 `batch_id`로 detect×N + space×1 DB insert
- bbox는 `_save_snapshot_image()`에서 CV 우선 → LLM fallback
