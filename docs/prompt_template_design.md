# LLM 프롬프트 템플릿 설계 — 설계 문서

## 1. 목적 및 범위

객체 행동 관찰 전문가 역할을 위한 시스템 프롬프트, 상태 변화 보고용 유저 프롬프트 구조, LLM 호출 조건을 정의한다.

---

## 2. 시스템 프롬프트 설계

### 2.1 역할 정의

```python
SYSTEM_PROMPT = """
You are an object behavior observation specialist. Your task is to describe the movement and interactions of tracked objects in a concise, objective manner.

Rules:
- Output exactly ONE sentence per state change event.
- Use objective language only. No emotions, no speculation.
- Do NOT include information that was not provided.
- If multiple changes occur, combine them into a single coherent sentence.
"""
```

### 2.2 출력 형식 제약

- **한 문장 강제:** 상태 변화가 복합적일 경우에도 단일 문장으로 통합
- **금지 사항:** 추측 ("아마", "probably"), 감정어, 확인되지 않은 정보
- **길이 제한:** 최대 150자 권장 (토큰 절약)

### 2.3 언어 선택 기준

- 기본 언어: 한국어 (`ko`)
- `config/spaces.yaml`에서 `llm.language` 필드로 공간별/전역 설정 가능
- 허용 값: `ko`, `en`

---

## 3. 단일 카메라 상태 보고용 프롬프트 템플릿

### 3.1 템플릿 구조

```python
USER_PROMPT_TEMPLATE = """
Timestamp: {timestamp}
Camera: {camera_id}
Space: {space_name}

Tracked objects:
{object_states}

Previous states (N frames ago):
{previous_states}

Describe the current state changes in one sentence.
"""
```

### 3.2 데이터 주입 형식

**`{object_states}` — 현재 상태 블록:**
```
- ID:{track_id}, Class:{class_name}: State={movement_state}, Speed={speed}px, Direction={direction}
  Interaction: {interaction_type} with {target_class}
```

**`{previous_states}` — 이전 상태 블록:**
```
- ID:{track_id}: State={prev_movement_state}, Interaction={prev_interaction_type}
```

### 3.3 예시 렌더링 결과

```
Timestamp: 2025-01-15T14:30:22.123Z
Camera: cam_01
Space: 거실

Tracked objects:
- ID:3, Class:cat: State=FAST_MOVE, Speed=45px, Direction=northeast
  Interaction: NEARBY with couch

Previous states (5 frames ago):
- ID:3: State=SLOW_MOVE, Interaction=NONE

Describe the current state changes in one sentence.
```

**기대 응답:** "거실 cam_01에서 고양이(ID:3)가 소파 근처로 빠르게 이동하며 속도가 빨라졌다."

---

## 4. 다중 카메라 공간별 종합 프롬프트 템플릿

### 4.1 템플릿 구조

```python
MULTI_CAMERA_PROMPT_TEMPLATE = """
Space: {space_name}
Timestamp: {timestamp}

Camera reports:
{camera_reports}

Synthesize the information from multiple cameras in this space into one coherent summary sentence.
"""
```

### 4.2 데이터 주입 형식

**`{camera_reports}` — 각 카메라별 개별 보고:**
```
cam_01: {single_camera_llm_response_1}
cam_02: {single_camera_llm_response_2}
```

### 4.3 예시 렌더링 결과

```
Space: 거실
Timestamp: 2025-01-15T14:30:22.123Z

Camera reports:
cam_01: 고양이(ID:3)가 소파 근처로 빠르게 이동하며 속도가 빨라졌다.
cam_02: 고양이(ID:7)가 정지해 있었으나 이제 천천히 움직이기 시작했다.

Synthesize the information from multiple cameras in this space into one coherent summary sentence.
```

**기대 응답:** "거실에서 두 마리의 고양이가 동시에 활동하기 시작했으며, 하나는 소파 쪽으로 빠르게 이동하고 다른 하나는 천천히 움직이고 있다."

### 4.4 카메라 간 중복 처리 (Phase 2)

- Re-ID 미구현 시점에는 LLM에게 "동일 객체일 가능성" 판단을 맡기지 않음
- 규칙 기반 필터링: 동일 시간대, 인접 공간의 카메라에서 같은 `class_name`이 감지되면 경고 로그만 기록
- Phase 4 (Re-ID 구현) 이후에 중복 제거 로직 추가

---

## 5. 상태 변화 감지 로직 — LLM 호출 조건

### 5.1 변화 감지 기준

LLM은 다음 중 하나라도 발생 시 호출된다:

| 조건 | 설명 |
|------|------|
| 이동 상태 변경 | `STOPPED → SLOW_MOVE`, `SLOW_MOVE → FAST_MOVE` 등 |
| 상호작용 시작/종료 | `NONE → NEARBY`, `CONTACT → NONE` 등 |
| 객체 등장 | 새로운 `track_id`가 감지됨 |
| 객체 소실 | 기존 `track_id`가 lost 상태가 됨 |

### 5.2 디바운스 로직

```python
class LLMCallDebouncer:
    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown = cooldown_seconds
        self.last_call_time: Dict[str, float] = {}  # key: "{space_id}_{track_id}"

    def should_call(self, space_id: str, track_id: int) -> bool:
        key = f"{space_id}_{track_id}"
        now = time.time()
        if key in self.last_call_time:
            if now - self.last_call_time[key] < self.cooldown:
                return False
        self.last_call_time[key] = now
        return True
```

- **쿨다운:** 동일 공간·동일 객체에 대해 최소 3초 간격으로 LLM 호출
- **이유:** 임계값 근처에서 상태가 왔다갔다 할 때 (예: `SLOW_MOVE ↔ FAST_MOVE`) 불필요한 호출 방지

### 5.3 배치 vs 즉시 호출

- **Phase 1 (단일 카메라):** 상태 변화 감지 시점 → 디바운스 확인 → 즉시 LLM 호출
- **Phase 2 (다중 카메라):** 동일 공간 내 N초(기본 5초) 동안의 변화를 수집 후 단일 LLM 호출로 배치 처리
- **이유:** 다중 카메라 환경에서 개별 호출은 비용이 기하급수적으로 증가하므로, 공간별 배치가 효율적

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
| bbox 좌표 제외 | LLM이 해석할 필요 없는 숫자 데이터는 프롬프트에 포함 안 함 |
| 변경 필드만 전달 | 디프 포맷: 이전 상태와 현재 상태의 차이만 주입 |
| 방향명 단순화 | 8방위 (north, northeast, ...) 사용, 각도값은 제외 |

### 7.2 호출 빈도 감소

| 전략 | 설명 |
|------|------|
| 디바운스 | 동일 객체/공간에서 3초 간격으로 LLM 호출 제한 |
| 배치 처리 | Phase 2에서 N초 단위 공간별 변경 사항 수집 후 단일 호출 |
| 경량 필터링 | "정지 → 정지" 같은 동일 상태는 LLM 호출 스킵 |

### 7.3 모델 선택 전략

| 시나리오 | 권장 모델 크기 | 이유 |
|----------|---------------|------|
| 단일 카메라 보고 | 소형 (예: gpt-4o-mini) | 단순 서술 작업, 복잡한 추론 불필요 |
| 다중 카메라 종합 | 중형 이상 (예: gpt-4o) | 통합 추론 필요 |

### 7.4 캐싱 고려사항

- 동일 패턴의 LLM 응답을 캐시하는 것은 현재 설계에서는 구현하지 않음
- 이유: 상태 변화는 매번 다른 객체/상황에 발생하므로 캐시 히트율이 낮을 것으로 예상
- Phase 4에서 필요 시 도입 가능
