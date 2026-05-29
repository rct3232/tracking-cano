# Interface Contract — 설계 문서

## 1. 목적 및 범위

모듈 간 데이터 전달 형식과 함수 인터페이스를 명확히 정의하여, `pipeline.py` 통합 시 인터페이스 불일치로 인한 버그를 방지한다.

---

## 2. 공통 데이터 구조체

### 2.1 BBox — YOLO 감지 결과

```python
@dataclass
class BBox:
    x1: int          # 왼쪽 위 x 좌표
    y1: int          # 왼쪽 위 y 좌표
    x2: int          # 오른쪽 아래 x 좌표
    y2: int          # 오른쪽 아래 y 좌표
    confidence: float  # 신뢰도 (0.0 ~ 1.0)
    class_id: int     # COCO 클래스 ID
    class_name: str   # 클래스명 (예: "cat", "person")
```

**선택 이유:** `x1y1x2y2` 형식을 채택한다. ByteTrack의 입력 형식과 호환되며, IoU 계산 시 직관적이다.

### 2.2 TrackedBBox — 추적 상태 포함 bbox

```python
@dataclass
class TrackedBBox:
    track_id: int           # ByteTrack가 부여한 고유 ID (0 이상)
    frame_id: int           # 현재 프레임 번호
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str
    prev_bbox: Optional[tuple[int, int, int, int]]  # 이전 프레임 bbox (속도 계산용)
    state: Optional[MovementState]                   # 현재 이동 상태
    speed: float = 0.0
    acceleration: float = 0.0
    prev_speed: float = 0.0
```

### 2.3 InteractionResult — 상호작용 감지 결과

```python
@dataclass
class InteractionResult:
    track_id: int              # 상호작용 대상의 track_id
    class_name: str            # 상호작용 대상 클래스 (예: "couch")
    relation_type: str         # "contact" | "nearby" | "interacting"
    distance: float            # 중심점 간 거리 (px)
```

---

## 3. 추적 ID 스키마

### 3.1 할당 규칙

- ByteTrack가 내부적으로 부여하는 `track_id`를 그대로 사용 (int, 0 이상)
- tracker.py는 ByteTrack의 ID 매핑을 투명하게 전달

### 3.2 유지 방식

- 프레임 간 매칭 시 동일 객체에 같은 ID 할당
- 객체 손실 후 재탐지 시 ByteTrack가 새 ID 부여 → 기존 ID는 유효하지 않음
- analyzer.py는 `track_id` 기준으로 이전/현재 상태 비교

### 3.3 다중 카메라 환경 (Phase 2)

- 각 카메라가 독립적인 Tracker 인스턴스를 가지므로 track_id는 카메라별로 로컬함
- 전역 식별은 camera_id 기준으로 로그에서 구분

---

## 4. 상태 분류 정의

### 4.1 MovementState enum

```python
from enum import Enum, auto

class MovementState(Enum):
    STOPPED      = auto()   # 정지: 속도 < speed_slow
    SLOW_MOVE    = auto()   # 천천히 이동: speed_slow ≤ 속도 < speed_fast
    FAST_MOVE    = auto()   # 빠르게 이동: 속도 ≥ speed_fast && 가속도 < dash_threshold
    DASHING      = auto()   # 돌진: 가속도 ≥ dash_threshold
    ROTATING     = auto()   # 회전: 방향 변화율 > rotation_threshold && 속도 < 일정치
```

### 4.2 InteractionResult.relation_type

`InteractionResult`의 `relation_type`은 문자열로 표현된다:

| 값 | 의미 |
|----|------|
| `"interacting"` | 겹침 + 근접 조건 모두 만족 또는 완전 포함 |
| `"contact"` | 겹침 조건만 만족 |
| `"nearby"` | 근접 조건만 만족 |

### 4.3 임계값 상수

모든 임계값은 `config/spaces.yaml` 또는 `.env`에서 동적 로드하며, 기본값은 다음과 같다:

| 상수 | 기본값 | 단위 | 설명 |
|------|--------|------|------|
| `speed_slow` | 20 | px/frame | 정지/천천히 이동 구분 기준 |
| `speed_fast` | 40 | px/frame | 천천히/빠르게 이동 구분 기준 |
| `overlap` | 0.3 | — | IoU 기반 겹침 임계값 |
| `distance` | 50 | px | 중심점 간 거리 임계값 |
| `dash_threshold` | 15 | px/frame² | 돌진 판단 가속도 임계값 |
| `rotation_threshold` | 45 | deg | 회전 판단 방향 변화 임계값 |
| `hysteresis` | 5 | px/frame | 상태 전환 히스테리시스 |
| `min_frames` | 3 | frame | 상태 유지 최소 프레임 수 (플리커 방지) |
| `surge_min_frames` | 2 | frame | 급속 가속 감지 최소 프레임 수 |

---

## 5. 모듈 인터페이스 계약

### 5.1 detector / tracker 통합

`detector.py`는 더 이상 사용되지 않는다. YOLO 추론과 ByteTrack 추적은 `tracker.py`의 `Tracker.update()`에서 통합 처리된다. 자세한 인터페이스는 5.2절을 참조.

- 모델 로드: `Tracker._ensure_loaded()`에서 지연 로드
- YOLO 추론: `HybridDetector.detect()`를 통해 실행 (타일 폴백 지원)
- GPU/CPU는 ultralytics 내부 자동 감지
- 예외 상황: 빈 리스트 반환 (예외 발생 안 함)

### 5.2 tracker.py — ByteTrack 추적 (detector + tracker 통합)

```python
def update(frame: np.ndarray, target_classes: List[str], frame_id: int, interaction_classes: Optional[List[str]] = None) -> tuple[List[TrackedBBox], List[TrackedBBox]]
```

- **입력:**
  - `frame`: BGR 형식 단일 프레임 (`np.ndarray`, shape: H×W×3)
  - `target_classes`: 추적 대상 COCO 클래스명 목록
  - `frame_id`: 현재 프레임 번호
  - `interaction_classes`: 상호작용 대상 COCO 클래스명 목록 (선택)
- **출력:** `(List[TrackedBBox], List[TrackedBBox])` — (추적된 target 목록, 상호작용 대상 목록)
- **특이사항:**
  - YOLO 추론 + ByteTrack 추적을 내부에서 한 번에 수행 (detector.py와 통합)
  - 이전 프레임의 bbox를 `prev_bbox` 필드에 캐시, 속도/가속도 자동 계산
  - `target_classes`와 `interaction_classes`를 각각 필터링하여 분리 반환
  - `_CLASS_NAME_MAP`으로 COCO class ID → 문자열 매핑

### 5.3 analyzer.py — 이동 상태 분류

```python
def classify_movement(tracked: TrackedBBox, thresholds: Thresholds) -> Tuple[MovementState, Dict[str, Any]]
```

- **입력:**
  - `tracked`: 현재 프레임의 추적된 객체 (이전 프레임 bbox 포함)
  - `thresholds`: 임계값 설정 (config에서 로드)
- **출력:**
  - `MovementState`: 분류된 이동 상태
  - `Dict[str, Any]`: 메타데이터 (`speed`, `acceleration`, `direction_angle` 등)
- **특이사항:**
  - 첫 프레임 (`prev_bbox`가 None)인 경우 → `STOPPED` 반환
  - 속도 계산: `sqrt((x2-x1)^2 + (y2-y1)^2)` (이전/현재 중심점 간 유클리드 거리)
  - 가속도 계산: 현재 속도 - 이전 속도 (이전 속도가 없으면 0)

### 5.4 interaction_detector.py — 상호작용 감지

```python
def detect(self, target: TrackedBBox, interactions: List[TrackedBBox]) -> List[InteractionResult]
```

- **입력:**
  - `target`: 상호작용 판단 대상 객체
  - `interactions`: tracker.update()에서 반환된 interaction 대상 목록
- **출력:** `List[InteractionResult]` — 감지된 상호작용 목록
- **특이사항:**
  - target 자신과 동일 track_id는 비교에서 제외 안 함 (interactions에 target이 포함되어 있지 않다고 가정)
  - 거리 계산 + 겹침 계산 → relation_type 결정
  - `is_contained` (dist==0.0)인 경우도 "interacting"으로 분류
  - 동시 다중 상호작용 가능

---

## 6. 데이터 흐름도

```
frame (np.ndarray, BGR H×W×3)
    │
    ▼
┌──────────────────────┐
│ detector.detect()     │  ← target_classes: config에서 로드
└──────────────────────┘
    │ List[BBox]
    ▼
┌──────────────────────┐
│ tracker.update()      │  ← prev_tracked: 이전 프레임 캐시
└──────────────────────┘
    │ List[TrackedBBox]
    ├─────────────────────────────────────────┐
    ▼                                         ▼
┌──────────────────────┐           ┌──────────────────────────┐
│ analyzer             │           │ interaction_detector     │
│ .classify_movement()  │           │ .detect_interactions()   │
└──────────────────────┘           └──────────────────────────┘
    │ (MovementState, meta)              │ List[Interaction]
    ▼                                    ▼
┌───────────────────────────────────────────────────────┐
│ nlp_logger.log()                                      │
│   ← MovementState + Interaction + 메타데이터           │
└───────────────────────────────────────────────────────┘
    │ str (자연어 로그)
    ▼
  콘솔 출력 + logs/ 파일 저장
```

**병렬 처리 가능성:** analyzer와 interaction_detector는 동일한 `List[TrackedBBox]`를 입력으로 받지만 서로 독립적이므로, Phase 2 다중 카메라 환경에서 이 두 단계는 스레드 풀로 병렬화 가능.

---

## 7. 에러 처리 정책

| 상황 | 처리 방식 |
|------|----------|
| YOLO 추론 실패 | 빈 리스트 반환 (프레임 스킵) |
| ByteTrack 매칭 실패 | 기존 추적 상태 유지 (변경 없음) |
| 카메라 연결 끊김 | 재연결 시도 → N회 실패 시 해당 파이프라인 종료 |
| LLM API 호출 실패 | 에러 로그 기록 + 프레임 처리 계속 (LLM 호출 스킵) |
| config 파일 파싱 오류 | 에러 메시지 출력 + 프로그램 종료 |

---

## 8. Config 기반 동적 값 매핑

모든 임계값은 `config/spaces.yaml` → `config_manager.py` → `orchestrator._make_pipeline_config()`를 통해 `PipelineConfig`에 로드된다:

```
spaces.yaml ──→ config_manager.load_config() ──→ AppConfig.thresholds/llm
                                                       │
                                          orchestrator._make_pipeline_config()
                                                       │
                                          PipelineConfig(thresholds, llm, ...)
```

핫리로드 시 `ConfigWatcher`가 변경을 감지하면 orchestrator가 해당 camera의 pipeline을 재시작하여 새 설정이 적용된다.
