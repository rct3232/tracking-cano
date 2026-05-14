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
```

### 2.3 Interaction — 상호작용 감지 결과

```python
@dataclass
class Interaction:
    target_track_id: int       # 추적 대상의 track_id
    object_class_name: str     # 상호작용 대상 클래스 (예: "couch")
    interaction_type: InteractionType  # 상호작용 유형
    overlap_ratio: float        # IoU 비율 (디버깅용)
    distance_px: float         # 중심점 간 거리 (디버깅용)
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

- 전역 고유성 보장: `{camera_id}_{track_id}` 조합 사용
- 예: `cam_01_3` → cam_01에서 track_id=3인 객체
- orchestrator.py가 조합/분리 로직을 담당

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

### 4.2 InteractionType enum

```python
class InteractionType(Enum):
    NONE         = auto()   # 상호작용 없음
    NEARBY       = auto()   # 근처 (거리만 만족)
    CONTACT      = auto()   # 접촉 (겹침만 만족)
    INTERACTING  = auto()   # 상호작용 중 (둘 다 만족)
```

### 4.3 임계값 상수

모든 임계값은 `config/spaces.yaml`에서 동적 로드하며, 기본값은 다음과 같다:

| 상수 | 기본값 | 단위 | 설명 |
|------|--------|------|------|
| `speed_slow` | 20 | px/frame | 정지/천천히 이동 구분 기준 |
| `speed_fast` | 40 | px/frame | 천천히/빠르게 이동 구분 기준 |
| `overlap` | 0.3 | — | IoU 기반 겹침 임계값 |
| `distance` | 50 | px | 중심점 간 거리 임계값 |

---

## 5. 모듈 인터페이스 계약

### 5.1 detector.py — YOLO26 감지

```python
def detect(frame: np.ndarray, target_classes: List[str]) -> List[BBox]
```

- **입력:**
  - `frame`: BGR 형식 단일 프레임 (`np.ndarray`, shape: H×W×3)
  - `target_classes`: 추적 대상 클래스명 목록 (예: `["cat", "person"]`)
- **출력:** `List[BBox]` — 감지된 객체들의 bbox 목록
- **특이사항:**
  - `target_classes`가 빈 배열인 경우 → 모든 COCO 클래스를 허용
  - GPU/CPU는 자동 감지 (ultralytics 내부 로직 의존)
  - 모델은 첫 호출 시 한 번만 로드 (싱글톤 패턴 권장)
  - 예외 상황: YOLO 추론 실패 시 빈 리스트 반환 (예외 발생 안 함)

### 5.2 tracker.py — ByteTrack 추적

```python
def update(prev_tracked: List[TrackedBBox], new_detections: List[BBox], frame_id: int) -> List[TrackedBBox]
```

- **입력:**
  - `prev_tracked`: 이전 프레임의 추적 상태 (첫 프레임 시 빈 리스트)
  - `new_detections`: detector.py에서 반환한 현재 프레임 감지 결과
  - `frame_id`: 현재 프레임 번호
- **출력:** `List[TrackedBBox]` — 업데이트된 추적 상태
- **특이사항:**
  - ByteTrack 내부적으로 bbox 매칭 수행
  - 이전 프레임의 bbox를 `prev_bbox` 필드에 캐시
  - lost 객체 처리: ByteTrack의 lost threshold 이후에는 리스트에서 제거

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
def detect_interactions(
    target_tracked: TrackedBBox,
    all_tracked: List[TrackedBBox],
    thresholds: Thresholds,
    interaction_classes: Optional[List[str]] = None
) -> List[Interaction]
```

- **입력:**
  - `target_tracked`: 상호작용 판단 대상 객체
  - `all_tracked`: 현재 프레임의 모든 추적 객체 (타겟 포함)
  - `thresholds`: 임계값 설정
  - `interaction_classes`: 상호작용 대상 클래스 필터 (None이면 전체 허용)
- **출력:** `List[Interaction]` — 감지된 상호작용 목록
- **특이사항:**
  - 타겟 객체 자신을 제외한 나머지 객체와 비교
  - 거리 계산 먼저 → 겹침 계산 후 (단락 평가 최적화)
  - 동시 다중 상호작용 가능 (하나의 객체가 여러 대상과 동시에 상호작용할 수 있음)

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

모든 임계값은 `config/spaces.yaml` → `config_manager.py`를 통해 로드되며, 모듈은 직접 YAML을 읽지 않는다:

```
pipeline.py ──→ config_manager.get_thresholds(camera_id)
              ──→ analyzer.classify_movement(..., thresholds)
              ──→ interaction_detector.detect_interactions(..., thresholds)
```

핫리로드 시 `config_manager`가 새 값을 반환하면, 다음 프레임부터 즉시 적용된다.
