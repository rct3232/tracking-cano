# 데이터 플로우 상세 설계 — 설계 문서

## 1. 목적 및 범위

프레임별 처리 파이프라인의 단계별 흐름, 상태 전이도, 상태 변화 감지 알고리즘, 상호작용 판단 로직, 다중 카메라 병렬 처리 모델, 모드 분기 로직을 정의한다.

---

## 2. 단일 프레임 처리 파이프라인

### 2.1 단계별 흐름도

```
```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: Vision Capture (_BatchCollector — daemon thread)      │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Timer-based (collect_interval=0.5s) encode & buffer       │ │
│  │  Read every frame (RTSP buffer mgmt), encode only on timer │ │
│  │  buffer: deque[_FrameEntry](maxlen=collect_count=5)        │ │
│  │  buffer.clear() on reconnect                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Main Loop (per frame — parallel with Layer 1)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Step 1: Frame Capture                                     │ │
│  │    ├─ read() from cv2.VideoCapture (every frame, always)   │ │
│  │    └─ frame_skip for main pipeline inference throttle      │ │
│  │         ↓                                                  │ │
│  │  Step 2: Detection & Tracking (tracker.py)                 │ │
│  │    Input: frame, target_classes, frame_id                  │ │
│  │    Output: (List[TrackedBBox], List[TrackedBBox])          │ │
│  │    Budget: ~112ms (YOLO26s) / ~68ms (YOLO26n)             │ │
│  │         ↓                                                  │ │
│  │  Step 3a: Movement Analysis (analyzer.py)                  │ │
│  │    Input: TrackedBBox, thresholds                          │ │
│  │    Output: (MovementState, metadata_dict)                  │ │
│  │    Budget: ~1ms                                            │ │
│  │         ↓                                                  │ │
│  │  Step 3b: Interaction Detection                            │ │
│  │          (interaction_detector.py)                         │ │
│  │    Input: target_tracked, all_tracked, thresholds           │ │
│  │    Output: List[Interaction]                               │ │
│  │    Budget: ~2ms                                            │ │
│  │         ↓                                                  │ │
│  │  Step 4: State Change Detection                            │ │
│  │    ├─ prev_state vs current_state 비교                     │ │
│  │    └─ 변화 감지 → LLM 호출 (디바운스 3s)                  │ │
│  │         ↓                                                  │ │
│  │  Step 5: Text Logging (NLPLogger)                          │ │
│  │    └─ 자연어 로그 (연속 파이프라인)                        │ │
│  │                                                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Layer 2: Space Scheduler (_VisionScheduler — 100ms polling)    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  per-space state machine                                   │ │
│  │  DETECTING → (target_found) LOGGING+COOLING                │ │
│  │                (all false)  → immediate DETECTING restart  │ │
│  │  Camera health: healthy → LLM detect / degraded → skip     │ │
│  │                              dead → skip                    │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```
```

### 2.2 프레임 드롭 조건 및 백프레시 전략

- **Main pipeline:** `frame_skip` 파라미터로 추론 간격 제어 (기본 15프레임마다 1회 추론)
- **Layer 1 (_BatchCollector):** 모든 프레임 read (RTSP 버퍼 관리), `collect_interval`(0.5s) 타이머에만 encode + buffer 저장
- **이유:** 실시간 모드에서 지연 누적 방지가 최우선, 동시에 버퍼는 일정 간격으로 신선한 이미지 유지

### 2.3 상태 캐시 생명주기

```
초기화: pipeline 시작 시 → {track_id: StateCacheEntry} = {}
갱신:  매 프레임 처리 후 → 해당 track_id의 캐시 갱신
삭제:  ByteTrack가 lost 객체로 판단 시 → 캐시에서 제거
```

```python
@dataclass
class StateCacheEntry:
    prev_movement_state: MovementState
    prev_interaction_type: InteractionType
    prev_speed: float
    prev_direction_angle: float
    frame_count: int          # 상태 유지 프레임 수 (플리커 방지용)
```

---

## 2.4 Vision 2-Layer Architecture

### 2.4.1 Layer 1: _BatchCollector (per-camera daemon thread)

모든 카메라에 대해 독립적인 `_BatchCollector` 데몬 스레드가 실행된다:

| 속성 | 값 |
|------|-----|
| Capture interval | `collect_interval` (기본 0.5s) |
| Buffer size | `collect_count` (기본 5, sliding window) |
| Encode | JPEG quality `vision_quality` (기본 60), resize to `vision_max_width` (1024px) |
| Stale threshold | `max_stale_threshold` (기본 10s) — 초과 시 degraded 처리 |

**동작:** read()는 매 프레임 수행 (RTSP UDP 버퍼 관리), encode + buffer 저장은 타이머에만 수행.
**Reconnect:** `buffer.clear()`를 호출하여 reconnect 직후 stale 이미지 전송 방지.
**동기화:** `start_event`를 통해 여러 카메라의 첫 캡처 시점을 동기화 (`math.ceil` alignment).

### 2.4.2 Layer 2: _VisionScheduler (per-space state machine)

`orchestrator.py`의 `_VisionScheduler`가 100ms 간격으로 polling하며, 한 사이클에 한 space씩 처리:

```
DETECTING ──→ target_found ──→ LOGGING+COOLING ──→ timer expire ──→ DETECTING (다음 사이클)
    │                                              (cooldown - early_trigger)
    └── all false ──→ immediate DETECTING restart
```

| 상태 | 설명 |
|------|------|
| `DETECTING` | `NLPLogger.vision_detect()` 호출 — 버퍼에서 최신 프레임을 LLM vision API로 전송 |
| `LOGGING+COOLING` | detect 결과 processing + `SpaceLogger.flush_vision()` 호출; 쿨다운 타이머 동작 |
| Immediate restart | 모든 카메라가 false detection이면 타이머 대기 없이 즉시 DETECTING 재진입 |

**Space 독립성:** 각 space의 state machine은 완전히 독립적 — 한 space가 LOGGING 중이어도 다른 space는 DETECTING 가능.

### 2.4.3 Camera Health 추적

| 상태 | 조건 | 처리 |
|------|------|------|
| `healthy` | 버퍼 최신 entry가 `max_stale_threshold` 이내 | LLM detect에 이미지 포함 |
| `degraded` | 버퍼 최신 entry가 `max_stale_threshold` 초과 | detect skip, 텍스트에 "(degraded)" 표기 |
| `dead` | 버퍼가 비어 있거나 collector thread 종료됨 | detect skip, 텍스트에 "(dead)" 표기 |

---

## 3. 이동 상태 전이도

### 3.1 상태 정의 및 수학식

| 상태 | 조건 |
|------|------|
| `STOPPED` | 속도 < `speed_slow` |
| `SLOW_MOVE` | `speed_slow` ≤ 속도 < `speed_fast` && 가속도 < `dash_threshold` |
| `FAST_MOVE` | 속도 ≥ `speed_fast` && 가속도 < `dash_threshold` && 방향 변화율 < `rotation_threshold` |
| `DASHING` | 가속도 ≥ `dash_threshold` (일시적 속도 급증) |
| `ROTATING` | 방향 변화율 > `rotation_threshold` && 속도 < `speed_slow` |

### 3.2 상태 전이 표

```
                    ┌───────────┐
              ┌──── │  STOPPED  │
              │    └──────┬────┘
              │           │ 속도 ≥ speed_slow
              │           ▼
              │    ┌──────────────┐
              │    │ SLOW_MOVE    │◄──── DASHING 종료
              │    └──────┬───────┘      (가속도 < dash_threshold)
              │           │ 속도 ≥ speed_fast
              │           ▼
              │    ┌──────────────┐
              │    │  FAST_MOVE   │◄──── DASHING 종료
              │    └──────┬───────┘      (가속도 < dash_threshold)
              │           │ 가속도 ≥ dash_threshold
              │           ▼               │ 방향 변화율 > rotation_threshold
              │    ┌──────────────┐       │ && 속도 < speed_slow
              └────│  DASHING    │        ▼
                   └──────┬──────┘   ┌──────────────┐
                           │               │ ROTATING   │
                           └───────────────┴────────────┘
```

**허용되지 않는 전이:** `STOPPED → DASHING` (속도 증가 없이 돌진 불가)

### 3.3 상태 유지 최소 프레임 수 (플리커 방지)

- **기본값:** 3 프레임
- **논리:** 상태 변경 감지 시, 동일 상태로 3프레임 연속 확인后才 확정
- **이유:** 임계값 근처에서 단일 프레임 노이즈로 인한 상태 왔다갔다 방지

### 3.4 히스테리시스 (Hysteresis)

```
정지 → 천천히 이동:      속도 ≥ speed_slow                (예: 20px)
천천히 이동 → 정지:      속도 < speed_slow - hysteresis    (예: 15px, 20-5)
```

역전 임계값을 약간 낮게 설정하여 상태 플리커를 추가로 방지한다.

---

## 4. 상태 변화 감지 알고리즘

### 4.1 비교 순서 흐름도

```
현재 프레임의 track_id 목록 확인
    │
    ├── track_id가 새로 등장?
    │     → 초기 상태 할당 (STOPPED 또는 SLOW_MOVE)
    │     → LLM 호출: "{class_name}이(가) 나타남"
    │
    ├── track_id가 사라짐 (lost)?
    │     → 캐시에서 제거
    │     → LLM 호출: "{class_name}(ID:{id})이(가) 화면에서 사라짐"
    │
    └── 기존 track_id?
          │
          ├── 현재 속도·가속도·방향 계산
          │
          ├── 상태 분류기 적용 → current_state
          │
          ├── prev_state == current_state ?
          │     ├─ Yes: LLM 호출 스킵, 프레임 카운터 증가
          │     └─ No:
          │           ├── 전이 표에서 허용 경로인가?
          │           │   ├─ Yes: LLM 호출 + 캐시 갱신
          │           │   └─ No: prev_state 유지 (플리커 필터링)
          │
          └── 상호작용 상태도 동일하게 비교
```

### 4.2 히스테리시스 적용 로직

```python
def classify_with_hysteresis(speed, acceleration, direction_change,
                              prev_state, thresholds):
    """히스테리시스를 고려한 상태 분류"""
    if prev_state == MovementState.STOPPED:
        slow_threshold = thresholds.speed_slow          # 20px
    else:
        slow_threshold = thresholds.speed_slow - 5      # 15px (역전 시)

    if speed < slow_threshold:
        return MovementState.STOPPED
    elif speed < thresholds.speed_fast:
        return MovementState.SLOW_MOVE
    # ... 이하 동일
```

---

## 5. 상호작용 판단 알고리즘

### 5.1 계산 순서 (단락 평가 최적화)

```
Step 1: tracker에서 현재 추적 중인 track_id의 bbox 추출
         ↓
Step 2: tracker.update()에서 반환된 객체 중 interaction_classes로 필터링
         ↓
Step 3: 거리 계산 먼저 (경량 연산)
         중심점 간 유클리드 거리 < distance_threshold → "NEARBY"
         ↓ (거리 조건 만족 시에만 계속)
Step 4: 겹침 계산 (비용 높은 연산)
         IoU > overlap_threshold → "CONTACT"
         ↓
Step 5: 둘 다 만족 → "INTERACTING"
```

### 5.2 bbox 교차 검증

- **부분 겹침 vs 완전 포함:** 현재 설계에서는 구분하지 않음 (IoU 값만으로 판단)
- Phase 4에서 필요 시 `containment_ratio` 추가

### 5.3 서지 필터링

- 단일 프레임에서 잠깐 겹친 후 떨어지는 경우: 실제 접촉으로 분류하지 않음
- **논리:** 겹침 상태가 최소 2프레임 지속되어야 "CONTACT" 확정

### 5.4 동시 다중 상호작용

- 한 객체가 여러 대상과 동시에 상호작용할 수 있음
- 예: 고양이가 소파 위에 있으면서 TV 근처에 있는 경우 → `INTERACTING(couch) + NEARBY(tv)`
- LLM 프롬프트에는 모든 상호작용을 포함하여 전달

---

## 6. 다중 카메라 병렬 처리 흐름

### 6.1 GIL 고려한 아키텍처 결정

Python의 GIL(Global Interpreter Lock)로 인해, 순수 Python 연산은 스레드로 진정한 병렬화가 불가능하다.

| 단계 | 연산 특성 | GIL 영향 |
|------|----------|---------|
| YOLO 추론 (tracker.update) | C++/CUDA 백엔드 | GIL 해제됨 → 스레드로 충분 |
| ByteTrack 매칭 (tracker) | Python + numpy | 부분적 GIL 영향 |
| 분석 로직 (analyzer) | 순수 Python | GIL 병목 가능 |
| LLM 호출 (nlp_logger) | 네트워크 I/O | 별도 처리 필요 |

### 6.2 모델 옵션 비교

| 모델 | 장점 | 단점 | 선택 기준 |
|------|------|-------|----------|
| 스레드 (ThreadPool) | 메모리 공유, 경량 | GIL 병목 | 카메라 ≤ 2개, GPU 사용 시 |
| 프로세스 (ProcessPool) | 진정한 병렬, GIL 무관 | IPC 오버헤드, 메모리 복제 | 카메라 ≥ 3개 또는 CPU 바운드 많을 때 |

### 6.3 추천: 하이브리드 모델

```
┌─────────────────────────────────────────────┐
│              Orchestrator                    │
│                                             │
│  ┌──────────┐   ┌──────────┐                │
│  │ Camera A  │   │ Camera B  │   ...          │
│  │ Pipeline  │   │ Pipeline  │     (스레드)    │
│  └──────────┘   └──────────┘                │
│                                             │
│  ┌─────────────────────────────┐             │
│  │   LLM Call Dispatcher       │             │
│  │   (별도 스레드, I/O 바운드)    │             │
│  └─────────────────────────────┘             │
└─────────────────────────────────────────────┘
```

- **카메라 파이프라인:** 스레드로 실행 (YOLO 추론이 GIL 해제하므로 전체 파이프라인의 병목은 YOLO에 집중됨)
- **LLM 호출:** 별도 스레드池中에서 처리 (네트워크 I/O 바운드이므로 분석 파이프라인과 격리)

### 6.4 오케스트레이터 역할 정의

1. 카메라별 파이프라인 시작/종료/재시작 (`_CameraWorker`)
2. 공간별 로그 수집 및 LLM 호출 디스패치 (`SpaceLogger.flush()` / `flush_vision()`)
3. **Vision Layer 2 state machine 운영 (`_VisionScheduler`)**
   - 100ms polling loop, space별 독립 상태 관리
   - `_process_detection_step()` → `_transition_to_logging()` 체인
4. **Camera health 추적 (`_get_camera_health()`):** 각 카메라 버퍼의 staleness 확인 → healthy/degraded/dead 결정
5. **Capture timing 동기화:** `start_event` + `math.ceil` alignment으로 다중 카메라 첫 캡처 시점 정렬
6. 구성 변경 시 안전한 전환 (실행 중인 프레임 처리 완료 대기), `diff_configs()` hot-reload

---

## 7. 오프라인 모드 vs 실시간 모드의 차이점

### 7.1 분기 시점

`main.py`에서 CLI 인파스에 따라 초기화 경로 분리:

```
python main.py --live        → CameraManager: webcam/RTSP 연결 → 무한 루프
python main.py --video <path> → VideoReader: 파일 오픈 → 프레임 수만큼 루프 후 종료
```

### 7.2 핵심 차이점 비교

| 구분 | 실시간 (--live) | 오프라인 (--video) |
|------|-----------------|-------------------|
| 입력 소스 | `cv2.VideoCapture(0~N)` 또는 RTSP URL | `cv2.VideoCapture(filepath)` |
| 루프 조건 | 무한 (Ctrl+C로 종료) | `cap.isOpened()` && `ret=True` |
| 프레임 스킵 | 필요 시 FPS 제한용 | 일반적으로 스킵 없음 |
| 상태 캐시 초기화 | 시작 시 + 카메라 추가 시 | 영상 파일 변경 시 |
| 에러 복구 | 카메라 연결 끊김 시 재연결 대기 | 프레임 읽기 실패 시 스킵 또는 종료 |

### 7.3 공유 로직

`pipeline.py` 내부의 `tracker.update → analyzer → logger` 흐름은 모드에 관계없이 동일해야 함 (DRY 원칙). (detector는 tracker로 통합됨)

### 7.4 다중 카메라 혼합 모드 (Vision 2-Layer)

`_BatchCollector`가 소스 타입에 따라 다른 동작:

| 소스 | _BatchCollector 동작 | 종료 조건 |
|------|---------------------|-----------|
| RTSP/Webcam | 타이머 기반 무한 캡처 (`_run` → `while not stop_event`) | SIGINT/SIGTERM |
| Video file | 프레임 종료 시 stop (`_run_video` → `loop_count` 도달) | 파일 끝 + on_finished 콜백 |
| Directory | 파일逐个 순회 (`resolve_source` → file list) | 모든 파일 완료 |

`_VisionScheduler`는 live/file 모두 동일하게 동작 — 카메라 health 상태에 따라 처리 여부 결정.
`main.py --live`로 시작 시 단일 인자 없으면 multi-camera mode (YAML config 기반).
