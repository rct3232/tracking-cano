# PLAN — Space-level Vision Aggregator (llm_vision 모드)

## SPEC

**Objective:** MODE=llm_vision 시, 동일 공간(space)에 속한 여러 카메라의 이미지를 취합하여 LLM이 하나의 통합된 공간으로 인식하도록 다중-시점 비전 분석 구현.

**Scope:**
- 각 `_VisionOnlyWorker`가 수집한 이미지 batch를 `SpaceLogger.vision_collect()`로 라우팅
- SpaceLogger는 camera_id별 최신 버퍼 유지 (최신만, 누적 아님 — temporal context 확보)
- 어떤 카메라든 새로운 batch 도착 시 → 모든 카메라의 버퍼 이미지를 취합해 즉시 LLM 호출
- 공간 내 temporal context: 각 카메라 snapshot_count=5 프레임이 FIFO로 유지되며 LLM에 전달됨
- Prompt 우선순위: space-level `llm_system_prompt`가 있으면 사용, 없으면 기본 VISION_SPACE_SYSTEM_PROMPT

**Success Criteria:**
1. 거실(3 카메라)의 이미지를 하나의 LLM call로 통합 분석됨 (debounce key = space_id_vision)
2. 각 카메라 5프레임 × 3카메라 = 최대 15장의 이미지가 LLM에 전달됨
3. 이미지 간 `[camera_id]` 라벨로 시점 구분 가능
4. 기존 `vision_log()` 직접 호출 경로(스페이스 없는 카메라)는 영향 없음

## 아키텍처

```
[VisionWorker cam_A: 5 frames] ──→ vision_collect(space_id, cam_A, images)
[VisionWorker cam_B: 5 frames] ──┤
[VisionWorker cam_C: 5 frames] ──┘
        ↓ SpaceLogger._vision_buffer[space_id][cam_X] = latest_images (최신만 유지)
        ↓ (어느 카메라든 도착 시 즉시 — flush 없음)
    [NLPLogger._process_vision_batch_space()]
        → all cameras' images + [camera_id] labels → 단일 LLM call
```

## 진행 상황

### Step 1: Config 계층 — SpaceConfig에 `llm_system_prompt` 추가 ✅ 완료

**파일**: `core/config_manager.py`, `config/spaces.yaml.example`

- `SpaceConfig.__init__`: `self.llm_system_prompt = cfg.get("llm_system_prompt", None)` 필드 추가
- `spaces.yaml.example`: space-level prompt 예시 주석 포함

### Step 2: SpaceLogger에 vision 버퍼 + collect 메서드 추가 ✅ 완료

**파일**: `nlp/logger.py`

- `VISION_SPACE_SYSTEM_PROMPT` 상수 정의 (다중 시점 통합 분석 지시)
- `SpaceLogger._vision_buffer`: `{space_id: {camera_id: [image_b64_batch]}}` — 최신만 유지
- `SpaceLogger.vision_collect()`: camera의 최신 batch를 버퍼에 저장 + **모든 카메라 도착 확인 후** LLM 호출

### Step 2.1: 모든 카메라 도착 조건 체크 ✅ 완료

**파일**: `nlp/logger.py`

- `vision_collect()`에서 `_camera_counts[space_id]`와 `_vision_buffer[space_id]`의 길이 비교
- 모든 카메라가 버퍼에 등록되지 않으면 LLM 호출 스킵 (`logger.debug("[space:%s][vision] waiting for all cameras: %d/%d")`)
- 마지막 카메라 도착 시에만 combined 이미지를 LLM에 전달

**테스트 결과:** livingroom(50초) → livingfront(54초) → hallway(58초) 제출 후 3/3 확인 → 단일 LLM 호출 (15장). 이전에는 livingroom이 먼저 호출됨.

### Step 3: NLPLogger에 다중 카메라 vision LLM call 메서드 추가 ✅ 완료

**파일**: `nlp/logger.py`

- `NLPLogger._process_vision_batch_space()`:
  - `images: List[Tuple[camera_id, image_b64]]` — camera 라벨 포함
  - `[camera_id]` 텍스트를 이미지 사이에 삽입하여 LLM이 시점 구분 가능
  - space-level llm_system_prompt가 있으면 사용, 없으면 기본 prompt만
  - target_classes는 공간 내 모든 카메라의 class를 합쳐 전달

### Step 4: _VisionOnlyWorker에 aggregator 경로 추가 ✅ 완료

**파일**: `core/vision_worker.py`

- 신규 파라미터: `on_vision_collect`, `space_id`
- `_run()`에서 batch 수집 시 분기:
  - `on_vision_collect + space_id` 있으면 → aggregator 경로 (SpaceLogger.vision_collect)
  - 없으면 → 기존 `on_batch_ready` 직접 호출

### Step 5: Orchestrator wiring 변경 ✅ 완료

**파일**: `core/orchestrator.py`

- llm_vision 모드에서 NLPLogger를 orchestrator 레벨 공유 인스턴스로 생성 (`_ensure_vision_nlp`)
- space_id가 있는 카메라는 aggregator 경로, 없는 카메라는 기존 직접 호출 유지
- target_classes는 공간 내 모든 카메라의 class 합쳐서 전달

## Gotchas

- **이미지 수**: N카메라 × snapshot_count=5 = LLM call당 최대 5N장. gpt-4o-mini는 지원하지만 비용 증가
- **debounce key 변경**: 기존 `{camera_id}_vision` → `{space_id}_vision`으로 변경 (공간 단위 cooldown)
- **thread safety**: SpaceLogger._lock 하에서 버퍼 접근, LLM 호출은 lock 밖 수행
- **buffer 교체 시점**: vision_collect()에서 최신 batch로 덮어씀 — 이전 이미지는 다음 call에 포함 안됨
