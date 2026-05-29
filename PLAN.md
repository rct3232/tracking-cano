# PLAN.md — SpaceLogger 연결 작업 ✅ 완료

## SPEC

### 목적
SpaceLogger가 Orchestrator → Pipeline → NLPLogger 파이프라인에 실제로 연결되어, 동일 공간의 다중 카메라 로그를 LLM으로 종합 로깅하도록 보완.

### 범위
- Orchestrator에 SpaceLogger 주입
- Pipeline에 SpaceLogger + space_id 주입, collect() 호출
- run_multi에 SpaceLogger flush 루프 (이벤트 기반 + 주기적 안전망)
- Hot-reload 시 공간 추가/삭제 처리

### 성공 기준
1. `python main.py --live` 실행 시, 동일 공간의 여러 카메라에서 상태 변화가 발생하면 SpaceLogger가 종합 문장을 생성하여 로그 파일에 저장 ✅
2. `spaces.yaml` 변경 시 추가된 공간의 SpaceLogger 버퍼가 초기화, 삭제된 공간의 버퍼가 정리됨 ✅
3. 기존 단일 카메라 모드(`--live <url>`, `--video`)는 영향 없음 ✅

---

## 작업 목록 ✅ 12/12 완료

### 1. Orchestrator에 SpaceLogger 주입 + camera→space 매핑 ✅
- [x] `Orchestrator.__init__`에 `space_logger: SpaceLogger` 파라미터 추가 → `orchestrator.py:88`
- [x] `Orchestrator.add_camera`에 `space_id` 파라미터 추가 → `orchestrator.py:110`
- [x] camera→space 매핑: `spaces.yaml`의 `spaces[].cameras[]`로 카메라 ID → 공간 ID 역매핑 → `orchestrator.py:172-177` `_build_cam_to_space`
- [x] `_CameraWorker`에 `space_id` 전달 → `orchestrator.py:116`

### 2. Pipeline에 SpaceLogger + space_id 주입, collect() 호출 ✅
- [x] `Pipeline.__init__`에 `space_logger: Optional[SpaceLogger]`, `space_id: Optional[str]` 파라미터 추가 → `pipeline.py:14`
- [x] `Pipeline.process_frame`에서 NLPLogger.log()가 텍스트를 리턴할 때 `space_logger.collect()` 호출 → `pipeline.py:95-98` `_collect`

### 3. run_multi에 SpaceLogger flush 루프 ✅
- [x] `run_multi`에 flush 스케줄러 추가 (10초 주기적) → `main.py:124-132`
- [x] `Orchestrator`에 `flush_spaces()` 메서드 추가 → `orchestrator.py:155-161`

### 4. _on_config_change에 공간 변경 처리 ✅
- [x] `diff.added_spaces` → `set_camera_count()` 호출 → `main.py:150-153`
- [x] `diff.removed_spaces` → SpaceLogger.flush() 후 버퍼 정리 → `main.py:154-158`
- [x] 카메라 재할당 감지 (기존 공간 → 새 공간) → `main.py:140-143` + `orchestrator.py:131-147`

---

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `core/orchestrator.py` | SpaceLogger 주입, camera→space 매핑, flush_spaces() |
| `core/pipeline.py` | SpaceLogger + space_id 주입, collect() 호출 |
| `main.py` | run_multi flush 루프, _on_config_change 공간 처리 |

---

# PLAN.md — 프레임 처리 속도 최적화

## SPEC

### 목적
프레임당 처리 속도를 개선하여 실시간 추적 성능 향상.

### 범위
- 모델 교체 (nano / 양자화)
- 프레임 스킵
- NLP 로깅 비동기화
- Tracker 내부 캐싱

### 성공 기준
1. 벤치마크(livingfront 비디오, 30초)에서 처리 프레임 수 2배 이상 증가
2. LLM API 호출이 프레임 처리 스레드를 블로킹하지 않음
3. 기존 단일 카메라 모드(`--live <url>`, `--video`)는 영향 없음

---

## 백그라운드: YOLO 클래스 제한 무효화

벤치마크 결과, `yolo_classes` 필터는 추론 **후** 필터링만 수행하며 YOLO 내부에서는 80개 클래스 전체를 항상 추론한다. 처리 속도, 메모리 점유 모두 차이가 없음. 따라서 추론 수준의 최적화가 필요.

---

## 작업 목록

### 1. 모델 교체 (nano + 양자화)

#### 1-1. yolo26n 모델 지원 ✅ 완료
- [x] `YOLOConfig`에 `model_size` 필드 추가 (`n`, `s`, `m`, `x` 중 선택, default `s`) → `config.py:24`
- [x] `model_path`를 `yolo26{n}.pt`로 동적 생성 → `orchestrator.py:69`
- [x] `spaces.yaml`에 `model_size: n` 옵션 추가 → CameraConfig에 필드 존재
- [x] `YOLOConfig.model_path`가 명시되면 동적 생성을 덮어쓰기 → `orchestrator.py:69` `camera.model_path or ...`
- **예상 효과**: s → n 모델로 VRAM 40% 감소, 추론 속도 1.5~2배
- **변경 파일**: `config/config.py`, `core/config_manager.py`, `config/spaces.yaml`

#### 1-2. INT8 양자화 지원 ⚠️ 부분완료 (1/3)
- [x] `YOLOConfig`에 `quantize: bool` 필드 추가 (default `False`) → `config.py:26`
- [x] `Tracker._ensure_loaded()`에서 모델 로드 후 `model.quantize()` 호출 → `tracker.py:46-47`
- [ ] 양자화된 모델의 정확도 저하 벤치마크 (livingfront 비디오로 100프레임 검증)
- **예상 효과**: 추론 속도 1.5~2배, VRAM 절반
- **변경 파일**: `config/config.py`, `modules/tracker.py`

---

### 3. 프레임 스킵 ✅ 완료

**개념**: 매 프레임을 추론하지 않고 N프레임마다 추론. 추적은 계속되므로 cat의 위치는 이전 추론 결과로 유지.

#### 3-1. 구현 ✅
- [x] `YOLOConfig`에 `frame_skip: int` 필드 추가 (default `0` = 스킵 없음) → `config.py:27`
- [x] `_CameraWorker._run`에서 프레임 카운터로 스킵 로직 추가 → `orchestrator.py:38-60`
- [x] `Pipeline.process_frame()`이 호출되지 않는 프레임은 추론/추적 모두 스킵

#### 3-2. 추적 상태 유지 ✅
- [x] 스킵 프레임에서는 `process_frame` 호출 자체를 하지 않음 → `Tracker._history`가 유지됨
- [x] 스킵 프레임에서 `Tracker.update()`를 호출하지 않음 → 추적 상태 그대로 유지
- [x] 스킵 프레임에서 `_check_disappeared`도 호출되지 않음 → 사라진 target 감지 지연 (trade-off)

#### 3-3. 구성 옵션 ✅
- [x] `spaces.yaml`에 `frame_skip` 옵션 추가 (카메라별 설정 가능) → CameraConfig에 필드 존재
- **예상 효과**: `frame_skip=2` 시 추론 3분의 1로 감소 → 처리량 3배
- **변경 파일**: `config/config.py`, `core/orchestrator.py`, `config/spaces.yaml`

---

### 4. NLP 로깅 비동기화 ✅ 완료

**문제**: `NLPLogger.log()`는 LLM API 호출을 동기적으로 수행하며, API 응답 대기 시간(1-3초)이 프레임 처리 스레드를 블로킹한다.

**선택: 완전 비동기(A안)** — `log()`는 큐 enqueue 후 즉시 `None` 반환. SpaceLogger는 10초 주기적 flush에만 의존.

#### 4-1. `nlp/logger.py` 변경 ✅
- [x] `_LogTask` 데이터클래스 추가 → `nlp/logger.py:43-46`
- [x] `NLPLogger.__init__`에 큐 + 워커 스레드 → `nlp/logger.py:70-73`
- [x] `log()` 비동기화 (큐 enqueue 후 즉시 `None` 반환) → `nlp/logger.py:82-92`
- [x] `_worker` + `_process_task` 메서드 추가 → `nlp/logger.py:98-118`
- [x] `stop()` 메서드 추가 → `nlp/logger.py:94-96`

#### 4-2. `core/pipeline.py` 변경 ✅
- [x] `_collect` 호출 제거 + `_state_summary`로 콘솔 로그 유지 → `core/pipeline.py:71-91`
- [x] `stop()` 메서드 추가 → `core/pipeline.py:97-98`

#### 4-3. `core/orchestrator.py` 변경 ✅
- [x] `_CameraWorker.stop()`에서 `pipeline.stop()` 호출 → `core/orchestrator.py:30`

#### 4-4. `main.py` 변경 ✅
- [x] `run_live`/`run_video`에서 `Pipeline.stop()` 호출 → `main.py:82`, `main.py:112`

#### 4-5. 변경 파일 요약

| 파일 | 변경 내용 |
|------|----------|
| `nlp/logger.py` | `_LogTask` 추가, `log()` 비동기화, `_worker` 스레드, `stop()` |
| `core/pipeline.py` | `_collect` 호출 제거, `_state_summary`로 콘솔 로그 유지, `stop()` |
| `core/orchestrator.py` | `_CameraWorker.stop()`에서 Pipeline.stop() 호출 |
| `main.py` | `run_live`/`run_video`에서 Pipeline.stop() 호출 |

---

### 5. Tracker 내부 캐싱 ✅ 완료

**문제**: `Tracker.update()`에서 매 프레임 `class_name_map`(80개 dict), `target_id_set`, `interaction_id_set`을 재생성한다.

#### 5-1. `modules/tracker.py` 변경 ✅
- [x] `class_name_map` → `Tracker._CLASS_NAME_MAP` 클래스 변수 → `modules/tracker.py:38-56`
- [x] `update()`에서 클래스 변수 참조 → `modules/tracker.py:97`

#### 5-2. 변경 파일 요약

| 파일 | 변경 내용 |
|------|----------|
| `modules/tracker.py` | `_CLASS_NAME_MAP` 클래스 변수 추가, `update()`에서 참조 |

---

## 실행 계획 ✅ 완료

### Step 1: Tracker 내부 캐싱 ✅
- **변경 파일**: `modules/tracker.py` 1개
- **변경량**: ~5줄 (클래스 변수 추가 + 참조 변경)

### Step 2: NLP 로깅 비동기화 ✅
- **변경 파일**: `nlp/logger.py`, `core/pipeline.py`, `core/orchestrator.py`, `main.py` 4개
- **변경량**: ~80줄
- **리스크**: SpaceLogger collect() 흐름 변경 → 주기적 flush만으로 동작

---

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `config/config.py` | YOLOConfig에 model_size, quantize, frame_skip 필드 추가 |
| `config/spaces.yaml` | model_size, frame_skip 옵션 |
| `core/config_manager.py` | CameraConfig에 model_size 필드 |
| `core/orchestrator.py` | _make_pipeline_config에서 프레임 스킵 설정, Pipeline.stop() 호출 |
| `core/pipeline.py` | 프레임 스킵 로직, NLP 비동기화 대응 |
| `modules/tracker.py` | 양자화, 캐싱 |
| `nlp/logger.py` | NLP 로깅 비동기화 |
| `modules/tile_detector.py` | 양자화 모델 지원 |

---

# PLAN.md — 디렉토리 기반 영상 순차 처리

## SPEC

### 목적
카메라의 `source`에 디렉토리 경로를 설정하면, 디렉토리 내 영상 파일을 문자열 오름차순으로 순차 처리하여, 연속 녹화 영상을 시간순으로 처리. 모든 카메라의 모든 영상 처리 완료 시 종료.

### 범위
- `source`가 디렉토리인 경우: 디렉토리 내 영상 파일(`.mp4`, `.avi`, `.mkv`)을 문자열 오름차순 정렬 후 순차 처리
- `source`가 RTSP/HTTP URL인 경우: 기존 재연결 로직 그대로 유지
- `source`가 단일 파일인 경우: 파일 끝나는 대로 종료 (재연결 없음)
- 모든 카메라의 영상 처리 완료 시 `run_multi` 루프 종료
- 디렉토리 모드와 RTSP 모드는 분리해서 사용 (혼합하지 않음)

### 성공 기준
1. 디렉토리 source 설정 시, 디렉토리 내 영상 파일을 문자열 오름차순으로 순차 처리
2. 파일 전환 시 Pipeline 상태 초기화 (새 Pipeline 생성)
3. 모든 카메라의 모든 영상 처리 완료 시, 프로그램 정상 종료
4. RTSP 모드 동작에 영향 없음

---

## 진행 상황

### 1. 단일 파일 단발성 처리 ✅ 완료

**변경 파일**: `core/orchestrator.py`, `main.py`

- [x] `_is_stream_source()` — source가 스트림 URL인지 판별
- [x] `_CameraWorker._run()` — 파일 소스는 재연결 없이 종료
- [x] `_CameraWorker`에 `on_finished` 콜백, `_finished` 플래그 (double-release 방지)
- [x] `Orchestrator.worker_finished()` — worker가 `_workers`에서 자동 제거
- [x] `Orchestrator.all_finished` 속성 — `_workers` 비었는지 확인
- [x] `run_multi()` — `all_finished` 체크 후 종료

**변경 파일 요약:**

| 파일 | 변경 내용 |
|------|----------|
| `core/orchestrator.py` | `_is_stream_source()`, `_CameraWorker` 파일 종료/콜백, `Orchestrator.worker_finished()`, `Orchestrator.all_finished` |
| `main.py` | `run_multi()`에 `all_finished` 체크 |

---

## 설계 (미구현)

### 2. `_CameraWorker` — 디렉토리 처리 로직

**변경 파일**: `core/orchestrator.py`

- `source`가 디렉토리인지 확인 (`os.path.isdir()`)
- 디렉토리면 `.mp4`, `.avi`, `.mkv` 파일 필터링 → `sorted()` (문자열 오름차순)
- 각 파일을 순차 처리:
  - 파일마다 **새 Pipeline 생성** (이전 Pipeline `stop()` → 버그)
  - 새 Pipeline = Tracker 상태 완전 초기화 → 객체 ID 꼬임 없음
- RTSP/HTTP URL: 기존 재연결 로직 그대로

**구조:**
```
_CameraWorker._run():
  if is_dir(source):
      files = sorted_video_files(source)
      for file_path in files:
          if stop_event: break
          pipeline = new Pipeline(config, camera_id, space_logger, space_id)
          cap = create_capture(file_path)
          process_frames(cap, pipeline)   # 파일 끝날 때까지
          cap.release()
          pipeline.stop()
  else:
      # 기존 RTSP 재연결 로직 그대로
```

---

## 작업 목록

### 1. 단일 파일 단발성 처리 ✅ 완료
- [x] `_CameraWorker._run()` — 파일 소스는 재연결 없이 종료
- [x] `Orchestrator.worker_finished()` + `all_finished` 속성
- [x] `run_multi()` — `all_finished()` 체크 후 종료

### 2. `_CameraWorker` 디렉토리 처리 (미구현)
- [ ] `source`가 디렉토리인지 판별
- [ ] 디렉토리 내 영상 파일 필터링 (`.mp4`, `.avi`, `.mkv`) + 문자열 오름차순 정렬
- [ ] 파일마다 새 Pipeline 생성 → 처리 → `stop()`
- [ ] 파일 전환 시 프레임 카운터 초기화

---

## 고려사항

- **파일 정렬**: 문자열 오름차순 (`sorted()`) — `cam_rec_20260528-100600.mp4.mp4` 패턴이면 시간순 보장
- **Pipeline 재생성**: 파일마다 새 Pipeline. YOLO 모델은 `Tracker._ensure_loaded()` 지연 로드로 이미 메모리에 있을 것
- **스레드 종료**: worker가 `daemon=True`이므로, `Orchestrator._workers`에서 제거해야 `all_finished`가 true가 됨
- **Docker 볼륨**: `./data:/app/data`로 마운트 시 컨테이너 내부에서 디렉토리 접근 가능
