# AGENTS.md — tracking-cano

## 프로젝트 개요
Camera → YOLO26 → ByteTrack → Movement Analyzer → Interaction Detector → NLPLogger → SpaceLogger → LLM
- Orchestrator: `_CameraWorker`(daemon thread)로 다중 카메라 관리, stream은 5회 실패 시 재시작, file/디렉토리는 단발
- SpaceLogger: 동일 공간 카메라 로그 취합 → 주기적 flush(10s) + 이벤트 기반 try_flush()로 LLM 요약

## Commandments
0. **디버깅 시 DEBUG 로깅 필수** — 문제 진단이 필요한 경우 반드시 `--verbose`/`-v`(또는 `LOG_LEVEL=DEBUG`)를 추가하여 실행하고, `docker logs`(또는 stdout) 출력에서 증거를 수집할 것. 추측으로 원인을 단정하지 말 것.
1. **코드가 진실** — 설정값/env var/CLI 플래그는 config.py, .env.example, main.py --help를 직접 읽고 AGENTS.md에 값 나열 금지
2. **State hold 보호** — pipeline.py 수정 시 `_state_hold` → `_prev_states` → `_check_disappeared()` 체인과의 상호작용 반드시 고려
3. **지연 로드 원칙** — Tracker YOLO는 `_ensure_loaded()`로만 로드, `__init__`에서 로드 금지
4. **LLM 직접 호출 금지** — 반드시 `LLMCallDebouncer`(cooldown=3s) 경유
5. **환경변수 추가 시 양쪽 필수** — config.py + .env.example, 한쪽만 하면 버그
6. **벤치마크 기록 의무** — bench_rtsp.py 실행 후 BENCHMARKS.md 갱신 필수
7. **Docker 실행 의무** — 로컬에서 `python main.py` 직접 실행 금지, 반드시 Docker로 실행.

## 핵심 파일
| 파일 | 역할 |
|------|------|
| `core/pipeline.py` | `process_frame()` → state hold + disappear/interaction change 감지 |
| `core/orchestrator.py` | `_CameraWorker`(daemon, 재연결), `flush_spaces()`, `diff_configs()` hot-reload |
| `modules/tracker.py` | YOLO+ByteTrack, `_ensure_loaded()` 지연로드, HybridDetector(tile fallback) |
| `modules/analyzer.py` | `classify_movement()` → STOPPED/SLOW/FAST/DASH/ROTATE |
| `modules/interaction_detector.py` | IoU+거리 기반: interacting/contact/nearby |
| `nlp/logger.py` | `LLMCallDebouncer`(3s), `SpaceLogger`, `try_flush()` |
| `config/config.py` | Thresholds/YOLOConfig/LLMConfig/PipelineConfig dataclasses |
| `config/config_manager.py` | YAML 로딩, `diff_configs()`, watchdog hot-reload |
| `utils/video.py` | `create_capture()`, `resolve_source()` |

## Docker 실행 절차

1. `.env` 파일이 존재하는지 확인한다.
2. 존재하지 않으면 `.env.example`을 읽어 아래 **핵심 변수**만 유저에게 입력받고, 나머지는 기본값을 그대로 사용한다:
   - `RTSP_URL` — 카메라 RTSP 주소 (필수)
   - `API_KEY` — LLM API 키 (필수)
   - `API_BASE_URL` — LLM API 엔드포인트 (기본값: `https://api.openai.com/v1`)
   - `MODEL_NAME` — 사용할 LLM 모델명 (기본값: `gpt-4o-mini`)
3. 생성된 `.env` 전체 내용을 유저에게 보여주고 "이대로 진행할까요?"라고 확인받는다.
4. 유저가 승인하면 `docker compose up --build`(CPU) 또는 `-f docker-compose.gpu.yml`(GPU)를 실행한다.
   - GPU 여부는 유저에게 추가로 물어본다.

## Gotchas (2026-05 기준)
- `--multi` 플래그 없음. `--live` 단독 인자 = multi mode
- 타입 힌트 혼용 중: `Optional[str]`(old) vs `str \| None`(new). 기존 파일 스타일 유지
