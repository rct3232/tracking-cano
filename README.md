# tracking-cano

LLM Vision을 이용해 IP 카메라(RTSP/MP4) 영상에서 지정한 객체를 감지하고, 스냅샷 기반으로 공간 수준의 자연어 로깅을 수행하는 시스템.

추적 대상은 고양이뿐만 아니라 사람, 동물, 차량 등 COCO 클래스 중 무엇이든 설정 가능합니다.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│ Detection Layer (per-camera daemon)                                 │
│   Camera ─→ [_BatchCollector] ─→ [LLM Vision] ─→ target_present?   │
│   (llm_vision)                                                      │
└──────────────────────────────────────────────────────────────────────┘
                              │ (snapshot trigger: CV change / LLM detect)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Snapshot Layer (space-level)                                        │
│   [CameraSnapshot] ─→ buffer freeze ─→ [SpaceLogger.space_snapshot()]│
│   ├─ vision_enabled=true:  LLM (images + detect data) → JSON       │
│   └─ vision_enabled=false: LLM (detect data only) → JSON           │
│   ↓                                                                 │
│   [DB insert] detect×N + space×1 ──→ log_entries (batch_id 그룹핑)  │
│   [Image save]  snapshot images ──→ output/                         │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ REST API (FastAPI, port 8000)                                       │
│   /api/cameras/  /api/spaces/  /api/logs/  /api/config/            │
│   /api/health    /api/status   /api/logs/stream (SSE)              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

- **LLM Vision 기반 객체 감지** — OpenAI API 호환 LLM으로 이미지에서 대상 객체 감지
- **Snapshot-based space logging** — detection trigger → space-level LLM 분석 → DB 저장(batch_id 그룹핑) + 이미지 저장
- **다중 카메라·다중 공간** — YAML 구성 파일로 카메라와 공간의 관계 정의
- **동적 구성 핫리로드** — 실행 중 구성 변경 시 자동 반영 (watchdog)
- **자연어 로깅** — OpenAI API 호환 LLM을 통해 snapshot 시점 자연어 분석
- **REST API** — FastAPI 기반 카메라/공간/로그 CRUD + SSE 실시간 로그 스트리밍
- **DB 저장** — SQLite(기본) 또는 PostgreSQL, console-only fallback 지원

---

## Tech Stack

| 용도 | 도구 |
|------|------|
| Object Detection | LLM Vision (OpenAI API 호환) |
| NLP Logging | OpenAI API 호환 LLM |
| REST API | FastAPI + uvicorn |
| Database | SQLAlchemy + SQLite / PostgreSQL |
| Config | PyYAML + python-dotenv |
| Hot Reload | watchdog |

---

## Quick Start (Docker)

```bash
# 설정 파일 준비
cp configuration.yaml.example configuration.yaml   # RTSP URL 등 실제 값으로 수정
cp .env.example .env                               # LLM API key 입력

# 빌드 및 실행
docker compose up --build
```

## Configuration

### `configuration.yaml` — 전체 설정 파일 (`.gitignore` 제외)

```yaml
# === 공간 정의 (선택) ===
spaces:
  - id: room_living
    name: 거실
    cameras: [cam_01, cam_02]

# === 카메라 정의 (필수) ===
cameras:
  - id: cam_01
    source: rtsp://your-camera-ip:554/stream
    status: active
    target_classes: [cat, person]
```


`configuration.yaml.example`을 복사해서 사용하세요. `configuration.yaml`은 `.gitignore`에 제외되어 있습니다.

### `.env` — LLM API key + Database URL (`.gitignore` 제외)

```env
LLM_KEY=your_llm_api_key_here
# API_KEY=your_rest_api_key_here
# DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

---

## Usage

```bash
# 기본 실행 (Docker 권장)
docker compose up --build

# DEBUG 로깅 활성화
docker run --rm -e LOG_LEVEL=DEBUG tracking-cano python main.py --verbose
```

| 옵션 | 설명 |
|------|------|
| `--config <path>` | 구성 파일 경로 (기본: `configuration.yaml`) |
| `--verbose` / `-v` | DEBUG 레벨 로깅 활성화 |

---

## REST API

실행 시 자동으로 FastAPI 서버가 포트 8000에서 시작됩니다.

```bash
# 헬스체크
curl http://localhost:8000/api/health

# 상태 조회
curl http://localhost:8000/api/status

# 로그 조회 (subject_id 필터)
curl "http://localhost:8000/api/logs/?subject_id=livingroom"

# SSE 로그 스트리밍
curl -N http://localhost:8000/api/logs/stream
```

Swagger UI: http://localhost:8000/docs

---

## Debugging

### `--verbose` / `-v` 플래그

`main.py`에 `--verbose`를 추가하면 모든 모듈의 **DEBUG 레벨 로그**가 `docker logs`(또는 stdout/stderr)로 출력됩니다.

```bash
# Docker DEBUG 모드로 실행
docker run --rm -e LOG_LEVEL=DEBUG tracking-cano python main.py --verbose
```

### DEBUG 로그 종류

| 로그 | Level | 모듈 | 설명 |
|------|-------|------|------|
| `[detect:cam] target_present=True` | INFO | nlp.logger | LLM vision detect 결과 |
| `[snapshot:space] cameras=N reasoning=...` | INFO | nlp.logger | Snapshot LLM 응답 |
| `[space:room] snapshot debounce suppress` | DEBUG | nlp.logger | Snapshot 디바운스 |
| `[cam:X] collector encoded frame=N` | DEBUG | vision_worker | _BatchCollector 타이머 캡처 |
| `[cam:X] collector buffer cleared (reconnect)` | INFO | vision_worker | 재연결 시 버퍼 초기화 |
| `[space:X] cam=Y target_present=True → snapshot` | INFO | orchestrator | detect 결과 snapshot 트리거 |
| `[detect:cam] parse failed, raw: ...` | WARNING | nlp.logger | LLM 응답 JSON 파싱 실패 |

### `LOG_LEVEL` 환경변수

`LOG_LEVEL=DEBUG` 환경변수로도 동일하게 DEBUG 로깅을 활성화할 수 있습니다 (`--verbose`와 동일).

---


## Project Structure

```
tracking-cano/
├── main.py                          # 진입점
├── requirements.txt                 # 의존성
│
├── configuration.yaml               # 전체 설정 파일 (gitignore)
├── configuration.yaml.example       # 커밋용 템플릿
├── settings.py                      # LLMConfig/LogConfig/MinIOConfig/ReconnectConfig dataclasses
├── core/
│   ├── __init__.py
│   ├── config_manager.py            # YAML/DB 로딩 + 핫리로드(diff_configs)
│   ├── config_applier.py            # 구조적 변경 적용 + 값 변경 시 카메라 재시작
│   ├── config_listener.py           # PostgreSQL LISTEN/NOTIFY + polling 폴백
│   ├── orchestrator.py              # 다중 카메라 오케스트레이션 + snapshot 관리
│   ├── vision_worker.py             # _BatchCollector (LLM vision 전용 timer capture)
│   └── yaml_writer.py               # 원자적 YAML 읽기/쓰기 (REST API → config)
├── nlp/
│   ├── __init__.py
│   ├── logger.py                    # SpaceLogger + LLMCallDebouncer + snapshot 관리
│   └── prompts.py                   # 시스템 프롬프트 (SNAPSHOT_VISION/TRACKING/DETECT)
├── api/
│   ├── __init__.py
│   ├── server.py                    # FastAPI 앱 + uvicorn daemon thread
│   ├── auth.py                      # Bearer token 인증
│   ├── event_bus.py                 # Thread-safe pub/sub (SSE 스트리밍)
│   ├── models.py                    # Pydantic request/response 모델
│   └── routes/
│       ├── __init__.py
│       ├── cameras.py               # 카메라 CRUD
│       ├── spaces.py                # 공간 CRUD + flush
│       ├── logs.py                  # 로그 조회 + SSE
│       ├── status.py                # 상태/헬스체크
│       └── config.py                # 설정 읽기/쓰기
├── storage/
│   ├── __init__.py
│   ├── database.py                  # SQLAlchemy engine, LogEntry/AppSetting/ConfigVersion 모델
│   ├── repository.py                # LogRepository.save() 추상화
│   └── config_repository.py         # DB 설정 저장/조회 (PostgreSQL 전용)
├── utils/
│   ├── __init__.py
│   ├── video.py                     # create_capture()
│   └── image.py                     # draw_normalized_bbox()
├── scripts/
│   └── migrate_yaml_to_db.py        # YAML → DB 마이그레이션 스크립트
├── logs/                            # 로그/DB 디렉토리 (Docker volume)
├── output/                          # 스냅샷 이미지 저장 (Docker volume)
├── .env                             # 민감 정보 (gitignore)
├── .env.example                     # 템플릿
├── .gitignore                       # git 제외 패턴
├── Dockerfile                       # 컨테이너 이미지
└── docker-compose.yml               # 서비스 정의
```

---

## License

[Apache 2.0](LICENSE)
