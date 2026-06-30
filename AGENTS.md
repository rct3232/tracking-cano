# AGENTS.md — tracking-cano

## 프로젝트 개요
LLM Vision: `_BatchCollector`(0.5s timer, buffer maxlen=5) → `SpaceLogger.vision_detect()` → snapshot trigger
Snapshot: CameraSnapshot buffer freeze → `SpaceLogger.space_snapshot()` → DB(batch_id) + image save(output/)
- Orchestrator: `_BatchCollector`(daemon thread, per camera) + `_SimpleVisionDetector`(round-robin per space)
- Detection: `llm_vision`(LLM detect 시 snapshot) — `_SimpleVisionDetector` round-robin per space
- Snapshot: `CameraSnapshot` registry → `space_snapshot()` → LLM space-level 분석 → per-camera detect + space log insert (동일 batch_id)

## Commandments
0. **디버깅 시 DEBUG 로깅 필수** — 문제 진단이 필요한 경우 반드시 `--verbose`/`-v`(또는 `LOG_LEVEL=DEBUG`)를 추가하여 실행하고, `docker logs`(또는 stdout) 출력에서 증거를 수집할 것. 추측으로 원인을 단정하지 말 것.
1. **코드가 진실** — 설정값/env var/CLI 플래그는 settings.py, configuration.yaml.example, main.py --help를 직접 읽고 AGENTS.md에 값 나열 금지
2. **LLM 직접 호출 금지** — 반드시 `LLMCallDebouncer`(cooldown=5s) 경유 (`nlp/logger.py:72`)
3. **환경변수 추가 시 configuration.yaml.example 필수** — 새 설정 추가 시 configuration.yaml.example에 반드시 추가
4. **Docker 실행 의무** — 로컬에서 `python main.py` 직접 실행 금지, 반드시 Docker로 실행.
5. **DATABASE_URL 우선순위** — `DATABASE_URL` env var가 설정되면 해당 DB 사용. 미설정 시 기본값 `sqlite:///logs/tracking.db`. PostgreSQL은 `postgresql://user:pass@host:5432/dbname` 형식.
6. **DB 비종속 설계** — `repo`가 None이면 console-only로 fallback (DB 없이도 standalone 동작). `init_db()` 실패 시 앱은 console-only로 계속 실행.
7. **PostgreSQL 전용 기능 금지** — SQLite에서도 동일하게 동작해야 함. JSONB 대신 Text, array 대신 JSON string으로 저장.

## 핵심 파일
| 파일 | 역할 |
|------|------|

| `core/orchestrator.py` | `_SimpleVisionDetector`(round-robin per space), `Orchestrator`(camera 관리, snapshot registry) |
| `core/vision_worker.py` | `_BatchCollector` — timer-based capture, sliding buffer, reconnect 시 buffer.clear() |
| `core/config_manager.py` | YAML/DB 로딩, `diff_configs()`, watchdog hot-reload (`ConfigWatcher`) |
| `core/config_applier.py` | `apply_config_changes()` — 구조적 변경 적용 + 값 변경 시 카메라 재시작 |
| `core/config_listener.py` | `ConfigListener` — PostgreSQL LISTEN/NOTIFY + polling 폴백 |
| `core/yaml_writer.py` | 원자적 YAML 읽기/쓰기 (REST API → config) |

| `nlp/logger.py` | `LLMCallDebouncer`(5s), `SpaceLogger.vision_detect()`, `SpaceLogger.space_snapshot()`, `_snapshot_fallback()` |
| `nlp/prompts.py` | LLM 시스템 프롬프트 상수 (`DETECT_SYSTEM_PROMPT`, `SNAPSHOT_VISION_PROMPT`, `SNAPSHOT_TRACKING_PROMPT`) |

| `api/server.py` | FastAPI 앱 + uvicorn daemon thread |
| `api/event_bus.py` | Thread-safe pub/sub (SSE 스트리밍) |
| `api/auth.py` | Bearer token 인증 (`verify_token`) |
| `api/models.py` | Pydantic request/response 모델 |

| `storage/database.py` | SQLAlchemy engine, session, LogEntry/AppSetting/ConfigVersion 모델, `init_db()` |
| `storage/repository.py` | `LogRepository.save()` — DB insert 추상화 |
| `storage/config_repository.py` | `ConfigRepository` — DB 설정 저장/조회 (PostgreSQL 전용) |

| `settings.py` | LLMConfig/LogConfig/MinIOConfig/ReconnectConfig dataclasses |
| `utils/video.py` | `create_capture()` |
| `utils/image.py` | `draw_normalized_bbox()` — LLM 응답 bbox 시각화 |
| `scripts/migrate_yaml_to_db.py` | YAML → DB 마이그레이션 스크립트 |

## Docker 실행 절차

1. `configuration.yaml` 파일이 존재하는지 확인한다.
2. 존재하지 않으면 `configuration.yaml.example`을 복사하여 편집한다.
3. `.env` 파일이 존재하는지 확인한다.
4. 존재하지 않으면 `.env.example`을 읽어 `LLM_KEY`와 `API_KEY`를 입력받고 저장한다:
   - `LLM_KEY` — LLM API 키 (필수)
   - `API_KEY` — REST API 인증 토큰 (선택)
5. 생성된 `.env` 전체 내용을 유저에게 보여주고 "이대로 진행할까요?"라고 확인받는다.
6. 유저가 승인하면 `docker compose up --build`를 실행한다.

## Gotchas (2026-06 기준)
- 타입 힌트 혼용 중: `Optional[str]`(old) vs `str \| None`(new). 기존 파일 스타일 유지
- `max_stale_threshold`는 `_BatchCollector`가 아닌 `_SimpleVisionDetector._run_space()`에서 사용됨 — age-based eviction은 vision detector loop 레벨에서 동작
- 로그는 파일 대신 DB(`log_entries` 테이블)에 저장됨. Console 출력은 `logs/console_YYYYMMDD.log`에 파일 저장.
- `DATABASE_URL` env var가 설정되면 해당 DB 사용. 미설정 시 `configuration.yaml`의 `logging.db_url` 또는 기본값 `sqlite:///logs/tracking.db` 사용.
- DB 연결 실패해도 앱은 console-only로 계속 실행 (`repo=None` → 모든 `_db_insert`가 무시됨).
