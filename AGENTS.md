# AGENTS.md — tracking-cano

## 프로젝트 개요
LLM Vision: `_BatchCollector`(0.5s timer, buffer maxlen=5) → `SpaceLogger.vision_detect()` → snapshot trigger
Snapshot: CameraSnapshot buffer freeze → `SpaceLogger.space_snapshot()` → DB(batch_id) + image save(output/)
- Orchestrator: `_CameraWorker`(daemon thread)로 다중 카메라 관리 + `_SimpleVisionDetector`(round-robin detect)
- Detection: `llm_vision`(LLM detect 시 snapshot) — `_SimpleVisionDetector` round-robin per space
- Snapshot: `CameraSnapshot` registry → `space_snapshot()` → LLM space-level 분석 → per-camera detect + space log insert (동일 batch_id)

## Commandments
0. **디버깅 시 DEBUG 로깅 필수** — 문제 진단이 필요한 경우 반드시 `--verbose`/`-v`(또는 `LOG_LEVEL=DEBUG`)를 추가하여 실행하고, `docker logs`(또는 stdout) 출력에서 증거를 수집할 것. 추측으로 원인을 단정하지 말 것.
1. **코드가 진실** — 설정값/env var/CLI 플래그는 settings.py, configuration.yaml.example, main.py --help를 직접 읽고 AGENTS.md에 값 나열 금지

4. **LLM 직접 호출 금지** — 반드시 `LLMCallDebouncer`(cooldown=3s) 경유
5. **환경변수 추가 시 configuration.yaml.example 필수** — 새 설정 추가 시 configuration.yaml.example에 반드시 추가
6. **벤치마크 기록 의무** — bench.py 실행 후 BENCHMARKS.md 갱신 필수
7. **Docker 실행 의무** — 로컬에서 `python main.py` 직접 실행 금지, 반드시 Docker로 실행.
8. **DATABASE_URL 우선순위** — `DATABASE_URL` env var가 설정되면 해당 DB 사용. 미설정 시 기본값 `sqlite:///logs/tracking.db`. PostgreSQL은 `postgresql://user:pass@host:5432/dbname` 형식.
9. **DB 비종속 설계** — `repo`가 None이면 console-only로 fallback (DB 없이도 standalone 동작). `init_db()` 실패 시 앱은 console-only로 계속 실행.
10. **PostgreSQL 전용 기능 금지** — SQLite에서도 동일하게 동작해야 함. JSONB 대신 Text, array 대신 JSON string으로 저장.

## 핵심 파일
| 파일 | 역할 |
|------|------|

| `core/orchestrator.py` | `_CameraWorker`(daemon, 재연결), `_SimpleVisionDetector`, snapshot registry, `diff_configs()` hot-reload |
| `core/vision_worker.py` | `_BatchCollector` — LLM vision용 timer-based capture, sliding buffer, reconnect 시 buffer.clear() |

| `nlp/logger.py` | `LLMCallDebouncer`(3s), `SpaceLogger.vision_detect()`, `SpaceLogger.space_snapshot()`, `_snapshot_fallback()` |
| `settings.py` | LLMConfig/LogConfig/MinIOConfig/ReconnectConfig dataclasses |
| `core/config_manager.py` | YAML 로딩, `diff_configs()`, watchdog hot-reload |
| `utils/video.py` | `create_capture()`, `resolve_source()` |
| `utils/image.py` | `draw_normalized_bbox()` — LLM 응답 bbox 시각화 |
| `storage/database.py` | SQLAlchemy engine, session, LogEntry 모델, `init_db()` |
| `storage/repository.py` | `LogRepository.save()` — DB insert 추상화 |

## Docker 실행 절차

1. `configuration.yaml` 파일이 존재하는지 확인한다.
2. 존재하지 않으면 `configuration.yaml.example`을 복사하여 편집한다.
3. `.env` 파일이 존재하는지 확인한다.
4. 존재하지 않으면 `.env.example`을 읽어 `LLM_KEY`와 `API_KEY`를 입력받고 저장한다:
   - `LLM_KEY` — LLM API 키 (필수)
   - `API_KEY` — REST API 인증 토큰 (선택)
5. 생성된 `.env` 전체 내용을 유저에게 보여주고 "이대로 진행할까요?"라고 확인받는다.
6. 유저가 승인하면 `docker compose up --build`(CPU) 또는 `-f docker-compose.gpu.yml`(GPU)를 실행한다.
   - GPU 여부는 유저에게 추가로 물어본다.

## Gotchas (2026-06 기준)
- `--multi` 플래그 없음. `--live` 단독 인자 = multi mode
- 타입 힌트 혼용 중: `Optional[str]`(old) vs `str \| None`(new). 기존 파일 스타일 유지
- `LLMConfig.cooldown_seconds` 중복 선언: 하드코딩된 3.0s가 `llm.cooldown_seconds`(기본 30.0)로 override됨 — vision logging 후 일반 텍스트 LLM도 30s 쿨다운 적용됨 (버그 가능성 있음)
- `max_stale_threshold`는 `settings.py`/`configuration.yaml`에 선언되어 있으나 현재 `_BatchCollector`에서 사용되지 않음 — age-based eviction 미구현
- 로그는 파일 대신 DB(`log_entries` 테이블)에 저장됨. Console 출력은 `logs/console_YYYYMMDD.log`에 파일 저장.
- `DATABASE_URL` env var가 설정되면 해당 DB 사용. 미설정 시 `configuration.yaml`의 `logging.db_url` 또는 기본값 `sqlite:///logs/tracking.db` 사용.
- DB 연결 실패해도 앱은 console-only로 계속 실행 (`repo=None` → 모든 `_db_insert`가 무시됨).
