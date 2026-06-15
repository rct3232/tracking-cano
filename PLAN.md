# PLAN.md — DB 기반 설정 관리

## SPEC

### Objective
운영 모드(PostgreSQL)에서 configuration.yaml 대신 DB에 모든 설정을 저장하고, config 변경 시 multi-node 환경에서 LISTEN/NOTIFY로 실시간 감지하여 카메라 재시작. 개발 모드는 기존 YAML hot-reload 유지.

### Scope
- **운영**: DB(app_settings + config_version) ↔ AppConfig 양방향 관리
- **개발**: 기존 YAML + watchdog hot-reload (변경 없음)
- 설정 변경 시 영향받는 카메라 재시작 (lazy 적용 X)
- LLM API key, MinIO 설정은 env var 전용 유지

### Key Components
| 컴포넌트 | 역할 |
|----------|------|
| `app_settings` 테이블 | DB 기반 config 저장소 |
| `config_version` 테이블 | 변경 감지용 버전 카운터 |
| `ConfigRepository` | DB ↔ dict 변환, patch, notify |
| `load_from_db()` | DB → AppConfig 파싱 |
| `_apply_config_changes()` | 변경 diff → 카메라 재시작 |
| `LISTEN/NOTIFY listener` | multi-node 실시간 감지 + version 폴백 (30s) |

### Success Criteria
1. PostgreSQL 연결 시 DB에서 설정 로드, YAML 무시
2. API를 통해 thresholds/yolo/llm/camera/space 변경 → 즉시 DB 반영 + NOTIFY 발행
3. 다른 pod가 변경 수신 → diff 계산 → 영향받는 카메라 재시작
4. 개발 모드(SQLite/YAML) → 기존 동작 유지
5. 운영 DB 연결 실패 시 앱 시작 거부

---

## 스키마

### app_settings

```sql
app_settings (
    key          TEXT PRIMARY KEY,     -- prefix 내 고유 ('speed_slow', 'livingroom')
    key_prefix   TEXT NOT NULL,        -- 'thresholds' | 'yolo' | 'llm' | 'cameras' | 'spaces' | 'mode'
    value_text   TEXT,                 -- string / JSON (camera 전체, space 전체)
    value_number NUMERIC,              -- int / float
    value_bool   BOOLEAN,              -- true / false
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_app_settings_prefix ON app_settings(key_prefix);
```

### config_version

```sql
config_version (
    id      INTEGER PRIMARY KEY CHECK(id = 1),
    version INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### key 매핑

| key_prefix | key 예시 | 컬럼 | 내용 |
|-----------|---------|------|------|
| `mode` | — (key='mode') | value_text | `"cv_pipeline"` |
| `thresholds` | `speed_slow`, `overlap` | value_number | 20.0, 0.3 |
| `yolo` | `conf_threshold`, `tile_enabled` | value_number / value_bool | 0.25, true |
| `llm` | `model_name`, `cooldown_seconds` | value_text / value_number | `"gpt-4o-mini"`, 30.0 |
| `cameras` | `livingroom` | value_text (JSON) | 전체 camera 설정 |
| `spaces` | `room_livingroom` | value_text (JSON) | `{"name": "거실", "cameras": ["livingroom"], ...}` |

---

## 구현 단계

### Step 16-1: 스키마 마이그레이션 (alembic migration)

**파일**: `migrations/versions/XXXX_add_app_settings.py`
```python
def upgrade():
    op.create_table('app_settings', ...)
    op.create_index('idx_app_settings_prefix', 'app_settings', ['key_prefix'])
    op.create_table('config_version', ...)
    op.execute("INSERT INTO config_version (id, version) VALUES (1, 0)")

def downgrade():
    op.drop_table('config_version')
    op.drop_index('idx_app_settings_prefix')
    op.drop_table('app_settings')
```

### Step 16-2: ConfigRepository 구현

**파일**: `storage/config_repository.py` (신규)

```python
class ConfigRepository:
    def __init__(self, session_factory): ...

    # ── 읽기 ──
    def get_version(self) -> int: ...
    def get_full_config(self) -> Dict[str, Any]: ...
        # prefix별 배치 조회 → AppConfig로 변환

    # ── 쓰기 ──
    def patch_thresholds(self, updates: dict): ...   # value_number 업데이트
    def patch_yolo(self, updates: dict): ...          # mixed column
    def patch_llm(self, updates: dict): ...           # mixed column
    def save_camera(self, camera_id: str, data: dict): ...  # JSON 전체 덮어쓰기
    def remove_camera(self, camera_id: str): ...      # DELETE WHERE key_prefix='cameras' AND key=camera_id
    def save_space(self, space_id: str, data: dict): ...   # JSON 전체 덮어쓰기
    def remove_space(self, space_id: str): ...       # DELETE WHERE key_prefix='spaces' AND key=space_id

    # ── 공통 ──
    def _increment_version(self): ...                # config_version.version += 1
    def _notify(self): ...                          # NOTIFY config_changed
```

### Step 16-3: load_from_db() 구현

**파일**: `core/config_manager.py` (추가)

```python
def load_from_db(repo: ConfigRepository, llm_key: str = "") -> AppConfig:
    raw = repo.get_full_config()
    thresholds = Thresholds.from_dict(raw["thresholds"])
    yolo = YOLOConfig.from_dict(raw["yolo"])
    llm = LLMConfig.from_dict(raw["llm"])
    llm.api_key = llm_key  # env var override (기존 규칙)
    log = LogConfig(db_url=os.environ.get("DATABASE_URL", ""))
    cameras = [CameraConfig(c) for c in raw["cameras"]]
    spaces = [SpaceConfig(s) for s in raw["spaces"]]
    mode = raw["mode"]
    return AppConfig(cameras, spaces, thresholds, yolo, llm, log, mode)
```

### Step 16-4: main.py 분기 로직

**파일**: `main.py` → `run_multi()` 변경

```python
def run_multi(config_path: str, model_path=None, repo=None):
    db_url = os.environ.get("DATABASE_URL", "")
    is_postgres = db_url.startswith("postgresql://")

    if is_postgres:
        # DB 연결 확인 — 실패 시 앱 시작 거부
        engine = create_engine(db_url)
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error("DB connection failed, aborting: %s", e)
            sys.exit(1)

        config_repo = ConfigRepository(Session)
        app_config = load_from_db(config_repo, llm_key=os.environ.get("LLM_KEY", ""))
    else:
        # 기존 YAML 로드 (변경 없음)
        app_config = load_config(config_path)

    event_bus = EventBus()
    minio_cfg = MinIOConfig.from_env()
    space_logger = SpaceLogger(app_config.llm, repo=repo, event_bus=event_bus, minio_config=minio_cfg)
    orchestrator = Orchestrator(app_config, space_logger, default_model_path=model_path, repo=repo, event_bus=event_bus)

    start_api(orchestrator=orchestrator, space_logger=space_logger, repo=repo, event_bus=event_bus)
    orchestrator.start()

    if is_postgres:
        # LISTEN/NOTIFY listener 시작
        listener = ConfigListener(config_repo, orchestrator, space_logger)
        listener.start()
    else:
        # 기존 watchdog hot-reload (변경 없음)
        watcher = ConfigWatcher(config_path, lambda new_cfg, diff: _on_config_change(...))
        watcher.start()

    # ... main loop 동일
```

### Step 16-5: ConfigListener (LISTEN/NOTIFY + polling 폴백)

**파일**: `core/config_listener.py` (신규)

```python
class ConfigListener(threading.Thread):
    def __init__(self, repo: ConfigRepository, orchestrator, space_logger, db_url: str):
        self.repo = repo
        self.orchestrator = orchestrator
        self.space_logger = space_logger
        self.db_url = db_url
        self._last_version = repo.get_version()

    def run(self):
        # LISTEN 채널 구독
        conn = psycopg2.connect(self.db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        cur.execute("LISTEN config_changed")

        poll_interval = 30.0
        next_poll = time.monotonic() + poll_interval

        while True:
            now = time.monotonic()
            # LISTEN 이벤트 대기 (polling 간격만큼 timeout)
            remaining = max(0, next_poll - now)
            conn.notifies(timeout=remaining)  # block until NOTIFY or timeout

            # polling 폴백 체크
            if time.monotonic() >= next_poll:
                current_version = self.repo.get_version()
                if current_version != self._last_version:
                    self._apply_changes()
                next_poll = time.monotonic() + poll_interval

    def _apply_changes(self):
        new_cfg = load_from_db(self.repo, llm_key=os.environ.get("LLM_KEY", ""))
        diff = diff_configs(self.orchestrator.app_config, new_cfg)
        if not diff.is_empty:
            _on_config_change(self.orchestrator, self.space_logger, new_cfg, diff)
        # thresholds/yolo/llm 변경 감지 → 전체 카메라 재시작 필요 (별도 로직)
```

### Step 16-6: API 라우터 DB 연동

**파일**: `api/routes/cameras.py`, `spaces.py`, `config.py`

각 라우터에서 yaml_writer 호출을 ConfigRepository로 교체. 조건부 분기:

```python
# api/routes/config.py 예시
@router.put("/thresholds")
async def update_thresholds(body, token):
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")

    orch = _get_orchestrator()
    is_db_mode = orch and hasattr(orch, '_config_repo') and orch._config_repo is not None

    if is_db_mode:
        repo = orch._config_repo
        for k, v in updates.items():
            repo.patch_thresholds({k: v})  # value_number 업데이트 + version 증가 + NOTIFY
    else:
        update_yaml_config_section("thresholds", updates)

    # orchestrator에 새 config 적용 (카메라 재시작)
    if orch:
        new_cfg = load_from_db(repo) if is_db_mode else load_config()
        _apply_config_changes(orch, orch.space_logger, new_cfg)
```

### Step 16-7: _apply_config_changes — 설정 변경 → 카메라 재시작

**파일**: `main.py` (기존 `_on_config_change` 확장)

```python
def _apply_config_changes(orchestrator, space_logger, new_config):
    old = orchestrator.app_config
    diff = diff_configs(old, new)

    # 1. camera/space 추가·삭제·재할당 (기존 로직 유지)
    for cam_id in diff.added_cameras: ...
    for cam_id in diff.removed_cameras: ...
    for space_id in diff.added_spaces: ...
    for space_id in diff.removed_spaces: ...

    # 2. 설정 값 변경 감지 → 영향받는 카메라 재시작
    needs_restart = False
    if old.thresholds != new_config.thresholds: needs_restart = True
    if old.yolo != new_config.yolo:            needs_restart = True
    if old.llm != new_config.llm:              needs_restart = True

    # camera 개별 설정 변경 감지 (JSON 파싱 필요)
    for cam in new_config.cameras:
        old_cam = next((c for c in old.cameras if c.id == cam.id), None)
        if old_cam and _camera_values_differ(old_cam, cam):
            needs_restart = True

    if needs_restart:
        # 영향받는 카메라 재시작 (remove + add)
        for cam_id in list(orchestrator._workers.keys()) + list(orchestrator._collectors.keys()):
            orchestrator.remove_camera(cam_id)
            cam_obj = next((c for c in new_config.cameras if c.id == cam_id), None)
            if cam_obj:
                orchestrator.add_camera(cam_obj)

    orchestrator.update_config(new_config)
```

### Step 16-8: 마이그레이션 스크립트 (별도)

**파일**: `scripts/migrate_yaml_to_db.py` (신규)

```python
"""YAML → DB 마이그레이션 (수동 실행). 운영 DB에 기존 YAML 설정을 이관."""

from dotenv import load_dotenv
load_dotenv()

import os
from sqlalchemy import text
from core.config_manager import load_config
from storage.database import init_db
from storage.config_repository import ConfigRepository

def migrate():
    db_url = os.environ["DATABASE_URL"]  # 필수
    engine, Session = init_db(db_url)
    repo = ConfigRepository(Session)

    app_config = load_config("configuration.yaml")

    # mode
    repo.save_setting("mode", "mode", value_text=app_config.mode)

    # thresholds
    for k, v in app_config.thresholds.__dict__.items():
        repo.save_setting(k, "thresholds", value_number=v)

    # yolo (global)
    for k, v in app_config.yolo.__dict__.items():
        if isinstance(v, bool):
            repo.save_setting(k, "yolo", value_bool=v)
        elif isinstance(v, (int, float)):
            repo.save_setting(k, "yolo", value_number=v)

    # llm (api_key 제외 — env var 전용)
    for k, v in app_config.llm.__dict__.items():
        if k == "api_key":
            continue
        elif isinstance(v, bool):
            repo.save_setting(k, "llm", value_bool=v)
        elif isinstance(v, (int, float)):
            repo.save_setting(k, "llm", value_number=v)
        else:
            repo.save_setting(k, "llm", value_text=str(v))

    # cameras → JSON 전체 저장
    for cam in app_config.cameras:
        import json
        data = {
            "id": cam.id, "source": cam.source, "status": cam.status,
            "target_classes": cam.target_classes,
            "interaction_classes": cam.interaction_classes,
            "model_size": cam.model_size, "frame_skip": cam.frame_skip,
            "quantize": cam.quantize, "llm_system_prompt": cam.llm_system_prompt,
        }
        repo.save_camera(cam.id, data)

    # spaces → JSON 전체 저장
    for sp in app_config.spaces:
        import json
        data = {
            "name": sp.name, "cameras": sp.camera_ids,
            "llm_system_prompt": sp.llm_system_prompt,
        }
        repo.save_space(sp.id, data)

    print(f"Migrated: {len(app_config.cameras)} cameras, {len(app_config.spaces)} spaces")

if __name__ == "__main__":
    migrate()
```

---

## Progress

| Phase | Step | Status |
|-------|------|--------|
| **Phase 16-1: 스키마 마이그레이션** | app_settings + config_version 테이블 생성 (create_all) | ✅ |
| **Phase 16-2: ConfigRepository** | get_full_config, patch_*, save_camera/space, notify | ✅ |
| **Phase 16-3: load_from_db()** | DB → AppConfig 파싱 | ✅ |
| **Phase 16-4: main.py 분기** | PostgreSQL → DB / 개발 → YAML | ✅ |
| **Phase 16-5: ConfigListener** | LISTEN/NOTIFY + version polling 폴백 | ✅ |
| **Phase 16-6: API 라우터 교체** | yaml_writer → ConfigRepository (조건부) | ✅ |
| **Phase 16-7: _apply_config_changes** | 설정 변경 감지 → 카메라 재시작 | ✅ |
| **Phase 16-8: 마이그레이션 스크립트** | YAML → DB 이관 (config_version=58) | ✅ |

---

## Hot-reload 미감지 항목 (현재 문제점 기록)

`diff_configs()`는 camera/space 추가·삭제·재할당만 감지. 다음 변경은 무시됨:
- thresholds 전체, yolo 전체, llm 전체, mode 변경
- camera 내부 값 변경 (source, target_classes, frame_skip 등)
- space 내부 값 변경 (name, llm_system_prompt)

---

# Jenkins 등록 계획

## 아키텍처

| 서버 | 역할 |
|------|------|
| **10.0.0.100** | Jenkins (pipeline 실행, Helm chart 원본) |
| **10.0.0.105** | 빌드 서버 (docker build/push + helm deploy via kubeconfig → 200) |
| **10.0.0.200** | K3s control-plane (K8s 클러스터, worker-1.k8s 포함) |

## webdav-easyaccess 참고 패턴

- **Secret**: `webdav-secret` — 사전에 200번에서 직접 kubectl로 생성, Pipeline과 무관
- **ConfigMap**: Helm chart가 values.yaml에서 비민감 설정만 읽어 생성 (password 없음)
- **Deployment template**: `secretKeyRef`로 K8s Secret 이름만 참조
- **Pipeline**: `image.tag` 등 비민감 값만 넘김

## tracking-cano Pipeline 설계

**Trigger**: Generic Webhook — main 브랜치 push 시 (token: `tracking-cano`)

| Stage | 작업 | 비고 |
|-------|------|------|
| **Clone** | main 브랜치 clone | Dockerfile은 레포에 있으므로 `writeFile` 불필요 |
| **Sync to Build Server** | rsync로 소스 + chart → 105번 | `.env`, `.venv`, `node_modules`, `.git` 제외 — Secret에서 주입 |
| **Build & Push** | 105번에서 docker build/push | `images.plume7eat.com/tracking-cano:BUILD_NUMBER` + `latest`, CPU 전용 |
| **Deploy** | 105번에서 helm upgrade/install | kubeconfig → K8s(200), namespace default |

## K8s 리소스 설계

### Secret (`tracking-cano-secret`) — 200번에서 사전 생성

```bash
kubectl create secret generic tracking-cano-secret \
  --from-literal=LLM_KEY=sk-q0tfwmLMj4GJ074cw6Pp1g \
  --from-literal=API_KEY=38b8ba4686d6e49a3b2ea7fb555b35acb08817c23eef7358 \
  --from-literal=DATABASE_URL=postgresql://tracking_cano:maLe232633*@10.0.0.103:5432/tracking_cano \
  --from-literal=MINIO_SECRET_KEY=M9hx5Ey8VHdBWMcn1LojPpnS3wHYdhQ7s1DgCZGhnMg=
```

### ConfigMap (`tracking-cano-config`) — Helm chart template에서 values.yaml 기반 생성

| Key | Value |
|-----|-------|
| MINIO_ENDPOINT | static.plume7eat.com:443 |
| MINIO_ACCESS_KEY | tracking-cano |
| MINIO_BUCKET | tracking-cano |

### Deployment

- `secretKeyRef`: LLM_KEY, API_KEY, DATABASE_URL, MINIO_SECRET_KEY (from `tracking-cano-secret`)
- `configMapKeyRef`: MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_BUCKET (from `tracking-cano-config`)
- no nodeSelector (any node OK)

### Service: ClusterIP, port 80→8000

### Ingress: tracking.plume7eat.com, traefik ingressClass

## Helm Chart 구조

```
/mnt/charts/tracking-cano/          (100번 Jenkins 서버)
├── Chart.yaml                      # name: tracking-chart, version: 0.1.0
├── values.yaml                     # image repo/tag, port, ingress host — Secret 값 없음
└── templates/
    ├── deployment.yaml             # secretKeyRef + configMapKeyRef
    ├── service.yaml                # ClusterIP 80→8000
    ├── ingress.yaml                # tracking.plume7eat.com / traefik
    └── configmap.yaml              # .env 비민감 설정 (webdav 패턴)
```

## 실행 순서

```
1. K8s Secret 생성 (200번 kubectl) → verify: kubectl get secret tracking-cano-secret
2. Helm chart 파일들 생성 (100번 /mnt/charts/tracking-cano) → verify: 파일 구조 확인
3. Jenkins job 생성 (config.xml 또는 UI) → verify: webhook 테스트
4. 실제 빌드/배포 테스트 → verify: tracking.plume7eat.com 접근
```

## Progress

| Step | Status | 비고 |
|------|--------|------|
| K8s Secret 생성 (200번) | ✅ | `tracking-cano-secret` — LLM_KEY, API_KEY, DATABASE_URL, MINIO_SECRET_KEY |
| Helm chart 파일들 생성 (100번) | ✅ | /mnt/charts/tracking-cano/ — Chart.yaml, values.yaml, templates/* |
| Jenkins job 생성 | ✅ | config.xml 작성 완료, job 인식됨 |
| 빌드/배포 테스트 | ⬜ | GitHub webhook 설정 필요 |
