# REST API Reference

FastAPI 기반 REST API (`api/server.py`). 기본 포트 `8000`, `main.py --live` 실행 시 자동 시작.

## 인증

`API_KEY` 환경변수가 설정된 경우 모든 엔드포인트에 Bearer token 필요. 미설정 시 anonymous 통과.

```http
Authorization: Bearer <API_KEY>
```

## Endpoints

### Health & Status

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/health` | 서비스 헬스 체크 |
| GET | `/api/status` | 시스템 상태 (모드, 카메라, 공간, uptime) |

### Cameras

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/cameras/` | 전체 카메라 목록 |
| GET | `/api/cameras/{camera_id}` | 단일 카메라 정보 |
| POST | `/api/cameras/` | 카메라 추가 (YAML 작성 → hot-reload) |
| PUT | `/api/cameras/{camera_id}` | 카메라 설정 변경 (YAML 작성 → 재시작) |
| DELETE | `/api/cameras/{camera_id}` | 카메라 삭제 |
| POST | `/api/cameras/{camera_id}/restart` | 카메라 재시작 |

**CameraResponse:**
```json
{
  "id": "livingroom",
  "source": "rtsp://...",
  "status": "active",
  "target_classes": ["cat"],
  "interaction_classes": null,
  "model_size": "n",
  "frame_skip": 15,
  "worker_state": "running | stopped | collector"
}
```

### Spaces

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/spaces/` | 전체 공간 목록 |
| GET | `/api/spaces/{space_id}` | 단일 공간 정보 |
| POST | `/api/spaces/` | 공간 추가 |
| PUT | `/api/spaces/{space_id}` | 공간 수정 |
| DELETE | `/api/spaces/{space_id}` | 공간 삭제 |
| POST | `/api/spaces/{space_id}/flush` | 강제 snapshot 트리거 |

### Logs

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/logs/` | 로그 목록 조회 (필터/페이징) |
| GET | `/api/logs/{log_id}` | 단일 로그 조회 (by PK id) |
| GET | `/api/logs/recent` | 최근 N개 로그 |
| GET | `/api/logs/stream` | SSE 실시간 로그 스트리밍 |

**Query parameters (`GET /api/logs/`):**

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `log_type` | string | — | `detect` or `space` |
| `subject_id` | string | — | 카메라명(detect) 또는 공간명(space) |
| `limit` | int | 100 | 최대 row 수 (max 500) |
| `offset` | int | 0 | 페이징 오프셋 |

**LogEntryResponse:**
```json
{
  "id": 1,
  "timestamp": "2026-06-12T00:32:49.552432",
  "log_type": "detect",
  "subject_id": "livingroom",
  "target_present": true,
  "description": "The cat is sitting on the couch.",
  "target_coordinate": "[0.1, 0.2, 0.5, 0.6]"
}
```

- `log_type=detect`: per-camera 감지 결과. `subject_id` = 카메라 ID
- `log_type=space`: 공간 종합 결과. `subject_id` = space_id, `description`에 reasoning 통합

**SSE Stream (`GET /api/logs/stream`):**

```text
event: log
data: {"id":1,"log_type":"detect","subject_id":"livingroom",...}

event: ping
data:
```

30초 간격 keepalive ping 전송.

### Config

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/config/` | 전체 설정 dump (`configuration.yaml`) |
| PUT | `/api/config/thresholds` | 임계값 업데이트 |
| PUT | `/api/config/yolo` | YOLO 설정 업데이트 |
| PUT | `/api/config/llm` | LLM 설정 업데이트 |

## DB Schema

```sql
CREATE TABLE log_entries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        TEXT NOT NULL,          -- UUID, snapshot 그룹 키
    timestamp       DATETIME NOT NULL,
    log_type        TEXT NOT NULL,          -- 'detect' | 'space'
    subject_id      TEXT,                   -- detect=카메라명, space=space_id
    target_present  BOOLEAN,
    description     TEXT,                   -- detect=per-camera desc, space=reasoning 통합
    target_coordinate TEXT,                 -- JSON array [x1,y1,x2,y2] or NULL
    raw_json        TEXT,                   -- space log 전체 dump
    created_at      DATETIME NOT NULL
);
```

**Query examples:**
```sql
-- 특정 snapshot의 모든 데이터
SELECT * FROM log_entries WHERE batch_id = 'abc123...' ORDER BY log_type;

-- 최근 space 이벤트
SELECT * FROM log_entries WHERE log_type = 'space' ORDER BY id DESC LIMIT 10;

-- 특정 카메라의 detect 로그
SELECT * FROM log_entries WHERE subject_id = 'livingroom' AND log_type = 'detect';

-- 카메라별 detect 대상 조회 (subject_id + batch_id join)
SELECT d.* FROM log_entries d
JOIN log_entries s ON d.batch_id = s.batch_id
WHERE s.log_type = 'space' AND s.subject_id = 'room_livingroom'
  AND d.log_type = 'detect';
```
