# PLAN.md — target_present: true 인 캡처만 저장

## SPEC

### Objective
`target_present: false`인 프레임의 이미지캡처를 MinIO/output/ 디렉토리에 저장하지 않는다. DB 로그(`_db_insert`)는 유지하여 분석용 기록은 보존한다.

### Scope
- **변경 대상**: image 파일 저장 경로만 (S3/MinIO, local `output/`)
- **유지 대상**: `_db_insert()` 호출 — target_present=false 기록도 DB에 남김
- **CV pipeline**: `process_frame()→CameraSnapshot update→request_space_snapshot→space_snapshot→_save_snapshot_image`
- **LLM Vision pipeline**: `_BatchCollector→vision_detect→target_present 체크(기존)→_update_all_snapshots→request_space_snapshot→space_snapshot→_save_snapshot_image`

### Key Components

| 컴포넌트 | 역할 | 파일 | 변경 내용 |
|----------|------|------|----------|
| `CameraSnapshot update` | registry에 snapshot 저장 | `core/orchestrator.py:341` | target_present=false일 때 skip |
| `request_space_snapshot` | snapshots dict 빌드 | `core/orchestrator.py:440-445` | false인 카메라 제외 |
| `space_snapshot()` | per-camera image 저장 | `nlp/logger.py:466-468` | snap.target_present 체크 후 저장 |
| `_snapshot_fallback()` | fallback image 저장 | `nlp/logger.py:549-551` | snap.target_present 체크 후 저장 |

### Success Criteria
1. target_present=false인 카메라의 이미지는 output/ 또는 MinIO에 저장되지 않음
2. target_present=true인 카메라의 이미지는 정상 저장됨
3. DB `_db_insert()` 호출은 target_present 불문하고 유지됨
4. LLM Vision pipeline의 기존 target_present=True 필터링(orchestrator.py:145)는 영향 없음

---

## 구현 단계

### Step 1: `core/orchestrator.py` — CV pipeline snapshot registry에 false 저장 방지

**위치**: `_CameraWorker._run()`, line 341-342附近

```python
# Before (line 341-342):
if self.orchestrator:
    self.orchestrator.update_snapshot(self.camera_id, snap)

# After:
if self.orchestrator and detect.target_present:
    self.orchestrator.update_snapshot(self.camera_id, snap)
```

**Why**: target_present=false인 snapshot이 registry에 갱신되면, 이후 `request_space_snapshot()` 호출 시 false 상태의 데이터가 전달되어 불필요하게 image 저장 경로까지 흐른다. false일 때는 registry를 업데이트하지 않아 이전 true 상태를 유지한다.

---

### Step 2: `core/orchestrator.py` — `request_space_snapshot()` 내 snapshots 빌드 시 false 제외

**위치**: line 439-445附近

```python
# Before (line 440-444):
with self._snapshot_lock:
    for cam_id in space_cameras:
        snap = self._snapshots.get(cam_id)
        if snap is None:
            continue
        snapshots[cam_id] = snap

# After:
with self._snapshot_lock:
    for cam_id in space_cameras:
        snap = self._snapshots.get(cam_id)
        if snap is None or not snap.target_present:
            continue
        snapshots[cam_id] = snap
```

**Why**: registry에 false가 들어오지 않도록 Step 1에서 방어했지만, 기존에 저장된 false snapshot이 남아있을 수 있으므로 defense-in-depth로 여기서도 필터링한다.

---

### Step 3: `nlp/logger.py` — `space_snapshot()` 내 image 저장 전 target_present 체크

**위치**: line 466-468附近

```python
# Before (line 466-468):
if snap.image_b64 or snap.images:
    img_to_save = snap.images[-1] if snap.images else snap.image_b64
    self._save_snapshot_image(img_to_save, space_name, cam_id, timestamp, coord)

# After:
if (snap.image_b64 or snap.images) and snap.target_present:
    img_to_save = snap.images[-1] if snap.images else snap.image_b64
    self._save_snapshot_image(img_to_save, space_name, cam_id, timestamp, coord)
```

**Why**: `_save_snapshot_image()` 호출 직전에 target_present를 확인하여 false일 때 image 저장 자체를 건너뛴다. DB insert는 아래에서 유지됨.

---

### Step 4: `nlp/logger.py` — `_snapshot_fallback()` 내 image 저장 전 target_present 체크

**위치**: line 549-551附近

```python
# Before (line 549-551):
if snap.image_b64 or snap.images:
    img_to_save = snap.images[-1] if snap.images else snap.image_b64
    self._save_snapshot_image(img_to_save, space_name or space_id, cam_id, timestamp, coord)

# After:
if (snap.image_b64 or snap.images) and snap.target_present:
    img_to_save = snap.images[-1] if snap.images else snap.image_b64
    self._save_snapshot_image(img_to_save, space_name or space_id, cam_id, timestamp, coord)
```

**Why**: LLM API 호출 실패 시 fallback 경로에서도 동일한 필터링이 필요함.

---

## 검증 방법

1. Docker 빌드 후 테스트 영상/스트림으로 실행
2. target_present=false인 프레임이 있을 때 output/ 디렉토리(또는 MinIO)에 해당 시점의 이미지가 저장되지 않는지 확인
3. target_present=true인 프레임은 정상적으로 저장되는지 확인
4. DB `log_entries` 테이블에는 target_present=false 기록도 유지되는지 확인

---

## Progress

| Step | Status | 비고 |
|------|--------|------|
| Step 1: orchestrator.py — registry update 필터링 | ✅ | detect.target_present 체크 추가 (line 341) |
| Step 2: orchestrator.py — request_space_snapshot 빌드 필터링 | ✅ | snap.target_present 체크 추가 (line 443) |
| Step 3: logger.py — space_snapshot() image 저장 필터링 | ✅ | snap.target_present 체크 추가 (line 466) |
| Step 4: logger.py — _snapshot_fallback() image 저장 필터링 | ✅ | snap.target_present 체크 추가 (line 549) |
