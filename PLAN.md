# PLAN.md — SpaceLogger 연결 작업

## SPEC

### 목적
SpaceLogger가 Orchestrator → Pipeline → NLPLogger 파이프라인에 실제로 연결되어, 동일 공간의 다중 카메라 로그를 LLM으로 종합 로깅하도록 보완.

### 범위
- Orchestrator에 SpaceLogger 주입
- Pipeline에 SpaceLogger + space_id 주입, collect() 호출
- run_multi에 SpaceLogger flush 루프 (이벤트 기반 + 주기적 안전망)
- Hot-reload 시 공간 추가/삭제 처리

### 성공 기준
1. `python main.py --live` 실행 시, 동일 공간의 여러 카메라에서 상태 변화가 발생하면 SpaceLogger가 종합 문장을 생성하여 로그 파일에 저장
2. `spaces.yaml` 변경 시 추가된 공간의 SpaceLogger 버퍼가 초기화, 삭제된 공간의 버퍼가 정리됨
3. 기존 단일 카메라 모드(`--live <url>`, `--video`)는 영향 없음

---

## 작업 목록

### 1. Orchestrator에 SpaceLogger 주입 + camera→space 매핑
- [x] `Orchestrator.__init__`에 `space_logger: SpaceLogger` 파라미터 추가
- [x] `Orchestrator.add_camera`에 `space_id` 파라미터 추가
- [x] camera→space 매핑: `spaces.yaml`의 `spaces[].cameras[]`로 카메라 ID → 공간 ID 역매핑
- [x] `_CameraWorker`에 `space_id` 전달

### 2. Pipeline에 SpaceLogger + space_id 주입, collect() 호출
- [x] `Pipeline.__init__`에 `space_logger: Optional[SpaceLogger]`, `space_id: Optional[str]` 파라미터 추가
- [x] `Pipeline.process_frame`에서 NLPLogger.log()가 텍스트를 리턴할 때 `space_logger.collect(space_id, camera_id, text)` 호출

### 3. run_multi에 SpaceLogger flush 루프
- [x] `run_multi`에 flush 스케줄러 추가 (10초 주기적)
- [x] `Orchestrator`에 `flush_spaces()` 메서드 추가 → 각 공간별 SpaceLogger.flush() 호출

### 4. _on_config_change에 공간 변경 처리
- [x] `diff.added_spaces` → 로그 출력
- [x] `diff.removed_spaces` → SpaceLogger.flush() 후 버퍼 정리
- [ ] 카메라 재할당 감지 (기존 공간 → 새 공간)

---

## 변경 파일

| 파일 | 변경 내용 |
|------|----------|
| `core/orchestrator.py` | SpaceLogger 주입, camera→space 매핑, flush_spaces() |
| `core/pipeline.py` | SpaceLogger + space_id 주입, collect() 호출 |
| `main.py` | run_multi flush 루프, _on_config_change 공간 처리 |
