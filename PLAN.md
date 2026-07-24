# PLAN — Cooldown 리팩토링 + detect_cooldown 추가

## Objective
LLMCallDebouncer 제거, snapshot_cooldown을 YAML ↔ Orchestrator로 연결, detect_cooldown 기능 추가를 통해 LLM 호출 빈도를 명확하게 통제.

## Scope
- LLMCallDebouncer 클래스 및 사용 코드 완전 제거
- `snapshot_interval` → `snapshot_cooldown` 리네이밍 후 orchestrator debounce에 연결
- `detect_cooldown` 필드 추가: space detect loop 한 바퀴 완료 시 대기
- 사용하지 않는 YAML 필드 (`snapshot_count`, `cooldown_seconds`, `early_trigger`) 삭제
- AGENTS.md 규칙 수정

## Success Criteria
- [x] LLMCallDebouncer 클래스 및 모든 참조 제거됨
- [x] `snapshot_cooldown`이 YAML → LLMConfig → orchestrator debounce로 연결됨
- [x] `detect_cooldown`이 space detect loop 사이클 간 대기값으로 동작함
- [x] 사용하지 않는 YAML 필드 삭제됨
- [x] AGENTS.md 규칙이 현재 코드와 일치함

---

## Task Dependency Graph

```
T1 (settings.py) ────────┐
                         ├── T4 (orchestrator.py)
T2 (YAML 파일)           │       T5 (config hot-reload 검증)
                         │
T3 (logger.py, models.py)│
                         │
T6 (AGENTS.md) ──────────┘  (독립)
```

T1, T2, T3는 병렬 실행 가능. T4는 T1+T2 완료 후. T5는 T4 완료 후. T6은 독립.

---

### T1 — settings.py: LLMConfig 필드 추가/삭제

- **Input**: 계획된 필드 정의
- **Output**: `settings.py` 수정 완료
- **Dependencies**: none
- **변경:**
  - `LLMConfig`에 `snapshot_cooldown: float = 5.0` 추가
  - `LLMConfig`에 `detect_cooldown: float = 0.0` 추가
- **Verification**: `from_dict()`이 새 필드를 `__annotations__`로 자동 인식하는지 확인

---

### T2 — YAML 파일 정리

- **Input**: 변경 계획 (리네이밍, 삭제, 추가)
- **Output**: 두 YAML 파일 수정 완료
- **Dependencies**: none
- **변경 (`configuration.yaml`):**
  - line 14: `snapshot_interval: 30` → `snapshot_cooldown: 30`
  - line 13: `snapshot_count: 3` 삭제
  - line 15: `cooldown_seconds: 30` 삭제
  - line 16: `early_trigger: 5` 삭제
  - llm 섹션에 `detect_cooldown: 60` 추가
- **변경 (`configuration.yaml.example`):**
  - llm 섹션에 `# snapshot_cooldown: 30` 주석 예시 추가
  - llm 섹션에 `# detect_cooldown: 60` 주석 예시 추가
- **Verification**: YAML 구문 유효성, 필드명 정합성

---

### T3 — LLMCallDebouncer 제거 + api/models.py 정리

- **Input**: 계획된 삭제 대상
- **Output**: `nlp/logger.py`, `api/models.py` 수정 완료
- **Dependencies**: none
- **변경 (`nlp/logger.py`):**
  - line 47-58: `LLMCallDebouncer` 클래스 전체 삭제
  - line 72: `self.debouncer = LLMCallDebouncer(...)` 줄 삭제
  - line 382-384: `if not self.debouncer.should_call(...)` 블록 + return 삭제
- **변경 (`api/models.py`):**
  - line 92: `cooldown_seconds: Optional[float] = None` 필드 삭제
- **Verification**: grep으로 `debouncer`, `LLMCallDebouncer`, `cooldown_seconds` 잔여 확인

---

### T4 — orchestrator.py: cooldown 연결 + detect_cooldown 적용

- **Input**: T1(LLMConfig 새 필드), T2(YAML 값)
- **Output**: `core/orchestrator.py` 수정 완료
- **Dependencies**: T1, T2
- **변경:**
  - line 221: `if now - last < 5.0:` → `if now - last < self.space_logger.config.snapshot_cooldown:`
  - line 102: `self._stop_event.wait(0.1)` → `self._stop_event.wait(self._config.detect_cooldown if self._config.detect_cooldown > 0 else 0.1)`
- **Verification**: hot-reload 시 `update_config()`에서 `_vision_detector._config`가 갱신되므로 새 값 반영 확인 (orchestrator.py:268)

---

### T5 — config hot-reload 정합성 검증

- **Input**: T4 완료된 orchestrator.py
- **Output**: 코드 검토 완료
- **Dependencies**: T4
- **검증:**
  - `Orchestrator.update_config()` (line 261-273)에서 `space_logger.config`와 `_vision_detector._config`가 새 LLMConfig로 갱신되는지 확인
  - `snapshot_cooldown`, `detect_cooldown`이 hot-reload 시 반영되는지 확인
- **Verification**: 코드 정적 분석으로 충분 (런타임 불필요)

---

### T6 — AGENTS.md 규칙 수정

- **Input**: 계획된 규칙 텍스트
- **Output**: `AGENTS.md` 수정 완료
- **Dependencies**: none (독립 작업)
- **변경:**
  - line 13: `"LLM 직접 호출 금지 — 반드시 LLMCallDebouncer(cooldown=5s) 경유 (nlp/logger.py:72)"` → `"LLM 직접 호출 금지 — 모든 LLM 호출은 SpaceLogger를 경유해야 함. 빈도 통제는 orchestrator의 snapshot_cooldown과 detect_cooldown이 담당함."`
  - line 31 핵심 파일 테이블: `LLMCallDebouncer(5s),` 부분 삭제
- **Verification**: 텍스트 정합성 확인

---

## Key Decisions
- debouncer 제거 이유: snapshot에서만 동작하고 orchestrator debounce와 중복, vision_detect에는 미적용
- `detect_cooldown` 기본값 0 (비활성화): 기존 동작과 호환 유지
- `snapshot_cooldown` 기본값 5.0: 기존 하드코드 값과 동일
