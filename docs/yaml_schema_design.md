# YAML 구성 스키마 상세 설계 — 설계 문서

## 1. 목적 및 범위

`config/spaces.yaml`의 완전한 스키마 정의를 문서화하여, `config_manager.py` 구현 시 일관된 유효성 검사 규칙과 핫리로드 차분 계산의 근거를 제공한다.

---

## 2. 최상위 스키마 구조

```yaml
spaces:       # 배열 — 공간 정의 (선택적, 디폴트: 단일 기본 공간)
cameras:      # 배열 — 카메라 정의 (필수)
thresholds:   # 객체  — 임계값 설정 (선택적, 디폴트 존재)
llm:          # 객체  — LLM 설정 (선택적, 디폴트 존재)
```

- 허용되지 않는 최상위 키는 무시하며 경고 로그 출력
- `cameras` 배열은 최소 1개 이상이어야 함

---

## 3. spaces 객체 스키마

### 3.1 필드 정의

| 필드 | 타입 | 필수 여부 | 디폴트 | 설명 |
|------|------|-----------|--------|------|
| `id` | string | ✅ | — | 공간 고유 식별자 |
| `name` | string | ✅ | — | 표시용 이름 (한글/영어 모두 허용) |
| `cameras` | array[string] | ✅ | — | 소속 카메라 ID 목록 |

### 3.2 유효성 규칙

- **`id` 형식:** `[a-z][a-z0-9_]*` 패턴 권장, 중복 금지
- **`name`:** 한글/영어 모두 허용 (표시용이므로 제약 없음)
- **`cameras`:** 배열 내 ID는 최상위 `cameras`에 반드시 존재해야 함 (참조 무결성)
- **1대1 매핑:** 동일한 카메라 ID가 여러 space의 `cameras`에 중복 포함될 수 없음

### 3.3 예시

```yaml
spaces:
  - id: room_living
    name: 거실
    cameras: [cam_01, cam_02]
  - id: room_bedroom
    name: 침실
    cameras: [cam_03]
```

---

## 4. cameras 객체 스키마

### 4.1 필드 정의

| 필드 | 타입 | 필수 여부 | 디폴트 | 설명 |
|------|------|-----------|--------|------|
| `id` | string | ✅ | — | 카메라 고유 식별자 |
| `source` | string | ✅ | — | 영상 소스 경로/주소 |
| `status` | enum | ❌ | `"active"` | 활성화 여부 |
| `target_classes` | array[string] | ❌ | `[cat, person]` | 추적 대상 COCO 클래스 |

### 4.2 source의 타입 구분 방식

`utils/video.py`의 캡처 헬퍼가 source 문자열을 해석하는 논리:

| 유형 | 패턴 | 예시 | OpenCV 호출 방식 |
|------|------|------|------------------|
| 웹캠 (디바이스 경로) | `/dev/video*` | `/dev/video0` | `cv2.VideoCapture("/dev/video0")` |
| 웹캠 (숫자 인덱스) | 정수 문자열 | `"0"`, `"1"` | `cv2.VideoCapture(0)` |
| 파일 | 영상 확장자 포함 | `./sample.mp4` | `cv2.VideoCapture("./sample.mp4")` |
| RTSP | `rtsp://` 프리픽스 | `rtsp://192.168.1.100/stream` | `cv2.VideoCapture(url)` |
| HTTP 스트림 | `http://` 또는 `https://` | `https://stream.url/...` | `cv2.VideoCapture(url)` |

### 4.3 status 허용 값

| 값 | 의미 |
|----|------|
| `"active"` | 파이프라인 실행, 리소스 할당 |
| `"inactive"` | 파이프라인 실행 안함, 리소스 할당도 안함 |

- 핫리로드 시 `active → inactive`: pipeline 종료 + OpenCV 캡처 리소스 해제
- 핫리로드 시 `inactive → active`: 새로운 pipeline 시작

### 4.4 target_classes 정의

- COCO 80 클래스 명칭 사용 (`cat`, `dog`, `person`, `car` 등)
- 대소문자 구분: 소문자로 정규화 후 비교
- 빈 배열 허용: 모든 감지된 객체를 추적 대상으로 함
- Phase 3 (Custom Fine-Tuning)에서 custom 클래스 추가 시, 검증 로직은 "문자열 배열"로 유연하게 유지

### 4.5 예시

```yaml
cameras:
  - id: cam_01
    source: /dev/video0
    status: active
    target_classes: [cat, person]
  - id: cam_02
    source: rtsp://192.168.1.100/stream
    status: active
    target_classes: [cat]
  - id: cam_03
    source: /home/user/videos/bedroom.mp4
    status: inactive
    target_classes: [cat, dog]
```

---

## 5. thresholds 스키마

### 5.1 필드 정의

| 필드 | 타입 | 필수 여부 | 디폴트 | 설명 |
|------|------|-----------|--------|------|
| `overlap` | float | ❌ | `0.3` | bbox 겹침률 임계값 (IoU) |
| `distance` | int | ❌ | `50` | 중심점 간 거리 임계값 (px) |
| `speed_slow` | int | ❌ | `20` | 정지/천천히 이동 기준 속도 (px/frame) |
| `speed_fast` | int | ❌ | `40` | 천천히/빠르게 이동 기준 속도 (px/frame) |

### 5.2 유효성 검사 규칙

- **`overlap`:** `[0.0, 1.0]` 범위 — IoU 값이므로
- **`distance`:** 양의 정수 (`> 0`)
- **`speed_slow`:** 양의 정수 (`> 0`)
- **`speed_fast`:** 양의 정수 (`> 0`) && `speed_fast > speed_slow` 관계 유지

### 5.3 속도 상태 분류 로직

```
속도 < speed_slow          → STOPPED 또는 SLOW_MOVE (히스테리시스로 판단)
speed_slow ≤ 속도 < speed_fast → SLOW_MOVE
속도 ≥ speed_fast                 → FAST_MOVE 또는 DASHING (가속도로 판단)
```

### 5.4 확장성 고려

- `acceleration_threshold`: 가속도 기반 "돌진" 판단 임계값 — Phase 1에는 코드 내장 상수, 추후 YAML로 분리 가능
- `interaction_classes`: 상호작용 대상 클래스 필터 (couch, chair, dining table, tv 등) — 현재는 YOLO26이 감지하는 모든 클래스가 후보지만, 명시적 필터링을 위한 필드 예약

---

## 6. llm 스키마

### 6.1 필드 정의

| 필드 | 타입 | 필수 여부 | 디폴트 | 설명 |
|------|------|-----------|--------|------|
| `provider` | enum | ❌ | `"openai"` | LLM 제공자 |
| `model` | string | ✅ | — | 사용할 모델 ID |
| `api_endpoint` | string | ❌ | `https://api.openai.com/v1` | API 엔드포인트 URL |
| `temperature` | float | ❌ | `0.7` | 생성 온도 |

### 6.2 provider 허용 값

- `"openai"` — OpenAI 호환 API
- 추가 제공자 확장 가능성: `"anthropic"`, `"gemini"` 등 (향후)

### 6.3 api_endpoint 유효성 검사

- `https://` 또는 `http://` 프리픽스 필수
- `/v1` 경로 포함 권장 (OpenAI 호환 형식)

### 6.4 API key 관리

- **YAML에는 평문 API 키를 포함하지 않음**
- `.env`의 `LLM_API_KEY`에서 읽음
- 이 분리 원칙을 문서화: 민감 정보는 `.env`, 구성 정보는 `spaces.yaml`

### 6.5 추가 고려 사항

- `max_tokens`: LLM 응답 길이 제한 (디폴트: 100, 한 문장 강제)
- `language`: 프롬프트 출력 언어 (`ko` 디폴트, `en` 허용)

---

## 7. 유효성 검사 규칙 종합

### 7.1 전역 규칙

1. 최상위 키 외의 키는 무시 (경고 로그 출력)
2. `spaces`와 `cameras` 배열 내 `id` 중복 금지 — 에러 발생 시 프로그램 종료
3. space의 `cameras` 참조가 실제 존재하는 camera id인지 검증 — 참조 무결성 에러
4. `cameras` 배열은 최소 1개 이상이어야 함

### 7.2 타입 강제

- YAML에서 타입이 맞지 않는 값은 명시적 에러
  - 예: `overlap: "0.3"` → float 변환 시도 후 실패 시 에러
- 배열 내 중복 요소:
  - `target_classes`: 중복 제거 권장, 에러 아님
  - space의 `cameras`: 중복 허용 안 함 (에러)

### 7.3 디폴트 적용 순서

1. YAML에 값 존재 → 사용
2. YAML에 값 없음 → 스키마 디폴트 적용
3. 필수가 누락됨 → 에러 발생 + 누락된 필드명 출력

---

## 8. 핫리로드 차분 계산 메타데이터

### 8.1 차분 계산 알고리즘

| 변경 유형 | 감지 방법 | 액션 |
|----------|-----------|------|
| 카메라 추가 | 새 `id`가 `cameras` 배열에 나타남 | 해당 space의 pipeline 시작 |
| 카메라 삭제 | 기존 `id`가 사라짐 | pipeline 종료 + OpenCV 캡처 리소스 해제 |
| 카메라 재할당 | camera의 `id`는 동일하나 소속 `space.id` 변경 | pipeline을 새 space로 이동 |
| 상태 전환 | `status: active ↔ inactive` | pipeline 시작/종료 |
| 임계값 변경 | `thresholds` 필드 값 변경 | analyzer/interaction_detector에 실시간 전달 (pipeline 재시작 불필요) |
| LLM 설정 변경 | `llm` 필드 값 변경 | nlp_logger 재초기화 |

### 8.2 안전한 전환 규칙

1. 변경 적용 전: 현재 프레임 처리 완료 대기
2. 카메라 삭제 시: ByteTrack 상태 정리 + 추적 객체 ID 풀 반환
3. 공간 삭제 시: 해당 space의 LLM 컨텍스트 정리 (로그 기록 후 폐기)
4. **원자성:** 모든 변경이 한 번에 적용됨 (부분적 적용 금지)

### 8.3 변경 이력 로깅

핫리로드 발생 시 로그 파일에 변경 사항 요약 기록:

```
[HOTRELOAD] {timestamp}: added=[cam_ids], removed=[cam_ids], reassigned=[{old_space→new_space}], threshold_changed=[fields]
```

---

## 9. 예시 — 완전한 spaces.yaml

```yaml
# === 공간 정의 ===
spaces:
  - id: room_living
    name: 거실
    cameras: [cam_01, cam_02]
  - id: room_bedroom
    name: 침실
    cameras: [cam_03]

# === 카메라 정의 ===
cameras:
  - id: cam_01
    source: /dev/video0                  # 웹캠 (디바이스 경로)
    status: active
    target_classes: [cat, person]
  - id: cam_02
    source: rtsp://192.168.1.100/stream  # RTSP 스트림
    status: active
    target_classes: [cat]
  - id: cam_03
    source: /home/user/videos/bedroom.mp4  # 오프라인 영상 파일
    status: inactive                      # 비활성화 상태
    target_classes: [cat, dog]

# === 임계값 ===
thresholds:
  overlap: 0.3          # bbox 겹침률 (IoU)
  distance: 50          # 중심점 간 거리 (px)
  speed_slow: 20        # 정지/천천히 이동 기준 (px/frame)
  speed_fast: 40        # 천천히/빠르게 이동 기준 (px/frame)

# === LLM 설정 ===
llm:
  provider: openai
  model: gpt-4o-mini
  api_endpoint: https://api.openai.com/v1
  temperature: 0.7
```

---

## 10. 구현 시 고려사항

### 10.1 YAML 스키마 검증 라이브러리

`pydantic` 또는 `cerberus` 등 스키마 검증 도구를 `config_manager.py`에서 사용할지 여부:
- **권장:** `pydantic` — 타입 안전성 + 자동 유효성 검사 + 직관적인 에러 메시지
- 위 스키마 정의를 pydantic 모델로 변환하는 작업이 추가로 필요

### 10.2 확장성

Phase 3 (Custom Fine-Tuning)에서 custom 클래스가 추가될 경우:
- `target_classes`에 COCO 외 클래스를 허용할 수 있도록 검증 로직의 확장 가능성 확보
- 현재는 COCO 클래스 명칭으로 제한하되, 검증 규칙은 "문자열 배열"로 유연하게 정의

### 10.3 다국어

- `name` 필드에 한글을 허용하는 것은 한국어 사용자가 주요 타겟임을 반영
- `id` 필드는 ASCII만 허용하여 코드에서의 안정성 유지
