# 🎯 tracking-cano

YOLO26을 이용해 웹캠 영상에서 지정한 객체를 추적하고, 주변 물체와의 상호작용까지 종합 판단하여 자연어 한 문장으로 이동 상태를 로깅하는 시스템.

추적 대상은 고양이뿐만 아니라 사람, 동물, 차량 등 COCO 클래스 중 무엇이든 설정 가능합니다.

---

## Architecture

```
Camera ─→ [YOLO26] → [ByteTrack] → [Spatial Analyzer] → [Interaction Detector]
  ...     ──────────────────────────────────────────────────────────────────────
                                                            │
                                                            ▼
                                      [Orchestrator: 공간별 그룹핑 + LLM 로깅]
```

---

## Features

- **YOLO26 기반 객체 추적** — 실시간 감지 및 ByteTrack 기반 ID 유지, 추적 대상 클래스 구성 가능
- **상호작용 감지** — bbox 겹침 + 거리 결합으로 주변 물체 상호작용 판단
- **다중 카메라·다중 공간** — YAML 구성 파일로 카메라와 공간의 관계 정의
- **동적 구성 핫리로드** — 실행 중 구성 변경 시 자동 반영 (추가/삭제/재할당)
- **자연어 로깅** — OpenAI API 호환 LLM을 통해 상태 변화 시점만 자연어로 표현

---

## Tech Stack

| 용도 | 도구 |
|------|------|
| Object Detection | YOLO26 (ultralytics) |
| Video Capture | OpenCV |
| Object Tracking | ByteTrack |
| NLP Logging | OpenAI API 호환 LLM |
| Config | PyYAML + python-dotenv |
| Hot Reload | watchdog |

---

## Quick Start

```bash
# 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env   # LLM API key 입력

# 실시간 웹캠 모드
python main.py --live

# 오프라인 영상 분석
python main.py --video ./sample.mp4
```

---

## Configuration

### `config/spaces.yaml`

```yaml
spaces:
  - id: room_living
    name: 거실
    cameras: [cam_01, cam_02]
  - id: room_bedroom
    name: 침실
    cameras: [cam_03]

cameras:
  - id: cam_01
    source: /dev/video0
    status: active
    target_classes: [cat]       # 고양이 추적
  - id: cam_02
    source: /dev/video1
    status: active
    target_classes: [person]    # 사람 추적
  - id: cam_03
    source: /dev/video2
    status: active
    target_classes: [cat]
```

### `.env`

```env
API_BASE_URL=https://api.openai.com/v1  # 또는 다른 OpenAI 호환 엔드포인트
API_KEY=your_api_key_here
MODEL_NAME=gemma4-e4b                    # 사용하려는 모델명
```

---

## Usage

| 옵션 | 설명 |
|------|------|
| `--live` | 실시간 웹캠 모드 (구성 파일 기반) |
| `--video <path>` | 오프라인 영상 분석 |
| `--config <path>` | 구성 파일 경로 (기본: `config/spaces.yaml`) |

---

## Project Structure

```
tracking-cano/
├── main.py                  # 진입점
├── config/
│   └── spaces.yaml          # 동적 구성 파일
├── core/
│   ├── config_manager.py    # YAML 읽기 + 핫리로드
│   ├── pipeline.py          # 단일 카메라 파이프라인
│   └── orchestrator.py      # 다중 카메라 오케스트레이션
├── modules/
│   ├── detector.py          # YOLO26 감지
│   ├── tracker.py           # ByteTrack 추적
│   ├── analyzer.py          # 이동 상태 분류
│   └── interaction_detector.py  # 상호작용 판단
├── nlp/
│   └── logger.py            # LLM 자연어 로깅
├── utils/
│   └── video.py             # 영상 캡처 헬퍼
└── logs/                    # 로그 출력 디렉토리
```

---

## License

[AGPL-3.0](LICENSE)
