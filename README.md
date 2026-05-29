# 🎯 tracking-cano

YOLO26을 이용해 IP 카메라(RTSP/MP4) 영상에서 지정한 객체를 추적하고, 주변 물체와의 상호작용까지 종합 판단하여 자연어 한 문장으로 이동 상태를 로깅하는 시스템.

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

## Quick Start (Docker)

```bash
# 설정 파일 준비
cp config/spaces.yaml.example config/spaces.yaml   # RTSP URL 등 실제 값으로 수정
cp .env.example .env                               # LLM API key 입력

# 빌드 및 실행 (CPU)
docker compose up --build

# GPU 모드
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

## Quick Start (로컬)

```bash
# 의존성 설치
pip install -r requirements.txt

# 설정 파일 준비
cp config/spaces.yaml.example config/spaces.yaml   # RTSP URL 등 실제 값으로 수정
cp .env.example .env                               # LLM API key 입력

# 실시간 모션 모드
python main.py --live

# 오프라인 영상 분석
python main.py --video ./sample.mp4
```

---

## Configuration (로컬 실행 시)

### `config/spaces.yaml` — RTSP URL 등 설정 (`.gitignore` 제외)

```yaml
cameras:
  - id: cam_01
    source: rtsp://your-camera-ip:554/stream   # ← 실제 RTSP URL로 변경
    status: active                              # active | inactive
    target_classes: [cat, person]               # COCO 클래스명

thresholds:
  overlap: 0.3        # bbox 겹침률 (IoU)
  distance: 50        # 중심점 간 거리 (px)
  speed_slow: 20      # 정지/천천히 이동 기준 (px/frame)
  speed_fast: 40      # 천천히/빠르게 이동 기준 (px/frame)

llm:
  provider: openai
  model: gpt-4o-mini
  api_endpoint: https://api.openai.com/v1
  temperature: 0.7
```

`config/spaces.yaml.example`을 복사해서 사용하세요. `.gitignore`에 `config/spaces.yaml`이 제외되어 있습니다.

### `.env` — LLM API key (`.gitignore` 제외)

```env
API_BASE_URL=https://api.openai.com/v1  # 또는 다른 OpenAI 호환 엔드포인트
API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini                    # 사용하려는 모델명
```

---

## Usage

| 옵션 | 설명 |
|------|------|
| `--live` | 실시간 모션 모드 (구성 파일 기반) |
| `--video <path>` | 오프라인 영상 분석 |
| `--config <path>` | 구성 파일 경로 (기본: `config/spaces.yaml`) |

---

## Debugging

### `--verbose` / `-v` 플래그

`main.py`에 `--verbose`를 추가하면 모든 모듈의 **DEBUG 레벨 로그**가 `docker logs`(또는 stdout/stderr)로 출력됩니다.

```bash
# Docker (multi-camera 모드)
docker run --rm --network host \
  -v ./config:/app/config:ro -v ./logs:/app/logs -v ./.env:/app/.env:ro \
  tracking-cano python main.py --live --verbose

# 로컬 (single-live 모드)
python main.py --live rtsp://... --verbose
```

### DEBUG 로그 종류

| 로그 | Level | 모듈 | 설명 |
|------|-------|------|------|
| `target N state OLD->NEW hold=M/MIN` | DEBUG | pipeline | 상태 전환 시도와 hold 진행률 |
| `target N interactions changed` | DEBUG | pipeline | 상호작용 변화 감지 |
| `frame=N skipped (interval=16)` | DEBUG | orchestrator | skip으로 건너뛴 프레임 |
| `target N: speed=X acc=Y angle=Z thresh=W -> STATE` | DEBUG | analyzer | 이동 분류 결정과 threshold 값 |
| `target N vs person M: iou=X dist=Y -> nearby` | DEBUG | interaction_detector | IoU/거리 기반 관계 분류 |
| `No detections` / `No tracking IDs` | DEBUG | tracker | 탐지 실패 vs 추적 실패 구분 |
| `Detect: N boxes, M tracked, K interaction` | DEBUG | tracker | 추론 결과 요약 |
| `[logger] enqueue: key (qsize=N)` | DEBUG | nlp.logger | LLM 큐 enqueue와 backlog |
| `[logger] LLM call: camera=X prompt=N chars` | DEBUG | nlp.logger | LLM API 호출 시점과 비용 |
| `[space:X] try_flush skipped: N < M` | DEBUG | nlp.logger | SpaceLogger flush 조건 미달 |
| `disappeared: target N (cat)` | INFO | pipeline | target이 화면에서 사라짐 |
| `Model loaded: yolo26n.pt` | INFO | tracker | YOLO 모델 적재 완료 |
| `[CAM FPS] livingroom frame=312 fps=14.9` | INFO | orchestrator | 5초 간격 카메라별 FPS |

### `LOG_LEVEL` 환경변수

`LOG_LEVEL=DEBUG` 환경변수로도 동일하게 DEBUG 로깅을 활성화할 수 있습니다 (`--verbose`와 동일).

---

## Benchmark

### RTSP 스트림 벤치마크 (`bench_rtsp.py`)

RTSP 스트림에서 다중 카메라를 동시 처리하는 벤치마크. LLM 호출 포함 전체 파이프라인 측정.

```bash
# Docker
docker run --rm --network host -v ./config:/app/config:ro -v ./logs:/app/logs tracking-cano python bench_rtsp.py --runtime 10 --frame-skip 15

# 로컬
python bench_rtsp.py --runtime 10 --frame-skip 15
```

| 옵션 | 설명 |
|------|------|
| `--runtime <sec>` | 실행 시간 (기본: 10초) |
| `--frame-skip <N>` | 프레임 스킵 (N프레임마다 1회 추론, 기본: 15) |
| `--config <path>` | 설정 파일 경로 (기본: `config/spaces.yaml`) |

### 로컬 영상 벤치마크 (`benchmark.py`, `bench_10s.py`)

`benchmark.py`는 `data/` 디렉토리의 영상으로 처리 가능한 프레임 수를 측정합니다.
`bench_10s.py`는 고정 10초 실행 벤치마크로, `Dockerfile.bench`의 기본 진입점입니다.

```bash
python benchmark.py
python bench_10s.py --runtime 10 --frame-skip 15
```

---

## Project Structure

```
tracking-cano/
├── main.py                          # 진입점
├── Dockerfile                       # CPU 베이스 컨테이너
├── Dockerfile.gpu                   # GPU 옵션 컨테이너
├── docker-compose.yml               # 기본 서비스 정의
├── docker-compose.gpu.yml           # GPU 오버레이
├── config/
│   ├── __init__.py
│   ├── spaces.yaml                  # RTSP URL 등 민감 정보 (gitignore)
│   └── spaces.yaml.example          # 커밋용 템플릿
├── core/
│   ├── __init__.py
│   ├── config_manager.py            # YAML 읽기 + 핫리로드
│   ├── pipeline.py                  # 단일 카메라 파이프라인
│   └── orchestrator.py              # 다중 카메라 오케스트레이션
├── modules/
│   ├── __init__.py
│   ├── detector.py                  # YOLO26 감지
│   ├── tracker.py                   # ByteTrack 추적
│   ├── analyzer.py                  # 이동 상태 분류
│   └── interaction_detector.py      # 상호작용 판단
├── nlp/
│   ├── __init__.py
│   └── logger.py                    # LLM 자연어 로깅
├── utils/
│   ├── __init__.py
│   └── video.py                     # 영상 캡처 헬퍼
├── logs/                            # 로그 출력 디렉토리
├── .env                             # 민감 정보 (gitignore)
├── .env.example                     # 템플릿
├── .gitignore                       # git 제외 패턴
├── requirements.txt                 # 의존성
└── MASTER_PLAN.md                   # 프로젝트 마스터 플랜
```

---

## License

[AGPL-3.0](LICENSE)
