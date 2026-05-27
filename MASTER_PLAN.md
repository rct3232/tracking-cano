# Tracking-Cano — Master Plan

## 프로젝트 개요

YOLO26을 이용해 webcam stream 영상으로부터 지정한 객체를 추적하고, 주변 물체와의 상호작용까지 종합 판단하여 자연어 한 문장으로 이동 상태를 로깅하는 시스템.

추적 대상은 고양이뿐만 아니라 사람, 동물, 차량 등 COCO 클래스 중 무엇이든 설정 가능합니다.

---

## 핵심 아키텍처

```
Camera A ─→ [YOLO26] → [ByteTrack] → [Spatial Analyzer] → [Interaction Detector]
Camera B ─→ [YOLO26] → [ByteTrack] → [Spatial Analyzer] → [Interaction Detector]
  ...      ──────────────────────────────────────────────────────────────────────
                                                                                    │
                                                                                    ▼
                                                          [Orchestrator: 공간별 그룹핑]
                                                                                    │
                                                                                    ▼
                                               [OpenAI API 호환 LLM: 자연어 로깅]
```

---

## 스택

| 용도 | 도구 |
|------|------|
| Object Detection | `ultralytics` — YOLO26 (yolo26s.pt 권장) |
| Video Capture | `opencv-python-headless` |
| Object Tracking | `bytetrack` |
| Movement + Interaction Analysis | `numpy` |
| NLP Logging | OpenAI API 호환 LLM |
| Config Management | `python-dotenv`, `pyyaml` |
| Hot Reload | `watchdog` |

---

## Phase 1 — 단일 카메라 기본 파이프라인

### 1.1 프로젝트 스캐폴딩
- [ ] 디렉토리 구조 생성
- [ ] `requirements.txt` 작성
- [ ] `.env` 템플릿 작성 (LLM API key)
- [ ] `.gitignore` 작성

### 1.2 설정 시스템
- [x] `config.py` — YOLO 모델, threshold, LLM 설정 로드
- [x] `python-dotenv`로 API key 관리
- [x] 기본값 제공 (thresholds: speed_slow=20, speed_fast=40, overlap=0.3, distance=50px)

### 1.3 YOLO26 감지 모듈
- [x] `detector.py` — YOLO26 모델 로드 및 추론
- [x] 단일 프레임에 대한 bbox + class 반환
- [x] target_classes 기반 필터링 (COCO 클래스)
- [x] GPU/CPU 자동 감지

### 1.4 ByteTrack 추적 모듈
- [x] `tracker.py` — ByteTrack 초기화 및 업데이트 (ultralytics 내장)
- [x] 객체 ID 할당 및 유지
- [x] 이전 프레임 → 현재 프레임 매칭

### 1.5 이동 상태 분석기
- [x] `analyzer.py` — 속도/가속도/방향 계산
- [x] 상태 분류: 정지, 천천히 이동, 빠르게 이동, 돌진, 회전
- [x] 임계값 기반 판단 (config에서 로드)

### 1.6 자연어 로깅
- [x] `nlp/logger.py` — OpenAI API 호환 LLM 연결
- [x] 상태 변화 감지 시 LLM 호출
- [x] 프롬프트 템플릿 설계 (객체 행동 관찰 전문가)
- [x] 로그 출력 + 파일 저장 (`logs/` 디렉토리)

### 1.7 단일 카메라 파이프라인 통합
- [x] `core/pipeline.py` — detector → tracker → analyzer → logger 연결
- [x] 실시간 웹캠 모드 (`--live`)
- [x] 오프라인 영상 모드 (`--video <path>`)
- [x] 프레임별 처리 루프 + 상태 변화 감지 로직

### 1.8 진입점 및 CLI
- [x] `main.py` — argparse로 실행 모드 선택
- [x] `python main.py --live --camera 0`
- [x] `python main.py --video ./sample.mp4`
- [x] Graceful shutdown (Ctrl+C)

---

## Phase 2 — 상호작용 감지 + 다중 카메라 + 동적 구성

### 2.1 공간·카메라 구성 시스템
- [x] `config/spaces.yaml` 스키마 설계
  - spaces: id, name, cameras[]
  - cameras: id, source, status, target_classes[]
  - thresholds: overlap, distance, speed_slow, speed_fast 등
  - llm: provider, model
- [x] `config_manager.py` — YAML 로드 + 유효성 검사
- [x] 구성 파일 변경 감지 (watchdog)

### 2.2 go2rtc 스트림 자동 해결
- [x] ~~`.env`에 `GO2RTC_URL=http://host:port` 설정~~ — **go2rtc 제거됨, RTSP 직접 사용**
- [x] ~~`source` 필드에 `go2rtc:스트림명` 형식 지원~~ — **go2rtc 제거됨**
- [x] ~~`utils/video.py` — `resolve_source()` 헬퍼~~ — **go2rtc 제거됨**
- [x] ~~`config_manager.py` — YAML 파싱 시 `resolve_source()` 자동 적용~~ — **go2rtc 제거됨**

### 2.3 상호작용 감지 모듈
- [x] `interaction_detector.py` — bbox 기반 상호작용 판단
- [x] 겹침 계산: 추적 대상 bbox ∩ 객체 bbox > overlap_threshold → "접촉"
- [x] 거리 계산: 중심점 간 거리 < distance_threshold → "근처"
- [x] 둘 다 만족 → "상호작용 중"
- [x] 상호작용 대상 클래스 필터 (couch, chair, dining table, tv 등)

### 2.4 다중 카메라 오케스트레이터
- [x] `orchestrator.py` — N개 카메라 병렬 실행
- [x] 각 카메라별 독립 파이프라인 (pipeline.py 재사용)
- [x] 공간별 카메라 그룹핑
- [x] 실시간 + 오프라인 혼합 지원

### 2.5 공간별 LLM 종합 로깅
- [x] 동일 공간의 여러 카메라 로그 수집 (SpaceLogger 클래스 구현)
- [x] 통합 프롬프트: "[방: 거실] cam_01에서 ~, cam_02에서 ~"
- [x] LLM이 종합 자연어 표현 생성
- [x] 상태 변화 시점만 호출 (비용 최적화) — **N개 카메라 수집 시 즉시 flush**

### 2.6 핫리로드 구현
- [x] `config_manager.py` — 파일 감시 루프
- [x] 구성 변경 시 차분 계산:
  - [x] 추가된 카메라 → pipeline 시작
  - [x] 제거된 카메라 → pipeline 종료 + 리소스 정리
  - [x] 재할당된 카메라 → pipeline 공간 이동
  - [x] 추가된 공간 → LLM 컨텍스트 생성
  - [x] 삭제된 공간 → LLM 컨텍스트 정리
- [x] 안전한 전환 (실행 중인 프레임 처리 완료 후 적용)

### 2.7 CLI 확장
- [x] `python main.py --live` — 구성 파일 기반 다중 카메라 실행
- [x] `python main.py --video <path>` — 단일 영상 모드 유지
- [x] 구성 파일 경로 옵션 (`--config <path>`)

### 2.8 SpaceLogger 실제 연결
- [x] Orchestrator → Pipeline → SpaceLogger 연결
- [x] Pipeline에서 SpaceLogger.collect() 호출
- [x] SpaceLogger.flush() 주기적/이벤트 기반 전략 (N개 카메라 수집 시 즉시 flush + 10초 안전망)
- [x] Hot-reload 시 공간 추가/삭제 처리

---

## Phase 3 — Custom Fine-Tuning (선택적)

### 3.1 데이터셋 준비
- [ ] 가정 내 장난감/특정 물품 이미지 수집
- [ ] 라벨링 (COCO/YOLO 형식)
- [ ] 데이터 증강 설정

### 3.2 YOLO26 Fine-Tuning
- [ ] `yolo train model=yolo26s.pt data=<custom_dataset.yaml>`
- [ ] 하이퍼파라미터 튜닝
- [ ] 검증 및 평가

### 3.3 통합
- [ ] 학습된 모델로 교체
- [ ] 새로운 클래스가 상호작용 감지에 반영되도록 수정

---

## Phase 4 — (선택적) 고급 기능

### 4.1 3D 공간 매핑
- [ ] 카메라 캘리브레이션 (OpenCV stereo)
- [ ] 3D 좌표 변환
- [ ] 통합 추적 → 정확한 위치/이동 판단

### 4.2 Re-ID 기반 객체 연결
- [ ] 각 카메라에서 추적 대상 이미지 특징 추출
- [ ] 동일 개체 판별 → 카메라 간 연결

### 4.3 시각화 대시보드
- [ ] 실시간 영상 + bbox 오버레이
- [ ] 로그 타임라인
- [ ] 공간별 상태 표시

---

## 프로젝트 구조

```
tracking-cano/
├── main.py                  # 진입점 (CLI: --live / --video)
├── config/
│   ├── __init__.py
│   └── spaces.yaml          # 동적 구성 파일 (Phase 2+)
├── core/
│   ├── __init__.py
│   ├── config_manager.py    # YAML 읽기 + watchdog 핫리로드
│   ├── pipeline.py          # 단일 카메라 파이프라인
│   └── orchestrator.py      # N개 카메라 병렬 + 공간별 그룹핑
├── modules/
│   ├── __init__.py
│   ├── detector.py          # YOLO26 감지 (다중 클래스)
│   ├── tracker.py           # ByteTrack 추적
│   ├── analyzer.py          # 이동 상태 분류
│   ├── interaction_detector.py  # bbox 상호작용 판단
│   └── tile_detector.py     # 타일링 폴백 감지 (전체화면 실패 시)
├── nlp/
│   ├── __init__.py
│   └── logger.py            # LLM 자연어 로깅 (NLPLogger + SpaceLogger)
├── utils/
│   ├── __init__.py
│   └── video.py             # RTSP/파일 소스 캡처 헬퍼
├── logs/                    # 로그 출력 디렉토리
├── .env                     # LLM API 설정
├── .env.example             # 템플릿
├── .gitignore
├── requirements.txt
├── PLAN.md                  # 현재 작업 계획
└── MASTER_PLAN.md           # 이 파일
```

---

## 진행 상황

- Phase 1: ✅ 완료 (1.2~1.8 구현)
- Phase 2: ✅ 완료 (2.1~2.8 구현)
- Phase 3: ⬜ 시작 전 (선택적)
- Phase 4: ⬜ 시작 전 (선택적)
