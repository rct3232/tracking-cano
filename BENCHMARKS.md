# Benchmarks — CPU 환경 (3대 RTSP 카메라 동시)

## YOLO26s vs n 단일 추론 속도

| 모델 | 평균 추론 시간 | 초당 처리 (단일) |
|------|---------------|-----------------|
| yolo26s | 112ms | ~9 fps |
| yolo26n | 71ms | ~14 fps |

## bench.py 결과 (30초, frame_skip=15, 3대 동시)

### yolo26s
| 카메라 | frames read | inferred | infer/s | avg ms/infer |
|--------|------------|----------|---------|-------------|
| livingroom | 442 | 28 | 0.9 | 111.4 |
| livingfront | 424 | 27 | 0.9 | 113.3 |
| hallway | 401 | 26 | 0.9 | 111.5 |
| **Total** | **1267** | **81** | **2.7** | **112.1** |

### yolo26n
| 카메라 | frames read | inferred | infer/s | avg ms/infer |
|--------|------------|----------|---------|-------------|
| livingroom | 445 | 28 | 0.9 | 82.6 |
| livingfront | 421 | 27 | 0.9 | 59.0 |
| hallway | 400 | 25 | 0.8 | 61.4 |
| **Total** | **1266** | **80** | **2.7** | **68.0** |

## 실제 실행 환경 (docker-compose.yml, yolo26n)

| 카메라 | 평균 추론 시간 |
|--------|---------------|
| livingroom | ~50ms |
| livingfront | ~52ms |
| hallway | ~50ms |

**실제 환경이 bench보다 빠른 이유:** bench는 3대가 동시에 추론하여 CPU 리소스 경쟁 발생. 실제 환경에서는 3대가 순차적으로 추론이 분산됨.

## 테스트 환경

- CPU: Intel Core i7 (6코어)
- OMP_NUM_THREADS=1, MKL_NUM_THREADS=1
- frame_skip=15
