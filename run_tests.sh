#!/bin/bash
set -e

REPORT=""
for mode in "cv_no_vision" "cv_vision" "llm_vision"; do
  CONFIG="config_test_${mode}.yaml"
  OUTFILE="logs/test_${mode}.log"

  echo "============================================"
  echo " TEST: $mode"
  echo " Config: $CONFIG"
  echo "============================================"
  START=$(date +%s%N)

  docker compose run --rm \
    -v $(pwd)/${CONFIG}:/app/config_test.yaml:ro \
    tracking-cano \
    python main.py --live "" --config /app/config_test.yaml --verbose 2>&1 | tee "$OUTFILE"

  END=$(date +%s%N)
  DURATION_MS=$(( (END - START) / 1000000 ))
  echo ""
  echo "=== DURATION: $(($DURATION_MS / 1000)).$(($DURATION_MS % 1000))s ==="
  REPORT="${REPORT}TEST ${mode}: $(($DURATION_MS / 1000)).$(($DURATION_MS % 1000))s\n"
done

echo ""
echo "============================================"
echo " RESULTS"
echo "============================================"
echo -e "$REPORT"
