#!/bin/bash
set -e

echo "=== Cleaning DB only (output preserved) ==="
rm -f logs/tracking.db

for config in config_test_cv_no_vision.yaml config_test_cv_vision.yaml config_test_llm_vision.yaml; do
    echo ""
    echo "=========================================="
    echo " RUNNING: $config"
    echo "=========================================="
    timeout 55 docker compose run --rm -e LOG_LEVEL=DEBUG tracking-cano \
        python main.py --live --config "$config" 2>&1 | grep -E 'snapshot:|fallback|All workers' || true
done

echo ""
echo "=========================================="
echo " FINAL OUTPUT DIRECTORY:"
echo "=========================================="
ls -la output/
