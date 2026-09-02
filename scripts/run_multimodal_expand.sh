#!/usr/bin/env bash
# Tuần tự: retry 110 trang lỗi -> classify 1591 external. --resume idempotent.
set -u
cd "$(dirname "$0")/.." || exit 1
M=cx/gpt-5.4-mini
echo "=== [$(date +%H:%M:%S)] STAGE 1: retry 110 error pages ==="
python3 scripts/classify_pages_vlm.py data/_retry_http_err data/vlm_retry_round.jsonl --model "$M" --resume 2>&1
echo "=== [$(date +%H:%M:%S)] STAGE 1 done. retry rows: $(wc -l < data/vlm_retry_round.jsonl 2>/dev/null) ==="
echo "=== [$(date +%H:%M:%S)] STAGE 2: classify 1591 external rendered pages ==="
python3 scripts/classify_pages_vlm.py data/rendered_external data/vlm_external_cv_candidates.jsonl --model "$M" --resume 2>&1
echo "=== [$(date +%H:%M:%S)] STAGE 2 done. external rows: $(wc -l < data/vlm_external_cv_candidates.jsonl 2>/dev/null) ==="
echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
