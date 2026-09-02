#!/usr/bin/env bash
# Tuần tự: bù 97 trang external lỗi (--resume) -> 110 trang retry nội bộ. Idempotent.
set -u
cd "$(dirname "$0")/.." || exit 1
M=cx/gpt-5.4-mini
echo "=== [$(date +%H:%M:%S)] STAGE A: bù external (97 trang còn thiếu) ==="
python3 scripts/classify_pages_vlm.py data/rendered_external data/vlm_external_cv_candidates.jsonl --model "$M" --resume 2>&1
echo "=== [$(date +%H:%M:%S)] STAGE A done. external rows: $(wc -l < data/vlm_external_cv_candidates.jsonl 2>/dev/null) ==="
echo "=== [$(date +%H:%M:%S)] STAGE B: retry 110 trang nội bộ lỗi ==="
python3 scripts/classify_pages_vlm.py data/_retry_round2 data/vlm_retry_round.jsonl --model "$M" --resume 2>&1
echo "=== [$(date +%H:%M:%S)] STAGE B done. retry rows: $(wc -l < data/vlm_retry_round.jsonl 2>/dev/null) ==="
echo "=== ALL DONE [$(date +%H:%M:%S)] ==="
