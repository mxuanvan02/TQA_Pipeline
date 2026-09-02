#!/usr/bin/env bash
# Classify the 251 fitz-mined candidate pages from 11 internal TEXT books.
# fitz already filtered (cheap), VLM is the acceptance gate. Sequential -> avoid 429.
set -u
cd /home/node/.openclaw/workspace/working/Research/07_Projects/DHH2026/TQA_Pipeline || exit 1
IMG_DIR=data/rendered_internal_textcand
OUT=data/vlm_internal_textcand.jsonl
MODEL=cx/gpt-5.4-mini
echo "=== [$(date +%H:%M:%S)] START classify internal text-candidates ==="
n=$(ls "$IMG_DIR"/*.jpg 2>/dev/null | wc -l)
echo "images to classify: $n  -> $OUT  model=$MODEL"
NINEROUTER_VLM_MODEL="$MODEL" python3 scripts/classify_pages_vlm.py "$IMG_DIR" "$OUT" --model "$MODEL" --resume
echo "=== [$(date +%H:%M:%S)] DONE classify; rows: $(wc -l < "$OUT" 2>/dev/null) ==="
