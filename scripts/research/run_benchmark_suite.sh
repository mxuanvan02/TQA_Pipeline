#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_DATASET="data/output/processed/dataset_eval_ready.clean.jsonl"
if [[ ! -f "$DEFAULT_DATASET" ]]; then
  DEFAULT_DATASET="data/output/processed/dataset_eval_ready.jsonl"
fi
DATASET="${1:-$DEFAULT_DATASET}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
OUT_DIR="${3:-research/results/benchmarks}"
SPLIT="${4:-test}"
GPU_UTIL="${GPU_UTIL:-0.90}"
TP="${TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

MODEL_STEM="${MODEL##*/}"

echo "[1/2] Dual-mode benchmark (single model load)..."
python scripts/research/eval_benchmark.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --split "$SPLIT" \
  --gpu-util "$GPU_UTIL" \
  --tensor-parallel "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --both-modes \
  --run-name "${MODEL_STEM}_${SPLIT}"

echo "[2/2] Aggregate benchmark tables..."
python scripts/research/build_benchmark_tables.py \
  --report-dir "$OUT_DIR" \
  --out-main-csv "$OUT_DIR/main_benchmark_table.csv" \
  --out-bloom-csv "$OUT_DIR/bloom_breakdown.csv" \
  --out-modality-csv "$OUT_DIR/modality_breakdown.csv" \
  --out-tex "$OUT_DIR/main_benchmark_table.tex"

echo "DONE. Reports written to $OUT_DIR"
