#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_BASE="${TQA_DATA_DIR:-$ROOT_DIR/data/output}"
RAW_DIR="${ROOT_DIR}/data/raw"
INTERIM_DIR="${DATA_BASE}/interim"

mkdir -p research/artifacts research/results research/human_eval

echo "[0/7] Validate current pipeline state..."
python scripts/research/validate_pipeline_state.py --data-base "${DATA_BASE}" || true

echo "[1/7] Prune orphan contexts against raw sources..."
python scripts/research/prune_orphan_contexts.py \
  --data-base "${DATA_BASE}" \
  --raw-dir "${RAW_DIR}" \
  --report-out research/results/prune_orphan_contexts_report.json

echo "[2/7] Build document-level splits..."
python scripts/research/build_doc_splits.py \
  --contexts "${INTERIM_DIR}/multimodal_contexts.json" \
  --out research/artifacts/doc_splits.json \
  --manifest-out research/artifacts/context_manifest.json \
  --seed 42 --train-ratio 0.70 --dev-ratio 0.15

echo "[3/7] Run quality gates..."
python scripts/research/quality_gates.py \
  --raw-dir "${RAW_DIR}" \
  --interim-dir "${INTERIM_DIR}" \
  --contexts "${INTERIM_DIR}/multimodal_contexts.json" \
  --raw-qa "${INTERIM_DIR}/raw_qa_pairs.json" \
  --qa-chunks-dir "${INTERIM_DIR}/qa_chunks" \
  --out research/results/quality_gates_pre_s3s4.json

echo "[4/7] Run integrity checks for raw QA..."
python scripts/research/data_integrity_check.py \
  --dataset "${INTERIM_DIR}/raw_qa_pairs.json" \
  --manifest research/artifacts/context_manifest.json \
  --out research/results/integrity_raw_qa.json

echo "[5/7] Generate human-eval sheets (n=240)..."
HUMAN_SOURCE="${INTERIM_DIR}/filtered_qa_pairs.json"
if [[ ! -f "$HUMAN_SOURCE" ]]; then
  HUMAN_SOURCE="${INTERIM_DIR}/raw_qa_pairs.json"
fi
python scripts/research/make_human_eval_sample.py \
  --input "$HUMAN_SOURCE" \
  --n 240 \
  --seed 42 \
  --out-dir research/human_eval \
  --qa-chunks-dir "${INTERIM_DIR}/qa_chunks"

echo "[6/7] Build main-results summary..."
python scripts/research/build_main_results.py \
  --split-json research/artifacts/doc_splits.json \
  --quality-gates-json research/results/quality_gates_pre_s3s4.json \
  --raw-qa "${INTERIM_DIR}/raw_qa_pairs.json" \
  --qa-chunks-dir "${INTERIM_DIR}/qa_chunks" \
  --out-json research/results/main_results_summary.json \
  --out-csv research/results/main_results_table.csv \
  --out-tex research/results/main_results_macros.tex

echo "[7/7] Coverage audit snapshot..."
python scripts/research/pipeline_coverage_audit.py \
  --data-base "${DATA_BASE}" \
  --raw-dir "${RAW_DIR}" \
  --out research/results/pipeline_coverage_audit.json

echo "DONE. Artifacts generated under research/artifacts, research/results, research/human_eval"
