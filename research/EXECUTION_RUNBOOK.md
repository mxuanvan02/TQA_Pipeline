# Execution Runbook (Findings Track, 2-3 Months)

This runbook operationalizes the accepted plan into concrete actions and commands.

## 0) Directory convention

- Input base (current): `data/output`
- Research artifacts: `research/artifacts`
- Metrics/reports: `research/results`

Create folders:

```bash
cd /content/TQA_Pipeline
mkdir -p research/artifacts research/results research/human_eval
```

## 1) Freeze task, RQ, and hypothesis

Use:
- `research/RQ_HYPOTHESES.md`
- `research/PAPER_OUTLINE.md`

Deliverable:
- Frozen claim + RQ/Hypothesis section in draft paper.

## 1.5) Prune orphan contexts (must match raw sources)

```bash
python scripts/research/prune_orphan_contexts.py \
  --data-base data/output \
  --raw-dir data/raw \
  --report-out research/results/prune_orphan_contexts_report.json
```

Deliverable:
- `research/results/prune_orphan_contexts_report.json`
- `data/output/interim/multimodal_contexts.orphan_backup.json`

## 2) Build document-level split (70/15/15, no doc leakage)

```bash
python scripts/research/build_doc_splits.py \
  --contexts data/output/interim/multimodal_contexts.json \
  --out research/artifacts/doc_splits.json \
  --manifest-out research/artifacts/context_manifest.json \
  --seed 42 --train-ratio 0.70 --dev-ratio 0.15
```

Deliverable:
- `research/artifacts/doc_splits.json`
- `research/artifacts/context_manifest.json`

## 3) Run quality gates before Stage 3/4 experiments

```bash
python scripts/research/quality_gates.py \
  --raw-dir data/raw \
  --interim-dir data/output/interim \
  --contexts data/output/interim/multimodal_contexts.json \
  --raw-qa data/output/interim/raw_qa_pairs.json \
  --qa-chunks-dir data/output/interim/qa_chunks \
  --out research/results/quality_gates_pre_s3s4.json
```

Deliverable:
- `research/results/quality_gates_pre_s3s4.json`

## 4) Integrity checks (schema, duplicates, leakage)

For raw QA:

```bash
python scripts/research/data_integrity_check.py \
  --dataset data/output/interim/raw_qa_pairs.json \
  --manifest research/artifacts/context_manifest.json \
  --out research/results/integrity_raw_qa.json
```

For final dataset (after Stage 4):

```bash
python scripts/research/data_integrity_check.py \
  --dataset data/output/processed/dataset.jsonl \
  --manifest research/artifacts/context_manifest.json \
  --out research/results/integrity_dataset_jsonl.json
```

## 5) Human evaluation sampling (200-300 items)

Use filtered QA (preferred) or raw QA:

```bash
python scripts/research/make_human_eval_sample.py \
  --input data/output/interim/filtered_qa_pairs.json \
  --n 240 \
  --seed 42 \
  --out-dir research/human_eval
```

Deliverables:
- `research/human_eval/human_eval_master.csv`
- `research/human_eval/annotator_1.csv`
- `research/human_eval/annotator_2.csv`

## 6) Agreement after annotation

```bash
python scripts/research/compute_agreement.py \
  --ann1 research/human_eval/annotator_1.csv \
  --ann2 research/human_eval/annotator_2.csv \
  --out research/results/human_agreement.json
```

## 7) Statistical reporting (95% CI)

Prepare a CSV with a numeric metric column (e.g., exact_correct).

```bash
python scripts/research/bootstrap_ci.py \
  --csv research/results/downstream_eval.csv \
  --column exact_correct \
  --n-bootstrap 2000 \
  --seed 42 \
  --out research/results/downstream_ci_exact_correct.json
```

## 8) Required paper tables/figures mapping

- Dataset stats table: from split + quality gate + integrity reports.
- Baseline/ablation table: from experiment matrix in `research/EXPERIMENT_MATRIX.md`.
- Human agreement table: from `human_agreement.json`.
- Cost/runtime table: log manually per run.
- Error taxonomy figure: curate at least 30 failures.

Baseline/ablation command set:
- `research/EXPERIMENT_COMMANDS.md`

## 9) Exit criteria before submission

- All checks in `PRE_SUBMISSION_CHECKLIST.md` marked complete.
- Main claims answered by at least one table + one qualitative section.
- Repro commands for every table recorded in paper appendix.
