# Experiment Commands (Baseline + Ablation)

Set data path first:

```bash
export TQA_DATA_DIR=/content/TQA_Pipeline/data/output
cd /content/TQA_Pipeline
```

## Primary system

```bash
python -m src.03_qag_generator
python -m src.04_evaluate
```

## Baselines

### B1 Text-only generation

```bash
python -m src.03_qag_generator --text-only
python -m src.04_evaluate
```

### B2 No-judge filtering

```bash
python -m src.04_evaluate --skip-judge
```

### B3 Cross-judge subset

```bash
python -m src.04_evaluate --limit 300 --model-name <OTHER_JUDGE_MODEL>
```

Note:
- Stage 4 now defaults to a separate judge model from `EvalConfig.judge_model_name`.
- Using the same generation and judge model is blocked by default; override only with:
```bash
python -m src.04_evaluate --allow-same-model
```

## Ablations

### A1 Disable multimodal alignment criterion

```bash
python -m src.04_evaluate --disable-multimodal-alignment
```

### A2 No chunk cleaning

```bash
python -m src.02_structuring --disable-cleaning
python -m src.03_qag_generator
python -m src.04_evaluate
```

### A3 No legal syllogism

```bash
python -m src.03_qag_generator --no-legal-syllogism
python -m src.04_evaluate
```

### A4 Threshold sensitivity

```bash
python -m src.04_evaluate \
  --groundedness-threshold 0.8 \
  --multimodal-threshold 0.8 \
  --legal-fluency-threshold 0.8 \
  --overall-threshold 0.8
```

## Notes

- Keep random seed and split manifest fixed across all runs.
- Record run metadata (model, command, timestamp, GPU, output paths).
