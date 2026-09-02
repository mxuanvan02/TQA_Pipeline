# TQA Pipeline on Google Colab

This guide is optimized for running `TQA_Pipeline` on Colab with rented GPU (e.g., L4/T4).

## 1) Mount Google Drive

Run this in the first Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2) Open project in `/content`

If your repo is on GitHub:

```bash
cd /content
git clone <YOUR_REPO_URL> TQA_Pipeline
cd /content/TQA_Pipeline
```

If project is already in Drive, copy/sync it into `/content/TQA_Pipeline`.

## 3) Install dependencies

```bash
cd /content/TQA_Pipeline
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

## 4) Prepare input PDFs

Put PDFs into one of these layouts:

- Standard: `/content/TQA_Pipeline/data/raw/*.pdf`
- Legacy: `/content/TQA_Pipeline/data/output/raw/*.pdf`

`src/config.py` now auto-detects both layouts.  
You can also force a custom data base dir:

```bash
export TQA_DATA_DIR=/content/TQA_Pipeline/data/output
```

## 5) Run full pipeline

### Full run

```bash
cd /content/TQA_Pipeline
bash run_pipeline.sh --yes
```

### Quick test run

```bash
cd /content/TQA_Pipeline
bash run_pipeline.sh --limit 2 --yes
```

Notes:

- `--yes` avoids interactive prompt if Drive is not mounted.
- `--require-drive` forces abort when Drive is not mounted.

## 6) Optional env overrides

```bash
export TQA_ROOT=/content/TQA_Pipeline
export TQA_BACKUP_BASE=/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup
```

## 7) Outputs

- Main output JSONL:
  - `/content/TQA_Pipeline/data/processed/dataset.jsonl`
  - or `/content/TQA_Pipeline/data/output/processed/dataset.jsonl` (legacy layout)
- Drive backup base:
  - `/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup`

## 8) Resume after Colab disconnect

When a new session starts:

1. Mount Drive again.
2. Reinstall dependencies.
3. Run `bash run_pipeline.sh --yes`.

The pipeline restores/continues from Drive checkpoints where available.
