# HOEIT-LegalQA Pipeline

This repository contains the construction and benchmarking code for **HOEIT-LegalQA**, a Bloom-structured Vietnamese legal textbook question-answering benchmark.

## Links

- Dataset: https://huggingface.co/datasets/maixuanvan/dhh2026-tqa-output
- Code: https://github.com/mxuanvan02/TQA_Pipeline

## Current Paper-Aligned Release

- Source documents: 48 university-level law textbooks from the Institute of Open Education and Information Technology, Hue University
- Full public release: 14,998 records
- Eval-ready benchmark subset: 14,210 records
- Document-aware train/dev/test split: 9,894 / 2,144 / 2,172 records
- Bloom levels: Remember, Understand, Apply
- Evaluated model families: Gemma-2-9B, Llama-3-8B, Mistral-7B, Qwen2.5-7B

The released dataset is intended for Vietnamese legal NLP, legal-education question answering, Bloom-level reasoning analysis, and retrieval-grounded multiple-choice benchmarking. It is not a source of legal advice and should not be treated as a high-stakes assessment instrument without independent expert validation.

## Pipeline

The construction pipeline has four stages.

1. PDF digitization: `marker-pdf` converts the 48 source PDFs to Markdown and extracts page-level visual assets.
2. Multimodal structuring: Markdown is split into source-traceable legal contexts using document headers and Vietnamese legal structural markers; Vintern-1B-v3 enriches image-bearing contexts.
3. QA generation: Qwen2.5-7B-Instruct-AWQ generates Vietnamese multiple-choice questions across Remember, Understand, and Apply Bloom levels, with legal-syllogism rationales.
4. Quality filtering: Gemma-2-2B-IT acts as a cross-family judge for groundedness, multimodal alignment, legal fluency, and taxonomy consistency.

Benchmark preparation is separate from the four-stage construction pipeline. It applies answer normalization, language-sanity cleanup, document-aware splitting, and deterministic option-position rebalancing.

## Repository Layout

- `src/01_digitize.py`: PDF-to-Markdown digitization.
- `src/02_structuring.py`: Markdown chunking and multimodal context construction.
- `src/03_qag_generator.py`: Bloom-level QA generation.
- `src/04_evaluate.py`: LLM-as-judge filtering.
- `scripts/research/`: benchmark preparation, integrity checks, figures, and statistical summaries.
- `run_pipeline.sh`, `run_stage3.sh`, `run_stage4.sh`: convenience entry points for pipeline execution.

Generated data, raw PDFs, intermediate outputs, benchmark artifacts, and paper build products are intentionally excluded from Git. The public dataset release is hosted on Hugging Face.

## Benchmark Metrics

Accuracy is the proportion of correctly answered multiple-choice items on the held-out test split. The benchmark compares two zero-shot settings:

- `None`: question and answer options only.
- `With`: question, answer options, and the gold source context.

Context gain is the paired item-level difference between the `With` and `None` settings, reported in percentage points. Accuracy intervals use Wilson 95% confidence intervals; context-gain intervals use paired item-level differences; and p-values use a continuity-corrected McNemar test over discordant item outcomes.

## Minimal Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

GPU execution is recommended for the full construction pipeline because Stage 1, Stage 2, Stage 3, and Stage 4 use OCR/VLM/LLM components. CPU-only execution is suitable mainly for lightweight integrity checks and post-processing scripts.

## Citation

```bibtex
@dataset{hoeitlegalqa2026,
  title     = {HOEIT-LegalQA: A Bloom-Structured Vietnamese Legal Textbook Question Answering Benchmark},
  author    = {Mai, Xuan Van and Nguyen, Tuong Tri},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/maixuanvan/dhh2026-tqa-output}
}
```
