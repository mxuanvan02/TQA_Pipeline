# TQA Pipeline: HOEIT-LegalQA and ECM-TQAG

This repository contains two separate Vietnamese legal-textbook QA tracks:

- **HOEIT-LegalQA benchmark:** the established Bloom-structured benchmark pipeline. Dataset: [maixuanvan/dhh2026-tqa-output](https://huggingface.co/datasets/maixuanvan/dhh2026-tqa-output).
- **ECM-TQAG experimental protocol:** a reproducible evidence-chain generation and audit pipeline. Its intended derivative dataset location is [maixuanvan/ECM-TQAG](https://huggingface.co/datasets/maixuanvan/ECM-TQAG), but no ECM artifact is released until generation, quality, and rights gates pass.

The two tracks must not be conflated. HOEIT-LegalQA is an established benchmark release; ECM-TQAG is a source-bound experimental study.

> **Release boundary:** this repository does not grant redistribution rights for source textbooks, page images, or raw model ledgers. A parsed ECM record only means its structural/provenance contract passed; it does not prove legal correctness, pedagogical quality, unique-best-answer validity, or visual grounding.

## HOEIT-LegalQA benchmark

### Current paper-aligned release

- Source documents: 48 university-level law textbooks from the Institute of Open Education and Information Technology, Hue University.
- Full public release: 14,998 records; eval-ready subset: 14,210 records.
- Document-aware train/dev/test split: 9,894 / 2,144 / 2,172 records.
- Bloom levels: Remember, Understand, Apply.
- Evaluated model families: Gemma-2-9B, Llama-3-8B, Mistral-7B, Qwen2.5-7B.

The benchmark supports Vietnamese legal NLP, legal-education QA, Bloom-level reasoning analysis, and retrieval-grounded multiple-choice evaluation. It is not legal advice or a high-stakes assessment instrument without independent expert validation.

### Pipeline

1. PDF digitization: `marker-pdf` converts source PDFs to Markdown and extracts page-level visual assets.
2. Multimodal structuring: Markdown is split into source-traceable legal contexts; Vintern-1B-v3 enriches image-bearing contexts.
3. QA generation: Qwen2.5-7B-Instruct-AWQ generates Vietnamese MCQs across the three Bloom levels.
4. Quality filtering: Gemma-2-2B-IT checks groundedness, multimodal alignment, legal fluency, and taxonomy consistency.

Benchmark preparation is separate: it applies answer normalization, language-sanity cleanup, document-aware splitting, and deterministic option-position rebalancing.

### Benchmark metrics

Accuracy is the proportion of correct held-out MCQ answers. The benchmark compares `None` (question and options only) with `With` (question, options, and gold source context). Context gain is their paired item-level difference in percentage points. Accuracy intervals use Wilson 95% confidence intervals; context-gain intervals use paired differences; p-values use continuity-corrected McNemar tests over discordant outcomes.

## ECM-TQAG: graph-program multimodal TQA generation

### Method contract

For every frozen chunk, ECM-TQAG uses three evidence conditions:

- **T:** extracted text only.
- **TL_struct:** text plus declared document structure.
- **TLV:** text, structure, and attached image pixels.

It produces one candidate item under each protocol:

- **Direct:** creates an MCQ from one directly supported proposition.
- **Answer-first:** locks a supported answer first, then builds a question and same-domain distractors.
- **ECM (Evidence-Chain Method):** makes a planner call followed, when the plan passes deterministic checks, by a realization call. The planner proposes a source-bound document graph and a closed-catalog motif request. Local code matches the motif, compiles and executes a restricted graph program, and derives locked answer atoms and provenance traces. A realizer receives **only this locked construction**, not the original text or image pixels, and creates one MCQ.

The validator rejects invalid IDs/roles, nodes not bound to frozen evidence, unmatched motifs, missing visual nodes in ECM--TLV plans, incomplete traces/anchors, and changed executor-derived answer atoms. The full matrix is `8 chunks × 3 conditions × 3 protocols = 72` cells. Direct and Answer-first each make one request per cell; ECM uses planner plus realization when planning succeeds, so the complete design plans up to **96 API calls**.

### Repository layout

```text
scripts/research/build_ecm_8chunk_manifest.py  # builds immutable 8×3 manifest
scripts/research/run_qwen37_tqa_pilot.py       # scoped pilot or full matrix
scripts/research/audit_strict_tqa_results.py   # deterministic provenance audit
tests/test_qwen37_tqa_pilot.py                 # ECM contract tests
research/artifacts/                            # local, git-ignored manifests
research/results/                              # local, git-ignored ledgers/audits
```

The runner filename is retained for compatibility with existing scripts; it supports both a scoped evaluation and the complete matrix.

### Requirements

- Python 3.10+; the ECM runner itself uses only the standard library.
- An OpenRouter key authorized for `qwen/qwen3.7-plus`.
- A local immutable manifest and its referenced image files.

For the broader OCR/chunking pipeline:

```bash
python -m pip install -r requirements.txt
```

Never commit `.env` files, API keys, raw textbooks, extracted images, or raw ledgers.

### Build immutable evidence packages

The generation runner consumes a frozen manifest rather than arbitrary source text. This makes the T/TL_struct/TLV inputs explicit and verifies image size/hash before a request.

```bash
python scripts/research/build_ecm_8chunk_manifest.py \
  --contexts data/output/interim/multimodal_contexts.json \
  --pilot-manifest research/artifacts/pilot_evaluation_manifest.json \
  --out research/artifacts/ecm_inputs_8chunks_v3.json
```

The input contract requires exactly eight chunks and T, TL_struct, TLV packages for each chunk. T has text only; TL_struct adds declared structure; TLV adds verified images. Image markdown, filenames, and page separators are removed from shared text; pixels are attached only for TLV. When OCR splitting leaves a package without sufficient semantic context, the package must be augmented only with demonstrably adjacent source context; unrelated neighbouring text is not inserted.

### Validate before API calls

```bash
python -m unittest tests/test_qwen37_tqa_pilot.py -v

RUN_ID="ecm_graph_program_matrix72_YYYYMMDD"
OUT_DIR="research/results/${RUN_ID}"

python scripts/research/run_qwen37_tqa_pilot.py \
  --manifest research/artifacts/ecm_inputs_8chunks_v3.json \
  --out-dir "$OUT_DIR" \
  --experiment-id "$RUN_ID" \
  --model qwen/qwen3.7-plus \
  --api-key-env OPENROUTER_API_KEY \
  --seed 20260824 \
  --all --dry-run
```

The dry run should report 72 cells and 96 planned calls; it neither reads an API key nor writes a result directory.

### Run frozen chunks to a results ledger

Set the key in the same terminal that starts the runner; never place it in Git, notebooks, issues, or chat logs.

```bash
export OPENROUTER_API_KEY='replace-with-your-local-secret'

RUN_ID="ecm_graph_program_matrix72_YYYYMMDD"
OUT_DIR="research/results/${RUN_ID}"
test ! -e "$OUT_DIR" || { echo "Refusing to overwrite $OUT_DIR"; exit 1; }

python scripts/research/run_qwen37_tqa_pilot.py \
  --manifest research/artifacts/ecm_inputs_8chunks_v3.json \
  --out-dir "$OUT_DIR" \
  --experiment-id "$RUN_ID" \
  --model qwen/qwen3.7-plus \
  --base-url https://openrouter.ai/api/v1/chat/completions \
  --api-key-env OPENROUTER_API_KEY \
  --seed 20260824 \
  --timeout-sec 180 \
  --retries 2 \
  --all
```

The non-overwriting directory contains `results.jsonl` (append-only cell ledger), `prompt_audit.jsonl` (prompt hashes and field receipts), and `summary.json`. A nonzero exit records rejects/errors without deleting the ledger; do not rerun into the same output directory.

For a nine-cell scoped pilot, use all three `--condition` values and one `--chunk-id` in a distinct output directory.

### Mechanical audit

```bash
python scripts/research/audit_strict_tqa_results.py \
  --results "$OUT_DIR/results.jsonl" \
  --manifest research/artifacts/ecm_inputs_8chunks_v3.json \
  --output "$OUT_DIR/mechanical_audit.json"
```

The audit replays record shape, distinct options, answer/choice consistency where applicable, source-bound typed graph nodes and edges, closed-catalog motif predicates, exact restricted-program compiler output, executor-derived answer atoms, construction receipts, program traces, locked graph-node anchors, and the ECM--TLV image-node requirement. It is not a semantic, legal, pedagogical, or image-grounding judge; those require documented human/expert review.

### ECM-TQAG evaluation result (72-cell matrix)

The completed Qwen3.7-plus matrix contains 72 cells. After rerunning the 18 cells belonging to two under-contextualized chunks with an augmented evidence manifest, 60 cells parsed (83.3%) and 12 were rejected (16.7%). The final result is reported with mixed provenance: 54 unaffected cells use the original manifest and 18 rerun cells use the augmented manifest. The two subsets are audited separately against their own manifest digests.

The 12 rejections are classified as follows:

| Scenario | Count | Interpretation |
|---|---:|---|
| Insufficient evidence | 5 | The supplied source context did not support a complete construction; four occurred in a title-and-image-only package. |
| Non-literal source grounding | 4 | A graph node or anchor paraphrased instead of reproducing a contiguous source span. |
| ECM answer not bound to executor atom | 2 | The selected option was not mechanically bound to the derived answer atom. |
| Invalid ECM evidence anchor | 1 | The final anchor did not match the locked graph node. |

These are structural/provenance outcomes, not judgments of legal correctness or educational quality. Detailed records remain local because raw model outputs, textbook excerpts, and page images are not redistributable.

## Transparency and citation

Runtime prompts are in `scripts/research/run_qwen37_tqa_pilot.py`; prompt version and SHA-256 receipts are stored locally per cell. Release only code, synthetic fixtures, non-sensitive schemas, and derivative records cleared for redistribution. Do not upload raw books, scans, figure pixels, long verbatim excerpts, credentials, or unreviewed raw model output.

```bibtex
@dataset{hoeitlegalqa2026,
  title     = {HOEIT-LegalQA: A Bloom-Structured Vietnamese Legal Textbook Question Answering Benchmark},
  author    = {Mai, Xuan Van and Nguyen, Tuong Tri},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/maixuanvan/dhh2026-tqa-output}
}
```

For the legacy Google Colab OCR/QAG workflow, see [README_COLAB.md](README_COLAB.md).
