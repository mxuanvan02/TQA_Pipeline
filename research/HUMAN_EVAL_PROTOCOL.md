# Human Evaluation Protocol

## Goal

Validate data quality beyond automatic LLM-judge metrics.

## Sampling

- Target size: 200-300 QA pairs.
- Stratified by:
  - Bloom level (Remember/Understand/Apply)
  - Multimodal vs text-only
  - Major legal topic (via doc_id mapping)

## Annotators

- 2-3 annotators with legal background.
- Double annotation for all selected items.

## Rubric fields (binary or 3-level)

1. Legal correctness
2. Groundedness to provided context
3. Question quality (clear, answerable, non-trivial)
4. Distractor quality
5. Visual alignment (for multimodal items)

## Adjudication

- Disagreement review by lead annotator.
- Store final decision and reason.

## Data logging rules

- Keep `qa_id`, annotator ID (anonymized), timestamp, and comments.
- Do not delete conflicting entries; keep trace for auditability.

## Report

- Percent agreement per field.
- Cohen's kappa per field.
- Failure pattern summary with examples.

