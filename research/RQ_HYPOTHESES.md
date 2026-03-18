# Research Questions and Hypotheses (Frozen)

## Claim

Multimodal, quality-controlled synthesis can produce a Vietnamese legal QA dataset that is higher quality than text-only/no-filter synthesis and improves downstream legal QA.

## RQ1

Does the proposed pipeline produce better QA data quality than text-only and no-filter baselines?

### H1

Adding multimodal grounding increases groundedness and reduces unsupported questions.

## RQ2

Which components contribute most to quality?

### H2

LLM-judge filtering and chunk cleaning provide the largest quality gains, while legal-syllogism prompting improves legal fluency.

## RQ3

Does training or retrieval augmentation using this dataset improve downstream legal QA?

### H3

Models adapted with the proposed dataset outperform the same base model without this dataset on a held-out human gold set.

## Acceptance mapping

- RQ1: Main baseline table + intrinsic metrics + human subset audit.
- RQ2: Ablation table.
- RQ3: Downstream table with confidence intervals.

