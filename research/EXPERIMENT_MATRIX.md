# Experiment Matrix (Findings)

## Primary system

- Full pipeline: Stage 1-4 with multimodal context, legal-syllogism prompt, judge filtering.

## Baselines

1. B1 Text-only generation
- Remove visual context from Stage 3 prompts.

2. B2 No-judge filtering
- Keep Stage 3 raw QA without Stage 4 filtering.

3. B3 Cross-judge subset
- Evaluate a fixed subset with an external judge model (different family from generator).

## Ablations

1. A1 No multimodal alignment
- Ignore multimodal criterion during filtering.

2. A2 No chunk cleaning
- Disable Stage 2 cleaning heuristics.

3. A3 No legal syllogism
- Remove legal-syllogism constraint from Stage 3 prompt.

4. A4 Threshold sensitivity
- Compare strict `1.0` thresholds vs softer dev-tuned thresholds.

## Metrics

### Intrinsic

- Groundedness
- Legal fluency
- Multimodal alignment
- Parse-valid rate
- Duplicate QA rate
- Diversity (distinct question ratio)

### Human

- Legal correctness
- Groundedness
- Question quality
- Distractor quality
- Visual alignment
- Inter-annotator agreement (kappa + percent agreement)

### Downstream

- Exact/partial correctness (or EM/F1 if task framing supports it)
- 95% CI via bootstrap

## Compute policy (T4/free)

- Full run for primary system.
- Cross-judge on fixed subset.
- Downstream on compact model and fixed split.
- Keep all seeds/configs fixed for reproducibility.

