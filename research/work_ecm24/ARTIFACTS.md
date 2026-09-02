# ECM-TQAG ICTC2026 — Reproducibility Artifacts

Generator: `qwen/qwen3-vl-32b-instruct` (OpenRouter).
Judges: `claude-sonnet-5`, `openai/gpt-5.6-terra` — blinded, cross-family.

**Scope caveat.** All scores here are *model-based* and describe inter-judge
agreement only. They are **not** evidence of semantic or legal validity.
No human anchor was collected.

## Files

| File | Content |
|---|---|
| `run_ecm24_matrix.py` | 144-cell generation runner (16 chunks x 3 conditions x 3 methods) |
| `judge_blinded.py` | Blinded judging harness (method/condition/generator masked, choices permuted) |
| `stats_tests.py` | Friedman + Wilcoxon/Holm + Cliff's delta + bootstrap CI |
| `runs/judge_full_sonnet5.jsonl` | 143 scored cells, judge 1 |
| `runs/judge_full_terra.jsonl` | 142 scored cells + 1 rejected (truncated JSON), judge 2 |
| `audit/grounding_check_v3.json` | Per-cell mechanical grounding metrics |
| `audit/generator_compare_qwen_vs_claude.json` | Qwen vs Claude generator comparison |
| `audit/judge_agreement.json` | Exact / within-1 / QWK / Pearson r |
| `audit/stats_tests.json` | Full statistical test output |
| `artifacts_sha256.txt` | SHA-256 of every file above |

Excluded on purpose: `ecm_inputs_final_v2.json` (contains verbatim textbook
text) and page images — copyright-restricted.

## Headline numbers

Cross-judge agreement (n=142):

| Dimension | Exact | Within-1 | QWK | r |
|---|---|---|---|---|
| faithfulness | 0.901 | 0.965 | 0.466 | 0.592 |
| answerability | 0.894 | 0.958 | 0.686 | 0.693 |
| distractor_quality | 0.549 | 0.979 | 0.369 | 0.401 |
| depth | 0.585 | 0.986 | 0.416 | 0.550 |

Method comparison (Friedman, unit = chunk, n=14 complete blocks):
**no dimension reaches Holm-corrected significance.** `depth` has an
uncorrected Friedman p=0.0243 but every post-hoc pair
fails after Holm correction. Condition comparison: all n.s.

Bootstrap: 10000 iterations, seed 20260806.
