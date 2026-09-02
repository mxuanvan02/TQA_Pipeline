# Pre-Submission Checklist (TQA Pipeline)

This checklist is tailored for the current Vietnamese legal multimodal QA pipeline.

Execution assets added in this repo:
- `research/EXECUTION_RUNBOOK.md`
- `research/EXPERIMENT_COMMANDS.md`
- `scripts/research/run_research_prep.sh`

## 1) What to do to reach a useful, publishable study

1. Define one clear claim:
- Example: "Multimodal + quality-controlled synthesis improves Vietnamese legal QA quality over text-only synthesis."

2. Lock the task definition and scope:
- Domain(s): civil law only vs multi-domain.
- Input/output schema frozen (no last-minute field changes).

3. Build strong baselines:
- Text-only generation baseline.
- No-judge filtering baseline.
- Alternative judge model baseline (different family than generator).

4. Add ablation experiments:
- Remove image features.
- Remove chunk cleaning.
- Remove legal rationale constraint.
- Remove checkpoint resume artifacts from evaluation set.

5. Add human evaluation:
- Sample stratified subsets (easy/medium/hard).
- At least 2 annotators.
- Report agreement (Cohen’s kappa or Krippendorff alpha).

6. Improve reproducibility:
- Fix random seeds.
- Pin model versions/dependency versions.
- Log exact prompts, config, and run command.

7. Add quality and risk analysis:
- Error taxonomy (hallucination, wrong legal grounding, visual mismatch).
- License/data provenance table.
- Ethical and misuse statement.

8. Add efficiency and practicality reporting:
- GPU type, runtime, throughput, cost per 1k QA.
- Resume behavior after Colab interruption.

## 2) Checklist (tick before submission)

### A. Research Framing
- [ ] Research question is one sentence and falsifiable.
- [ ] Hypothesis is explicit and testable.
- [ ] Contribution list is specific (dataset / method / evaluation).
- [ ] Every major claim has a claim-to-evidence mapping (why + proof + artifact pointer).

### B. Data & Provenance
- [ ] Data sources are listed with license/usage terms.
- [ ] Inclusion/exclusion criteria are documented.
- [ ] Dedup policy is documented.
- [ ] Train/dev/test split protocol is fixed and documented.
- [ ] Leakage risk assessment is documented.

### C. Pipeline Soundness
- [ ] Stage 1 output coverage reported (PDF -> MD success rate).
- [ ] Stage 2 context quality checks reported.
- [ ] Stage 3 generation constraints logged (Bloom levels, prompt template).
- [ ] Stage 4 filtering thresholds justified (not arbitrary).
- [ ] Resume/checkpoint behavior tested and documented.

### D. Baselines & Ablations
- [ ] Text-only baseline implemented and reported.
- [ ] No-judge baseline implemented and reported.
- [ ] Cross-judge baseline (different judge model) reported.
- [ ] Core ablations completed (no-image, no-cleaning, no-rationale).
- [ ] All comparisons use same evaluation protocol.

### E. Evaluation Quality
- [ ] Automatic metrics are defined and justified.
- [ ] Human evaluation protocol is documented.
- [ ] Inter-annotator agreement is reported.
- [ ] Statistical significance/confidence intervals are reported.
- [ ] Error analysis includes representative failure cases.

### F. Reproducibility
- [ ] All key seeds are fixed and logged.
- [ ] Dependency versions are pinned.
- [ ] Config snapshot per run is saved.
- [ ] Commands to reproduce main tables are provided.
- [ ] Hardware/runtime details are provided.

### G. Ethics, Safety, Limitations
- [ ] Legal/ethical risks are discussed.
- [ ] Intended use and out-of-scope use are stated.
- [ ] Known limitations are explicit.
- [ ] Model/data bias discussion included.

### H. Writing & Artifact Readiness
- [ ] Abstract states problem, method, results, impact.
- [ ] Figures/tables are readable and self-contained.
- [ ] Claims match evidence (no over-claiming).
- [ ] No unsupported claims: any new statement includes rationale + quantitative/qualitative evidence + reproducible source.
- [ ] Release package includes scripts + configs + schema docs.
- [ ] Anonymous version (if double-blind) is clean.

## 3) Minimum evidence bar (recommended)

- At least 3 baselines.
- At least 4 ablations.
- Human evaluation on >= 200 items (stratified).
- Agreement metric reported.
- Cost/runtime table included.
- Repro scripts run end-to-end on a clean environment.
