# ECM--TQAG final experiment protocol

**Status:** frozen design; no result is asserted by this document.

## Corpus

The evaluation population is the complete set of **16 image-bearing,
text-sufficient chunks from nine Vietnamese-law textbooks** that pass the frozen
screening rules. The upstream store contains 24 multimodal declarations, but seven
non-manifest records contain only 80--215 text characters or cover artwork. The
remaining excluded record (`10. XAY DUNG VAN BAN PHAPLUAT_chunk_178`) has a
2,454-character passage but its only recovered visual is publisher/logo material;
its exclusion is recorded in `image_refetch_report.json`. There is therefore no
unexplained 17-to-16 attrition and no claimed expansion headroom.

The unit of analysis is the chunk. Calls, retries and repeated answers are not
independent units. Results are bounded-census estimates and are not generalized
outside this corpus, language or domain.

## Primary question and endpoint

The **single confirmatory contrast** is full ECM--TQAG versus the
caption-mediated baseline on gate-valid yield per chunk. A success satisfies the
construction contract and all frozen item gates. Analysis is a two-sided exact
McNemar test on the 16 paired chunks, with the discordant table and exact
Clopper--Pearson intervals reported. At unadjusted alpha 0.05, at least six
one-directional discordances are required; a null result is not called equivalence.

Image perturbation is secondary and is named **paired perturbation sensitivity**
(`Delta_perm`), not a natural direct effect. It is the paired accuracy difference
between control and pixel-permuted images. It carries no mediation interpretation.
If the sensitivity-floor check fails, no `Delta_perm` estimate is reported.

## Construction arms

All arms use the same frozen chunks, decoding, seed, retry policy, output schema and
accounting.

1. `full`: pixels-only reader -> closed graph -> answer-first planner -> compiler ->
   sealed realizer -> gates.
2. `caption_mediated`: pixels-only free-text caption -> the same answer-first and
   realization path, with the caption in the structure slot.
3. `text_only`: OCR only; defines the degeneracy floor.
4. `text_assisted_reader`: reader receives pixels and OCR but must emit the same
   closed graph; this is the controlled isolation ablation.
5. `direct`: standard one-call text/layout/pixel MCQ baseline.
6. `gates_off`: deterministic re-scoring of `full`; no API call.

The planner and realizer are byte-identical for the two graph-reader arms. Only the
reader payload varies. `caption_mediated`, `text_only`, and `direct` are external
baselines and need not be falsely described as one-factor ablations.

## Secondary measurement

Before probing generated items, two answerer families must pass a fixed sensitivity
floor containing five positive controls and five text-sufficient negative controls.
Required separation is at least 8/10 for each family; control-replicate disagreement
must not exceed 10%. On failure, the probe is labeled insensitive and stops.

For each eligible item, frozen free-text answering is run under control PNG, a
control replicate, label permutation, block shuffle, and text-anchor removal.
Choice selection is secondary. Repeats are aggregated within item and never used as
sample-size inflation. `Delta_perm` uses paired item outcomes and exact paired or
document-clustered bootstrap intervals. Occlusion and image deletion are diagnostic
only and are never pooled with pixel permutations.

## Gates and judging

The lexical novelty statistic previously failed its discrimination criterion: no
threshold in 0.60--0.95 both rejected four of five known text-sufficient items and
preserved the protected fixture. This is reported as a gate-validity failure, not a
corpus property. It may be shown as a sensitivity curve but cannot silently decide
the primary endpoint. The confirmatory gate set therefore consists of source
support, single-best-answer, generic-cue, contract, and seal checks; lexical novelty
is exploratory.

Model-based judging is exploratory. Answerability is the only model-judge dimension
eligible for a secondary claim unless final agreement reaches quadratic-weighted
kappa >= 0.60. A fixed, arm-blinded sample of 40 generated items is selected
independently of release outcome, so an empty release set does not erase gate
precision evaluation.

### Fixed 40-of-80 judging frame

The judging candidate pool is exactly 80 items: the five item-generating arms
(`full`, `caption_mediated`, `text_only`, `text_assisted_reader`, `direct`) contribute
one candidate per census chunk, 5 x 16. `gates_off` is a deterministic re-scoring of
`full` and contributes no second candidate, so the pool is not inflated to 96.

From that pool, exactly 8 items per arm are retained, giving a frame of 40. Selection
is deterministic and outcome-independent: within each arm, candidates are ranked by
SHA-256 of the freeze identity, the arm name, and the item identifier, and the first
eight are taken. A second frozen hash sort removes arm blocks from the presentation
order so the judging sequence itself carries no arm signal.

Blinding precedes judging. The private frame retains routing fields for later
reconciliation; the frame handed to judges exposes only an opaque `judge_item_id` and
a payload commitment (`item_payload_sha256`), with no arm, chunk token, path, or
condition metadata. Two blinded model judges each rate all 40 items, so the judging
phase costs 80 calls.

Frame construction fails closed. A pool that is not exactly 80 with 16 per arm, a
duplicate or blank item identifier, an unexpected arm (including `gates_off`), or a
malformed freeze identity raises `BLOCKED_PROTOCOL:*` and no frame is emitted. The
frame is never re-drawn after outcomes are observed.

## Stopping and accounting

- Dry run is the default; HTTP requires `--execute`.
- Freeze manifest, image, graph/caption, prompt, runner and model-role hashes first.
- Run all 16 paired cells for the primary arms; no outcome-dependent re-prompting.
- If the full method yields fewer than four contract-parsed items, component probes
  are not interpreted because the prespecified discordance target is unreachable.
- Extraction is counted per declared image. Two blinded model judges each rate the
  fixed 40-item frame. Missing cells are itemized, never dropped or imputed.
- Transport errors, schema rejections, abstentions and blocked perturbations remain
  separate outcomes and all 16 chunks remain in the intention-to-test ledger.

### Study-level call accounting (550-call hard cap)

Accounting is study-level rather than per replacement freeze. Before the final
replacement freeze, the append-only ledgers recorded **96 actual HTTP attempts**.
These attempts remain spent regardless of transport outcome or local schema validity.

Offline cross-freeze verification covers 44 distinct one-call extraction tasks:
42 responses parse to usable interfaces and two are terminal `SCHEMA_REJECTED`
outcomes. All 44 are completed intention-to-test observations and are not called
again; downstream work that requires either rejected interface follows the frozen
missingness policy.

| Term | Calls |
| --- | ---: |
| Prior spent attempts | 96 |
| Gross frozen base plan | 486 |
| Verified completed extraction tasks | 44 |
| Remaining frozen base calls | 442 |
| Study base after verified reuse | 538 |
| Retry reserve constrained by hard cap | 12 |
| Study worst case | 550 |
| Hard operational cap | 550 |
| Remaining unallocated headroom | 0 |
| Replacement-freeze ledger cap (`550 - 96`) | 454 |

The 486-call gross plan decomposes into six role smokes, 54 per-image extraction
calls (18 declared images x 3 interfaces), 128 construction calls, 40
sensitivity-floor calls, at most 160 conditional secondary probes, 18 image audits,
and 80 blinded judging calls. The 44 verified extraction tasks reduce only the calls
still required; they do not erase historical expenditure or alter the scientific
plan. The six final-freeze smoke calls are part of the gross base plan and are carried
into the replacement-run ledger before paid phases.

Every attempt, retries included, is metered by an append-only ledger; a
`CALL_STARTED` record is itself a spent token, so a crashed call is never silently
reclaimed. The replacement-freeze ledger is capped at 454. Exceeding the cap fails
closed rather than degrading the design: the budget planner raises
`BLOCKED_BUDGET:study_worst_case_exceeds_cap:<total>/<cap>`, the ledger raises
`BLOCKED_BUDGET:http_cap_exhausted`, and the execution gate refuses to authorize a
freeze whose recorded worst case exceeds its recorded cap.

### Source freeze policy

Implementation integrity is frozen by discovery policy rather than a hand-maintained
list, so the freezable set does not have to be guessed before the code is complete.
Policy `ecm-tqag.source-discovery.v1` hashes `dry_run.py`, `run_smoke.py`, and every
`**/*.py` under the `ecm_tqag` package, excluding `__pycache__`, `.pytest_cache`,
`tests`, `runs`, and `.git`. Support modules that a hand list tends to omit, including
`ecm_tqag/io.py` and every package `__init__.py`, are inside the frozen surface and
are additionally pinned by a required-files floor.

Discovery is deterministic and sorted. It fails closed in both directions: a missing
required module raises `BLOCKED_FREEZE:source_discovery_incomplete`, an absent package
root raises `BLOCKED_FREEZE:source_package_missing`, and any file whose digest changes
or which appears after freezing blocks execution with `source_hash_drift`. Code that
was never hashed therefore cannot run under a frozen record.

## Model-role validity constraints

At least three model families are used. Answerers are distinct from the generator
and from each other. Automated image auditor and model judge are distinct from the
generator. Every record stores provider, model, prompt hash, decoding, call count,
retry count, token usage, input hash and timestamps. No identifier, path, split,
condition or development metadata enters a model prompt.

## Required release artifacts

`FREEZE_MANIFEST.json`, frozen graph and caption records, per-arm JSONL and summary,
prompt audit, call-parity report, gate ledger, sensitivity-floor result,
counterfactual images and probe records, primary statistics, fixed human-evaluation
sample, figure sources, manuscript source, PDF, checksums, and a clean source ZIP.
Any deviation is timestamped in `deviations.md` and reported.
