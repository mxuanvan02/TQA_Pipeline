# Prespecified protocol amendments

All amendments below were recorded before network execution and before any outcome
was observed. No amendment is outcome-dependent.

## Pre-execution call-accounting amendment

Before any network execution or outcome observation, exhaustive phase-level call
accounting replaced the construction-only estimate. The current-freeze plan contains
486 base calls: six role smokes, 54 per-image extraction calls, 128 construction
calls, 40 sensitivity-control calls, at most 160 conditional probes, 18 image audits,
and 80 blinded ratings. Every attempt, including a retry, is metered by an
append-only run ledger. The construction design, endpoints, analysis units, stopping
rules, and model-role validity constraints are unchanged.

## Study-level accounting supersedes the earlier 535 worst case

The earlier statement of a 49-call retry reserve and a 535-call worst case computed
the reserve on the 486 current-freeze base calls alone, which silently excluded the
six HTTP attempts already spent on prior role smokes. Those attempts were real and
must be carried, so accounting is now study-level:

- prior spent smoke attempts: 6
- current-freeze base calls: 486
- study base calls: 492
- retry reserve at 10% of study base, rounded up: 50
- study worst case: 542, against the unchanged hard cap of 550
- current-run ledger cap: 536, that is 542 minus the 6 already spent

The hard operational cap of 550 is unchanged; the change is that the study worst case
is stated as 542 rather than 535, and the current run is metered against 536 so the
study total cannot exceed 542. This amendment increases conservatism only. No phase
was enlarged and no scientific parameter changed.

## Fixed 40-of-80 judging frame made explicit

The protocol already required a fixed, arm-blinded 40-item judging sample selected
independently of release outcome. The frame is now fully specified rather than left
to run-time discretion. The candidate pool is exactly 80: the five item-generating
arms contribute one candidate per census chunk, 5 x 16. `gates_off` is a
deterministic re-scoring of `full` and contributes no second candidate, so the pool
is not 96. Exactly 8 items per arm are retained, giving 40.

Selection is deterministic and outcome-independent: SHA-256 rank over the freeze
identity, arm, and item identifier within each arm, followed by a frozen hash sort of
the presentation order so the judging sequence carries no arm signal. Blinding occurs
before judging; judges receive only an opaque `judge_item_id` and a payload
commitment. Frame construction fails closed on a pool that is not 5 x 16 = 80, on a
duplicate or blank item identifier, on an unexpected arm, or on a malformed freeze
identity. The judging phase cost is unchanged at 80 calls (40 items x 2 judge
families), and the budget planner now rejects any judge sample size other than the
fixed 40.

## Source integrity frozen by discovery policy

Freezing source hashes to a hand-maintained list risked omitting modules that were
still being written, and had already omitted `ecm_tqag/io.py`, `ecm_tqag/protocol.py`,
and every package `__init__.py`. Source integrity is now frozen by policy
`ecm-tqag.source-discovery.v1`: `dry_run.py`, `run_smoke.py`, and every `**/*.py`
under the `ecm_tqag` package, excluding `__pycache__`, `.pytest_cache`, `tests`,
`runs`, and `.git`. A required-files floor additionally pins `io.py`, `protocol.py`,
and the package initializers.

This is a strict widening of what is hashed. Discovery fails closed when a required
module is unreachable or a package root is absent, and the execution gate blocks on
any digest change or on a file that appears after freezing. Consequently, code that
was never hashed cannot execute under a frozen record.

## Pre-outcome bbox-coordinate and cumulative-accounting amendment

Before any construction endpoint or comparative study outcome was observed, two
fail-closed extraction runs showed that provider-side image resizing made pixel
coordinates unverifiable against local source dimensions. The reader contract was
therefore changed to a fixed 0--1000 full-image grid, converted deterministically to
the planner's [0,1] coordinates. Out-of-range or reversed boxes are rejected; no box
is clipped, repaired, or rescaled from an inferred provider resolution.

## Pre-outcome caption-schema enforcement amendment

A subsequent fail-closed extraction attempt returned 16 caption relations although
both the frozen prompt and local parser cap the interface at 12. No construction or
comparative endpoint had yet been observed. The request now supplies a strict JSON
Schema expressing the already frozen three-key caption contract and its 1--12
relation bound. The local parser remains authoritative and performs no truncation,
repair, or coercion. The eight attempts from that failed run and the six preceding
replacement-freeze smoke attempts remain in the cumulative ledger.

## Cumulative accounting after the caption amendment

All 40 HTTP attempts spent before this replacement freeze are carried forward. With
486 current-freeze base calls, the study base is 526. The unchanged hard cap of 550
therefore leaves a 24-call retry reserve; the nominal 53-call 10% reserve is reduced
rather than expanding the cap. No experimental phase or endpoint was enlarged.

## Pre-outcome cumulative accounting after verified cross-freeze import

Before construction or any comparative endpoint was observed, the append-only ledgers
were reconciled across all replacement freezes. The cumulative count is 96 actual HTTP
attempts: 40 attempts recorded before the caption-schema replacement, followed by six
role-smoke attempts and 25 extraction attempts under `paid_caption_v2`, six role-smoke
attempts under `smoke_import_v4`, and 19 further extraction attempts under
`paid_import_v4`. Attempts are counted even when the returned body fails local schema
validation.

A pure fail-closed importer independently verifies the origin freeze, request payload,
provider/model identity, task and image identity, response path and byte hash, and local
schema outcome. The two extraction ledgers jointly cover 44 distinct frozen extraction
tasks. Forty-two responses parse to usable interfaces; two are preserved as terminal
`SCHEMA_REJECTED` intent-to-treat outcomes (one out-of-grid graph and one overlong
caption). A rejected one-call task is not retried or converted into a successful record,
but downstream tasks that require its interface are marked unavailable under the frozen
missingness policy. Consequently, all 44 paid task calls are satisfied, the replacement
freeze has 486 gross base calls, 442 remaining base calls, and a study base of 538
(= 96 + 442). The unchanged hard cap of 550 permits at most 12 retry attempts, so the
replacement run ledger is capped at 454 (= 442 + 12). No scientific endpoint, arm,
sample, prompt contract, or stopping rule was changed.

## Post-execution reconciliation note

This note records accounting after the frozen run; it is not a protocol amendment.
Append-only ledgers contain 246 actual HTTP attempts across all replacement freezes
and smoke runs: 119 attempts before the final freeze, six final-freeze role smokes,
and 121 fresh paid-run attempts (81 construction and 40 sensitivity-control calls).
Every final paid-run attempt has one start and one terminal record, HTTP status 200,
and zero retries. Imported extraction and construction records were not billed again.
The cumulative total remained below the unchanged hard cap of 550.
