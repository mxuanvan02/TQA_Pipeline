"""Visual-dependence criterion for ECM-TQAG cross-modal items (v4, step 5).

WHY THIS EXISTS
---------------
`guard_plan_v3` + `verify_visual_anchors` prove that the PLANNER used the image:
a cross-modal motif cannot compile without a visual anchor, and that anchor must
name a declared image, commit to a localising bbox, and report a fact. That is a
statement about CONSTRUCTION.

It is NOT a statement about the QUESTION. A question can be built through an
image and still be answerable from prose alone -- in which case calling it
"multimodal" is unfalsifiable. This module supplies the missing half: an item
counts as visual-dependent only if an independent answerer NEEDS the image to
answer it.

DESIGN (pre-registered before any API call)
-------------------------------------------
Paired trials on the SAME frozen question, differing in exactly one factor:

    arm "with_image"     : evidence.text + document_structure + page image
    arm "without_image"  : evidence.text + document_structure           (image removed)

Everything else is held fixed: identical question string, identical choice
strings in identical order, identical text, identical decoding, same model, same
prompt template. The answerer returns only {"answer_index": N}; it never sees the
gold answer and never rewrites the item.

Per-item verdict (k repeats per arm, majority vote):

    VISUAL_DEPENDENT   correct with image  AND  incorrect without image
    TEXT_SUFFICIENT    correct without image            -> image not needed
    UNANSWERABLE       incorrect with image             -> item is broken, not
                                                          evidence about vision
    INDETERMINATE      arms disagree across repeats without a majority

Only VISUAL_DEPENDENT items enter the reported TLV set. TEXT_SUFFICIENT items are
NOT deleted from the ledger -- they are retained and reported, because the ratio
    n_visual_dependent / n_cross_modal_parsed
is itself a headline result about how often a cross-modal construction actually
yields a vision-requiring question.

WHY MAJORITY VOTE OVER k REPEATS
--------------------------------
A single trial at temperature 0 is still one sample of a stochastic system
(provider-side nondeterminism, image re-encoding). One flipped trial would
otherwise promote or demote an item. k=3 with a strict majority makes a verdict
require agreement, and `INDETERMINATE` keeps unstable items out of the claim
rather than silently rounding them in.

GUARD AGAINST THE OBVIOUS FAILURE MODE
--------------------------------------
An item whose choices are guessable (e.g. one plausible option, three absurd
ones) can be "correct without image" for reasons unrelated to text sufficiency,
and an item can be "correct with image" by luck. `chance_rate` is therefore
reported alongside, and `assert_pairing_valid()` refuses to score any pair whose
question/choices/text are not byte-identical across arms -- if the pairing is
broken, the comparison is meaningless and must fail loudly, not quietly.

No function here calls a model: the runner supplies observed answers, so every
verdict is deterministic and replayable from the ledger without network access.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

SCHEMA = "ecm-tqag.visual-dependence.v1"

ARM_WITH = "with_image"
ARM_WITHOUT = "without_image"
ARMS = (ARM_WITH, ARM_WITHOUT)

VERDICT_DEPENDENT = "VISUAL_DEPENDENT"
VERDICT_TEXT_SUFFICIENT = "TEXT_SUFFICIENT"
VERDICT_UNANSWERABLE = "UNANSWERABLE"
VERDICT_INDETERMINATE = "INDETERMINATE"

N_CHOICES = 4
CHANCE_RATE = 1.0 / N_CHOICES

# Repeats per arm. Odd so a strict majority always exists for a 2-outcome
# (correct / incorrect) vote when every repeat is observed.
DEFAULT_REPEATS = 3


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def item_fingerprint(question: str, choices: list[str], evidence_text: str) -> str:
    """Identity of everything that MUST be identical across the two arms.

    The image is deliberately excluded -- it is the manipulated factor.
    """
    return sha256_text(canonical({
        "question": question, "choices": choices, "evidence_text": evidence_text,
    }))


def assert_pairing_valid(trials: list[dict[str, Any]]) -> None:
    """Refuse to score unless the pair differs ONLY in image presence.

    A silent pairing break (different text, reordered choices, a re-generated
    question) would turn this experiment into noise while still producing
    plausible-looking numbers. It must therefore raise, not warn.
    """
    if not trials:
        raise ValueError("BLOCKED_PAIRING:no_trials")
    fingerprints = {t["item_fingerprint"] for t in trials}
    if len(fingerprints) != 1:
        raise ValueError(f"BLOCKED_PAIRING:fingerprint_mismatch:{sorted(fingerprints)}")
    arms = {t["arm"] for t in trials}
    if arms != set(ARMS):
        raise ValueError(f"BLOCKED_PAIRING:arms_present={sorted(arms)}")
    for t in trials:
        if t["arm"] == ARM_WITH and not t.get("image_attached"):
            raise ValueError("BLOCKED_PAIRING:with_image_arm_has_no_image")
        if t["arm"] == ARM_WITHOUT and t.get("image_attached"):
            raise ValueError("BLOCKED_PAIRING:without_image_arm_received_image")


def _majority_correct(trials: list[dict[str, Any]]) -> bool | None:
    """Majority correctness over repeats of ONE arm. None when no majority.

    Trials whose answer was never obtained (transport/contract failure) are
    excluded from the vote rather than counted as wrong: a provider error is not
    evidence that the model could not answer.
    """
    votes = [t["is_correct"] for t in trials if t.get("is_correct") is not None]
    if not votes:
        return None
    counts = Counter(votes)
    if counts[True] * 2 > len(votes):
        return True
    if counts[False] * 2 > len(votes):
        return False
    return None


def classify_item(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign one visual-dependence verdict to one frozen item.

    `trials` must contain both arms for the SAME item; each trial carries
    {arm, item_fingerprint, image_attached, is_correct}. `is_correct` may be None
    for a failed trial.
    """
    assert_pairing_valid(trials)
    with_arm = [t for t in trials if t["arm"] == ARM_WITH]
    without_arm = [t for t in trials if t["arm"] == ARM_WITHOUT]

    correct_with = _majority_correct(with_arm)
    correct_without = _majority_correct(without_arm)

    if correct_with is None or correct_without is None:
        verdict = VERDICT_INDETERMINATE
    elif not correct_with:
        # The item cannot be answered even WITH the image: this is a defective
        # item, and reporting it as "needs the image" would be an artefact.
        verdict = VERDICT_UNANSWERABLE
    elif correct_without:
        verdict = VERDICT_TEXT_SUFFICIENT
    else:
        verdict = VERDICT_DEPENDENT

    def _rate(arm_trials: list[dict[str, Any]]) -> float | None:
        got = [t["is_correct"] for t in arm_trials if t.get("is_correct") is not None]
        return round(sum(got) / len(got), 4) if got else None

    return {
        "verdict": verdict,
        "majority_correct_with_image": correct_with,
        "majority_correct_without_image": correct_without,
        "accuracy_with_image": _rate(with_arm),
        "accuracy_without_image": _rate(without_arm),
        "n_trials_with_image": len(with_arm),
        "n_trials_without_image": len(without_arm),
        "n_failed_trials": sum(1 for t in trials if t.get("is_correct") is None),
        "item_fingerprint": trials[0]["item_fingerprint"],
    }


def aggregate(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Corpus-level summary. The headline number is `visual_dependence_rate`.

    Denominator is every item submitted for testing (i.e. every cross-modal
    PARSED item), NOT only the ones that passed -- otherwise the rate would be
    trivially 1.0 and would claim nothing.
    """
    counts = Counter(v["verdict"] for v in verdicts)
    n = len(verdicts)
    dependent = counts[VERDICT_DEPENDENT]

    accs_with = [v["accuracy_with_image"] for v in verdicts
                 if v["accuracy_with_image"] is not None]
    accs_without = [v["accuracy_without_image"] for v in verdicts
                    if v["accuracy_without_image"] is not None]

    return {
        "schema": SCHEMA,
        "n_items_tested": n,
        "verdict_counts": dict(counts),
        "n_visual_dependent": dependent,
        "visual_dependence_rate": round(dependent / n, 4) if n else None,
        "mean_accuracy_with_image": round(sum(accs_with) / len(accs_with), 4) if accs_with else None,
        "mean_accuracy_without_image": round(sum(accs_without) / len(accs_without), 4) if accs_without else None,
        "chance_rate": CHANCE_RATE,
        "interpretation": (
            "visual_dependence_rate is the fraction of cross-modal PARSED items for "
            "which an independent answerer was correct WITH the page image and "
            "incorrect WITHOUT it, under identical question, choices and text. It "
            "measures whether the image is REQUIRED to answer, which construction-side "
            "checks (motif modality contract, bbox verification) cannot establish. "
            "It is a model-based operationalisation of visual necessity for this "
            "answerer, not a claim about human respondents."
        ),
    }


def accepted_items(verdicts: list[dict[str, Any]],
                   items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The reported TLV set: items whose verdict is VISUAL_DEPENDENT.

    Pairing is by item_fingerprint, so a mis-ordered verdict list cannot silently
    admit the wrong items.
    """
    keep = {v["item_fingerprint"] for v in verdicts
            if v["verdict"] == VERDICT_DEPENDENT}
    by_fp: dict[str, dict[str, Any]] = {}
    for it in items:
        fp = item_fingerprint(it["question"], it["choices"], it["evidence_text"])
        if fp in by_fp:
            raise ValueError(f"BLOCKED_PAIRING:duplicate_item_fingerprint:{fp[:16]}")
        by_fp[fp] = it
    missing = keep - set(by_fp)
    if missing:
        raise ValueError(f"BLOCKED_PAIRING:verdict_without_item:{sorted(missing)[:1]}")
    return [by_fp[fp] for fp in by_fp if fp in keep]
