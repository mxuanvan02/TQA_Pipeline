#!/usr/bin/env python3
"""Deterministic tests for the visual-dependence criterion (v4 step 5).

No network. Every verdict path, every pairing guard, and the aggregate maths are
exercised from synthetic trial ledgers so a reviewer can replay the criterion
without an API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import visual_dependence as vd  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(name: str, cond: bool) -> None:
    RESULTS.append((bool(cond), name))


def trial(arm: str, correct: bool | None, fp: str = "FP") -> dict:
    return {"arm": arm, "item_fingerprint": fp,
            "image_attached": arm == vd.ARM_WITH, "is_correct": correct}


def pair(with_vals: list[bool | None], without_vals: list[bool | None],
         fp: str = "FP") -> list[dict]:
    return ([trial(vd.ARM_WITH, v, fp) for v in with_vals]
            + [trial(vd.ARM_WITHOUT, v, fp) for v in without_vals])


# ---------------------------------------------------------------- verdicts
v = vd.classify_item(pair([True, True, True], [False, False, False]))
check("correct w/ image + wrong w/o -> VISUAL_DEPENDENT",
      v["verdict"] == vd.VERDICT_DEPENDENT)
check("accuracy_with_image == 1.0", v["accuracy_with_image"] == 1.0)
check("accuracy_without_image == 0.0", v["accuracy_without_image"] == 0.0)

v = vd.classify_item(pair([True, True, True], [True, True, False]))
check("correct w/o image -> TEXT_SUFFICIENT",
      v["verdict"] == vd.VERDICT_TEXT_SUFFICIENT)

v = vd.classify_item(pair([False, False, True], [False, False, False]))
check("wrong even w/ image -> UNANSWERABLE",
      v["verdict"] == vd.VERDICT_UNANSWERABLE)

v = vd.classify_item(pair([True, False], [False, False]))
check("tie in with-arm -> INDETERMINATE",
      v["verdict"] == vd.VERDICT_INDETERMINATE)

v = vd.classify_item(pair([True, True], [True, False]))
check("tie in without-arm -> INDETERMINATE",
      v["verdict"] == vd.VERDICT_INDETERMINATE)

# Majority, not unanimity: 2/3 correct with image, 0/3 without -> dependent.
v = vd.classify_item(pair([True, True, False], [False, False, False]))
check("2/3 majority still VISUAL_DEPENDENT",
      v["verdict"] == vd.VERDICT_DEPENDENT)

# Failed trials are excluded from the vote, not counted as wrong.
v = vd.classify_item(pair([True, None, None], [False, None, None]))
check("failed trials excluded, not counted wrong",
      v["verdict"] == vd.VERDICT_DEPENDENT and v["n_failed_trials"] == 4)

v = vd.classify_item(pair([None, None], [False, False]))
check("all-failed arm -> INDETERMINATE",
      v["verdict"] == vd.VERDICT_INDETERMINATE)

# ---------------------------------------------------------------- guards
def raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


check("fingerprint mismatch raises", raises(lambda: vd.classify_item(
    [trial(vd.ARM_WITH, True, "A"), trial(vd.ARM_WITHOUT, False, "B")])))

check("missing arm raises", raises(lambda: vd.classify_item(
    [trial(vd.ARM_WITH, True), trial(vd.ARM_WITH, True)])))

check("no trials raises", raises(lambda: vd.classify_item([])))

bad_with = [{"arm": vd.ARM_WITH, "item_fingerprint": "FP",
             "image_attached": False, "is_correct": True},
            trial(vd.ARM_WITHOUT, False)]
check("with-arm lacking image raises", raises(lambda: vd.classify_item(bad_with)))

leak = [trial(vd.ARM_WITH, True),
        {"arm": vd.ARM_WITHOUT, "item_fingerprint": "FP",
         "image_attached": True, "is_correct": False}]
check("without-arm receiving image raises", raises(lambda: vd.classify_item(leak)))

# ---------------------------------------------------------------- aggregate
verdicts = [
    vd.classify_item(pair([True] * 3, [False] * 3, "F1")),
    vd.classify_item(pair([True] * 3, [False] * 3, "F2")),
    vd.classify_item(pair([True] * 3, [True] * 3, "F3")),
    vd.classify_item(pair([False] * 3, [False] * 3, "F4")),
]
agg = vd.aggregate(verdicts)
check("aggregate n_items_tested==4", agg["n_items_tested"] == 4)
check("aggregate n_visual_dependent==2", agg["n_visual_dependent"] == 2)
check("visual_dependence_rate==0.5", agg["visual_dependence_rate"] == 0.5)
check("denominator = all tested (not just passed)",
      agg["verdict_counts"].get(vd.VERDICT_TEXT_SUFFICIENT) == 1
      and agg["verdict_counts"].get(vd.VERDICT_UNANSWERABLE) == 1)
check("chance_rate == 0.25", agg["chance_rate"] == 0.25)
check("aggregate empty -> rate None", vd.aggregate([])["visual_dependence_rate"] is None)

# ---------------------------------------------------------------- selection
items = [
    {"question": "q1", "choices": ["a", "b", "c", "d"], "evidence_text": "t1"},
    {"question": "q2", "choices": ["a", "b", "c", "d"], "evidence_text": "t2"},
    {"question": "q3", "choices": ["a", "b", "c", "d"], "evidence_text": "t3"},
]
fps = [vd.item_fingerprint(i["question"], i["choices"], i["evidence_text"]) for i in items]
sel_verdicts = [
    vd.classify_item(pair([True] * 3, [False] * 3, fps[0])),   # dependent
    vd.classify_item(pair([True] * 3, [True] * 3, fps[1])),    # text sufficient
    vd.classify_item(pair([True] * 3, [False] * 3, fps[2])),   # dependent
]
kept = vd.accepted_items(sel_verdicts, items)
check("accepted_items keeps only dependent", len(kept) == 2)
check("accepted_items keeps the right ones",
      {k["question"] for k in kept} == {"q1", "q3"})

check("fingerprint ignores image, includes text",
      vd.item_fingerprint("q", ["a", "b", "c", "d"], "t1")
      != vd.item_fingerprint("q", ["a", "b", "c", "d"], "t2"))
check("fingerprint sensitive to choice order",
      vd.item_fingerprint("q", ["a", "b", "c", "d"], "t")
      != vd.item_fingerprint("q", ["b", "a", "c", "d"], "t"))

dup = items + [dict(items[0])]
check("duplicate item fingerprint raises",
      raises(lambda: vd.accepted_items(sel_verdicts, dup)))

# ---------------------------------------------------------------- report
passed = sum(1 for ok, _ in RESULTS if ok)
total = len(RESULTS)
print(f"=== {passed}/{total} {'PASS' if passed == total else 'FAIL'} ===\n")
for ok, name in RESULTS:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
sys.exit(0 if passed == total else 1)
