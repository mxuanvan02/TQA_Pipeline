#!/usr/bin/env python3
"""Statistical comparison for ECM-TQAG work24 blinded-judge scores.

Design:
  * Unit of analysis = chunk (block). Complete blocks only (all 9 cells present).
  * Primary score = mean of the two blinded judges (sonnet5, terra) per cell,
    reported per rubric dimension plus a composite mean.
  * Method comparison: Friedman test across 3 methods (scores averaged over the
    3 conditions within each chunk), post-hoc Wilcoxon signed-rank with Holm
    correction, Cliff's delta effect size, bootstrap 95% CI of paired median diff.
  * Condition comparison: same procedure across 3 conditions.
  * No human anchor is used; every number here is model-based and is labelled as
    descriptive agreement, not semantic/legal validity.
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import stats

R = Path("/media/SAS/Van/DeTai2026/TQA_Pipeline/research/work_ecm24")
DIMS = ["faithfulness", "answerability", "distractor_quality", "depth"]
METHODS = ["direct", "answer_first", "ecm"]
CONDS = ["T", "TL_struct", "TLV"]
JUDGES = {"sonnet5": "runs/judge_full_sonnet5.jsonl",
          "terra": "runs/judge_full_terra.jsonl"}
BOOT_N = 10000
SEED = 20260806


def load_scores() -> dict:
    """cell key -> {dim: [judge scores]} plus metadata."""
    cells: dict[tuple[str, str, str], dict] = {}
    for jid, rel in JUDGES.items():
        for line in (R / rel).read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "SCORED":
                continue
            sc = row.get("scores") or {}
            if not all(isinstance(sc.get(d), int) for d in DIMS):
                continue
            key = (row["chunk_id"], row["method"], row["condition"])
            rec = cells.setdefault(key, {d: {} for d in DIMS})
            for d in DIMS:
                rec[d][jid] = sc[d]
    # keep only cells scored by both judges on all dims
    out = {}
    for key, rec in cells.items():
        if all(len(rec[d]) == len(JUDGES) for d in DIMS):
            out[key] = {d: mean(rec[d].values()) for d in DIMS}
            out[key]["composite"] = mean(out[key][d] for d in DIMS)
    return out


def complete_blocks(cells: dict) -> list[str]:
    chunks = {k[0] for k in cells}
    ok = []
    for c in sorted(chunks):
        if all((c, m, d) in cells for m in METHODS for d in CONDS):
            ok.append(c)
    return ok


def cliffs_delta(a: list[float], b: list[float]) -> float:
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def boot_ci_median_diff(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float]:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    meds = []
    for _ in range(BOOT_N):
        samp = [diffs[rng.randrange(n)] for _ in range(n)]
        meds.append(float(np.median(samp)))
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adj[k] = running
    return adj


def analyse(cells: dict, blocks: list[str], factor: str, levels: list[str],
            other_levels: list[str], dim: str, rng: random.Random) -> dict:
    """factor='method' averages over conditions; factor='condition' averages over methods."""
    per_level: dict[str, list[float]] = {}
    for lv in levels:
        vals = []
        for c in blocks:
            if factor == "method":
                keys = [(c, lv, o) for o in other_levels]
            else:
                keys = [(c, o, lv) for o in other_levels]
            vals.append(mean(cells[k][dim] for k in keys))
        per_level[lv] = vals

    fr_stat, fr_p = stats.friedmanchisquare(*[per_level[lv] for lv in levels])
    n = len(blocks)
    k = len(levels)
    kendall_w = float(fr_stat / (n * (k - 1))) if n and k > 1 else None

    raw_p: dict[str, float] = {}
    pair_stats: dict[str, dict] = {}
    for a, b in itertools.combinations(levels, 2):
        pa, pb = per_level[a], per_level[b]
        if all(abs(x - y) < 1e-12 for x, y in zip(pa, pb)):
            w_p, w_stat = 1.0, 0.0
        else:
            w_stat, w_p = stats.wilcoxon(pa, pb, zero_method="wilcox")
        tag = f"{a}_vs_{b}"
        raw_p[tag] = float(w_p)
        lo, hi = boot_ci_median_diff(pa, pb, rng)
        pair_stats[tag] = {
            "wilcoxon_stat": float(w_stat),
            "p_raw": float(w_p),
            "median_diff": float(np.median([x - y for x, y in zip(pa, pb)])),
            "boot_ci95_median_diff": [lo, hi],
            "cliffs_delta": cliffs_delta(pa, pb),
            "mean_a": mean(pa),
            "mean_b": mean(pb),
        }
    adj = holm(raw_p)
    for tag in pair_stats:
        pair_stats[tag]["p_holm"] = adj[tag]
        pair_stats[tag]["sig_holm_0.05"] = adj[tag] < 0.05

    return {
        "factor": factor,
        "dimension": dim,
        "n_blocks": n,
        "level_means": {lv: mean(v) for lv, v in per_level.items()},
        "level_medians": {lv: float(np.median(v)) for lv, v in per_level.items()},
        "friedman_stat": float(fr_stat),
        "friedman_p": float(fr_p),
        "kendall_w": kendall_w,
        "pairwise": pair_stats,
        "raw_values": {lv: [round(x, 4) for x in v] for lv, v in per_level.items()},
    }


def main() -> None:
    rng = random.Random(SEED)
    cells = load_scores()
    blocks = complete_blocks(cells)
    dims_all = DIMS + ["composite"]

    result = {
        "schema": "ecm24.stats.v1",
        "generator_model": "qwen/qwen3-vl-32b-instruct",
        "judges": list(JUDGES),
        "judge_note": "Model-based scores only. Descriptive agreement, NOT semantic/legal validity. No human anchor.",
        "n_cells_both_judges": len(cells),
        "n_complete_blocks": len(blocks),
        "complete_block_chunks": blocks,
        "bootstrap_iterations": BOOT_N,
        "seed": SEED,
        "method_tests": {},
        "condition_tests": {},
    }
    for dim in dims_all:
        result["method_tests"][dim] = analyse(cells, blocks, "method", METHODS, CONDS, dim, rng)
        result["condition_tests"][dim] = analyse(cells, blocks, "condition", CONDS, METHODS, dim, rng)

    out = R / "audit" / "stats_tests.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    # console report
    print(f"n cells (both judges) = {len(cells)} | complete blocks (chunks) = {len(blocks)}")
    for factor_key, label, levels in [("method_tests", "METHOD", METHODS),
                                      ("condition_tests", "CONDITION", CONDS)]:
        print(f"\n{'='*74}\n{label}  (Friedman + Wilcoxon/Holm, unit = chunk, n={len(blocks)})\n{'='*74}")
        print(f"{'dim':<20}" + "".join(f"{lv:>13}" for lv in levels) + f"{'chi2':>9}{'p':>9}{'W':>7}")
        for dim in dims_all:
            r = result[factor_key][dim]
            row = f"{dim:<20}" + "".join(f"{r['level_means'][lv]:>13.3f}" for lv in levels)
            print(row + f"{r['friedman_stat']:>9.2f}{r['friedman_p']:>9.4f}{r['kendall_w']:>7.2f}")
        print("\n  post-hoc (Holm-adjusted):")
        for dim in dims_all:
            r = result[factor_key][dim]
            if r["friedman_p"] >= 0.05:
                print(f"  {dim:<20} Friedman n.s. -> post-hoc not interpreted")
                continue
            for tag, ps in r["pairwise"].items():
                mark = "*" if ps["sig_holm_0.05"] else " "
                print(f"  {dim:<20} {tag:<28} p_holm={ps['p_holm']:.4f}{mark} "
                      f"delta={ps['cliffs_delta']:+.3f} meddiff={ps['median_diff']:+.3f} "
                      f"CI[{ps['boot_ci95_median_diff'][0]:+.3f},{ps['boot_ci95_median_diff'][1]:+.3f}]")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
