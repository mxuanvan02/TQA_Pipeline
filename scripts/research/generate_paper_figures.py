#!/usr/bin/env python3
"""Generate paper-ready figures from current release artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "research" / "artifacts"
RESULTS = ROOT / "research" / "results"
FIGURES = ROOT / "research" / "paper" / "figures"
FULL_RELEASE = ROOT / "data" / "output" / "processed" / "dataset.jsonl"
CLEAN_BENCHMARK = ROOT / "data" / "output" / "processed" / "dataset_eval_ready.clean.jsonl"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "axes.facecolor": "#fbfbfb",
            "figure.facecolor": "white",
        }
    )


def plot_release_funnel() -> Path:
    stats = _load_json(ARTIFACTS / "dataset_stats.json")
    split_meta = _load_json(ARTIFACTS / "doc_splits.json")
    clean_rows = _load_jsonl(CLEAN_BENCHMARK)
    full_rows = _load_jsonl(FULL_RELEASE)

    stages = [
        ("Raw QA\n(Stage 3)", stats["scale"]["n_raw_qa_pairs"]),
        ("Filtered QA\n(Stage 4)", stats["scale"]["n_filtered_qa_pairs"]),
        ("Public full\nrelease", len(full_rows)),
        ("Clean eval-ready\nbenchmark", len(clean_rows)),
    ]
    labels = [label for label, _ in stages]
    counts = [count for _, count in stages]
    colors = ["#0f4c81", "#2d728f", "#5aa469", "#e39b2d"]

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bars = ax.barh(labels, counts, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Number of QA pairs")
    ax.set_title("Release funnel from generated QA to benchmark-safe rows")

    raw_total = counts[0]
    for i, (bar, count) in enumerate(zip(bars, counts)):
        label = f"{count:,}"
        if i > 0:
            label += f"  ({count / counts[i - 1] * 100:.1f}% of previous)"
        else:
            label += "  (100.0%)"
        ax.text(count + 120, bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=8)

    ax.set_xlim(0, max(counts) * 1.28)
    fig.text(
        0.5,
        0.01,
        f"Source side: {split_meta['meta']['total_docs']} textbooks -> {split_meta['meta']['total_contexts']:,} contexts before QA generation.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = FIGURES / "release_funnel.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_split_balance() -> Path:
    rows = _load_jsonl(CLEAN_BENCHMARK)
    order = ["train", "dev", "test"]
    blooms = ["Remember", "Understand", "Apply"]
    colors = {"Remember": "#1f6aa5", "Understand": "#4aa382", "Apply": "#e39b2d"}

    split_counts = {split: Counter() for split in order}
    totals = Counter()
    for row in rows:
        split = str(row.get("split", "unknown"))
        bloom = str(row.get("bloom_level", "Unknown"))
        if split in split_counts and bloom in colors:
            split_counts[split][bloom] += 1
            totals[split] += 1

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    bottom = [0, 0, 0]
    for bloom in blooms:
        values = [split_counts[split][bloom] for split in order]
        ax.bar(order, values, bottom=bottom, color=colors[bloom], label=bloom, width=0.65)
        bottom = [b + v for b, v in zip(bottom, values)]

    for idx, split in enumerate(order):
        ax.text(idx, totals[split] + 120, f"{totals[split]:,}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Number of benchmark QA pairs")
    ax.set_xlabel("Split")
    ax.set_title("Bloom balance remains stable across train/dev/test")
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="upper right")
    ax.set_ylim(0, max(totals.values()) * 1.12)
    fig.tight_layout()
    out = FIGURES / "split_balance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_source_coverage_curve() -> Path:
    rows = _load_jsonl(CLEAN_BENCHMARK)
    counts = Counter(str(row.get("doc_id", "unknown")).strip() or "unknown" for row in rows)
    ordered = sorted(counts.values(), reverse=True)
    cumulative = pd.Series(ordered).cumsum() / len(rows) * 100
    x = list(range(1, len(ordered) + 1))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.plot(x, cumulative, color="#0f4c81", linewidth=2.2)
    ax.scatter([10, 20, len(ordered)], [cumulative.iloc[9], cumulative.iloc[19], cumulative.iloc[-1]],
               color="#e39b2d", zorder=3, s=24)
    ax.axvline(10, color="#cccccc", linestyle="--", linewidth=0.9)
    ax.axvline(20, color="#cccccc", linestyle="--", linewidth=0.9)
    ax.text(10.3, cumulative.iloc[9] - 5.5, f"Top 10 books: {cumulative.iloc[9]:.1f}%", fontsize=8, color="#444444")
    ax.text(20.3, cumulative.iloc[19] - 5.5, f"Top 20 books: {cumulative.iloc[19]:.1f}%", fontsize=8, color="#444444")
    ax.set_xlim(1, len(ordered))
    ax.set_ylim(0, 103)
    ax.set_xlabel("Number of source textbooks (sorted by QA count)")
    ax.set_ylabel("Cumulative share of clean benchmark (%)")
    ax.set_title("Benchmark coverage is broad but still shows a long-tail source distribution")
    fig.tight_layout()
    out = FIGURES / "source_coverage_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_answer_position_bias() -> Path:
    rows = _load_jsonl(CLEAN_BENCHMARK)
    labels = ["A", "B", "C", "D"]
    before_counts = Counter()
    after_counts = Counter()
    for row in rows:
        meta = row.get("eval_ready_meta", {})
        source_idx = meta.get("source_gold_index")
        if isinstance(source_idx, int) and 0 <= source_idx < 4:
            before_counts[labels[source_idx]] += 1
        gold = str(row.get("gold_letter", "")).strip()
        if gold in labels:
            after_counts[gold] += 1
    before = [before_counts.get(k, 0) for k in labels]
    after = [after_counts.get(k, 0) for k in labels]
    before_total = sum(before)
    after_total = sum(after)
    before_pct = [100 * x / before_total for x in before]
    after_pct = [100 * x / after_total for x in after]

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar([i - width / 2 for i in x], before_pct, width=width, label="Before standardization", color="#b94e48")
    ax.bar([i + width / 2 for i in x], after_pct, width=width, label="Released eval-ready", color="#3a7d44")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of gold answers (%)")
    ax.set_xlabel("Correct option position")
    ax.set_title("Answer-position bias before standardization and in the released eval-ready subset")
    ax.legend(frameon=False, fontsize=8)
    for i, (b, a) in enumerate(zip(before_pct, after_pct)):
        ax.text(i - width / 2, b + 1.0, f"{b:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, a + 1.0, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, max(max(before_pct), max(after_pct)) + 8)
    fig.tight_layout()
    out = FIGURES / "answer_position_bias.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _apply_style()
    outputs = [
        plot_release_funnel(),
        plot_split_balance(),
        plot_source_coverage_curve(),
        plot_answer_position_bias(),
    ]
    for output in outputs:
        print(f"[OK] Wrote figure: {output}")


if __name__ == "__main__":
    main()
