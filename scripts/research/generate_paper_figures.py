#!/usr/bin/env python3
"""
Generate paper-ready figures from research artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "research" / "artifacts"
RESULTS = ROOT / "research" / "results"
FIGURES = ROOT / "research" / "paper" / "figures"


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


def plot_bloom_distribution() -> Path:
    df = pd.read_csv(ARTIFACTS / "bloom_dist.csv")
    order = ["Remember", "Understand", "Apply"]
    df["bloom_level"] = pd.Categorical(df["bloom_level"], categories=order, ordered=True)
    df = df.sort_values("bloom_level")

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    bars = ax.bar(df["bloom_level"], df["count"], color=["#1f6aa5", "#4aa382", "#e39b2d"])
    ax.set_ylabel("Number of QA pairs")
    ax.set_xlabel("Bloom level")
    ax.set_title("Bloom-level distribution in the final dataset")
    for bar, count, pct in zip(bars, df["count"], df["pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60, f"{int(count):,}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = FIGURES / "bloom_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_top_domains() -> Path:
    df = pd.read_csv(ARTIFACTS / "domain_dist.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bars = ax.barh(df["domain"], df["count"], color="#2d728f")
    ax.set_xlabel("Number of QA pairs")
    ax.set_ylabel("Domain")
    ax.set_title("Top-10 legal domains in the final dataset")
    for bar, count in zip(bars, df["count"]):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2, f"{int(count):,}",
                va="center", ha="left", fontsize=8)
    fig.tight_layout()
    out = FIGURES / "top_domains.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_answer_position_bias() -> Path:
    report = json.loads((RESULTS / "eval_ready_dataset_report.json").read_text(encoding="utf-8"))
    labels = ["A", "B", "C", "D"]
    before = [report["gold_position_before"].get(k, 0) for k in labels]
    after = [report["gold_position_after"].get(k, 0) for k in labels]
    before_total = sum(before)
    after_total = sum(after)
    before_pct = [100 * x / before_total for x in before]
    after_pct = [100 * x / after_total for x in after]

    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar([i - width / 2 for i in x], before_pct, width=width, label="Before standardization", color="#b94e48")
    ax.bar([i + width / 2 for i in x], after_pct, width=width, label="After standardization", color="#3a7d44")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of gold answers (%)")
    ax.set_xlabel("Correct option position")
    ax.set_title("Answer-position bias before and after standardization")
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
        plot_bloom_distribution(),
        plot_top_domains(),
        plot_answer_position_bias(),
    ]
    for output in outputs:
        print(f"[OK] Wrote figure: {output}")


if __name__ == "__main__":
    main()
