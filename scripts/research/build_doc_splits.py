#!/usr/bin/env python3
"""
Build document-level train/dev/test splits for multimodal legal QA contexts.

Key properties:
- No document leakage across splits.
- Approximate stratification by multimodal ratio and sample volume.
- Reproducible with a fixed seed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_doc_stats(contexts: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for ctx in contexts:
        grouped[str(ctx.get("doc_id", "unknown_doc"))].append(ctx)

    stats: dict[str, dict] = {}
    for doc_id, items in grouped.items():
        multimodal_count = sum(1 for x in items if bool(x.get("is_multimodal", False)))
        text_lens = [len(str(x.get("text", ""))) for x in items]
        stats[doc_id] = {
            "doc_id": doc_id,
            "n_contexts": len(items),
            "n_multimodal": multimodal_count,
            "multimodal_ratio": (multimodal_count / len(items)) if items else 0.0,
            "avg_text_len": (sum(text_lens) / len(text_lens)) if text_lens else 0.0,
        }
    return stats


def _choose_split(
    split_state: dict[str, dict],
    target_counts: dict[str, int],
    doc_n: int,
    doc_mm: int,
) -> str:
    # First prefer splits still under target, by largest remaining quota.
    deficits = []
    for name, state in split_state.items():
        deficit = target_counts[name] - state["n"]
        deficits.append((deficit, name))
    deficits.sort(reverse=True)
    if deficits[0][0] > 0:
        return deficits[0][1]

    # If all targets exceeded, place in split with smallest absolute overflow after add.
    best_name = "train"
    best_score = float("inf")
    for name, state in split_state.items():
        overflow = (state["n"] + doc_n) - target_counts[name]
        mm_balance = abs((state["mm"] + doc_mm) / max(1, state["n"] + doc_n))
        score = max(0, overflow) + mm_balance * 0.01
        if score < best_score:
            best_score = score
            best_name = name
    return best_name


def build_splits(contexts: list[dict], seed: int, train_ratio: float, dev_ratio: float) -> dict:
    doc_stats = _build_doc_stats(contexts)
    docs = list(doc_stats.keys())
    rng = random.Random(seed)
    rng.shuffle(docs)

    # Place larger docs first to reduce later balancing drift.
    docs.sort(key=lambda d: doc_stats[d]["n_contexts"], reverse=True)

    total_n = sum(doc_stats[d]["n_contexts"] for d in docs)
    total_mm = sum(doc_stats[d]["n_multimodal"] for d in docs)

    targets = {"train": train_ratio, "dev": dev_ratio, "test": 1.0 - train_ratio - dev_ratio}
    target_counts = {k: int(total_n * v) for k, v in targets.items()}
    # Keep exact total by assigning residual to train.
    assigned = sum(target_counts.values())
    if assigned < total_n:
        target_counts["train"] += total_n - assigned
    split_state = {
        "train": {"docs": [], "n": 0, "mm": 0},
        "dev": {"docs": [], "n": 0, "mm": 0},
        "test": {"docs": [], "n": 0, "mm": 0},
    }

    for doc_id in docs:
        d = doc_stats[doc_id]
        chosen = _choose_split(
            split_state=split_state,
            target_counts=target_counts,
            doc_n=d["n_contexts"],
            doc_mm=d["n_multimodal"],
        )
        split_state[chosen]["docs"].append(doc_id)
        split_state[chosen]["n"] += d["n_contexts"]
        split_state[chosen]["mm"] += d["n_multimodal"]

    # Build chunk-level manifest
    doc_to_split = {}
    for split_name, info in split_state.items():
        for doc in info["docs"]:
            doc_to_split[doc] = split_name

    manifest = []
    for ctx in contexts:
        doc_id = str(ctx.get("doc_id", "unknown_doc"))
        manifest.append(
            {
                "chunk_id": str(ctx.get("chunk_id", "")),
                "doc_id": doc_id,
                "is_multimodal": bool(ctx.get("is_multimodal", False)),
                "split": doc_to_split.get(doc_id, "train"),
            }
        )

    summary = {}
    for split_name in ("train", "dev", "test"):
        n = split_state[split_name]["n"]
        mm = split_state[split_name]["mm"]
        summary[split_name] = {
            "n_docs": len(split_state[split_name]["docs"]),
            "n_contexts": n,
            "n_multimodal": mm,
            "context_ratio": (n / total_n) if total_n else 0.0,
            "multimodal_ratio": (mm / n) if n else 0.0,
        }

    return {
        "meta": {
            "seed": seed,
            "train_ratio": train_ratio,
            "dev_ratio": dev_ratio,
            "test_ratio": 1.0 - train_ratio - dev_ratio,
            "total_contexts": total_n,
            "total_multimodal_contexts": total_mm,
            "total_docs": len(docs),
        },
        "doc_splits": {k: v["docs"] for k, v in split_state.items()},
        "summary": summary,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document-level splits for contexts JSON.")
    parser.add_argument("--contexts", type=Path, required=True, help="Path to multimodal_contexts.json")
    parser.add_argument("--out", type=Path, required=True, help="Output split JSON path")
    parser.add_argument("--manifest-out", type=Path, default=None, help="Optional manifest JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.dev_ratio <= 0 or (args.train_ratio + args.dev_ratio) >= 1:
        raise ValueError("Invalid split ratios: require train>0, dev>0, train+dev<1")

    contexts = _load_json(args.contexts)
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("Contexts file must be a non-empty JSON list")

    result = build_splits(
        contexts=contexts,
        seed=args.seed,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "manifest"}, f, ensure_ascii=False, indent=2)

    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_out, "w", encoding="utf-8") as f:
            json.dump(result["manifest"], f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote split summary to: {args.out}")
    if args.manifest_out:
        print(f"[OK] Wrote manifest to: {args.manifest_out}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
