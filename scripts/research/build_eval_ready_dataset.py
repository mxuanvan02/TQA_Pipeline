#!/usr/bin/env python3
"""
Build an evaluation-ready MCQ dataset from the processed JSONL release.

Goals:
- Keep only 4-option records with resolvable ground truth.
- Normalize candidate texts by removing option labels and trailing notes.
- Rebalance the final gold-answer position across A/B/C/D deterministically.
- Emit a machine-readable report for paper experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mcq_utils import canonicalize_text, resolve_ground_truth_index, strip_choice_label, strip_trailing_notes

_DOMAIN_MAP: dict[str, str] = {
    "4": "History of Law",
    "10": "Legislative Drafting",
    "11": "Legal Theory",
    "12": "Legal Theory",
    "13": "Civil Law I",
    "14": "Civil Law II",
    "15": "Commercial Law I",
    "16": "Commercial Law II",
    "17": "Constitutional Law VN",
    "18": "Constitutional Law VN",
    "19": "Comparative Constitutional",
    "20": "Comparative Law",
    "21": "Administrative Law",
    "22": "Admin. Procedure",
    "23": "Environmental Law",
    "24": "Civil Procedure",
    "25": "Family Law",
    "26": "Criminal Law (General)",
    "27": "Criminal Law (Specific)",
    "28": "Criminal Procedure",
    "29": "Tax / Budget Law",
    "30": "Banking Law",
    "31": "International Law",
    "32": "International Law",
    "33": "Private International Law",
    "34": "Intl. Commercial Law",
    "36": "Intl. Economic Law",
    "37": "Land Law",
    "38": "Labor Law",
    "39": "IP Law",
    "40": "Forensic Psychology",
    "41": "Competition Law",
    "42": "Civil Registration",
    "43": "Legal Practice",
    "46": "Criminology",
    "47": "Criminal Qualification",
    "48": "Social Security Law",
    "49": "Notarization Law",
    "50": "Inheritance Law",
    "51": "Real Estate Law",
    "52": "Land Dispute Law",
    "53": "Contract Skills",
}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _clean_option(text: str) -> str:
    cleaned = strip_trailing_notes(strip_choice_label(str(text)))
    return " ".join(cleaned.split())


def _rebalance_options(candidates: list[str], gold_index: int, target_index: int, rng: random.Random) -> tuple[list[str], int]:
    order = list(range(len(candidates)))
    rng.shuffle(order)
    shuffled = [candidates[i] for i in order]
    current_gold = order.index(gold_index)
    shuffled[current_gold], shuffled[target_index] = shuffled[target_index], shuffled[current_gold]
    return shuffled, target_index


def _parse_qa_id(qa_id: str) -> tuple[str, str]:
    qa_id = str(qa_id)
    if "_chunk_" not in qa_id:
        return qa_id, qa_id
    left, right = qa_id.split("_chunk_", 1)
    chunk_num = right.split("_", 1)[0]
    chunk_id = f"{left}_chunk_{chunk_num}"
    return left, chunk_id


def _load_manifest_map(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {}
    out: dict[str, dict] = {}
    for row in data:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id", "")).strip()
        if chunk_id:
            out[chunk_id] = row
    return out


def _infer_domain(record: dict) -> str:
    domain_tag = str(record.get("domain_tag", "")).strip()
    if domain_tag and domain_tag not in {"civil_law", "unknown", "Unknown"}:
        return domain_tag

    qa_id = str(record.get("qa_id", ""))
    match = re.match(r"^(\d+)", qa_id)
    if match:
        return _DOMAIN_MAP.get(match.group(1), "Other")
    return "Other"


def build_eval_ready(records: list[dict], seed: int, manifest_map: dict[str, dict] | None = None) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    target_positions = [0, 1, 2, 3]
    target_cursor = 0
    manifest_map = manifest_map or {}

    cleaned_rows: list[dict] = []
    reasons = Counter()
    before_pos = Counter()
    after_pos = Counter()
    by_bloom = defaultdict(lambda: {"total": 0, "multimodal": 0})
    by_domain = Counter()
    by_split = Counter()

    for record in records:
        resolution = resolve_ground_truth_index(record.get("candidate_answers", []), str(record.get("ground_truth", "")))
        matched_index = resolution["matched_index"]
        candidate_count = resolution["candidate_count"]

        if candidate_count != 4:
            reasons["not_4_options"] += 1
            continue
        if matched_index is None or matched_index < 0 or matched_index > 3:
            reasons[resolution["method"]] += 1
            continue

        raw_candidates = [str(x) for x in record["candidate_answers"]]
        cleaned_candidates = [_clean_option(x) for x in raw_candidates]
        if any(not x for x in cleaned_candidates):
            reasons["empty_cleaned_option"] += 1
            continue

        canonical_set = [canonicalize_text(x) for x in cleaned_candidates]
        if len(set(canonical_set)) != 4:
            reasons["duplicate_cleaned_options"] += 1
            continue

        gold_text = cleaned_candidates[matched_index]
        before_pos[chr(65 + matched_index)] += 1

        target_index = target_positions[target_cursor % 4]
        target_cursor += 1
        balanced_candidates, balanced_gold_index = _rebalance_options(cleaned_candidates, matched_index, target_index, rng)
        balanced_gold_text = balanced_candidates[balanced_gold_index]
        after_pos[chr(65 + balanced_gold_index)] += 1

        doc_id, chunk_id = _parse_qa_id(str(record.get("qa_id", "")))
        manifest_row = manifest_map.get(chunk_id, {})
        split = str(manifest_row.get("split", "")).strip() or str(record.get("split", "")).strip() or "unknown"

        row = dict(record)
        row["candidate_answers_raw"] = raw_candidates
        row["ground_truth_raw"] = str(record.get("ground_truth", ""))
        row["candidate_answers"] = balanced_candidates
        row["ground_truth"] = balanced_gold_text
        row["gold_index"] = balanced_gold_index
        row["gold_letter"] = chr(65 + balanced_gold_index)
        row["doc_id"] = str(manifest_row.get("doc_id", "")).strip() or doc_id
        row["chunk_id"] = chunk_id
        row["split"] = split
        row["eval_ready"] = True
        row["eval_ready_meta"] = {
            "source_match_method": resolution["method"],
            "source_gold_index": matched_index,
            "position_rebalanced": True,
            "seed": seed,
        }
        cleaned_rows.append(row)

        bloom = str(record.get("bloom_level", "Unknown"))
        by_bloom[bloom]["total"] += 1
        if bool(record.get("is_multimodal", False)):
            by_bloom[bloom]["multimodal"] += 1
        by_domain[_infer_domain(record)] += 1
        by_split[split] += 1

    report = {
        "seed": seed,
        "input_records": len(records),
        "output_records": len(cleaned_rows),
        "retention_ratio": round(len(cleaned_rows) / len(records), 4) if records else 0.0,
        "drop_reasons": dict(reasons),
        "gold_position_before": dict(before_pos),
        "gold_position_after": dict(after_pos),
        "by_split": dict(by_split),
        "by_bloom": {k: v for k, v in sorted(by_bloom.items())},
        "top_domains": dict(by_domain.most_common(10)),
    }
    return cleaned_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation-ready dataset from processed JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output eval-ready dataset.jsonl")
    parser.add_argument("--report-out", type=Path, required=True, help="Output JSON report")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional context manifest for split assignment")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = _load_jsonl(args.input)
    manifest_map = _load_manifest_map(args.manifest)
    cleaned_rows, report = build_eval_ready(records, seed=args.seed, manifest_map=manifest_map)

    _write_jsonl(args.out, cleaned_rows)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote eval-ready dataset: {args.out}")
    print(f"[OK] Wrote eval-ready report : {args.report_out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
