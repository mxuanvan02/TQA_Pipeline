#!/usr/bin/env python3
"""
Audit coverage across pipeline artifacts:
raw PDFs -> interim markdown/contexts -> qa_chunks -> evaluated -> dataset.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pipeline coverage and completeness.")
    parser.add_argument("--data-base", type=Path, default=Path("data/output"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    interim = args.data_base / "interim"
    processed = args.data_base / "processed"

    raw_pdfs = sorted([p for p in args.raw_dir.glob("*.pdf")] + [p for p in args.raw_dir.glob("*.PDF")])
    raw_pdf_stems = {_norm(p.stem) for p in raw_pdfs}
    md_files = sorted(interim.glob("*.md"))

    ctx_path = interim / "multimodal_contexts.json"
    contexts = _load_json(ctx_path) if ctx_path.exists() else []
    contexts = contexts if isinstance(contexts, list) else []

    expected_chunk_set = {_norm(str(c.get("chunk_id", ""))) for c in contexts if str(c.get("chunk_id", "")).strip()}
    expected_chunk_set.discard("")
    expected_docs = {_norm(str(c.get("doc_id", ""))) for c in contexts if str(c.get("doc_id", "")).strip()}
    expected_docs.discard("")

    qa_chunk_files = sorted((interim / "qa_chunks").glob("*.json"))
    qa_records = []
    qa_chunk_ids = []
    for f in qa_chunk_files:
        try:
            d = _load_json(f)
        except Exception:
            continue
        if isinstance(d, list):
            for x in d:
                if isinstance(x, dict):
                    qa_records.append(x)
                    cid = _norm(str(x.get("chunk_id", "")))
                    if cid:
                        qa_chunk_ids.append(cid)
        elif isinstance(d, dict):
            qa_records.append(d)
            cid = _norm(str(d.get("chunk_id", "")))
            if cid:
                qa_chunk_ids.append(cid)
    qa_chunk_id_set = set(qa_chunk_ids)

    missing_chunk_ids = sorted(expected_chunk_set - qa_chunk_id_set)
    unknown_chunk_ids = sorted(qa_chunk_id_set - expected_chunk_set)

    exp_doc_chunks: dict[str, set[str]] = defaultdict(set)
    for c in contexts:
        d = _norm(str(c.get("doc_id", "")))
        cid = _norm(str(c.get("chunk_id", "")))
        if d and cid:
            exp_doc_chunks[d].add(cid)

    act_doc_chunks: dict[str, set[str]] = defaultdict(set)
    for r in qa_records:
        d = _norm(str(r.get("doc_id", "")))
        cid = _norm(str(r.get("chunk_id", "")))
        if d and cid:
            act_doc_chunks[d].add(cid)

    per_doc = []
    for d in sorted(exp_doc_chunks):
        exp = len(exp_doc_chunks[d])
        act = len(act_doc_chunks.get(d, set()))
        per_doc.append(
            {
                "doc_id": d,
                "expected_chunks": exp,
                "qa_chunks_present": act,
                "coverage": (act / exp) if exp else 0.0,
            }
        )
    per_doc_sorted = sorted(per_doc, key=lambda x: x["coverage"])

    raw_qa = _load_json(interim / "raw_qa_pairs.json") if (interim / "raw_qa_pairs.json").exists() else []
    raw_qa = raw_qa if isinstance(raw_qa, list) else []
    filtered = _load_json(interim / "filtered_qa_pairs.json") if (interim / "filtered_qa_pairs.json").exists() else []
    filtered = filtered if isinstance(filtered, list) else []
    eval_files = list((interim / "evaluated_qa").glob("*.json")) if (interim / "evaluated_qa").exists() else []

    dataset_path = processed / "dataset.jsonl"
    dataset_n = 0
    if dataset_path.exists():
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dataset_n += 1

    qa_id_counter = Counter(_norm(str(r.get("qa_id", ""))) for r in qa_records if str(r.get("qa_id", "")).strip())
    qa_id_dup = sum(v - 1 for v in qa_id_counter.values() if v > 1)
    sig_counter = Counter(
        (
            " ".join(str(r.get("question_content", "")).lower().split()),
            " ".join(str(r.get("ground_truth", "")).lower().split()),
        )
        for r in qa_records
    )
    sig_dup = sum(v - 1 for v in sig_counter.values() if v > 1)

    docs_in_context_not_in_raw = sorted(expected_docs - raw_pdf_stems)
    raw_not_in_context = sorted(raw_pdf_stems - expected_docs)

    report = {
        "counts": {
            "raw_pdfs": len(raw_pdfs),
            "interim_mds": len(md_files),
            "contexts_total": len(contexts),
            "context_docs": len(expected_docs),
            "context_chunks": len(expected_chunk_set),
            "qa_chunk_files": len(qa_chunk_files),
            "qa_records": len(qa_records),
            "qa_unique_chunks": len(qa_chunk_id_set),
            "raw_qa_pairs": len(raw_qa),
            "filtered_qa_pairs": len(filtered),
            "evaluated_qa_files": len(eval_files),
            "dataset_jsonl_records": dataset_n,
        },
        "coverage": {
            "qa_chunk_over_context_chunk_ratio": (len(qa_chunk_id_set) / len(expected_chunk_set)) if expected_chunk_set else 0.0,
            "missing_context_chunks_without_qa": len(missing_chunk_ids),
            "unknown_qa_chunks_not_in_contexts": len(unknown_chunk_ids),
        },
        "consistency": {
            "qa_id_duplicates": qa_id_dup,
            "question_ground_truth_duplicates": sig_dup,
            "docs_in_contexts_not_in_raw_pdfs": len(docs_in_context_not_in_raw),
            "raw_pdfs_without_context_docs": len(raw_not_in_context),
        },
        "samples": {
            "missing_chunk_ids_head": missing_chunk_ids[:30],
            "unknown_chunk_ids_head": unknown_chunk_ids[:30],
            "docs_in_contexts_not_in_raw_pdfs": docs_in_context_not_in_raw[:30],
            "raw_pdfs_without_context_docs": raw_not_in_context[:30],
            "lowest_doc_coverage_head": per_doc_sorted[:15],
            "highest_doc_coverage_head": per_doc_sorted[-15:],
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote coverage audit: {args.out}")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

