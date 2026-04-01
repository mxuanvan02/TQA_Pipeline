#!/usr/bin/env python3
"""
Quality gates for Stage 1-4 artifacts.

Gates:
- A: OCR/chunk coverage (raw PDF count vs markdown count)
- B: Valid context ratio
- C: Parse-valid QA ratio
- D: Duplicate QA rate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_qa(raw_qa_path: Path, qa_chunks_dir: Path | None) -> tuple[list[dict], str]:
    records: list[dict] = []

    if raw_qa_path.exists():
        data = _load_json(raw_qa_path)
        if isinstance(data, list):
            records.extend(x for x in data if isinstance(x, dict))
        if records:
            return records, "raw_qa_pairs"

    if qa_chunks_dir and qa_chunks_dir.exists():
        for p in sorted(qa_chunks_dir.glob("*.json")):
            try:
                data = _load_json(p)
                if isinstance(data, list):
                    records.extend(x for x in data if isinstance(x, dict))
                elif isinstance(data, dict):
                    records.append(data)
            except Exception:
                continue
        if records:
            return records, "qa_chunks_fallback"

    return records, "none"


def _norm_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _gate_report(
    raw_dir: Path,
    interim_dir: Path,
    contexts_path: Path,
    raw_qa_path: Path,
    qa_chunks_dir: Path | None,
) -> dict:
    raw_pdfs = list(raw_dir.glob("*.pdf")) + list(raw_dir.glob("*.PDF"))
    md_files = list(interim_dir.glob("*.md"))
    raw_pdf_stems = {p.stem for p in raw_pdfs}
    md_stems = {p.stem for p in md_files}
    matched_md_stems = raw_pdf_stems & md_stems
    extra_md_stems = md_stems - raw_pdf_stems
    missing_md_stems = raw_pdf_stems - md_stems

    contexts = _load_json(contexts_path) if contexts_path.exists() else []
    if not isinstance(contexts, list):
        contexts = []

    valid_contexts = 0
    for c in contexts:
        if not isinstance(c, dict):
            continue
        if str(c.get("doc_id", "")).strip() and str(c.get("chunk_id", "")).strip() and str(c.get("text", "")).strip():
            valid_contexts += 1

    qas, qa_source = _collect_qa(raw_qa_path, qa_chunks_dir)
    required = {"qa_id", "question_content", "candidate_answers", "ground_truth", "legal_rationale"}
    valid_qas = 0
    uniq = set()
    dup = 0
    for qa in qas:
        if required.issubset(set(qa.keys())) and str(qa.get("question_content", "")).strip():
            valid_qas += 1
        sig = (_norm_text(qa.get("question_content", "")), _norm_text(qa.get("ground_truth", "")))
        if sig in uniq:
            dup += 1
        else:
            uniq.add(sig)

    gate_a_coverage = (len(matched_md_stems) / len(raw_pdf_stems)) if raw_pdf_stems else 0.0
    gate_b_valid_context_ratio = (valid_contexts / len(contexts)) if contexts else 0.0
    gate_c_parse_valid_qa_ratio = (valid_qas / len(qas)) if qas else 0.0
    gate_d_duplicate_qa_rate = (dup / len(qas)) if qas else 0.0

    return {
        "counts": {
            "raw_pdfs": len(raw_pdfs),
            "interim_markdown_docs": len(md_files),
            "matched_markdown_docs": len(matched_md_stems),
            "extra_markdown_docs": len(extra_md_stems),
            "missing_markdown_docs": len(missing_md_stems),
            "contexts_total": len(contexts),
            "contexts_valid": valid_contexts,
            "qa_total": len(qas),
            "qa_valid": valid_qas,
            "qa_duplicates": dup,
            "qa_source": qa_source,
        },
        "gates": {
            "gate_a_ocr_chunk_coverage": gate_a_coverage,
            "gate_b_valid_context_ratio": gate_b_valid_context_ratio,
            "gate_c_parse_valid_qa_ratio": gate_c_parse_valid_qa_ratio,
            "gate_d_duplicate_qa_rate": gate_d_duplicate_qa_rate,
        },
        "diagnostics": {
            "extra_markdown_doc_stems": sorted(extra_md_stems),
            "missing_markdown_doc_stems": sorted(missing_md_stems),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quality gates for TQA pipeline artifacts.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--interim-dir", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--raw-qa", type=Path, required=True)
    parser.add_argument("--qa-chunks-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = _gate_report(
        raw_dir=args.raw_dir,
        interim_dir=args.interim_dir,
        contexts_path=args.contexts,
        raw_qa_path=args.raw_qa,
        qa_chunks_dir=args.qa_chunks_dir,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote quality gate report: {args.out}")
    print(json.dumps(report["gates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
