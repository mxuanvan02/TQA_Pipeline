#!/usr/bin/env python3
"""
Audit and conservatively clean the multimodal slice of the public release.

Goals:
- identify why current multimodal rows look low-quality,
- demote low-value visual rows back to text-only instead of deleting them, and
- emit a before/after report with explicit reasons.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INTERIM_CONTEXTS = ROOT / "data" / "output" / "interim" / "multimodal_contexts.json"
FULL_DATASET = ROOT / "data" / "output" / "processed" / "dataset.jsonl"
EVAL_READY_FILES = {
    "train": ROOT / "data" / "output" / "processed" / "eval_ready" / "train.jsonl",
    "dev": ROOT / "data" / "output" / "processed" / "eval_ready" / "dev.jsonl",
    "test": ROOT / "data" / "output" / "processed" / "eval_ready" / "test.jsonl",
}

VISUAL_CUE_RE = re.compile(
    r"\b(sơ đồ|bản đồ|biểu đồ|hình|hình vẽ|quy trình|cấu trúc|mô hình|giai đoạn|tòa|"
    r"quan hệ|trình tự|bộ máy|thủ tục|hệ thống)\b",
    re.IGNORECASE,
)
HARD_ARTIFACT_RE = re.compile(
    r"logo|publishing house|first part|second part|title of|legal text, titled|"
    r"\btập i\b|\btập ii\b|nhà xuất bản|lưu hành nội bộ|trường đại học",
    re.IGNORECASE,
)
DECORATIVE_RE = re.compile(
    r"courtroom setting|legal or judicial environment|bureaucraticive|books, suggesting|"
    r"\bcabinet\b|generic legal|stock",
    re.IGNORECASE,
)
MALFORMED_SUMMARY_RE = re.compile(r'^\{|"entities"\s*:|"relationships"\s*:', re.IGNORECASE)
PAGE_ZERO_RE = re.compile(r"_page_0_picture_", re.IGNORECASE)
STRUCTURAL_SUMMARY_RE = re.compile(
    r"diagram|map|chart|hierarch|flow|structure|stages|court|relationship|illustrates",
    re.IGNORECASE,
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _chunk_id_from_qa_id(qa_id: str) -> str:
    m = re.match(r"(.+_chunk_\d+)_(remember|understand|apply)_\d+$", qa_id)
    return m.group(1) if m else qa_id


def _joined_summary(ctx: dict[str, Any]) -> str:
    descs = ctx.get("visual_descriptions") or []
    return " | ".join(str(d.get("summary") or "") for d in descs)


def _joined_questions(rows: list[dict[str, Any]]) -> str:
    return " ".join(str(r.get("question_content") or "") for r in rows)


def classify_chunk(ctx: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_id = str(ctx.get("chunk_id") or rows[0].get("chunk_id") or "")
    context_text = str(ctx.get("text") or "")
    summary = _joined_summary(ctx)
    questions = _joined_questions(rows)
    visuals = ctx.get("visuals") or rows[0].get("visuals") or []
    visuals_text = " ".join(str(v) for v in visuals)

    positive_signals = []
    if VISUAL_CUE_RE.search(context_text):
        positive_signals.append("context_visual_cue")
    if VISUAL_CUE_RE.search(questions):
        positive_signals.append("question_visual_cue")
    if STRUCTURAL_SUMMARY_RE.search(summary):
        positive_signals.append("summary_structural")

    negative_reasons = []
    if PAGE_ZERO_RE.search(visuals_text):
        negative_reasons.append("page_zero_image")
    if HARD_ARTIFACT_RE.search(summary) or HARD_ARTIFACT_RE.search(questions) or HARD_ARTIFACT_RE.search(context_text):
        negative_reasons.append("cover_logo_or_front_matter")
    if DECORATIVE_RE.search(summary):
        negative_reasons.append("decorative_or_generic_visual")
    if MALFORMED_SUMMARY_RE.search(summary):
        negative_reasons.append("malformed_vlm_summary")

    keep = True
    decision_reason = "retain_structural_visual"

    if "cover_logo_or_front_matter" in negative_reasons:
        keep = False
        decision_reason = "drop_cover_logo_or_front_matter"
    elif "page_zero_image" in negative_reasons and "context_visual_cue" not in positive_signals:
        keep = False
        decision_reason = "drop_page_zero_non_structural_image"
    elif "decorative_or_generic_visual" in negative_reasons and "question_visual_cue" not in positive_signals:
        keep = False
        decision_reason = "drop_decorative_or_generic_visual"
    elif "summary_structural" not in positive_signals and "context_visual_cue" not in positive_signals:
        keep = False
        decision_reason = "drop_no_structural_visual_evidence"
    elif "malformed_vlm_summary" in negative_reasons and "question_visual_cue" not in positive_signals:
        keep = False
        decision_reason = "drop_malformed_summary_without_question_grounding"

    return {
        "chunk_id": chunk_id,
        "doc_id": str(ctx.get("doc_id") or rows[0].get("doc_id") or ""),
        "n_rows": len(rows),
        "visuals": visuals,
        "question_examples": [str(r.get("question_content") or "") for r in rows[:3]],
        "summary": summary,
        "positive_signals": positive_signals,
        "negative_reasons": negative_reasons,
        "keep_multimodal": keep,
        "decision_reason": decision_reason,
    }


def _demote_row(row: dict[str, Any]) -> dict[str, Any]:
    new_row = dict(row)
    new_row["is_multimodal"] = False
    new_row["visuals"] = []
    new_row["image_file_name"] = None
    new_row["image_file_names"] = []
    if isinstance(new_row.get("context_payload"), dict):
        cp = dict(new_row["context_payload"])
        cp["visuals"] = []
        new_row["context_payload"] = cp
    return new_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and clean multimodal release rows.")
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Rewrite processed JSONL files in place.")
    args = parser.parse_args()

    contexts = json.loads(INTERIM_CONTEXTS.read_text(encoding="utf-8"))
    ctx_by_chunk = {str(c.get("chunk_id")): c for c in contexts}

    eval_rows_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in EVAL_READY_FILES.values():
        for row in _iter_jsonl(path):
            if row.get("is_multimodal"):
                eval_rows_by_chunk[str(row.get("chunk_id"))].append(row)

    audits = []
    for chunk_id, rows in sorted(eval_rows_by_chunk.items()):
        ctx = ctx_by_chunk.get(chunk_id, {"chunk_id": chunk_id})
        audits.append(classify_chunk(ctx, rows))

    kept_chunks = {a["chunk_id"] for a in audits if a["keep_multimodal"]}
    dropped_chunks = {a["chunk_id"] for a in audits if not a["keep_multimodal"]}

    before_rows = sum(a["n_rows"] for a in audits)
    after_rows = sum(a["n_rows"] for a in audits if a["keep_multimodal"])
    before_page_zero = sum(1 for a in audits if any(PAGE_ZERO_RE.search(v) for v in a["visuals"]))
    after_page_zero = sum(
        1 for a in audits if a["keep_multimodal"] and any(PAGE_ZERO_RE.search(v) for v in a["visuals"])
    )
    before_question_visual = sum(
        a["n_rows"] for a in audits if "question_visual_cue" in a["positive_signals"]
    )
    after_question_visual = sum(
        a["n_rows"] for a in audits if a["keep_multimodal"] and "question_visual_cue" in a["positive_signals"]
    )

    report = {
        "before": {
            "multimodal_contexts": len(audits),
            "multimodal_rows": before_rows,
            "page_zero_contexts": before_page_zero,
            "question_visual_cue_rows": before_question_visual,
        },
        "after": {
            "multimodal_contexts": len(kept_chunks),
            "multimodal_rows": after_rows,
            "page_zero_contexts": after_page_zero,
            "question_visual_cue_rows": after_question_visual,
        },
        "dropped": {
            "contexts": len(dropped_chunks),
            "rows": before_rows - after_rows,
            "decision_reasons": Counter(a["decision_reason"] for a in audits if not a["keep_multimodal"]),
        },
        "kept": {
            "contexts": len(kept_chunks),
            "rows": after_rows,
            "decision_reasons": Counter(a["decision_reason"] for a in audits if a["keep_multimodal"]),
        },
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.details_out.parent.mkdir(parents=True, exist_ok=True)
    args.details_out.write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.apply:
        for split_name, path in EVAL_READY_FILES.items():
            rows = []
            changed = 0
            for row in _iter_jsonl(path):
                chunk_id = str(row.get("chunk_id"))
                if chunk_id in dropped_chunks and row.get("is_multimodal"):
                    rows.append(_demote_row(row))
                    changed += 1
                else:
                    rows.append(row)
            _write_jsonl(path, rows)
            print(f"[apply] {split_name}: demoted {changed} rows")

        full_rows = []
        full_changed = 0
        for row in _iter_jsonl(FULL_DATASET):
            chunk_id = _chunk_id_from_qa_id(str(row.get("qa_id") or ""))
            if chunk_id in dropped_chunks and row.get("is_multimodal"):
                full_rows.append(_demote_row(row))
                full_changed += 1
            else:
                full_rows.append(row)
        _write_jsonl(FULL_DATASET, full_rows)
        print(f"[apply] full: demoted {full_changed} rows")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
