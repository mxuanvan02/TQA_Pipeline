#!/usr/bin/env python3
"""
Prune orphan context artifacts whose doc_id is not present in raw PDF stems.

Targets:
- interim/multimodal_contexts.json
- interim/contexts/*.json
- interim/qa_chunks/*.json (records/files by orphan doc_id)
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip())


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune orphan context artifacts by raw PDF stems.")
    parser.add_argument("--data-base", type=Path, default=Path("data/output"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    interim = args.data_base / "interim"
    ctx_path = interim / "multimodal_contexts.json"
    contexts_dir = interim / "contexts"
    qa_chunks_dir = interim / "qa_chunks"

    if not ctx_path.exists():
        raise FileNotFoundError(f"Missing contexts file: {ctx_path}")

    raw_stems = {_norm(p.stem) for p in args.raw_dir.glob("*.pdf")} | {_norm(p.stem) for p in args.raw_dir.glob("*.PDF")}
    contexts = _load_json(ctx_path)
    if not isinstance(contexts, list):
        raise ValueError("multimodal_contexts.json must be a list")

    kept_contexts = []
    pruned_contexts = []
    for c in contexts:
        if not isinstance(c, dict):
            continue
        doc_id = _norm(str(c.get("doc_id", "")))
        if doc_id in raw_stems:
            kept_contexts.append(c)
        else:
            pruned_contexts.append(c)

    pruned_doc_ids = sorted({_norm(str(c.get("doc_id", ""))) for c in pruned_contexts if str(c.get("doc_id", "")).strip()})

    # Save pruned contexts snapshot for traceability
    snapshot_path = interim / "multimodal_contexts.orphan_backup.json"
    _save_json(snapshot_path, pruned_contexts)
    _save_json(ctx_path, kept_contexts)

    deleted_context_files = []
    if contexts_dir.exists():
        for p in sorted(contexts_dir.glob("*.json")):
            if _norm(p.stem) in pruned_doc_ids:
                p.unlink(missing_ok=True)
                deleted_context_files.append(str(p))

    # prune qa_chunks
    pruned_qa_records = 0
    deleted_qa_chunk_files = []
    if qa_chunks_dir.exists():
        for p in sorted(qa_chunks_dir.glob("*.json")):
            try:
                data = _load_json(p)
            except Exception:
                continue

            if isinstance(data, list):
                kept = []
                removed = 0
                for r in data:
                    doc_id = _norm(str(r.get("doc_id", ""))) if isinstance(r, dict) else ""
                    if doc_id in pruned_doc_ids:
                        removed += 1
                    else:
                        kept.append(r)
                if removed > 0:
                    pruned_qa_records += removed
                    if kept:
                        _save_json(p, kept)
                    else:
                        p.unlink(missing_ok=True)
                        deleted_qa_chunk_files.append(str(p))
            elif isinstance(data, dict):
                doc_id = _norm(str(data.get("doc_id", "")))
                if doc_id in pruned_doc_ids:
                    pruned_qa_records += 1
                    p.unlink(missing_ok=True)
                    deleted_qa_chunk_files.append(str(p))

    report = {
        "raw_pdf_count": len(raw_stems),
        "contexts_before": len(contexts),
        "contexts_after": len(kept_contexts),
        "pruned_contexts": len(pruned_contexts),
        "pruned_doc_ids": pruned_doc_ids,
        "backup_orphan_contexts": str(snapshot_path),
        "deleted_context_files_count": len(deleted_context_files),
        "deleted_qa_chunk_files_count": len(deleted_qa_chunk_files),
        "pruned_qa_records": pruned_qa_records,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    _save_json(args.report_out, report)
    print(f"[OK] Pruned orphan contexts. Report: {args.report_out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

