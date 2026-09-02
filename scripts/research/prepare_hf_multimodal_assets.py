#!/usr/bin/env python3
"""
Prepare Hugging Face-friendly image references for multimodal rows.

This script does two things:
1. rewrites the public JSONL files with relative image references via
   `image_file_name` and `image_file_names`, and
2. exports a true image-dataset layout for the eval-ready subset under
   `processed/eval_ready_hf/{train,validation,test}/metadata.jsonl`, with
   split-local `images/` folders and a `file_name` column that Hugging Face
   can auto-cast to an image feature in the Viewer.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "output" / "processed"
IMAGES_OUT = PROCESSED / "eval_ready" / "images"
HF_EVAL_READY_OUT = PROCESSED / "eval_ready_hf"
HF_FULL_OUT = PROCESSED / "full_hf" / "data"
PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9s6nXfoAAAAASUVORK5CYII="
)
COLAB_PREFIXES = (
    "/content/TQA_Pipeline/data/interim/images/",
    "/content/TQA_Pipeline/data/output/interim/images/",
)
LOCAL_PREFIXES = (
    str((ROOT / "data" / "output" / "interim" / "images").resolve()) + "/",
    str((ROOT / "data" / "interim" / "images").resolve()) + "/",
    str((ROOT / "data" / "TQA_Pipeline_Backup" / "interim" / "images").resolve()) + "/",
)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _resolve_source_image(raw_path: str) -> tuple[Path, Path]:
    raw = str(raw_path)
    for prefix in COLAB_PREFIXES:
        if raw.startswith(prefix):
            rel = Path(raw[len(prefix):])
            for local_prefix in LOCAL_PREFIXES:
                candidate = Path(local_prefix) / rel
                if candidate.exists():
                    return candidate, rel
    p = Path(raw)
    if p.exists():
        for local_prefix in LOCAL_PREFIXES:
            try:
                rel = p.resolve().relative_to(Path(local_prefix).resolve())
                return p, rel
            except Exception:
                continue
    raise FileNotFoundError(raw)


def _convert_paths_for_file(rows: list[dict], metadata_path: Path, copied: dict[Path, Path]) -> tuple[list[dict], int]:
    updated_rows: list[dict] = []
    changed = 0
    for row in rows:
        visuals = row.get("visuals") or row.get("context_payload", {}).get("visuals") or []
        rel_paths: list[str] = []
        for visual in visuals:
            _, rel = _resolve_source_image(str(visual))
            repo_image_path = IMAGES_OUT / rel
            rel_paths.append(os.path.relpath(repo_image_path, start=metadata_path.parent).replace("\\", "/"))
        new_row = dict(row)
        first = rel_paths[0] if rel_paths else None
        if new_row.get("image_file_name") != first or new_row.get("image_file_names") != rel_paths:
            changed += 1
        new_row["image_file_name"] = first
        new_row["image_file_names"] = rel_paths
        updated_rows.append(new_row)
    return updated_rows, changed


def _copy_split_image(raw_path: str, split_dir: Path) -> str:
    src, rel = _resolve_source_image(raw_path)
    dst = split_dir / "images" / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return os.path.relpath(dst, start=split_dir).replace("\\", "/")


def _ensure_placeholder(split_dir: Path) -> str:
    dst = split_dir / "images" / "_placeholder.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_bytes(PLACEHOLDER_PNG)
    return os.path.relpath(dst, start=split_dir).replace("\\", "/")


def _export_hf_metadata(rows: list[dict], split_dir: Path) -> dict[str, int]:
    hf_rows: list[dict] = []
    multimodal_rows = 0
    copied_images: set[str] = set()
    placeholder_rel = _ensure_placeholder(split_dir)

    for row in rows:
        visuals = row.get("visuals") or row.get("context_payload", {}).get("visuals") or []
        rel_paths = [_copy_split_image(str(visual), split_dir) for visual in visuals]

        new_row = dict(row)
        if isinstance(new_row.get("context_payload"), dict):
            new_context_payload = dict(new_row["context_payload"])
            new_context_payload.pop("visuals", None)
            new_row["context_payload"] = new_context_payload
        new_row.pop("visuals", None)
        new_row.pop("image_file_name", None)
        new_row.pop("image_file_names", None)
        new_row["file_name"] = rel_paths[0] if rel_paths else placeholder_rel
        hf_rows.append(new_row)

        if rel_paths:
            multimodal_rows += 1
            copied_images.update(rel_paths)
        else:
            copied_images.add(placeholder_rel)

    _write_jsonl(split_dir / "metadata.jsonl", hf_rows)
    return {
        "rows": len(hf_rows),
        "multimodal_rows": multimodal_rows,
        "images_copied": len(copied_images),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HF multimodal image assets.")
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            PROCESSED / "dataset.jsonl",
            PROCESSED / "eval_ready" / "train.jsonl",
            PROCESSED / "eval_ready" / "dev.jsonl",
            PROCESSED / "eval_ready" / "test.jsonl",
        ],
    )
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    IMAGES_OUT.mkdir(parents=True, exist_ok=True)
    copied: dict[Path, Path] = {}
    file_reports: dict[str, dict] = {}

    # First pass: discover and copy unique images.
    all_visuals: set[str] = set()
    for metadata_path in args.files:
        for row in _iter_jsonl(metadata_path):
            visuals = row.get("visuals") or row.get("context_payload", {}).get("visuals") or []
            for visual in visuals:
                all_visuals.add(str(visual))

    for visual in sorted(all_visuals):
        src, rel = _resolve_source_image(visual)
        dst = IMAGES_OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        copied[src] = dst

    # Second pass: rewrite metadata files with relative image fields.
    for metadata_path in args.files:
        rows = list(_iter_jsonl(metadata_path))
        updated_rows, changed = _convert_paths_for_file(rows, metadata_path, copied)
        _write_jsonl(metadata_path, updated_rows)
        mm_rows = sum(1 for row in updated_rows if row.get("image_file_name"))
        file_reports[str(metadata_path.relative_to(ROOT))] = {
            "rows": len(updated_rows),
            "multimodal_rows": mm_rows,
            "rows_touched": changed,
        }

    report = {
        "images_copied": len(copied),
        "images_dir": str(IMAGES_OUT.relative_to(ROOT)),
        "files": file_reports,
    }

    split_source_map = {
        "train": PROCESSED / "eval_ready" / "train.jsonl",
        "validation": PROCESSED / "eval_ready" / "dev.jsonl",
        "test": PROCESSED / "eval_ready" / "test.jsonl",
    }
    hf_layout_report: dict[str, dict[str, int | str]] = {}
    for split_name, source_path in split_source_map.items():
        rows = list(_iter_jsonl(source_path))
        split_dir = HF_EVAL_READY_OUT / split_name
        split_stats = _export_hf_metadata(rows, split_dir)
        hf_layout_report[split_name] = {
            "source": str(source_path.relative_to(ROOT)),
            "metadata": str((split_dir / "metadata.jsonl").relative_to(ROOT)),
            **split_stats,
        }
    report["hf_eval_ready_layout"] = hf_layout_report

    full_rows = list(_iter_jsonl(PROCESSED / "dataset.jsonl"))
    full_stats = _export_hf_metadata(full_rows, HF_FULL_OUT)
    report["hf_full_layout"] = {
        "source": str((PROCESSED / "dataset.jsonl").relative_to(ROOT)),
        "metadata": str((HF_FULL_OUT / "metadata.jsonl").relative_to(ROOT)),
        **full_stats,
    }

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
