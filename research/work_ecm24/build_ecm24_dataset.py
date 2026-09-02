#!/usr/bin/env python3
"""Build a SELF-CONTAINED 24-chunk multimodal ECM dataset in an isolated working dir.

Why this script exists
----------------------
The 8-chunk dataset (`research/artifacts/ecm_inputs_8chunks_v3.json`) is a subset
of the real multimodal pool. The pool declares image paths under a stale
`/content/...` (Colab) prefix, so paths must be re-resolved against local disk by
basename before anything can be claimed about availability.

This script:
  1. reads the frozen multimodal pool,
  2. selects every chunk with `is_multimodal == true`,
  3. re-resolves each declared image to a real local file (basename index),
  4. applies EXPLICIT, RECORDED screening rules (no silent drops),
  5. copies surviving images into the working dir (self-contained, hashed),
  6. emits `ecm_inputs_24chunks_v1.json` using the SAME evidence contract as
     `ecm_inputs_8chunks_v3.json`: T / TL_struct / TLV.

It never contacts a network service and never fabricates content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/media/SAS/Van/DeTai2026/TQA_Pipeline")
POOL = ROOT / "data/output/interim/multimodal_contexts.json"
LEGACY_INPUTS = ROOT / "research/artifacts/ecm_inputs_8chunks_v3.json"
IMAGE_SEARCH_ROOTS = [
    ROOT / "data/output/interim/images",   # used by the legacy 8-chunk dataset
    ROOT / "data/images",                  # bulk extraction
]
IMAGE_EXT = {".jpeg", ".jpg", ".png", ".webp"}
CONDITIONS = ("T", "TL_struct", "TLV")

# ---- screening thresholds (recorded in the audit, not hidden in prose) -------
MIN_TEXT_CHARS = 200        # a chunk shorter than this cannot ground a 4-choice MCQ
MIN_IMAGE_PIXELS = 64       # min(width, height) below this carries no diagram content
MIN_IMAGE_BYTES = 2048      # guards against truncated / placeholder extractions
COVER_PAGE_PATTERN = re.compile(r"_page_0_", re.IGNORECASE)  # front-cover artwork


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    norm = unicodedata.normalize("NFKD", value)
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    norm = re.sub(r"[^A-Za-z0-9]+", "_", norm).strip("_")
    return norm[:80] or "chunk"


def build_image_index() -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for base in IMAGE_SEARCH_ROOTS:
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXT:
                index.setdefault(path.name, []).append(path)
    return index


def resolve_image(declared: str, index: dict[str, list[Path]]) -> tuple[Path | None, str]:
    """Resolve a declared (possibly stale) image path to a local file.

    Preference order: exact path -> parent-dir + basename match -> basename match.
    Returns (path, how_resolved).
    """
    declared_path = Path(declared)
    if declared_path.is_file():
        return declared_path, "exact_path"
    candidates = index.get(declared_path.name, [])
    if not candidates:
        return None, "unresolved"
    parent = declared_path.parent.name
    same_parent = [c for c in candidates if c.parent.name == parent]
    if len(same_parent) >= 1:
        return sorted(same_parent)[0], "basename+parent"
    return sorted(candidates)[0], "basename_only"


def image_dims(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image  # noqa: PLC0415
        with Image.open(path) as handle:
            return handle.size
    except Exception:  # noqa: BLE001 - unreadable image is a screening outcome
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the isolated 24-chunk multimodal ECM dataset.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--force", action="store_true", help="rebuild even if dataset already exists")
    args = parser.parse_args()

    work = args.out
    dataset_path = work / "ecm_inputs_24chunks_v1.json"
    if dataset_path.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite {dataset_path}; pass --force to rebuild.")

    images_dir = work / "images"
    audit_dir = work / "audit"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    if not POOL.is_file():
        raise SystemExit(f"Missing frozen pool: {POOL}")
    pool_sha = sha256_file(POOL)
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    if not isinstance(pool, list):
        raise SystemExit("Unexpected pool shape (expected a list of chunk records).")

    multimodal = [c for c in pool if isinstance(c, dict) and c.get("is_multimodal") is True]

    legacy_chunks: set[str] = set()
    if LEGACY_INPUTS.is_file():
        legacy = json.loads(LEGACY_INPUTS.read_text(encoding="utf-8")).get("packages", [])
        legacy_chunks = {p["chunk_id"] for p in legacy if isinstance(p, dict) and p.get("chunk_id")}

    index = build_image_index()
    audit_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    seen_text: dict[str, str] = {}
    seen_image_sha: dict[str, str] = {}

    for chunk in sorted(multimodal, key=lambda c: str(c.get("chunk_id"))):
        chunk_id = str(chunk.get("chunk_id"))
        doc_id = str(chunk.get("doc_id"))
        raw_text = chunk.get("text")
        text: str = raw_text if isinstance(raw_text, str) else ""
        raw_images = chunk.get("image_paths")
        declared_images: list[Any] = raw_images if isinstance(raw_images, list) else []
        row: dict[str, Any] = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "in_legacy_8chunk": chunk_id in legacy_chunks,
            "text_chars": len(text),
            "declared_image_count": len(declared_images),
            "images": [],
            "decision": "ACCEPT",
            "reasons": [],
        }

        if len(text) < MIN_TEXT_CHARS:
            row["decision"] = "REJECT"
            row["reasons"].append(f"text_too_short:{len(text)}<{MIN_TEXT_CHARS}")
        if not declared_images:
            row["decision"] = "REJECT"
            row["reasons"].append("no_declared_image")

        text_sha = sha256_text(re.sub(r"\s+", " ", text).strip())
        if text_sha in seen_text and row["decision"] == "ACCEPT":
            row["decision"] = "REJECT"
            row["reasons"].append(f"duplicate_text_of:{seen_text[text_sha]}")

        resolved: list[dict[str, Any]] = []
        for order, declared in enumerate(declared_images, 1):
            entry: dict[str, Any] = {"declared_order": order, "declared_path": str(declared)}
            path, how = resolve_image(str(declared), index)
            entry["resolution"] = how
            if path is None:
                entry["status"] = "UNRESOLVED"
                row["reasons"].append(f"image_unresolved:{Path(str(declared)).name}")
                row["decision"] = "REJECT"
                row["images"].append(entry)
                continue
            size = path.stat().st_size
            dims = image_dims(path)
            sha = sha256_file(path)
            entry.update({
                "local_path": str(path), "bytes": size, "sha256": sha,
                "dimensions": list(dims) if dims else None,
            })
            problems: list[str] = []
            if dims is None:
                problems.append("unreadable_image")
            elif min(dims) < MIN_IMAGE_PIXELS:
                problems.append(f"image_too_small:{dims[0]}x{dims[1]}")
            if size < MIN_IMAGE_BYTES:
                problems.append(f"image_bytes_too_small:{size}")
            if COVER_PAGE_PATTERN.search(path.name):
                problems.append("front_cover_artwork_not_content_figure")
            if sha in seen_image_sha:
                problems.append(f"duplicate_image_of:{seen_image_sha[sha]}")
            entry["problems"] = problems
            entry["status"] = "OK" if not problems else "SCREENED_OUT"
            row["images"].append(entry)
            if not problems:
                resolved.append(entry)

        if row["decision"] == "ACCEPT" and not resolved:
            row["decision"] = "REJECT"
            row["reasons"].append("no_usable_image_after_screening")

        if row["decision"] == "ACCEPT":
            seen_text[text_sha] = chunk_id
            copied: list[dict[str, Any]] = []
            for new_order, entry in enumerate(resolved, 1):
                src = Path(entry["local_path"])
                target = images_dir / f"{slug(chunk_id)}_{new_order}{src.suffix.lower()}"
                shutil.copyfile(src, target)
                if sha256_file(target) != entry["sha256"]:
                    raise SystemExit(f"Copy integrity failure for {src}")
                seen_image_sha[entry["sha256"]] = chunk_id
                copied.append({
                    "path": str(target), "bytes": entry["bytes"],
                    "sha256": entry["sha256"], "declared_order": new_order,
                    "origin_path": str(src), "dimensions": entry["dimensions"],
                })
            row["kept_image_count"] = len(copied)
            accepted.append({
                "chunk_id": chunk_id, "doc_id": doc_id, "text": text,
                "section_title": chunk.get("section_title"),
                "header_level": chunk.get("header_level"),
                "images": copied,
                "visual_descriptions": chunk.get("visual_descriptions"),
            })
        audit_rows.append(row)

    # ---- emit dataset in the legacy evidence contract -----------------------
    packages: list[dict[str, Any]] = []
    for chunk in accepted:
        structure = {
            "section_title": chunk["section_title"],
            "header_level": chunk["header_level"],
            "declared_image_order": [img["declared_order"] for img in chunk["images"]],
        }
        for condition in CONDITIONS:
            evidence: dict[str, Any] = {"text": chunk["text"]}
            if condition in ("TL_struct", "TLV"):
                evidence["document_structure"] = structure
            if condition == "TLV":
                evidence["images"] = [
                    {"path": i["path"], "bytes": i["bytes"], "sha256": i["sha256"],
                     "declared_order": i["declared_order"]}
                    for i in chunk["images"]
                ]
            packages.append({
                "item_id": chunk["chunk_id"], "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"], "split": "test",
                "condition": condition, "evidence": evidence,
            })

    dataset = {
        "schema": "ecm-tqag.multimodal-inputs.v4-work24",
        "created_utc": utc_now(),
        "purpose_vi": (
            "Bộ dữ liệu multimodal đầy đủ (mọi chunk có ảnh thật) dùng làm dataset chính "
            "cho bài ECM; dựng trong thư mục làm việc riêng, tự chứa ảnh, có audit sàng lọc."
        ),
        "working_dir": str(work),
        "source_pool": {"path": str(POOL), "sha256": pool_sha, "total_chunks": len(pool),
                        "multimodal_chunks": len(multimodal)},
        "legacy_subset": {"path": str(LEGACY_INPUTS), "chunk_count": len(legacy_chunks)},
        "screening_rules": {
            "min_text_chars": MIN_TEXT_CHARS,
            "min_image_min_dimension_px": MIN_IMAGE_PIXELS,
            "min_image_bytes": MIN_IMAGE_BYTES,
            "excluded_pattern": "front-cover artwork (_page_0_*)",
            "dedup": "chunk text sha256; image sha256",
        },
        "conditions": list(CONDITIONS),
        "chunk_count": len(accepted),
        "document_count": len({c["doc_id"] for c in accepted}),
        "image_count": sum(len(c["images"]) for c in accepted),
        "package_count": len(packages),
        "split_policy": "pilot_test_only_document_grouped; no training or model-selection claim",
        "excluded_fields": ["visual_descriptions (kept in audit only, never shown to generators/judges)"],
        "packages": packages,
    }
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    audit = {
        "schema": "ecm-tqag.multimodal-screening-audit.v1",
        "created_utc": utc_now(),
        "source_pool_sha256": pool_sha,
        "image_index_size": sum(len(v) for v in index.values()),
        "candidates": len(multimodal),
        "accepted": sum(1 for r in audit_rows if r["decision"] == "ACCEPT"),
        "rejected": sum(1 for r in audit_rows if r["decision"] == "REJECT"),
        "rows": audit_rows,
    }
    (audit_dir / "screening_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "BUILT",
        "candidates_multimodal": len(multimodal),
        "accepted_chunks": dataset["chunk_count"],
        "rejected_chunks": audit["rejected"],
        "documents": dataset["document_count"],
        "images_kept": dataset["image_count"],
        "packages": dataset["package_count"],
        "legacy_overlap": sum(1 for r in audit_rows if r["in_legacy_8chunk"] and r["decision"] == "ACCEPT"),
        "dataset": str(dataset_path),
        "audit": str(audit_dir / "screening_audit.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
