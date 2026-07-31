#!/usr/bin/env python3
"""Build the immutable v3 eight-chunk, three-condition evidence manifest.

Text is canonicalised once per chunk before it is copied to T, TL_struct, and
TLV.  Image markdown, standalone extracted-image filenames, and page-divider
markers are removed from that common text; image pixels are supplied only in
TLV.  No visual-description/alt-text fields are propagated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = "ecm-tqag.multimodal-inputs.v3"
CONDITIONS = ("T", "TL_struct", "TLV")
_IMAGE_MD = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$", re.MULTILINE)
_PAGE_SEPARATOR = re.compile(r"^\s*\{\d+\}\s*-{3,}\s*$", re.MULTILINE)
_IMAGE_FILENAME = re.compile(r"^\s*(?:[_\w .,-]+)?_page_\d+(?:_[\w.-]+)?\.(?:jpe?g|png|webp|gif)\s*$", re.I | re.MULTILINE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_evidence_text(text: str) -> str:
    """Remove extraction artefacts without removing ordinary document prose."""
    if not isinstance(text, str):
        raise ValueError("BLOCKED_INPUT_INTEGRITY:text_not_string")
    text = _IMAGE_MD.sub("", text)
    text = _PAGE_SEPARATOR.sub("", text)
    text = _IMAGE_FILENAME.sub("", text)
    text = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", text)
    return text.strip()


def source_path(root: Path, recorded: str) -> Path:
    candidate = Path(recorded)
    if candidate.is_file():
        return candidate.resolve()
    # Historical manifests recorded absolute paths from another checkout.
    for marker in ("/data/", "/output/"):
        if marker in recorded:
            suffix = recorded.split(marker, 1)[1]
            candidate = root / ("data" if marker == "/data/" else "data/output") / suffix
            if candidate.is_file():
                return candidate.resolve()
    raise ValueError(f"BLOCKED_INPUT_INTEGRITY:unresolvable_image_path:{recorded}")


def build_manifest(contexts_path: Path, pilot_path: Path, root: Path) -> dict[str, Any]:
    contexts = json.loads(contexts_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    records = pilot.get("records") if isinstance(pilot, dict) else None
    if not isinstance(contexts, list) or pilot.get("schema") != "ecm-tqag.pilot-evaluation-manifest.v1":
        raise ValueError("BLOCKED_INPUT_INTEGRITY:unexpected_source_schema")
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("BLOCKED_INPUT_INTEGRITY:expected_eight_pilot_records")
    by_id = {x.get("chunk_id"): x for x in contexts if isinstance(x, dict)}
    packages: list[dict[str, Any]] = []
    image_count = 0
    for record in records:
        chunk_id, doc_id = record.get("chunk_id"), record.get("doc_id")
        context, images = by_id.get(chunk_id), record.get("resolved_images")
        if not isinstance(chunk_id, str) or not isinstance(doc_id, str) or not isinstance(context, dict) or not isinstance(images, list) or not images:
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:missing_source_record:{chunk_id}")
        if context.get("doc_id") != doc_id:
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:context_mismatch:{chunk_id}")
        raw_text = context.get("text")
        if not isinstance(raw_text, str):
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:invalid_text:{chunk_id}")
        text = clean_evidence_text(raw_text)
        if not text:
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:empty_clean_text:{chunk_id}")
        title, level = context.get("section_title"), context.get("header_level")
        if not isinstance(title, str) or not isinstance(level, int):
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:invalid_structure:{chunk_id}")
        verified: list[dict[str, Any]] = []
        for ordinal, image in enumerate(images, start=1):
            if not isinstance(image, dict) or image.get("status") != "PASS":
                raise ValueError(f"BLOCKED_INPUT_INTEGRITY:unresolved_image:{chunk_id}")
            path = source_path(root, str(image.get("candidate_path", "")))
            size, digest = path.stat().st_size, sha256_file(path)
            if size != image.get("bytes") or digest != image.get("sha256"):
                raise ValueError(f"BLOCKED_INPUT_INTEGRITY:image_hash_or_size_mismatch:{path.name}")
            verified.append({"path": str(path), "bytes": size, "sha256": digest, "declared_order": ordinal})
        image_count += len(verified)
        # No filename, path, alt-text, caption, or page-layout coordinate enters structure.
        structure = {"section_title": title, "header_level": level,
                     "declared_image_order": [x["declared_order"] for x in verified]}
        for condition in CONDITIONS:
            evidence: dict[str, Any] = {"text": text}
            if condition != "T":
                evidence["document_structure"] = structure
            if condition == "TLV":
                evidence["images"] = verified
            packages.append({"item_id": chunk_id, "chunk_id": chunk_id, "doc_id": doc_id,
                             "split": "test", "condition": condition, "evidence": evidence})
    return {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "chunk_count": 8, "package_count": len(packages),
            "document_count": len({x["doc_id"] for x in packages}), "image_count": image_count,
            "conditions": list(CONDITIONS),
            "text_cleaning": ["markdown_image_placeholders", "standalone_image_filenames", "page_separators"],
            "excluded_fields": ["visual_descriptions", "image_filenames", "image_paths", "page_layout"],
            "source_contexts": {"path": str(contexts_path), "sha256": sha256_file(contexts_path)},
            "source_pilot_manifest": {"path": str(pilot_path), "sha256": sha256_file(pilot_path)},
            "packages": packages}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build v3 clean 8-chunk manifest.")
    parser.add_argument("--contexts", type=Path, default=root / "data/output/interim/multimodal_contexts.json")
    parser.add_argument("--pilot-manifest", type=Path, default=root / "research/artifacts/pilot_evaluation_manifest.json")
    parser.add_argument("--out", type=Path, default=root / "research/artifacts/ecm_inputs_8chunks_v3.json")
    args = parser.parse_args()
    if args.out.exists(): raise SystemExit(f"Refusing overwrite: {args.out}")
    manifest = build_manifest(args.contexts, args.pilot_manifest, root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS chunks=8 packages=24 images={manifest['image_count']} out={args.out}")

if __name__ == "__main__": main()
