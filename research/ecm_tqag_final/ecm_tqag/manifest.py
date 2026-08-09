from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import canonical, sha256_file

CONDITIONS = ("T", "TL_struct", "TLV")
MANIFEST_SCHEMA = "ecm-tqag.multimodal-inputs.v4-work24"


@dataclass(frozen=True)
class Corpus:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    tlv: tuple[dict[str, Any], ...]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(sorted(p["chunk_id"] for p in self.tlv))


def image_part(image: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = {"path", "bytes", "sha256", "declared_order"}
    if not isinstance(image, dict) or not required <= set(image):
        raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_image_record")
    path = Path(image["path"])
    if not path.is_file():
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:image_missing:{path.name}")
    actual_hash = sha256_file(path)
    if path.stat().st_size != image["bytes"] or actual_hash != image["sha256"]:
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:image_hash_or_size_mismatch:{path.name}")
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
    audit = {"path": str(path), "bytes": image["bytes"], "sha256": actual_hash,
             "declared_order": image["declared_order"]}
    return part, audit


def load_corpus(path: Path) -> Corpus:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:manifest_unreadable:{type(exc).__name__}") from exc
    if obj.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("BLOCKED_INPUT_INTEGRITY:unexpected_manifest_schema")
    packages = obj.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("BLOCKED_INPUT_INTEGRITY:no_packages")
    keyed: dict[tuple[str, str], dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or package.get("condition") not in CONDITIONS:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_package")
        cid = package.get("chunk_id")
        if not isinstance(cid, str) or not cid:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_chunk_id")
        key = (cid, package["condition"])
        if key in keyed:
            raise ValueError("BLOCKED_INPUT_INTEGRITY:duplicate_chunk_condition")
        evidence = package.get("evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("text"), str) or not evidence["text"].strip():
            raise ValueError("BLOCKED_INPUT_INTEGRITY:invalid_text")
        keyed[key] = package
    chunks = sorted({cid for cid, _ in keyed})
    if len(chunks) != 16 or len(packages) != 48:
        raise ValueError(f"BLOCKED_DESIGN:census_must_be_16x3:chunks={len(chunks)}:packages={len(packages)}")
    for cid in chunks:
        if any((cid, condition) not in keyed for condition in CONDITIONS):
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:incomplete_chunk:{cid}")
        t = keyed[(cid, "T")]["evidence"]["text"]
        tls = keyed[(cid, "TL_struct")]["evidence"]
        tlv = keyed[(cid, "TLV")]["evidence"]
        if t != tls["text"] or t != tlv["text"]:
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:unpaired_text:{cid}")
        if tls.get("document_structure") != tlv.get("document_structure"):
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:unpaired_structure:{cid}")
        images = tlv.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError(f"BLOCKED_INPUT_INTEGRITY:missing_images:{cid}")
        for image in images:
            image_part(image)
    tlv_rows = tuple(keyed[(cid, "TLV")] for cid in chunks)
    if sum(len(p["evidence"]["images"]) for p in tlv_rows) != 18:
        raise ValueError("BLOCKED_DESIGN:census_must_have_18_images")
    return Corpus(path, sha256_file(path), obj, tlv_rows)


def evidence_public_view(package: dict[str, Any]) -> dict[str, Any]:
    """The only evidence fields permitted in prompts; identifiers never enter."""
    evidence = package["evidence"]
    return {"text": evidence["text"], "document_structure": evidence.get("document_structure")}


def input_fingerprint(package: dict[str, Any]) -> str:
    from .io import sha256_bytes
    evidence = package["evidence"]
    payload = {"text": evidence["text"], "structure": evidence.get("document_structure"),
               "images": [{"sha256": x["sha256"], "declared_order": x["declared_order"]}
                          for x in evidence.get("images", [])]}
    return sha256_bytes(canonical(payload).encode("utf-8"))
