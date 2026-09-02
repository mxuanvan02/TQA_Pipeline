"""
Provenance / Audit Metadata for AI-generated dataset records
============================================================
Who:    Imported by Stage 3 (QAG) and any Source-B generator (reconstructed
        legal diagrams / graphs). Used offline by verify_provenance.py.
Where:  src/provenance.py
Why:    A generated benchmark (DHH2026 / ViLeGr-TQA) must let a reviewer or
        auditor answer three questions AFTER the fact:
          1. Reproducibility  — same model + prompt + params ⇒ same record?
          2. Groundedness      — which source chunk/statute produced this?
          3. Tamper-evidence   — was the record altered after generation?
        This module captures a "provenance envelope" AT GENERATION TIME.
        Provenance reconstructed later is worthless for audit; capture it now.

Design:
    - Pure standard library (hashlib, json, os, subprocess, platform, datetime).
    - A run-level envelope (identical for one generation run) + a per-record
      stamp (source hash, prompt-input hash, content hash).
    - Deterministic hashing (sorted-key canonical JSON) so hashes are
      reproducible across machines.
    - Schema loosely mirrors W3C PROV (Activity/Entity/Agent) and MLCommons
      Croissant provenance so it can be exported to those later.

Envelope is attached to each record under the reserved key ``_provenance``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA_VERSION = "tqa.provenance.v1"
PROVENANCE_KEY = "_provenance"


# ─────────────────────────────────────────────
# Deterministic hashing helpers
# ─────────────────────────────────────────────
def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON string: sorted keys, no whitespace jitter, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def hash_text(text: str) -> str:
    """SHA-256 of a UTF-8 string, prefixed with the algorithm name."""
    return _sha256_text(text or "")


def hash_obj(obj: Any) -> str:
    """SHA-256 of the canonical JSON of any JSON-serializable object."""
    return _sha256_text(canonical_json(obj))


def hash_file(path: str | Path, *, missing_ok: bool = True) -> str | None:
    """SHA-256 of a file's bytes (streamed). Returns None if missing and allowed."""
    p = Path(path)
    if not p.exists():
        if missing_ok:
            return None
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────────────────────────
# Git lineage (best-effort, never raises)
# ─────────────────────────────────────────────
def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_lineage(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    commit = _git(["rev-parse", "HEAD"], root)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    status = _git(["status", "--porcelain"], root)
    return {
        "commit": commit,
        "branch": branch,
        # dirty = uncommitted changes present at generation time (audit red flag)
        "dirty": bool(status) if status is not None else None,
    }


# ─────────────────────────────────────────────
# Envelope builders
# ─────────────────────────────────────────────
@dataclass
class ModelSpec:
    """Identity of the generating model (the PROV Agent)."""

    model_name: str
    engine: str = "unknown"           # vllm | huggingface | openai-compatible | ...
    quantization: str | None = None   # awq | 4bit-nf4 | None
    revision: str | None = None       # HF commit / model snapshot if known
    provider: str | None = None       # local | 9router | openai | ...
    endpoint: str | None = None        # redact keys; host only


@dataclass
class DecodingSpec:
    """Sampling parameters. Include seed when the backend supports it."""

    temperature: float | None = None
    top_p: float | None = None
    max_new_tokens: int | None = None
    repetition_penalty: float | None = None
    do_sample: bool | None = None
    seed: int | None = None


@dataclass
class PromptSpec:
    """Prompt identity. Hash the templates so wording changes are detectable."""

    system_prompt_sha256: str | None = None
    template_sha256: str | None = None
    template_id: str | None = None
    enforce_legal_syllogism: bool | None = None
    use_visual_context: bool | None = None


def build_run_envelope(
    *,
    stage: str,
    repo_root: str | Path,
    model: ModelSpec,
    decoding: DecodingSpec,
    prompt: PromptSpec,
    script_path: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the run-level envelope shared by every record of one generation run.

    Returned dict is JSON-serializable and safe to embed under each record's
    ``_provenance.run`` field, or to persist once as a sidecar manifest.
    """
    root = Path(repo_root)
    env = {
        "schema": PROVENANCE_SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "stage": stage,                       # PROV Activity
        "generated_at": now_utc_iso(),
        "generator": {
            "script": str(script_path) if script_path else None,
            "git": git_lineage(root),
            "host": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        },
        "model": {k: v for k, v in asdict(model).items()},         # PROV Agent
        "decoding": {k: v for k, v in asdict(decoding).items()},
        "prompt": {k: v for k, v in asdict(prompt).items()},
    }
    if extra:
        env["extra"] = extra
    return env


# Fields that are the model's *output* — hashed to detect post-hoc edits.
DEFAULT_CONTENT_FIELDS = (
    "question_content",
    "candidate_answers",
    "ground_truth",
    "legal_rationale",
)


def record_content_hash(
    record: dict[str, Any],
    content_fields: tuple[str, ...] = DEFAULT_CONTENT_FIELDS,
) -> str:
    """Hash only the generated-answer fields (order-stable)."""
    payload = {k: record.get(k) for k in content_fields}
    return hash_obj(payload)


def stamp_record(
    record: dict[str, Any],
    run_envelope: dict[str, Any],
    *,
    context_text: str | None = None,
    prompt_input: str | None = None,
    source_file: str | Path | None = None,
    content_fields: tuple[str, ...] = DEFAULT_CONTENT_FIELDS,
) -> dict[str, Any]:
    """
    Attach a per-record provenance stamp under ``_provenance`` (in place).

    - run:            the shared run envelope (reference-shared, not copied)
    - source:         doc/chunk ids + hash of the exact context the model saw
    - prompt_input:   hash of the reproducible prompt inputs for THIS record
    - content_sha256: hash of the model output (tamper-evidence)
    """
    ctx = context_text if context_text is not None else record.get("context_text", "")
    stamp: dict[str, Any] = {
        "run": run_envelope,
        "source": {
            "doc_id": record.get("doc_id"),
            "chunk_id": record.get("chunk_id"),
            "is_multimodal": record.get("is_multimodal", False),
            "context_sha256": hash_text(ctx),
            "visuals": record.get("context_visuals") or [],
            "source_file_sha256": hash_file(source_file) if source_file else None,
        },
        "content_sha256": record_content_hash(record, content_fields),
        "stamped_at": now_utc_iso(),
    }
    if prompt_input is not None:
        stamp["prompt_input_sha256"] = hash_text(prompt_input)
    record[PROVENANCE_KEY] = stamp
    return record


def verify_record(
    record: dict[str, Any],
    content_fields: tuple[str, ...] = DEFAULT_CONTENT_FIELDS,
) -> dict[str, Any]:
    """
    Re-check a stamped record. Returns a small report dict:
      { has_provenance, content_ok, context_ok(None if no stored hash), reasons }
    ``content_ok=False`` means the answer fields changed since stamping.
    """
    reasons: list[str] = []
    prov = record.get(PROVENANCE_KEY)
    if not isinstance(prov, dict):
        return {"has_provenance": False, "content_ok": None, "context_ok": None,
                "reasons": ["missing _provenance"]}

    stored = prov.get("content_sha256")
    recomputed = record_content_hash(record, content_fields)
    content_ok = (stored == recomputed)
    if not content_ok:
        reasons.append("content hash mismatch (record edited after generation)")

    context_ok: bool | None = None
    src = prov.get("source") or {}
    stored_ctx = src.get("context_sha256")
    if stored_ctx is not None and "context_text" in record:
        context_ok = (stored_ctx == hash_text(record.get("context_text", "")))
        if not context_ok:
            reasons.append("context hash mismatch (source context changed)")

    return {
        "has_provenance": True,
        "content_ok": content_ok,
        "context_ok": context_ok,
        "reasons": reasons,
    }


# ─────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run = build_run_envelope(
        stage="selftest",
        repo_root=Path(__file__).resolve().parent.parent,
        model=ModelSpec(model_name="demo/model", engine="huggingface", quantization="4bit-nf4"),
        decoding=DecodingSpec(temperature=0.7, top_p=0.9, max_new_tokens=512),
        prompt=PromptSpec(template_id="qag_v1", enforce_legal_syllogism=True),
        script_path=__file__,
    )
    rec = {
        "qa_id": "demo_chunk_0_remember_0",
        "doc_id": "demo",
        "chunk_id": "demo_chunk_0",
        "context_text": "Điều 1. Đây là ngữ cảnh mẫu.",
        "question_content": "Câu hỏi mẫu?",
        "candidate_answers": ["A. x", "B. y", "C. z", "D. w"],
        "ground_truth": "A. x",
        "legal_rationale": "Đại tiền đề... Kết luận...",
    }
    stamp_record(rec, run, prompt_input="ctx|Remember|syllogism=1")
    ok = verify_record(rec)
    assert ok["content_ok"] is True and ok["context_ok"] is True, ok
    # simulate tampering
    rec["ground_truth"] = "B. y"
    bad = verify_record(rec)
    assert bad["content_ok"] is False, bad
    print("provenance self-test OK")
    print(canonical_json(rec[PROVENANCE_KEY])[:400], "...")
