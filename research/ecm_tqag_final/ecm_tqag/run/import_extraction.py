"""Pure fail-closed verification of paid extraction artifacts across freezes.

An extraction call already paid for under an older freeze may be reused under the
current freeze only if it can be proven, offline, to be the *same* call: same
corpus image bytes, same request payload bytes, same provider identity, and the
same response bytes the provider actually returned.  This module performs that
proof and nothing else.

There is no network I/O here, no model output repair, and no write to any run
ledger.  Verification is per record and fail-closed: a record is admitted only
when every binding below holds, and a record that fails any binding is reported
as an explicit rejection rather than being silently dropped.

Bindings checked for each admitted record:

* the origin ledger has exactly one terminal for the task, with a matching
  write-ahead ``CALL_STARTED`` (no orphan attempt, no post-success terminal);
* the terminal outcome is ``OK`` with HTTP 200, phase ``extraction``, role
  ``generator``;
* ``freeze_sha256`` on the terminal equals the sha256 of the origin freeze file;
* the origin and current freezes agree on every field that could change what was
  asked (corpus manifest, census, decoding, prompt templates, generator roster);
* ``task_id``, ``extraction_kind``, ``image_index`` and ``image_sha256`` agree
  between the terminal and the current phase plan, and the image fingerprint
  agrees with the corpus;
* ``payload_sha256`` is reproduced by rebuilding the request with the *current*
  request builders over current corpus bytes -- this is what detects source
  drift, and it is the check that cannot be weakened;
* ``idempotency_key`` is reproduced from that payload plus frozen metadata;
* ``response_path`` is confined to ``<run>/responses/<response_sha256>.json``,
  the file bytes hash to ``response_sha256``, and the body is a JSON object;
* the response body parses under the phase's closed schema through the same
  shared parser live execution uses.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..interfaces import caption_request, ocr_assisted_graph_prompt
from ..io import canonical, sha256_bytes, sha256_file, write_json
from ..manifest import Corpus, image_part
from ..prompts import DECODING
from ..structure_reader import PROMPT
from .paid_worker import parse_extraction_body
from .transport import load_response_sidecar

IMPORT_SCHEMA = "ecm-tqag.extraction-import.v1"
EXTRACTION_KINDS = frozenset({"graph", "caption", "ocr_graph"})

#: Freeze fields that must be identical across the origin and current freeze for
#: a paid extraction to remain the same measurement.  ``full_call_budget`` and
#: ``source_sha256`` are deliberately absent: budget accounting and source-tree
#: identity do not change what was asked of the provider.
CROSS_FREEZE_INVARIANTS = (
    "census",
    "decoding",
    "input_fingerprints",
    "manifest_sha256",
    "prompt_templates",
)


class ImportVerificationBlocked(RuntimeError):
    """Stable fail-closed rejection for an unverifiable paid artifact."""


def _blocked(reason: str) -> ImportVerificationBlocked:
    return ImportVerificationBlocked(f"BLOCKED_EXTRACTION_IMPORT:{reason}")


def _reason(exc: ImportVerificationBlocked) -> str:
    return str(exc).removeprefix("BLOCKED_EXTRACTION_IMPORT:")


def _graph_request(prompt: str, data_url: str) -> dict[str, Any]:
    """Rebuild the blind/OCR-assisted graph request byte-for-byte.

    Kept structurally identical to ``paid_worker._graph_request``; the payload
    digest comparison in :func:`verify_extraction_record` is what proves the two
    have not drifted apart.
    """
    return {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "temperature": DECODING["temperature"],
        "max_tokens": DECODING["max_tokens"],
    }


def global_image_index(corpus: Corpus) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The one deterministic 18-image order shared with the paid worker."""
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for package in corpus.tlv:
        for image in sorted(package["evidence"]["images"],
                            key=lambda value: value["declared_order"]):
            rows.append((package, image))
    if len(rows) != 18:
        raise _blocked(f"image_census:{len(rows)}")
    return rows


def rebuild_extraction_payload(corpus: Corpus, *, kind: str,
                               image_index: int) -> tuple[dict[str, Any], str]:
    """Rebuild the provider-neutral extraction request from current corpus bytes.

    Returns the payload and the corpus image fingerprint, so a caller can prove
    the payload was built over the image it believes it was built over.
    """
    if kind not in EXTRACTION_KINDS:
        raise _blocked(f"kind_invalid:{kind}")
    if isinstance(image_index, bool) or not isinstance(image_index, int):
        raise _blocked("image_index_invalid")
    images = global_image_index(corpus)
    if not 1 <= image_index <= len(images):
        raise _blocked(f"image_index_invalid:{image_index}")
    package, image = images[image_index - 1]
    try:
        part, _audit = image_part(image)
    except Exception as exc:
        raise _blocked(f"image_unreadable:{type(exc).__name__}") from exc
    data_url = part["image_url"]["url"]
    if kind == "caption":
        payload = caption_request(data_url)
    elif kind == "graph":
        payload = _graph_request(PROMPT, data_url)
    else:
        payload = _graph_request(
            ocr_assisted_graph_prompt(package["evidence"]["text"]), data_url
        )
    return payload, str(image["sha256"])


def stamp_request(payload: Mapping[str, Any], *, provider: str,
                  model: str) -> dict[str, Any]:
    """Apply the transport's model/provider stamping to a neutral payload."""
    request = dict(payload)
    request["model"] = model
    if provider == "openrouter":
        request["provider"] = {"allow_fallbacks": False}
    return request


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(f"{label}_unreadable:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise _blocked(f"{label}_not_object")
    return value


def read_origin_terminals(path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Group origin ledger terminals by task and count write-ahead attempts.

    Structural ledger corruption raises.  Ambiguity that only affects one task
    (an orphan attempt, or a retried task with several terminals) is left for the
    per-record verifier to reject, so one damaged task cannot void the rest.
    """
    terminals: dict[str, list[dict[str, Any]]] = {}
    starts: dict[str, int] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise _blocked(f"ledger_unreadable:{type(exc).__name__}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise _blocked(f"ledger_invalid_json:{line_no}") from exc
        if not isinstance(row, dict):
            raise _blocked(f"ledger_not_object:{line_no}")
        record_type = row.get("record_type")
        if record_type not in {"CALL_STARTED", "CALL_TERMINAL"}:
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise _blocked(f"ledger_task_invalid:{line_no}")
        if record_type == "CALL_STARTED":
            starts[task_id] = starts.get(task_id, 0) + 1
            continue
        bucket = terminals.setdefault(task_id, [])
        if any(previous.get("outcome") == "OK" for previous in bucket):
            raise _blocked(f"terminal_after_success:{task_id}")
        bucket.append(row)
    for task_id, bucket in terminals.items():
        if starts.get(task_id, 0) < len(bucket):
            raise _blocked(f"terminal_without_attempt:{task_id}")
    return terminals, starts


def _check_cross_freeze(origin: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for field in CROSS_FREEZE_INVARIANTS:
        if field not in origin or field not in current:
            raise _blocked(f"cross_freeze_field_missing:{field}")
        if canonical(origin[field]) != canonical(current[field]):
            raise _blocked(f"cross_freeze_mismatch:{field}")
    origin_roles = origin.get("roles")
    current_roles = current.get("roles")
    if not isinstance(origin_roles, Mapping) or not isinstance(current_roles, Mapping):
        raise _blocked("cross_freeze_roster_missing")
    if canonical(origin_roles.get("generator")) != canonical(current_roles.get("generator")):
        raise _blocked("cross_freeze_mismatch:roles.generator")


def _generator(freeze: Mapping[str, Any]) -> dict[str, Any]:
    roles = freeze.get("roles")
    generator = roles.get("generator") if isinstance(roles, Mapping) else None
    if not isinstance(generator, Mapping):
        raise _blocked("generator_roster_missing")
    provider = generator.get("provider")
    model = generator.get("model")
    if not isinstance(provider, str) or not provider or not isinstance(model, str) or not model:
        raise _blocked("generator_identity_invalid")
    return dict(generator)


def verify_extraction_record(*, terminal: Mapping[str, Any], task: Mapping[str, Any],
                             corpus: Corpus, origin_run: Path,
                             origin_freeze_sha256: str,
                             generator: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one paid extraction terminal, or raise ``ImportVerificationBlocked``.

    Pure: reads only the preserved response sidecar and the corpus image bytes.
    """
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise _blocked("current_task_id_invalid")
    kind = task.get("extraction_kind")
    index = task.get("image_index")
    image_sha = task.get("image_sha256")
    if (task.get("phase") != "extraction" or task.get("calls") != 1
            or kind not in EXTRACTION_KINDS
            or isinstance(index, bool) or not isinstance(index, int)
            or not isinstance(image_sha, str) or len(image_sha) != 64):
        raise _blocked(f"current_task_invalid:{task_id}")

    expected = {
        "freeze_sha256": origin_freeze_sha256,
        "phase": "extraction",
        "role": "generator",
        "task_id": task_id,
        "extraction_kind": kind,
        "image_index": index,
        "image_sha256": image_sha,
        "provider": generator["provider"],
        "model": generator["model"],
        "outcome": "OK",
        "http_status": 200,
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise _blocked(f"terminal_metadata_mismatch:{field}")
    if terminal.get("returned_model") != generator["model"]:
        raise _blocked("returned_model_mismatch")

    payload, corpus_image_sha = rebuild_extraction_payload(
        corpus, kind=str(kind), image_index=index
    )
    if corpus_image_sha != image_sha:
        raise _blocked("corpus_image_fingerprint_mismatch")
    request = stamp_request(payload, provider=generator["provider"],
                            model=generator["model"])
    payload_sha = sha256_bytes(canonical(request).encode("utf-8"))
    if terminal.get("payload_sha256") != payload_sha:
        raise _blocked("payload_sha256_mismatch")
    prompt_sha = sha256_bytes(canonical(request.get("messages", [])).encode("utf-8"))
    if terminal.get("prompt_sha256") not in {None, prompt_sha}:
        raise _blocked("prompt_sha256_mismatch")

    metadata = {
        "phase": "extraction",
        "role": "generator",
        "task_id": task_id,
        "extraction_kind": kind,
        "image_index": index,
        "image_sha256": image_sha,
    }
    idempotency = sha256_bytes(canonical({
        "freeze_sha256": origin_freeze_sha256,
        "provider": generator["provider"],
        "model": generator["model"],
        "payload_sha256": payload_sha,
        "metadata": metadata,
    }).encode("utf-8"))
    if terminal.get("idempotency_key") != idempotency:
        raise _blocked("idempotency_key_mismatch")

    try:
        body = load_response_sidecar(Path(origin_run), dict(terminal))
    except Exception as exc:
        detail = str(exc).removeprefix("BLOCKED_EXECUTION:")
        raise _blocked(f"response_sidecar_invalid:{detail}") from exc
    if body.get("model") != generator["model"]:
        raise _blocked("response_model_mismatch")
    try:
        result = parse_extraction_body(
            body, kind=str(kind), image_index=index, image_sha256=image_sha
        )
    except Exception as exc:
        raise _blocked(f"response_schema_rejected:{type(exc).__name__}") from exc

    response_sha = terminal.get("response_sha256")
    if result.get("response_sha256") not in {None, response_sha}:
        raise _blocked("parsed_response_sha256_mismatch")
    return {
        "task_id": task_id,
        "extraction_kind": kind,
        "image_index": index,
        "image_sha256": image_sha,
        "status": result["status"],
        "result": result,
        "import_origin": {
            "origin_freeze_sha256": origin_freeze_sha256,
            "origin_run": Path(origin_run).name,
            "origin_task_id": task_id,
            "origin_idempotency_key": idempotency,
            "origin_payload_sha256": payload_sha,
            "origin_response_sha256": response_sha,
        },
    }


def verify_extraction_imports(*, origin_run: Path, origin_freeze_path: Path,
                              current_freeze_path: Path, corpus: Corpus,
                              current_plan: Mapping[str, Any],
                              task_ids: Iterable[str] | None = None) -> dict[str, Any]:
    """Build a canonical IMPORT_MANIFEST from one preserved paid run.

    Every selected record appears exactly once, in ``verified`` or in
    ``rejected`` with a stable reason.  Nothing is repaired and nothing is
    dropped, so ``verified + rejected == selected`` always holds.  When
    ``task_ids`` is omitted, every extraction terminal in the origin ledger is
    selected.
    """
    origin_run = Path(origin_run)
    origin_freeze_path = Path(origin_freeze_path)
    current_freeze_path = Path(current_freeze_path)

    origin_freeze = _read_object(origin_freeze_path, "origin_freeze")
    current_freeze = _read_object(current_freeze_path, "current_freeze")
    origin_freeze_sha = sha256_file(origin_freeze_path)
    current_freeze_sha = sha256_file(current_freeze_path)
    if origin_freeze.get("manifest_sha256") != corpus.manifest_sha256:
        raise _blocked("origin_manifest_mismatch")
    if current_freeze.get("manifest_sha256") != corpus.manifest_sha256:
        raise _blocked("current_manifest_mismatch")
    _check_cross_freeze(origin_freeze, current_freeze)
    generator = _generator(current_freeze)

    raw_tasks = current_plan.get("tasks")
    if current_plan.get("schema") != "ecm-tqag.phase-plan.v1" or not isinstance(raw_tasks, list):
        raise _blocked("current_plan_invalid")
    plan_tasks: dict[str, Mapping[str, Any]] = {}
    for row in raw_tasks:
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
            raise _blocked("current_plan_tasks_invalid")
        plan_tasks[str(row["task_id"])] = row
    if len(plan_tasks) != len(raw_tasks):
        raise _blocked("current_plan_tasks_invalid")

    terminals, starts = read_origin_terminals(origin_run / "RUN_LEDGER.jsonl")
    if task_ids is None:
        selected = sorted(
            task_id for task_id, bucket in terminals.items()
            if any(row.get("phase") == "extraction" for row in bucket)
        )
    else:
        selected = list(task_ids)
        if any(not isinstance(value, str) or not value for value in selected):
            raise _blocked("selection_invalid")
        if len(selected) != len(set(selected)):
            raise _blocked("selection_not_unique")
        selected = sorted(selected)

    verified: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for task_id in selected:
        bucket = terminals.get(task_id, [])
        task = plan_tasks.get(task_id)
        if not bucket:
            rejected.append({"task_id": task_id, "reason": "origin_terminal_missing"})
            continue
        if len(bucket) != 1:
            rejected.append({"task_id": task_id,
                             "reason": f"terminal_not_unique:{len(bucket)}"})
            continue
        if starts.get(task_id, 0) != 1:
            rejected.append({"task_id": task_id,
                             "reason": f"attempt_history_ambiguous:{starts.get(task_id, 0)}"})
            continue
        if task is None:
            rejected.append({"task_id": task_id, "reason": "current_plan_task_missing"})
            continue
        try:
            verified.append(verify_extraction_record(
                terminal=bucket[0], task=task, corpus=corpus,
                origin_run=origin_run, origin_freeze_sha256=origin_freeze_sha,
                generator=generator,
            ))
        except ImportVerificationBlocked as exc:
            rejected.append({"task_id": task_id, "reason": _reason(exc)})

    verified.sort(key=lambda row: str(row["task_id"]))
    rejected.sort(key=lambda row: str(row["task_id"]))
    records = {"verified": verified, "rejected": rejected}
    return {
        "schema": IMPORT_SCHEMA,
        "status": "VERIFIED" if not rejected else "VERIFIED_WITH_REJECTIONS",
        "origin_run": origin_run.name,
        "origin_freeze_sha256": origin_freeze_sha,
        "current_freeze_sha256": current_freeze_sha,
        "manifest_sha256": corpus.manifest_sha256,
        "selected_count": len(selected),
        "verified_count": len(verified),
        "rejected_count": len(rejected),
        "verified_task_ids": [str(row["task_id"]) for row in verified],
        "rejected_task_ids": [str(row["task_id"]) for row in rejected],
        "records_sha256": sha256_bytes(canonical(records).encode("utf-8")),
        "records": records,
    }


def validate_import_manifest(manifest: Mapping[str, Any]) -> None:
    """Re-check a manifest's own internal commitments before it is trusted."""
    if manifest.get("schema") != IMPORT_SCHEMA:
        raise _blocked("manifest_schema_invalid")
    if manifest.get("status") not in {"VERIFIED", "VERIFIED_WITH_REJECTIONS"}:
        raise _blocked("manifest_status_invalid")
    records = manifest.get("records")
    if not isinstance(records, Mapping):
        raise _blocked("manifest_records_missing")
    verified = records.get("verified")
    rejected = records.get("rejected")
    if not isinstance(verified, list) or not isinstance(rejected, list):
        raise _blocked("manifest_records_invalid")
    if manifest.get("records_sha256") != sha256_bytes(canonical(
            {"verified": verified, "rejected": rejected}).encode("utf-8")):
        raise _blocked("manifest_commitment_mismatch")
    if (manifest.get("verified_count") != len(verified)
            or manifest.get("rejected_count") != len(rejected)
            or manifest.get("selected_count") != len(verified) + len(rejected)):
        raise _blocked("manifest_counts_mismatch")
    if (manifest.get("status") == "VERIFIED") != (not rejected):
        raise _blocked("manifest_status_mismatch")
    task_ids = [row.get("task_id") for row in verified] + [row.get("task_id") for row in rejected]
    if len(task_ids) != len(set(task_ids)):
        raise _blocked("manifest_task_ids_not_disjoint")


def write_import_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write a validated manifest without overwriting a differing artifact."""
    validate_import_manifest(manifest)
    path = Path(path)
    payload = dict(manifest)
    if path.exists():
        existing = _read_object(path, "existing_manifest")
        if canonical(existing) != canonical(payload):
            raise _blocked("manifest_exists_with_different_content")
        return path
    write_json(path, payload)
    return path


__all__ = [
    "CROSS_FREEZE_INVARIANTS",
    "EXTRACTION_KINDS",
    "IMPORT_SCHEMA",
    "ImportVerificationBlocked",
    "global_image_index",
    "read_origin_terminals",
    "rebuild_extraction_payload",
    "stamp_request",
    "validate_import_manifest",
    "verify_extraction_imports",
    "verify_extraction_record",
    "write_import_manifest",
]
