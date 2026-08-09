"""Pure fail-closed verification of paid *construction planner* artifacts.

A construction planner call that already completed HTTP 200 under an older freeze
may be reused under the current freeze only if it can be proven, offline, to be
the *same* call: same corpus package bytes, same upstream extraction interface,
same request payload bytes, same provider identity, and the same response bytes
the provider actually returned.  This module performs that proof and nothing
else.

The motivating case is narrow and worth stating plainly.  One planner call was
paid for, the provider returned HTTP 200, and the returned completion is *not*
valid planner JSON.  Under the current source that is a terminal one-call ITT
outcome (``SCHEMA_REJECTED``), not an error to retry: retrying would spend a
second call on the same intention-to-treat cell and would bias the arm.  The old
worker crashed before it checkpointed, so the paid artifact exists in the origin
run ledger but not in the origin execution checkpoint.  Importing it is therefore
the only way to keep the call census honest without paying twice.

There is no network I/O here, no model output repair, and no write to any run
ledger.  Verification is per record and fail-closed.

Bindings checked for the planner record:

* the origin ledger has exactly one terminal for the task, with exactly one
  matching write-ahead ``CALL_STARTED`` (no orphan attempt, no retry history,
  no post-success terminal);
* the terminal outcome is ``OK`` with HTTP 200, phase ``construction``, role
  ``generator``, stage ``planner``;
* ``freeze_sha256`` on the terminal equals the sha256 of the origin freeze file;
* the origin and current freezes agree on every field that could change what was
  asked (corpus manifest, census, decoding, prompt templates, arm roster,
  per-chunk input fingerprints, generator roster);
* ``task_id``, ``arm``, ``chunk_id`` and ``construction_stage`` agree between the
  terminal and the current phase plan, and the plan task's
  ``input_fingerprint`` agrees with the corpus package;
* every upstream extraction artifact the planner prompt consumed is bound to the
  origin freeze through the origin execution checkpoint, hashes to its recorded
  digest, and carries status ``OK``;
* ``payload_sha256`` is reproduced by rebuilding the planner request with the
  *current* prompt builders over current corpus bytes and the origin extraction
  interfaces -- this is what detects source drift, and it is the check that
  cannot be weakened;
* ``prompt_sha256`` and ``idempotency_key`` are reproduced from that payload plus
  frozen metadata;
* ``response_path`` is confined to ``<run>/responses/<response_sha256>.json``,
  the file bytes hash to ``response_sha256``, and the body is a JSON object whose
  ``model`` is the frozen generator model;
* the response body is classified through the *same* shared envelope reader and
  planner parser live execution uses, and the classification is
  ``SCHEMA_REJECTED``.  A record that now parses cleanly is rejected rather than
  imported, because that would mean the frozen parser changed and the artifact is
  no longer the measurement it was.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..interface_merge import merge_caption_interfaces, merge_graph_interfaces
from ..io import canonical, sha256_bytes, sha256_file, write_json
from ..manifest import Corpus, input_fingerprint
from ..prompts import DECODING, planner_prompt, planner_program
from .envelope import response_content
from .import_extraction import ImportVerificationBlocked as _ExtractionBlocked
from .import_extraction import read_origin_terminals
from .transport import load_response_sidecar

IMPORT_SCHEMA = "ecm-tqag.construction-import.v1"

#: Planner arm -> (upstream extraction kind, planner interface kind).  Mirrors
#: ``PaidPhaseWorker._construct``; the payload digest comparison is what proves
#: the two have not drifted apart.
PLANNER_ARMS: Mapping[str, tuple[str, str]] = {
    "full": ("graph", "closed_graph"),
    "caption_mediated": ("caption", "caption"),
    "text_assisted_reader": ("ocr_graph", "closed_graph"),
}

#: Freeze fields that must be identical across the origin and current freeze for
#: a paid planner call to remain the same measurement.  ``full_call_budget``,
#: ``construction_call_budget``, ``operational_http_cap`` and ``source_sha256``
#: are deliberately absent: budget accounting and source-tree identity do not
#: change what was asked of the provider.
CROSS_FREEZE_INVARIANTS = (
    "arms",
    "census",
    "decoding",
    "input_fingerprints",
    "manifest_sha256",
    "prompt_templates",
)

#: The only outcome this module will import.  Kept explicit so a future caller
#: cannot widen it by accident.
IMPORTABLE_STATUS = "SCHEMA_REJECTED"


class ConstructionImportBlocked(RuntimeError):
    """Stable fail-closed rejection for an unverifiable paid planner artifact."""


def _blocked(reason: str) -> ConstructionImportBlocked:
    return ConstructionImportBlocked(f"BLOCKED_CONSTRUCTION_IMPORT:{reason}")


def _reason(exc: ConstructionImportBlocked) -> str:
    return str(exc).removeprefix("BLOCKED_CONSTRUCTION_IMPORT:")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(f"{label}_unreadable:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise _blocked(f"{label}_not_object")
    return value


def _check_cross_freeze(origin: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if origin.get("schema") != current.get("schema"):
        raise _blocked("cross_freeze_mismatch:schema")
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


def chunk_image_index(corpus: Corpus) -> dict[str, list[tuple[int, int]]]:
    """Map ``chunk_id -> [(global image index, declared order)]``.

    The one deterministic 18-image order shared with the paid worker and the live
    extraction loader.  A census that is not exactly 18 fails closed.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    global_index = 0
    for package in corpus.tlv:
        rows: list[tuple[int, int]] = []
        for image in sorted(package["evidence"]["images"],
                            key=lambda value: value["declared_order"]):
            global_index += 1
            rows.append((global_index, int(image["declared_order"])))
        out[str(package["chunk_id"])] = rows
    if global_index != 18:
        raise _blocked(f"image_census:{global_index}")
    return out


def read_origin_checkpoints(path: Path) -> dict[str, dict[str, Any]]:
    """Index the origin execution checkpoint by task id, rejecting duplicates."""
    rows: dict[str, dict[str, Any]] = {}
    target = Path(path)
    if not target.exists():
        return rows
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise _blocked(f"checkpoint_unreadable:{type(exc).__name__}") from exc
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise _blocked(f"checkpoint_invalid_json:{line_no}") from exc
        if not isinstance(row, dict):
            raise _blocked(f"checkpoint_not_object:{line_no}")
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise _blocked(f"checkpoint_task_invalid:{line_no}")
        if task_id in rows:
            raise _blocked(f"checkpoint_task_duplicate:{task_id}")
        rows[task_id] = row
    return rows


def load_origin_extraction_result(*, origin_run: Path, task_id: str,
                                 checkpoints: Mapping[str, Any],
                                 origin_freeze_sha256: str) -> dict[str, Any]:
    """Load one upstream extraction artifact, bound to the origin freeze.

    The planner prompt is a function of these artifacts, so each one is held to
    the same standard as the planner terminal itself: recorded in the origin
    checkpoint, confined to ``results/<sha(task)>.json``, hashing to its recorded
    digest, and carrying status ``OK``.
    """
    record = checkpoints.get(task_id)
    if not isinstance(record, Mapping):
        raise _blocked(f"upstream_checkpoint_missing:{task_id}")
    if record.get("freeze_sha256") != origin_freeze_sha256:
        raise _blocked(f"upstream_checkpoint_freeze_mismatch:{task_id}")
    if record.get("phase") != "extraction" or record.get("calls") != 1:
        raise _blocked(f"upstream_checkpoint_metadata_mismatch:{task_id}")
    relative_raw = record.get("result_path")
    digest = record.get("result_sha256")
    if not isinstance(relative_raw, str) or not isinstance(digest, str) or len(digest) != 64:
        raise _blocked(f"upstream_checkpoint_sidecar_metadata:{task_id}")
    relative = Path(relative_raw)
    expected = Path("results") / f"{sha256_bytes(canonical(task_id).encode('utf-8'))}.json"
    if relative.is_absolute() or ".." in relative.parts or relative != expected:
        raise _blocked(f"upstream_result_path_invalid:{task_id}")
    path = Path(origin_run) / relative
    if not path.is_file():
        raise _blocked(f"upstream_result_missing:{task_id}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise _blocked(f"upstream_result_hash_mismatch:{task_id}")
    try:
        result = json.loads(payload)
    except Exception as exc:
        raise _blocked(f"upstream_result_invalid_json:{task_id}") from exc
    if not isinstance(result, dict):
        raise _blocked(f"upstream_result_not_object:{task_id}")
    if result.get("task_id") != task_id or result.get("freeze_sha256") != origin_freeze_sha256:
        raise _blocked(f"upstream_result_metadata_mismatch:{task_id}")
    if result.get("status") != "OK":
        raise _blocked(f"upstream_result_not_ok:{task_id}:{result.get('status')}")
    if not isinstance(result.get("interface"), Mapping):
        raise _blocked(f"upstream_result_interface_missing:{task_id}")
    return result


def rebuild_origin_interface(*, corpus: Corpus, origin_run: Path, chunk_id: str,
                             kind: str, checkpoints: Mapping[str, Any],
                             origin_freeze_sha256: str
                             ) -> tuple[dict[str, Any], list[str]]:
    """Recompose the merged extraction interface the planner prompt consumed."""
    indexes = chunk_image_index(corpus)
    if chunk_id not in indexes:
        raise _blocked(f"chunk_not_found:{chunk_id}")
    records: list[dict[str, Any]] = []
    task_ids: list[str] = []
    for global_index, declared_order in indexes[str(chunk_id)]:
        task_id = f"extract:{kind}:{global_index:02d}"
        result = load_origin_extraction_result(
            origin_run=origin_run, task_id=task_id, checkpoints=checkpoints,
            origin_freeze_sha256=origin_freeze_sha256,
        )
        records.append({"declared_order": declared_order,
                        "interface": dict(result["interface"])})
        task_ids.append(task_id)
    try:
        merged = (merge_caption_interfaces(records) if kind == "caption"
                  else merge_graph_interfaces(records))
    except Exception as exc:
        raise _blocked(f"interface_merge_failed:{type(exc).__name__}") from exc
    if not isinstance(merged, Mapping):
        raise _blocked("interface_merge_not_object")
    return dict(merged), task_ids


def _package(corpus: Corpus, chunk_id: str) -> Mapping[str, Any]:
    for package in corpus.tlv:
        if package.get("chunk_id") == chunk_id:
            return package
    raise _blocked(f"chunk_not_found:{chunk_id}")


def rebuild_planner_payload(*, corpus: Corpus, chunk_id: str, arm: str,
                            interface: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the provider-neutral planner request from current source.

    Structurally identical to ``PaidPhaseWorker._construct``'s planner branch.
    """
    routing = PLANNER_ARMS.get(arm)
    if routing is None:
        raise _blocked(f"planner_arm_invalid:{arm}")
    _kind, interface_kind = routing
    package = _package(corpus, chunk_id)
    evidence = package["evidence"]
    try:
        prompt = planner_prompt(
            evidence["text"], evidence.get("document_structure") or {},
            dict(interface), interface_kind=interface_kind,
        )
    except Exception as exc:
        raise _blocked(f"planner_prompt_failed:{type(exc).__name__}") from exc
    return {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": DECODING["temperature"],
        "max_tokens": DECODING["max_tokens"],
    }


def stamp_request(payload: Mapping[str, Any], *, provider: str,
                  model: str) -> dict[str, Any]:
    """Apply the transport's model/provider stamping to a neutral payload."""
    request = dict(payload)
    request["model"] = model
    if provider == "openrouter":
        request["provider"] = {"allow_fallbacks": False}
    return request


def classify_planner_body(body: Mapping[str, Any], *, arm: str, chunk_id: str,
                          input_fingerprint_value: Any) -> dict[str, Any]:
    """Classify a preserved planner response exactly as the live worker would.

    Returns the worker result object for the ``SCHEMA_REJECTED`` branch, or
    raises when the body now parses (which would mean the frozen parser changed).
    """
    try:
        program_raw = response_content(body)
        planner_program(program_raw)
    except Exception as exc:
        return {
            "status": IMPORTABLE_STATUS,
            "calls_used": 1,
            "reason": f"planner_schema_rejected:{type(exc).__name__}:{exc}",
            "arm": arm,
            "chunk_id": chunk_id,
            "construction_stage": "planner",
            "input_fingerprint": input_fingerprint_value,
        }
    raise _blocked("outcome_changed:planner_body_now_parses")


def verify_construction_planner_record(*, terminal: Mapping[str, Any],
                                       task: Mapping[str, Any], corpus: Corpus,
                                       origin_run: Path,
                                       origin_freeze_sha256: str,
                                       checkpoints: Mapping[str, Any],
                                       generator: Mapping[str, Any]) -> dict[str, Any]:
    """Verify one paid planner terminal, or raise ``ConstructionImportBlocked``.

    Pure: reads only the preserved response sidecar, the origin extraction
    artifacts, and the corpus package bytes.
    """
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise _blocked("current_task_id_invalid")
    arm = task.get("arm")
    chunk_id = task.get("chunk_id")
    fingerprint = task.get("input_fingerprint")
    if (task.get("phase") != "construction" or task.get("calls") != 1
            or task.get("construction_stage") != "planner"
            or task.get("parent_task_id") is not None
            or task.get("deterministic_rescore") is True
            or arm not in PLANNER_ARMS
            or not isinstance(chunk_id, str) or not chunk_id
            or not isinstance(fingerprint, str) or len(fingerprint) != 64):
        raise _blocked(f"current_task_invalid:{task_id}")
    if task_id != f"construct:{arm}:{chunk_id}:planner":
        raise _blocked(f"current_task_id_shape_invalid:{task_id}")

    package = _package(corpus, str(chunk_id))
    try:
        corpus_fingerprint = input_fingerprint(package)
    except Exception as exc:
        raise _blocked(f"corpus_fingerprint_failed:{type(exc).__name__}") from exc
    if corpus_fingerprint != fingerprint:
        raise _blocked("corpus_input_fingerprint_mismatch")

    if task_id in checkpoints:
        # The origin run already recorded a result for this task, so the paid
        # bytes are reachable without re-derivation.  Re-deriving would risk
        # importing something that disagrees with what the origin run recorded.
        raise _blocked(f"origin_checkpoint_present:{task_id}")

    expected = {
        "freeze_sha256": origin_freeze_sha256,
        "phase": "construction",
        "role": "generator",
        "task_id": task_id,
        "construction_stage": "planner",
        "arm": arm,
        "chunk_id": chunk_id,
        "provider": generator["provider"],
        "model": generator["model"],
        "outcome": "OK",
        "http_status": 200,
        "retry_count": 0,
    }
    for field, value in expected.items():
        if terminal.get(field) != value:
            raise _blocked(f"terminal_metadata_mismatch:{field}")
    if terminal.get("returned_model") != generator["model"]:
        raise _blocked("returned_model_mismatch")

    kind, _interface_kind = PLANNER_ARMS[str(arm)]
    interface, upstream_task_ids = rebuild_origin_interface(
        corpus=corpus, origin_run=origin_run, chunk_id=str(chunk_id), kind=kind,
        checkpoints=checkpoints, origin_freeze_sha256=origin_freeze_sha256,
    )
    payload = rebuild_planner_payload(
        corpus=corpus, chunk_id=str(chunk_id), arm=str(arm), interface=interface
    )
    request = stamp_request(payload, provider=generator["provider"],
                            model=generator["model"])
    payload_sha = sha256_bytes(canonical(request).encode("utf-8"))
    if terminal.get("payload_sha256") != payload_sha:
        raise _blocked("payload_sha256_mismatch")
    prompt_sha = sha256_bytes(canonical(request.get("messages", [])).encode("utf-8"))
    if terminal.get("prompt_sha256") != prompt_sha:
        raise _blocked("prompt_sha256_mismatch")

    metadata = {
        "phase": "construction",
        "role": "generator",
        "task_id": task_id,
        "construction_stage": "planner",
        "arm": arm,
        "chunk_id": chunk_id,
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

    result = classify_planner_body(
        body, arm=str(arm), chunk_id=str(chunk_id),
        input_fingerprint_value=fingerprint,
    )
    return {
        "task_id": task_id,
        "arm": arm,
        "chunk_id": chunk_id,
        "construction_stage": "planner",
        "status": result["status"],
        "result": result,
        "upstream_task_ids": upstream_task_ids,
        "import_origin": {
            "origin_freeze_sha256": origin_freeze_sha256,
            "origin_run": Path(origin_run).name,
            "origin_task_id": task_id,
            "origin_idempotency_key": idempotency,
            "origin_payload_sha256": payload_sha,
            "origin_response_sha256": terminal.get("response_sha256"),
        },
    }


def verify_construction_imports(*, origin_run: Path, origin_freeze_path: Path,
                                current_freeze_path: Path, corpus: Corpus,
                                current_plan: Mapping[str, Any],
                                task_ids: Iterable[str]) -> dict[str, Any]:
    """Build a canonical construction IMPORT_MANIFEST from one preserved run.

    Every selected record appears exactly once, in ``verified`` or in
    ``rejected`` with a stable reason.  Nothing is repaired and nothing is
    dropped, so ``verified + rejected == selected`` always holds.  Unlike the
    extraction importer there is no implicit "all terminals" mode: a paid
    construction call is only ever imported by explicit task id.
    """
    origin_run = Path(origin_run)
    origin_freeze_path = Path(origin_freeze_path)
    current_freeze_path = Path(current_freeze_path)

    origin_freeze = _read_object(origin_freeze_path, "origin_freeze")
    current_freeze = _read_object(current_freeze_path, "current_freeze")
    origin_freeze_sha = sha256_file(origin_freeze_path)
    current_freeze_sha = sha256_file(current_freeze_path)
    if origin_freeze_sha == current_freeze_sha:
        raise _blocked("freeze_identical_import_unnecessary")
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

    selected = list(task_ids)
    if not selected:
        raise _blocked("selection_empty")
    if any(not isinstance(value, str) or not value for value in selected):
        raise _blocked("selection_invalid")
    if len(selected) != len(set(selected)):
        raise _blocked("selection_not_unique")
    selected = sorted(selected)

    try:
        terminals, starts = read_origin_terminals(origin_run / "RUN_LEDGER.jsonl")
    except _ExtractionBlocked as exc:
        raise _blocked(
            str(exc).removeprefix("BLOCKED_EXTRACTION_IMPORT:")
        ) from exc
    checkpoints = read_origin_checkpoints(origin_run / "EXECUTION.jsonl")

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
            verified.append(verify_construction_planner_record(
                terminal=bucket[0], task=task, corpus=corpus,
                origin_run=origin_run, origin_freeze_sha256=origin_freeze_sha,
                checkpoints=checkpoints, generator=generator,
            ))
        except ConstructionImportBlocked as exc:
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


def validate_construction_import_manifest(manifest: Mapping[str, Any]) -> None:
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
    for row in verified:
        if not isinstance(row, Mapping):
            raise _blocked("manifest_record_not_object")
        if row.get("status") != IMPORTABLE_STATUS:
            raise _blocked(f"manifest_status_not_importable:{row.get('status')}")
    task_ids = [row.get("task_id") for row in verified] + [
        row.get("task_id") for row in rejected if isinstance(row, Mapping)
    ]
    if len(task_ids) != len(set(task_ids)):
        raise _blocked("manifest_task_ids_not_disjoint")


def construction_import_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a verified record onto ``PhaseExecutor.import_completed`` kwargs.

    The returned mapping is exactly ``task_id``, ``result``, ``status`` and
    ``import_origin``, so a caller can splat it.  The provenance block is
    re-validated here (all-64-hex, ``origin_task_id`` bound to ``task_id``)
    because that is the shape the executor enforces on import.
    """
    task_id = record.get("task_id")
    result = record.get("result")
    status = record.get("status")
    origin = record.get("import_origin")
    if not isinstance(task_id, str) or not task_id:
        raise _blocked("entry_task_id_invalid")
    if not isinstance(result, Mapping) or result.get("status") != IMPORTABLE_STATUS:
        raise _blocked("entry_result_invalid")
    if result.get("calls_used") != 1:
        raise _blocked("entry_result_calls_invalid")
    if status != IMPORTABLE_STATUS:
        raise _blocked(f"entry_status_not_importable:{status}")
    if not isinstance(origin, Mapping):
        raise _blocked("entry_origin_missing")
    required = {
        "origin_freeze_sha256", "origin_run", "origin_task_id",
        "origin_idempotency_key", "origin_payload_sha256",
        "origin_response_sha256",
    }
    if set(origin) != required or origin.get("origin_task_id") != task_id:
        raise _blocked("entry_origin_invalid")
    for key in required - {"origin_run", "origin_task_id"}:
        value = origin.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise _blocked(f"entry_origin_invalid:{key}")
    if not isinstance(origin.get("origin_run"), str) or not origin["origin_run"]:
        raise _blocked("entry_origin_invalid:origin_run")
    return {
        "task_id": task_id,
        "result": dict(result),
        "status": status,
        "import_origin": dict(origin),
    }


def write_construction_import_manifest(path: Path, manifest: Mapping[str, Any]) -> Path:
    """Write a validated manifest without overwriting a differing artifact."""
    validate_construction_import_manifest(manifest)
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
    "IMPORTABLE_STATUS",
    "IMPORT_SCHEMA",
    "PLANNER_ARMS",
    "ConstructionImportBlocked",
    "chunk_image_index",
    "classify_planner_body",
    "construction_import_entry",
    "load_origin_extraction_result",
    "read_origin_checkpoints",
    "rebuild_origin_interface",
    "rebuild_planner_payload",
    "stamp_request",
    "validate_construction_import_manifest",
    "verify_construction_import_manifest_path" if False else "verify_construction_imports",
    "verify_construction_planner_record",
    "write_construction_import_manifest",
]
