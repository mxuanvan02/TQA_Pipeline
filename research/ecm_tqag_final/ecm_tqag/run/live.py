"""Live paid-run assembly helpers.

This module contains deterministic glue only: result loading, multi-image interface
composition, sensitivity-floor evaluation, judging-frame construction, and the
remaining HTTP-ledger budget.  Network egress remains confined to ``transport``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..interface_merge import merge_caption_interfaces, merge_graph_interfaces
from ..io import canonical, sha256_bytes
from ..manifest import Corpus
from ..protocol import JUDGING_ARMS, fixed_judging_frame
from ..stats.sensitivity import family_sensitivity, sensitivity_floor


def _blocked(reason: str) -> ValueError:
    return ValueError(f"BLOCKED_LIVE_PAID:{reason}")


def _result_path(run_dir: Path, task_id: str) -> Path:
    digest = hashlib.sha256(
        json.dumps(task_id, ensure_ascii=False, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path(run_dir) / "results" / f"{digest}.json"


def load_task_result(run_dir: Path, task_id: str) -> dict[str, Any]:
    """Load one deterministic task sidecar, rejecting malformed content."""
    if not isinstance(task_id, str) or not task_id:
        raise _blocked("task_id_invalid")
    path = _result_path(run_dir, task_id)
    if not path.is_file():
        raise FileNotFoundError(task_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(f"result_unreadable:{task_id}:{type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise _blocked(f"result_not_object:{task_id}")
    recorded = value.get("task_id")
    if recorded is not None and recorded != task_id:
        raise _blocked(f"result_task_mismatch:{task_id}")
    return value


def build_artifact_loader(run_dir: Path) -> Callable[[str], Mapping[str, Any]]:
    root = Path(run_dir)
    return lambda task_id: load_task_result(root, task_id)


def _image_indexes(corpus: Corpus) -> dict[str, list[tuple[int, int]]]:
    """Map chunk to (global image index, declared order), preserving freeze order."""
    out: dict[str, list[tuple[int, int]]] = {}
    global_index = 0
    for package in corpus.tlv:
        rows: list[tuple[int, int]] = []
        for image in sorted(package["evidence"]["images"],
                            key=lambda row: row["declared_order"]):
            global_index += 1
            rows.append((global_index, int(image["declared_order"])))
        out[str(package["chunk_id"])] = rows
    if global_index != 18:
        raise _blocked(f"image_census_mismatch:{global_index}")
    return out


def build_extraction_loader(*, corpus: Corpus, run_dir: Path
                            ) -> Callable[[dict[str, Any]], Mapping[str, Any]]:
    """Build a loader that composes every image belonging to a chunk."""
    indexes = _image_indexes(corpus)
    root = Path(run_dir)

    def load(task: dict[str, Any]) -> Mapping[str, Any]:
        chunk_id = task.get("chunk_id")
        kind = task.get("extraction_kind")
        if chunk_id not in indexes:
            raise _blocked(f"extraction_chunk_invalid:{chunk_id}")
        if kind not in {"graph", "caption", "ocr_graph"}:
            raise _blocked(f"extraction_kind_invalid:{kind}")
        records: list[dict[str, Any]] = []
        rejected: list[str] = []
        for global_index, declared_order in indexes[str(chunk_id)]:
            result = load_task_result(root, f"extract:{kind}:{global_index:02d}")
            if result.get("status") == "SCHEMA_REJECTED":
                rejected.append(f"extract:{kind}:{global_index:02d}")
                continue
            if result.get("status") not in {None, "OK"}:
                raise _blocked(f"extraction_not_ok:{kind}:{global_index}")
            interface = result.get("interface")
            if not isinstance(interface, Mapping):
                raise _blocked(f"extraction_interface_missing:{kind}:{global_index}")
            records.append({"declared_order": declared_order,
                            "interface": dict(interface)})
        if rejected:
            return {
                "status": "NOT_APPLICABLE",
                "reason": "upstream_unavailable",
                "source_reason": "schema_rejected",
                "calls_used": 0,
                "extraction_kind": kind,
                "chunk_id": chunk_id,
                "source_task_ids": rejected,
            }
        merged = (merge_caption_interfaces(records) if kind == "caption"
                  else merge_graph_interfaces(records))
        return {"status": "OK", "extraction_kind": kind,
                "chunk_id": chunk_id, "interface": merged}

    return load


def evaluate_sensitivity_results(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered 8/10 and <=10% replicate-disagreement floor."""
    materialized = [dict(row) for row in rows]
    family_records: list[dict[str, Any]] = []
    for family in ("answerer_a", "answerer_b"):
        selected = [row for row in materialized if row.get("answerer_role") == family]
        if len(selected) != 10:
            raise _blocked(f"sensitivity_count:{family}:{len(selected)}")
        positives = [row for row in selected if row.get("control_type") == "positive_visual"]
        negatives = [row for row in selected if row.get("control_type") == "negative_text_sufficient"]
        if len(positives) != 5 or len(negatives) != 5:
            raise _blocked(f"sensitivity_type_counts:{family}:{len(positives)}:{len(negatives)}")

        def first_correct(row: Mapping[str, Any]) -> bool:
            value = row.get("correct")
            if (not isinstance(value, list) or len(value) != 2
                    or any(not isinstance(x, bool) for x in value)):
                raise _blocked(f"sensitivity_correct_invalid:{family}")
            return value[0]

        agreements: list[bool] = []
        for row in positives + negatives:
            agreement = row.get("replicate_agreement")
            if not isinstance(agreement, bool):
                raise _blocked(f"sensitivity_agreement_invalid:{family}")
            agreements.append(agreement)
        family_records.append(family_sensitivity(
            [first_correct(row) for row in positives],
            [first_correct(row) for row in negatives],
            agreements,
            family=family,
        ))
    return sensitivity_floor(family_records)


def build_judging_frame_from_results(*, plan: Mapping[str, Any], run_dir: Path,
                                     freeze_sha256: str) -> dict[str, Any]:
    """Construct the frozen 40-item frame from complete common-arm chunks.

    Missingness is handled before outcome inspection: a chunk absent from any one
    generating arm is excluded from every arm.  At least 14/16 common chunks are
    required, so controlled schema rejection cannot silently erode the census.
    """
    tasks = plan.get("tasks")
    if not isinstance(tasks, list):
        raise _blocked("phase_tasks_missing")
    by_chunk: dict[str, dict[str, dict[str, Any]]] = {}
    unavailable: set[str] = set()
    for task in tasks:
        if not isinstance(task, Mapping) or task.get("phase") != "construction":
            continue
        arm = task.get("arm")
        stage = task.get("construction_stage")
        if arm not in JUDGING_ARMS or stage not in {"realizer", "direct"}:
            continue
        task_id = task.get("task_id")
        if not isinstance(task_id, str):
            raise _blocked("judging_task_id_invalid")
        result = load_task_result(Path(run_dir), task_id)
        chunk_id = task.get("chunk_id")
        if result.get("arm") != arm or result.get("chunk_id") != chunk_id:
            raise _blocked(f"judging_candidate_metadata_invalid:{task_id}")
        if (result.get("status") == "NOT_APPLICABLE"
                and result.get("reason") == "upstream_unavailable"):
            unavailable.add(str(chunk_id))
            continue
        if result.get("status") != "PARSED":
            raise _blocked(f"judging_candidate_invalid:{task_id}")
        item = result.get("item")
        if not isinstance(item, Mapping):
            raise _blocked(f"judging_item_missing:{task_id}")
        row = {
            "item_id": task_id,
            "arm": arm,
            "chunk_id": chunk_id,
            "item": dict(item),
            "item_payload_sha256": sha256_bytes(canonical(item).encode("utf-8")),
        }
        by_chunk.setdefault(str(chunk_id), {})[str(arm)] = row
    complete_chunks = sorted(
        chunk for chunk, arm_rows in by_chunk.items()
        if chunk not in unavailable and set(arm_rows) == set(JUDGING_ARMS)
    )
    if len(complete_chunks) < 14:
        raise _blocked(f"common_chunk_yield_below_14:{len(complete_chunks)}")
    candidates = [by_chunk[chunk][arm] for chunk in complete_chunks for arm in JUDGING_ARMS]
    frame = fixed_judging_frame(candidates, freeze_sha256=freeze_sha256)
    dropped_chunks = sorted(set(by_chunk) - set(complete_chunks) | unavailable)
    frame["common_chunk_count"] = len(complete_chunks)
    frame["dropped_chunks"] = dropped_chunks
    frame["excluded_chunks"] = dropped_chunks
    frame["missingness_policy"] = "listwise_common_chunk_before_outcome_inspection"
    return frame


def paid_ledger_limits(freeze: Mapping[str, Any], smoke: Mapping[str, Any]) -> tuple[int, int]:
    """Return remaining cap after carrying the six verified smoke attempts forward."""
    attempts = smoke.get("http_attempts_used")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts != 6:
        raise _blocked(f"smoke_attempt_count:{attempts}")
    budget = freeze.get("full_call_budget")
    if not isinstance(budget, Mapping):
        raise _blocked("budget_missing")
    cap = budget.get("current_ledger_cap")
    reserve = budget.get("retry_reserve")
    if (isinstance(cap, bool) or not isinstance(cap, int)
            or isinstance(reserve, bool) or not isinstance(reserve, int)
            or cap < attempts or reserve < 0):
        raise _blocked("budget_invalid")
    return cap - attempts, reserve


__all__ = [
    "build_artifact_loader", "build_extraction_loader",
    "build_judging_frame_from_results", "evaluate_sensitivity_results",
    "load_task_result", "paid_ledger_limits",
]
