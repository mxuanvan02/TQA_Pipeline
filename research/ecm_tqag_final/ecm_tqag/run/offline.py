"""Network-free rehearsal of a frozen phase plan.

This runner exercises checkpointing, result sidecars, call parity and resume
semantics without importing or invoking the HTTP transport.  It is deliberately
separate from the paid executor: a successful rehearsal is evidence about run
bookkeeping, never evidence that a model call happened.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from ..io import canonical, sha256_bytes

SCHEMA = "ecm-tqag.offline-run.v1"


def _blocked(reason: str) -> ValueError:
    return ValueError(f"BLOCKED_OFFLINE:{reason}")


def _validate_freeze_sha256(value: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise _blocked("invalid_freeze_sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise _blocked("invalid_freeze_sha256") from exc


def _read_records(
    path: Path,
    freeze_sha256: str,
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
        except Exception as exc:
            raise _blocked(f"invalid_checkpoint_line:{line_no}") from exc
        if not isinstance(record, dict):
            raise _blocked(f"checkpoint_not_object:{line_no}")
        if record.get("freeze_sha256") != freeze_sha256:
            raise _blocked("freeze_mismatch")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise _blocked(f"missing_task_id:{line_no}")
        if task_id in completed:
            raise _blocked(f"duplicate_task:{task_id}")
        planned = tasks_by_id.get(task_id)
        if planned is None:
            raise _blocked(f"checkpoint_task_not_in_plan:{task_id}")
        if record.get("phase") != planned.get("phase"):
            raise _blocked(f"checkpoint_metadata_mismatch:{task_id}:phase")
        result_path = record.get("result_path")
        digest = record.get("result_sha256")
        if not isinstance(result_path, str) or not result_path or not isinstance(digest, str) or len(digest) != 64:
            raise _blocked(f"checkpoint_metadata_mismatch:{task_id}:sidecar")
        sidecar = root / result_path
        try:
            payload = sidecar.read_bytes()
        except Exception as exc:
            raise _blocked(f"sidecar_unreadable:{task_id}") from exc
        if sha256_bytes(payload) != digest:
            raise _blocked(f"sidecar_hash_mismatch:{task_id}")
        try:
            result = json.loads(payload)
        except Exception as exc:
            raise _blocked(f"sidecar_invalid_json:{task_id}") from exc
        if (
            not isinstance(result, dict)
            or result.get("freeze_sha256") != freeze_sha256
            or result.get("task_id") != task_id
            or result.get("phase") != planned.get("phase")
            or result.get("planned_calls") != planned.get("calls")
            or result.get("network_called") is not False
            or result.get("status") != "OFFLINE_REHEARSED"
        ):
            raise _blocked(f"sidecar_metadata_mismatch:{task_id}")
        completed[task_id] = record
    return completed


def _append_0600(path: Path, value: Mapping[str, Any]) -> None:
    line = canonical(dict(value)) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def _validate_tasks(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema") != "ecm-tqag.phase-plan.v1":
        raise _blocked("invalid_plan_schema")
    raw = plan.get("tasks")
    if not isinstance(raw, list):
        raise _blocked("tasks_not_list")
    tasks: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, task in enumerate(raw):
        if not isinstance(task, dict):
            raise _blocked(f"task_not_object:{index}")
        task_id, phase, calls = task.get("task_id"), task.get("phase"), task.get("calls")
        if not isinstance(task_id, str) or not task_id or task_id in ids:
            raise _blocked(f"invalid_or_duplicate_task_id:{index}")
        if not isinstance(phase, str) or not phase:
            raise _blocked(f"invalid_phase:{task_id}")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise _blocked(f"invalid_calls:{task_id}")
        deterministic_rescore = (
            calls == 0 and phase == "construction"
            and task.get("arm") == "gates_off"
            and task.get("deterministic_rescore") is True
        )
        if calls == 0 and not deterministic_rescore:
            raise _blocked(f"invalid_calls:{task_id}")
        ids.add(task_id)
        tasks.append(dict(task))
    return tasks


def run_offline_plan(
    plan: Mapping[str, Any],
    run_dir: Path,
    *,
    freeze_sha256: str,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Replay every planned task into deterministic sidecars, with safe resume."""
    _validate_freeze_sha256(freeze_sha256)
    tasks = _validate_tasks(plan)
    if stop_after is not None and (isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after < 0):
        raise _blocked("invalid_stop_after")

    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "OFFLINE_RUN.jsonl"
    sidecar_dir = root / "offline_results"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_records(
        ledger_path,
        freeze_sha256,
        {task["task_id"]: task for task in tasks},
        root,
    )
    unknown = sorted(set(existing) - {task["task_id"] for task in tasks})
    if unknown:
        raise _blocked("checkpoint_task_not_in_plan:" + ",".join(unknown))

    newly_completed = 0
    skipped = 0
    for task in tasks:
        task_id = task["task_id"]
        if task_id in existing:
            skipped += 1
            continue
        if stop_after is not None and newly_completed >= stop_after:
            break
        result = {
            "schema": SCHEMA,
            "freeze_sha256": freeze_sha256,
            "task_id": task_id,
            "phase": task["phase"],
            "planned_calls": task["calls"],
            "network_called": False,
            "status": "OFFLINE_REHEARSED",
        }
        payload = canonical(result).encode("utf-8")
        digest = sha256_bytes(payload)
        sidecar = sidecar_dir / f"{digest}.json"
        if not sidecar.exists():
            sidecar.write_bytes(payload)
            os.chmod(sidecar, 0o600)
        record = {
            "schema": SCHEMA,
            "record_type": "TASK_TERMINAL",
            "freeze_sha256": freeze_sha256,
            "task_id": task_id,
            "phase": task["phase"],
            "result_sha256": digest,
            "result_path": str(sidecar.relative_to(root)),
            "http_attempts": 0,
        }
        _append_0600(ledger_path, record)
        existing[task_id] = record
        newly_completed += 1

    return {
        "schema": SCHEMA,
        "planned": len(tasks),
        "completed": len(existing),
        "newly_completed": newly_completed,
        "skipped_existing": skipped,
        "remaining": len(tasks) - len(existing),
        "http_attempts": 0,
        "complete": len(existing) == len(tasks),
    }


__all__ = ["run_offline_plan"]
