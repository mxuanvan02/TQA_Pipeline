"""Paid-run preflight and phase authorization primitives.

This module intentionally contains no network transport.  It binds the paid run to
one frozen manifest, one matching role-smoke result, the fixed controls, and the
pre-registered phase plan before a caller is allowed to construct a worker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from ..controls import load_controls
from ..freeze import validate_execution_gate
from ..io import sha256_file
from ..manifest import load_corpus
from .executor import ExecutionBlocked, PhaseExecutor
from .experiment import build_phase_plan


def _blocked(reason: str) -> ValueError:
    return ValueError(f"BLOCKED_PAID_PREFLIGHT:{reason}")


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise _blocked(f"{label}_unreadable:{type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise _blocked(f"{label}_not_object")
    return obj


def _current_ledger_cap(freeze: dict[str, Any]) -> int:
    budget = freeze.get("full_call_budget")
    cap = budget.get("current_ledger_cap") if isinstance(budget, dict) else None
    operational = freeze.get("operational_http_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise _blocked("current_ledger_cap_invalid")
    if isinstance(operational, bool) or not isinstance(operational, int) or cap > operational:
        raise _blocked("current_ledger_cap_invalid")
    return cap


def _validate_smoke_results(smoke: dict[str, Any], freeze: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate per-role smoke metadata before importing completed calls.

    Older synthetic preflight fixtures may omit ``results``; when a result list is
    present it is validated as one complete, exact copy of the frozen role roster.
    """
    raw = smoke.get("results", [])
    if raw == []:
        return []
    roles = freeze.get("roles")
    if not isinstance(roles, dict) or not isinstance(raw, list) or len(raw) != len(roles):
        raise _blocked("smoke_results_mismatch")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            raise _blocked("smoke_results_mismatch")
        role = row.get("role")
        if not isinstance(role, str) or role in seen or role not in roles:
            raise _blocked("smoke_results_mismatch")
        expected = roles[role]
        digest = row.get("response_sha256")
        if (
            row.get("provider") != expected.get("provider")
            or row.get("model") != expected.get("model")
            or row.get("vision_required") != expected.get("vision_required")
            or row.get("status") != "PASS"
            or not isinstance(row.get("idempotency_key"), str)
            or not row["idempotency_key"]
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise _blocked("smoke_results_mismatch")
        seen.add(role)
        out.append(dict(row))
    if seen != set(roles):
        raise _blocked("smoke_results_mismatch")
    return out


def preflight_paid_run(*, freeze_path: Path, smoke_path: Path,
                       manifest_path: Path, controls_path: Path,
                       execute: bool) -> dict[str, Any]:
    """Authorize the paid experiment without making an HTTP call."""
    if execute is not True:
        raise _blocked("execute_flag_missing")
    freeze = _read_object(freeze_path, "freeze")
    smoke = _read_object(smoke_path, "smoke")
    freeze_sha = sha256_file(Path(freeze_path))

    if smoke.get("schema") != "ecm-tqag.role-smoke.v1":
        raise _blocked("smoke_schema_invalid")
    if smoke.get("freeze_sha256") != freeze_sha:
        raise _blocked("smoke_freeze_mismatch")
    if smoke.get("status") != "PASS":
        raise _blocked("smoke_not_passed")
    expected_roles = set(freeze.get("roles", {}))
    passed_roles = smoke.get("passed_roles")
    if not isinstance(passed_roles, list) or set(passed_roles) != expected_roles:
        raise _blocked("smoke_roster_mismatch")
    smoke_results = _validate_smoke_results(smoke, freeze)

    try:
        validate_execution_gate(freeze, execute=True, smoke_passed_roles=expected_roles)
    except ValueError as exc:
        raise _blocked(str(exc).replace("BLOCKED_EXECUTION:", "execution_gate:")) from exc

    corpus = load_corpus(Path(manifest_path))
    controls = load_controls(Path(controls_path))
    control_ids = [row["control_id"] for row in controls["controls"]]
    try:
        plan = build_phase_plan(
            chunk_ids=corpus.chunk_ids,
            image_count=sum(len(row["evidence"]["images"]) for row in corpus.tlv),
            control_ids=control_ids,
            freeze=freeze,
            corpus=corpus,
        )
    except (ValueError, AssertionError) as exc:
        raise _blocked(f"phase_plan_invalid:{exc}") from exc

    return {
        "schema": "ecm-tqag.paid-preflight.v1",
        "status": "AUTHORIZED",
        "freeze_sha256": freeze_sha,
        "chunk_count": len(corpus.chunk_ids),
        "image_count": sum(len(row["evidence"]["images"]) for row in corpus.tlv),
        "control_count": len(control_ids),
        "phase_counts": dict(plan["counts"]),
        "current_ledger_cap": _current_ledger_cap(freeze),
        "retry_reserve": freeze["full_call_budget"]["retry_reserve"],
        "smoke_results": smoke_results,
        "plan": plan,
    }


class PaidRunBlocked(RuntimeError):
    """Stable outer error for paid-run orchestration failures."""


def _paid_blocked(reason: str) -> PaidRunBlocked:
    return PaidRunBlocked(f"BLOCKED_PAID_RUN:{reason}")


class PaidRunner:
    """Authorized phase facade; construction itself performs no work or egress."""

    def __init__(self, *, authorization: Mapping[str, Any], run_dir: Path,
                 worker: Callable[[dict[str, Any]], Mapping[str, Any]]):
        self.authorization = dict(authorization)
        self.executor = PhaseExecutor(
            Path(run_dir),
            freeze_sha256=str(self.authorization["freeze_sha256"]),
            plan=self.authorization["plan"],
            worker=worker,
        )
        for row in self.authorization.get("smoke_results", []):
            role = row["role"]
            self.executor.import_completed(
                f"smoke:{role}",
                {"smoke_result": row},
                status="IMPORTED_VERIFIED_SMOKE",
            )

    def run_phase(self, phase: str, *, floor_passed: bool | None = None) -> dict[str, Any]:
        if phase == "secondary_probes" and floor_passed is not True:
            raise _paid_blocked("secondary_requires_floor")
        try:
            return dict(self.executor.run(phase=phase, floor_passed=floor_passed))
        except ExecutionBlocked as exc:
            reason = str(exc).removeprefix("BLOCKED_EXECUTION:")
            raise _paid_blocked(reason) from exc


def build_paid_runner(*, freeze_path: Path, smoke_path: Path,
                      manifest_path: Path, controls_path: Path, run_dir: Path,
                      execute: bool,
                      worker: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None) -> PaidRunner:
    """Authorize and construct a resumable runner without invoking its worker."""
    if worker is None or not callable(worker):
        raise _paid_blocked("worker_missing")
    authorization = preflight_paid_run(
        freeze_path=freeze_path,
        smoke_path=smoke_path,
        manifest_path=manifest_path,
        controls_path=controls_path,
        execute=execute,
    )
    try:
        return PaidRunner(authorization=authorization, run_dir=run_dir, worker=worker)
    except ExecutionBlocked as exc:
        reason = str(exc).removeprefix("BLOCKED_EXECUTION:")
        raise _paid_blocked(reason) from exc


__all__ = ["PaidRunBlocked", "PaidRunner", "build_paid_runner", "preflight_paid_run"]
