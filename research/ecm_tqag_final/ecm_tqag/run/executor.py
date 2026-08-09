"""Fail-closed, resumable phase executor.

This layer only orchestrates already-approved tasks.  The worker is injected, so
unit tests can prove ordering, resume and integrity without making HTTP calls.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping


class ExecutionBlocked(RuntimeError):
    pass


def _blocked(reason: str) -> ExecutionBlocked:
    return ExecutionBlocked(f"BLOCKED_EXECUTION:{reason}")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class PhaseExecutor:
    def __init__(self, root: Path, *, freeze_sha256: str,
                 plan: Mapping[str, Any], worker: Callable[[dict[str, Any]], Mapping[str, Any]]):
        if not isinstance(freeze_sha256, str) or len(freeze_sha256) != 64:
            raise _blocked("invalid_freeze_sha256")
        if plan.get("schema") != "ecm-tqag.phase-plan.v1":
            raise _blocked("invalid_phase_plan")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.root / "EXECUTION.jsonl"
        self.freeze_sha256 = freeze_sha256
        self.plan = dict(plan)
        self.worker = worker
        tasks = plan.get("tasks")
        if not isinstance(tasks, list):
            raise _blocked("plan_tasks_missing")
        self.tasks = {str(t["task_id"]): dict(t) for t in tasks if isinstance(t, dict) and t.get("task_id")}
        if len(self.tasks) != len(tasks):
            raise _blocked("duplicate_or_invalid_task")
        for task_id, task in self.tasks.items():
            calls = task.get("calls")
            deterministic_rescore = (
                calls == 0
                and task.get("phase") == "construction"
                and task.get("arm") == "gates_off"
                and task.get("deterministic_rescore") is True
            )
            if (isinstance(calls, bool) or not isinstance(calls, int)
                    or calls < 0 or (calls == 0 and not deterministic_rescore)):
                raise _blocked(f"invalid_task_calls:{task_id}")
            phase = task.get("phase")
            if not isinstance(phase, str) or not phase:
                raise _blocked(f"invalid_task_phase:{task_id}")
        counts = plan.get("counts")
        if not isinstance(counts, Mapping):
            raise _blocked("phase_counts_missing")
        actual_counts = {
            phase: sum(int(task["calls"]) for task in self.tasks.values()
                       if task.get("phase") == phase)
            for phase in counts
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in counts.values()) or actual_counts != dict(counts):
            raise _blocked(f"phase_count_mismatch:{actual_counts}/{dict(counts)}")
        self.completed = self._load_checkpoint()

    def _load_checkpoint(self) -> dict[str, dict[str, Any]]:
        done: dict[str, dict[str, Any]] = {}
        if not self.checkpoint.exists():
            return done
        for no, line in enumerate(self.checkpoint.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception as exc:
                raise _blocked(f"checkpoint_invalid_json:{no}") from exc
            if not isinstance(rec, dict):
                raise _blocked(f"checkpoint_not_object:{no}")
            tid = rec.get("task_id")
            if tid not in self.tasks or tid in done:
                raise _blocked(f"checkpoint_task_invalid:{tid}")
            task = self.tasks[tid]
            if rec.get("freeze_sha256") != self.freeze_sha256 or rec.get("phase") != task.get("phase"):
                raise _blocked(f"checkpoint_metadata_mismatch:{tid}")
            rp, digest = rec.get("result_path"), rec.get("result_sha256")
            if not isinstance(rp, str) or not isinstance(digest, str) or len(digest) != 64:
                raise _blocked(f"checkpoint_sidecar_metadata:{tid}")
            relative = Path(rp)
            expected = Path("results") / f"{_sha(tid)}.json"
            if relative != expected or relative.is_absolute() or ".." in relative.parts:
                raise _blocked(f"result_path_invalid:{tid}")
            path = self.root / relative
            if not path.is_file():
                raise _blocked(f"result_missing:{tid}")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise _blocked(f"result_hash_mismatch:{tid}")
            try:
                result = json.loads(payload)
            except Exception as exc:
                raise _blocked(f"result_invalid_json:{tid}") from exc
            if not isinstance(result, dict) or result.get("task_id") != tid or result.get("freeze_sha256") != self.freeze_sha256:
                raise _blocked(f"result_metadata_mismatch:{tid}")
            done[tid] = rec
        return done

    def _append(self, rec: dict[str, Any]) -> None:
        payload = (json.dumps(rec, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(self.checkpoint, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(self.checkpoint, 0o600)

    @staticmethod
    def _write_private(path: Path, payload: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(path, 0o600)

    def import_completed(
        self,
        task_id: str,
        result: Mapping[str, Any],
        *,
        status: str | None = None,
        import_origin: Mapping[str, Any] | None = None,
    ) -> None:
        """Import one independently verified paid result without network egress.

        The checkpoint is rebound to the current freeze, while ``import_origin``
        preserves the immutable provenance of the HTTP attempt.  Only one-call
        smoke, extraction, and construction tasks are importable.  Construction
        imports are deliberately restricted to terminal planner schema rejections;
        this prevents this generic boundary from becoming a way to inject normal
        generated items or to bypass a paid phase.
        """
        if not isinstance(task_id, str) or task_id not in self.tasks:
            raise _blocked(f"import_task_invalid:{task_id}")
        if task_id in self.completed:
            return
        task = self.tasks[task_id]
        phase = task.get("phase")
        planned_calls = task.get("calls")
        if phase not in {"role_smoke", "extraction", "construction"} or planned_calls != 1:
            raise _blocked(f"import_task_not_allowed:{task_id}")
        if status is not None and (not isinstance(status, str) or not status):
            raise _blocked(f"import_status_invalid:{task_id}")
        if not isinstance(result, Mapping):
            raise _blocked(f"import_result_not_object:{task_id}")
        if phase == "construction" and (
            task.get("construction_stage") != "planner"
            or status != "SCHEMA_REJECTED"
            or result.get("status") != "SCHEMA_REJECTED"
            or result.get("calls_used") != 1
        ):
            raise _blocked(f"import_construction_not_schema_rejection:{task_id}")
        if phase in {"extraction", "construction"}:
            if not isinstance(import_origin, Mapping):
                raise _blocked(f"import_origin_missing:{task_id}")
            required = {
                "origin_freeze_sha256", "origin_run", "origin_task_id",
                "origin_idempotency_key", "origin_payload_sha256",
                "origin_response_sha256",
            }
            if set(import_origin) != required or import_origin.get("origin_task_id") != task_id:
                raise _blocked(f"import_origin_invalid:{task_id}")
            for key in required - {"origin_run", "origin_task_id"}:
                value = import_origin.get(key)
                if not isinstance(value, str) or len(value) != 64:
                    raise _blocked(f"import_origin_invalid:{task_id}:{key}")
        elif import_origin is not None:
            raise _blocked(f"import_origin_forbidden:{task_id}")

        imported = dict(result)
        imported.update({
            "schema": "ecm-tqag.execution-result.v1",
            "task_id": task_id,
            "phase": phase,
            "planned_calls": planned_calls,
            "calls_used": planned_calls,
            "freeze_sha256": self.freeze_sha256,
        })
        if status is not None:
            imported["status"] = status
        if import_origin is not None:
            imported["import_origin"] = dict(import_origin)
        payload = _canonical(imported)
        digest = hashlib.sha256(payload).hexdigest()
        rel = Path("results") / f"{_sha(task_id)}.json"
        self._write_private(self.root / rel, payload)
        rec = {
            "schema": "ecm-tqag.execution-checkpoint.v1",
            "task_id": task_id,
            "phase": phase,
            "calls": planned_calls,
            "freeze_sha256": self.freeze_sha256,
            "result_path": str(rel),
            "result_sha256": digest,
            "imported": True,
        }
        if status is not None:
            rec["status"] = status
        self._append(rec)
        self.completed[task_id] = rec

    def run(self, *, phase: str, floor_passed: bool | None = None) -> dict[str, int | str]:
        if phase not in set(self.plan.get("phases", ())) and not any(t.get("phase") == phase for t in self.tasks.values()):
            raise _blocked(f"unknown_phase:{phase}")
        if phase == "secondary_probes" and floor_passed is not True:
            raise _blocked("secondary_requires_floor")
        dependencies = self.plan.get("dependencies", {})
        if not isinstance(dependencies, Mapping):
            raise _blocked("dependencies_missing")
        prerequisites = dependencies.get(phase, [])
        if not isinstance(prerequisites, list) or any(not isinstance(p, str) for p in prerequisites):
            raise _blocked(f"dependencies_invalid:{phase}")
        for prerequisite in prerequisites:
            prerequisite_tasks = [t for t in self.tasks.values()
                                  if t.get("phase") == prerequisite]
            if not prerequisite_tasks:
                raise _blocked(f"prerequisite_phase_missing:{prerequisite}")
            if any(t["task_id"] not in self.completed for t in prerequisite_tasks):
                raise _blocked(f"prerequisite_incomplete:{prerequisite}")
        selected = [t for t in self.tasks.values() if t.get("phase") == phase]
        completed = skipped = 0
        for task in selected:
            tid = task["task_id"]
            if tid in self.completed:
                skipped += 1
                continue
            raw = self.worker(dict(task))
            if not isinstance(raw, Mapping):
                raise _blocked(f"worker_result_not_object:{tid}")
            result = dict(raw)
            calls_used = result.get("calls_used", task.get("calls"))
            if isinstance(calls_used, bool) or not isinstance(calls_used, int) or calls_used < 0:
                raise _blocked(f"call_parity_mismatch:{tid}:{calls_used}/{task.get('calls')}")
            upstream_unavailable = (
                result.get("status") == "NOT_APPLICABLE"
                and result.get("reason") == "upstream_unavailable"
                and calls_used == 0
                and task.get("phase") in {"construction", "image_audit"}
            )
            not_applicable_probe = (
                phase == "secondary_probes"
                and result.get("status") == "NOT_APPLICABLE"
                and result.get("reason") in {"full_item_not_eligible", "upstream_unavailable"}
                and calls_used == 0
            )
            if (not_applicable_probe or upstream_unavailable):
                pass
            elif calls_used != task.get("calls"):
                raise _blocked(f"call_parity_mismatch:{tid}:{calls_used}/{task.get('calls')}")
            result.update({"schema": "ecm-tqag.execution-result.v1", "task_id": tid,
                           "phase": phase, "planned_calls": task.get("calls"),
                           "calls_used": calls_used, "freeze_sha256": self.freeze_sha256})
            payload = _canonical(result)
            digest = hashlib.sha256(payload).hexdigest()
            rel = Path("results") / f"{_sha(tid)}.json"
            self._write_private(self.root / rel, payload)
            rec = {"schema": "ecm-tqag.execution-checkpoint.v1", "task_id": tid,
                   "phase": phase, "calls": task.get("calls"), "freeze_sha256": self.freeze_sha256,
                   "result_path": str(rel), "result_sha256": digest}
            self._append(rec)
            self.completed[tid] = rec
            completed += 1
        return {"phase": phase, "completed": completed, "skipped": skipped}


__all__ = ["ExecutionBlocked", "PhaseExecutor"]
