#!/usr/bin/env python3
"""Execute the frozen ECM--TQAG paid phases, one phase at a time.

The command is intentionally boring and fail-closed: it imports the verified smoke,
then runs extraction -> construction -> sensitivity -> probes -> audit -> judging.
It never starts a later phase when a prerequisite is incomplete.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ecm_tqag.controls import load_controls
from ecm_tqag.io import sha256_file
from ecm_tqag.manifest import load_corpus
from ecm_tqag.run.experiment import build_phase_plan
from ecm_tqag.run.live import (
    build_artifact_loader,
    build_extraction_loader,
    build_judging_frame_from_results,
    evaluate_sensitivity_results,
    load_task_result,
    paid_ledger_limits,
)
from ecm_tqag.run.paid import PaidRunBlocked, build_paid_runner, preflight_paid_run
from ecm_tqag.run.paid_worker import PaidPhaseWorker
from ecm_tqag.run.import_extraction import (
    verify_extraction_imports,
    write_import_manifest,
)
from ecm_tqag.run.import_construction import (
    construction_import_entry,
    verify_construction_imports,
    write_construction_import_manifest,
)
from ecm_tqag.run.transport import OpenAITransport, TransportConfig


def _key(path: Path) -> Path:
    if not path.is_file() or path.stat().st_mode & 0o077 or not path.read_text(encoding="utf-8").strip():
        raise ValueError(f"BLOCKED_PAID_CLI:key_invalid:{path}")
    return path


def _completion_url(base: str) -> str:
    """Accept an API root or an exact chat-completions endpoint."""
    value = base.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    return value + "/chat/completions"


def _configs(freeze: dict[str, Any], *, openrouter_key: Path, omniproxy_key: Path,
             openrouter_url: str, omniproxy_url: str) -> dict[str, TransportConfig]:
    out: dict[str, TransportConfig] = {}
    for role, record in freeze["roles"].items():
        provider = record["provider"]
        if provider == "openrouter":
            url, key = _completion_url(openrouter_url), openrouter_key
        elif provider == "omniproxy":
            url, key = _completion_url(omniproxy_url), omniproxy_key
        else:
            raise ValueError(f"BLOCKED_PAID_CLI:provider_invalid:{role}")
        out[role] = TransportConfig(provider=provider, model=record["model"],
                                    base_url=url, api_key_file=_key(key),
                                    allow_fallbacks=False, max_retries=0)
    return out


def _phase_rows(run_dir: Path, plan: dict[str, Any], phase: str) -> list[dict[str, Any]]:
    rows = []
    for task in plan["tasks"]:
        if task.get("phase") == phase:
            rows.append(load_task_result(run_dir, task["task_id"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Run frozen ECM--TQAG paid experiment")
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--smoke", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--openrouter-key", type=Path, required=True)
    ap.add_argument("--omniproxy-key", type=Path, required=True)
    ap.add_argument("--openrouter-url", required=True)
    ap.add_argument("--omniproxy-url", required=True)
    ap.add_argument("--import-extraction-run", type=Path, action="append", default=[])
    ap.add_argument("--import-extraction-freeze", type=Path, action="append", default=[])
    ap.add_argument("--import-construction-run", type=Path, action="append", default=[])
    ap.add_argument("--import-construction-freeze", type=Path, action="append", default=[])
    ap.add_argument("--import-construction-task-id", action="append", default=[])
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    corpus = load_corpus(args.manifest)
    controls = load_controls(args.controls)
    auth = preflight_paid_run(freeze_path=args.freeze, smoke_path=args.smoke,
                              manifest_path=args.manifest, controls_path=args.controls,
                              execute=args.execute)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # The paid executor owns the resumable checkpoint.  The transport ledger is
    # created below and is shared by every phase through the injected worker.
    from ecm_tqag.run.ledger import RunLedger
    smoke_record = json.loads(args.smoke.read_text(encoding="utf-8"))
    paid_cap, retry_reserve = paid_ledger_limits(freeze, smoke_record)
    ledger = RunLedger(run_dir / "RUN_LEDGER.jsonl", freeze_sha256=sha256_file(args.freeze),
                       cap=paid_cap, retry_reserve=retry_reserve)
    transport = OpenAITransport(ledger)
    configs = _configs(freeze, openrouter_key=args.openrouter_key,
                       omniproxy_key=args.omniproxy_key, openrouter_url=args.openrouter_url,
                       omniproxy_url=args.omniproxy_url)
    plan = auth["plan"]
    worker = PaidPhaseWorker(
        corpus=corpus, freeze=freeze, transport=transport, role_configs=configs,
        response_root=run_dir, extraction_loader=build_extraction_loader(corpus=corpus, run_dir=run_dir),
        artifact_loader=build_artifact_loader(run_dir), controls=controls,
    )
    # Build one executor for the entire run so smoke imports, dependency checks,
    # checkpoint replay, and resume semantics are shared across all phases.
    runner = build_paid_runner(
        freeze_path=args.freeze, smoke_path=args.smoke,
        manifest_path=args.manifest, controls_path=args.controls,
        run_dir=run_dir, execute=args.execute, worker=worker,
    )
    import_manifests: list[dict[str, Any]] = []
    construction_import_manifests: list[dict[str, Any]] = []
    if len(args.import_extraction_run) != len(args.import_extraction_freeze):
        raise ValueError("BLOCKED_PAID_CLI:import_arguments_must_be_paired")
    if not (
        len(args.import_construction_run)
        == len(args.import_construction_freeze)
        == len(args.import_construction_task_id)
    ):
        raise ValueError("BLOCKED_PAID_CLI:construction_import_arguments_must_be_paired")
    imported_task_ids: set[str] = set()
    for origin_run, origin_freeze in zip(
            args.import_extraction_run, args.import_extraction_freeze, strict=True):
        import_manifest = verify_extraction_imports(
            origin_run=origin_run,
            origin_freeze_path=origin_freeze,
            current_freeze_path=args.freeze,
            corpus=corpus,
            current_plan=plan,
        )
        if import_manifest["rejected_count"]:
            raise ValueError(
                f"BLOCKED_PAID_CLI:import_rejections:{origin_run}:"
                f"{import_manifest['rejected_count']}"
            )
        import_manifests.append(import_manifest)
        for entry in import_manifest["records"]["verified"]:
            if entry["task_id"] in imported_task_ids:
                raise ValueError(f"BLOCKED_PAID_CLI:duplicate_import:{entry['task_id']}")
            runner.executor.import_completed(
                entry["task_id"], entry["result"],
                status=entry["status"],
                import_origin=entry["import_origin"],
            )
            imported_task_ids.add(entry["task_id"])
    for origin_run, origin_freeze, task_id in zip(
            args.import_construction_run,
            args.import_construction_freeze,
            args.import_construction_task_id,
            strict=True):
        construction_manifest = verify_construction_imports(
            origin_run=origin_run,
            origin_freeze_path=origin_freeze,
            current_freeze_path=args.freeze,
            corpus=corpus,
            current_plan=plan,
            task_ids=[task_id],
        )
        if construction_manifest["rejected_count"]:
            raise ValueError(
                f"BLOCKED_PAID_CLI:construction_import_rejections:{origin_run}:"
                f"{construction_manifest['rejected_count']}"
            )
        construction_import_manifests.append(construction_manifest)
        for record in construction_manifest["records"]["verified"]:
            entry = construction_import_entry(record)
            if entry["task_id"] in imported_task_ids:
                raise ValueError(f"BLOCKED_PAID_CLI:duplicate_import:{entry['task_id']}")
            runner.executor.import_completed(**entry)
            imported_task_ids.add(entry["task_id"])
    expected_satisfied = freeze.get("full_call_budget", {}).get("satisfied_calls", 0)
    if len(imported_task_ids) != expected_satisfied:
        raise ValueError(
            f"BLOCKED_PAID_CLI:verified_import_count_mismatch:"
            f"{len(imported_task_ids)}/{expected_satisfied}"
        )
    if import_manifests or construction_import_manifests:
        combined = {
            "schema": "ecm-tqag.paid-import-set.v1",
            "status": "VERIFIED",
            "verified_count": len(imported_task_ids),
            "task_ids": sorted(imported_task_ids),
            "extraction_manifests": import_manifests,
            "construction_manifests": construction_import_manifests,
        }
        (run_dir / "IMPORT_MANIFEST_SET.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for index, manifest in enumerate(construction_import_manifests, 1):
            write_construction_import_manifest(
                run_dir / f"CONSTRUCTION_IMPORT_MANIFEST_{index:02d}.json", manifest
            )
    reports: dict[str, Any] = {}
    for phase in ("extraction", "construction", "sensitivity_floor"):
        reports[phase] = runner.run_phase(phase)
    sensitivity = evaluate_sensitivity_results(_phase_rows(run_dir, plan, "sensitivity_floor"))
    (run_dir / "SENSITIVITY_VERDICT.json").write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not sensitivity["passed"]:
        raise PaidRunBlocked("BLOCKED_PAID_RUN:sensitivity_floor_failed")
    reports["secondary_probes"] = runner.run_phase("secondary_probes", floor_passed=True)
    reports["image_audit"] = runner.run_phase("image_audit")
    frame = build_judging_frame_from_results(plan=plan, run_dir=run_dir, freeze_sha256=sha256_file(args.freeze))
    (run_dir / "JUDGING_FRAME.json").write_text(json.dumps(frame, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    worker.judging_frame = frame
    reports["judging"] = runner.run_phase("judging")
    result = {"schema": "ecm-tqag.paid-run.v1", "status": "COMPLETE", "freeze_sha256": sha256_file(args.freeze), "reports": reports, "sensitivity": sensitivity, "http_attempts_used": ledger.attempts_used}
    (run_dir / "PAID_RUN_SUMMARY.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "http_attempts_used": ledger.attempts_used, "sensitivity_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
