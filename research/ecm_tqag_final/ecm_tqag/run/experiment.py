"""Fail-closed phase planner for the frozen ECM--TQAG experiment.

This module deliberately plans work before any transport call.  It does not
invent controls or infer outcomes.  The execution layer may consume the plan
only after the freeze and smoke gates have passed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..arms import ARMS
from ..budget import full_call_plan
from ..manifest import Corpus, input_fingerprint
from ..io import sha256_bytes
from ..protocol import JUDGING_ARMS, JUDGING_FRAME_SIZE, JUDGING_POOL_SIZE

PROBE_CONDITIONS = (
    "control",
    "control_replicate",
    "label_permutation",
    "block_shuffle",
    "text_anchor_removal",
)
DIAGNOSTIC_CONDITIONS = ("occlusion", "image_deletion")

PHASES = (
    "role_smoke",
    "extraction",
    "construction",
    "sensitivity_floor",
    "secondary_probes",
    "image_audit",
    "judging",
)
GENERATING_ARMS = tuple(a.name for a in ARMS if not a.deterministic_rescore)


@dataclass(frozen=True)
class PhaseTask:
    phase: str
    task_id: str
    calls: int
    arm: str | None = None
    chunk_id: str | None = None
    deterministic_rescore: bool = False
    extraction_kind: str | None = None
    image_index: int | None = None
    image_sha256: str | None = None
    source_condition: str | None = None
    input_fingerprint: str | None = None
    construction_stage: str | None = None
    parent_task_id: str | None = None
    answerer_role: str | None = None
    control_id: str | None = None
    replicates: int | None = None
    probe_condition: str | None = None
    judge_role: str | None = None
    frame_index: int | None = None


def _blocked(reason: str) -> ValueError:
    return ValueError(f"BLOCKED_RUN:{reason}")


def validate_census(*, chunk_ids: Iterable[str], image_count: int) -> tuple[str, ...]:
    ids = tuple(sorted(chunk_ids))
    if len(ids) != 16 or len(set(ids)) != 16:
        raise _blocked(f"census_chunks:{len(ids)}")
    if image_count != 18:
        raise _blocked(f"census_images:{image_count}")
    if any(not isinstance(x, str) or not x for x in ids):
        raise _blocked("invalid_chunk_id")
    return ids


def build_phase_plan(*, chunk_ids: Iterable[str], image_count: int,
                     control_ids: Iterable[str] | None = None,
                     freeze: Mapping[str, Any] | None = None,
                     corpus: Corpus | None = None) -> dict[str, Any]:
    """Create the complete deterministic call plan without making calls.

    ``control_ids`` must be supplied by a pre-registered fixture file.  A missing
    list blocks the paid runner rather than allowing it to select controls after
    seeing results.
    """
    chunks = validate_census(chunk_ids=chunk_ids, image_count=image_count)
    controls = tuple(control_ids or ())
    if len(controls) != 10 or len(set(controls)) != 10 or any(not x for x in controls):
        raise _blocked("controls_must_be_fixed_10_unique_ids")
    if freeze is not None:
        if freeze.get("schema") is None or freeze.get("state") is None:
            raise _blocked("invalid_freeze_record")
        if freeze.get("judging_frame", {}).get("frame_size") != JUDGING_FRAME_SIZE:
            raise _blocked("judging_frame_size")
        if freeze.get("judging_frame", {}).get("candidate_pool_size") != JUDGING_POOL_SIZE:
            raise _blocked("judging_pool_size")

    tasks: list[PhaseTask] = []
    for role in ("generator", "answerer_a", "answerer_b", "image_auditor", "model_judge_a", "model_judge_b"):
        tasks.append(PhaseTask("role_smoke", f"smoke:{role}", 1))
    image_hashes: list[str] = []
    if corpus is not None:
        for row in corpus.tlv:
            image_hashes.extend(str(image["sha256"]) for image in row["evidence"]["images"])
    if len(image_hashes) != image_count:
        image_hashes = [sha256_bytes(f"planned-image:{i+1}".encode()) for i in range(image_count)]
    for i in range(image_count):
        for kind in ("graph", "caption", "ocr_graph"):
            tasks.append(PhaseTask("extraction", f"extract:{kind}:{i+1:02d}", 1,
                                   extraction_kind=kind, image_index=i + 1,
                                   image_sha256=image_hashes[i]))
    chunk_fps: dict[str, str] = {}
    if corpus is not None:
        packages = {row["chunk_id"]: row for row in corpus.tlv}
        chunk_fps = {chunk: input_fingerprint(packages[chunk]) for chunk in chunks}
    else:
        chunk_fps = {chunk: sha256_bytes(f"planned-chunk:{chunk}".encode()) for chunk in chunks}
    for chunk in chunks:
        for arm in ARMS:
            source_condition = "T" if arm.input_mode == "ocr_only" else "TLV"
            if arm.deterministic_rescore:
                source_id = f"construct:full:{chunk}:realizer"
                tasks.append(PhaseTask(
                    phase="construction",
                    task_id=f"construct:{arm.name}:{chunk}:rescore",
                    calls=0,
                    arm=arm.name,
                    chunk_id=chunk,
                    deterministic_rescore=True,
                    source_condition=source_condition,
                    input_fingerprint=chunk_fps[chunk],
                    construction_stage="rescore",
                    parent_task_id=source_id,
                ))
            elif arm.path == "planner_realizer":
                planner_id = f"construct:{arm.name}:{chunk}:planner"
                tasks.append(PhaseTask(
                    phase="construction", task_id=planner_id, calls=1,
                    arm=arm.name, chunk_id=chunk,
                    source_condition=source_condition,
                    input_fingerprint=chunk_fps[chunk],
                    construction_stage="planner", parent_task_id=None,
                ))
                tasks.append(PhaseTask(
                    phase="construction",
                    task_id=f"construct:{arm.name}:{chunk}:realizer", calls=1,
                    arm=arm.name, chunk_id=chunk,
                    source_condition=source_condition,
                    input_fingerprint=chunk_fps[chunk],
                    construction_stage="realizer", parent_task_id=planner_id,
                ))
            elif arm.path == "direct_generation":
                tasks.append(PhaseTask(
                    phase="construction",
                    task_id=f"construct:{arm.name}:{chunk}:direct", calls=1,
                    arm=arm.name, chunk_id=chunk,
                    source_condition=source_condition,
                    input_fingerprint=chunk_fps[chunk],
                    construction_stage="direct", parent_task_id=None,
                ))
            else:
                raise AssertionError(f"unknown construction path: {arm.path}")
    for family in ("answerer_a", "answerer_b"):
        for control in controls:
            tasks.append(PhaseTask(
                "sensitivity_floor", f"control:{family}:{control}", 2,
                answerer_role=family, control_id=control, replicates=2,
            ))
    # Probe tasks are conditional and must be activated only after the floor passes.
    for chunk in chunks:
        for condition in PROBE_CONDITIONS:
            for family in ("answerer_a", "answerer_b"):
                tasks.append(PhaseTask(
                    "secondary_probes", f"probe:{family}:{chunk}:{condition}", 1,
                    chunk_id=chunk, answerer_role=family,
                    probe_condition=condition,
                    parent_task_id=f"construct:full:{chunk}:realizer",
                    input_fingerprint=chunk_fps[chunk],
                ))
    for i in range(image_count):
        tasks.append(PhaseTask(
            "image_audit", f"audit:image:{i+1:02d}", 1,
            image_index=i + 1, image_sha256=image_hashes[i],
            parent_task_id=f"extract:graph:{i+1:02d}",
        ))
    for item in range(JUDGING_FRAME_SIZE):
        for judge in ("model_judge_a", "model_judge_b"):
            tasks.append(PhaseTask(
                "judging", f"judge:{judge}:{item+1:02d}", 1,
                judge_role=judge, frame_index=item + 1,
            ))

    counts = {phase: sum(t.calls for t in tasks if t.phase == phase) for phase in PHASES}
    expected = {"role_smoke": 6, "extraction": 54, "construction": 128,
                "sensitivity_floor": 40, "secondary_probes": 160,
                "image_audit": 18, "judging": 80}
    if counts != expected:
        raise AssertionError(f"phase call parity mismatch: {counts} != {expected}")
    dependencies = {
        "role_smoke": [],
        "extraction": ["role_smoke"],
        "construction": ["extraction"],
        "sensitivity_floor": ["construction"],
        "secondary_probes": ["construction", "sensitivity_floor"],
        "image_audit": ["extraction"],
        "judging": ["construction"],
    }
    prior_attempts = 6
    satisfied_calls = 0
    operational_http_cap = 550
    if freeze is not None:
        frozen_budget = freeze.get("full_call_budget")
        if not isinstance(frozen_budget, Mapping):
            raise _blocked("frozen_budget_missing")
        frozen_prior = frozen_budget.get("prior_attempts")
        frozen_satisfied = frozen_budget.get("satisfied_calls", 0)
        frozen_cap = freeze.get("operational_http_cap")
        if isinstance(frozen_prior, bool) or not isinstance(frozen_prior, int) or frozen_prior < 0:
            raise _blocked("frozen_prior_attempts_invalid")
        if (isinstance(frozen_satisfied, bool)
                or not isinstance(frozen_satisfied, int)
                or frozen_satisfied < 0):
            raise _blocked("frozen_satisfied_calls_invalid")
        if isinstance(frozen_cap, bool) or not isinstance(frozen_cap, int) or frozen_cap < 1:
            raise _blocked("frozen_operational_cap_invalid")
        prior_attempts = frozen_prior
        satisfied_calls = frozen_satisfied
        operational_http_cap = frozen_cap
    budget = full_call_plan(
        images=image_count,
        prior_attempts=prior_attempts,
        satisfied_calls=satisfied_calls,
        operational_http_cap=operational_http_cap,
    )
    return {"schema": "ecm-tqag.phase-plan.v1", "chunks": list(chunks),
            "controls": list(controls), "generating_arms": list(GENERATING_ARMS),
            "phases": list(PHASES), "counts": counts,
            "dependencies": dependencies,
            "tasks": [t.__dict__ for t in tasks], "budget": budget,
            "secondary_is_conditional": True}


def activate_secondary(plan: Mapping[str, Any], *, floor_passed: bool) -> list[dict[str, Any]]:
    if not floor_passed:
        return []
    return [dict(t) for t in plan.get("tasks", []) if t.get("phase") == "secondary_probes"]
