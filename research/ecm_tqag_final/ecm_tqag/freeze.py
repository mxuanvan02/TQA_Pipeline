from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .arms import ARMS
from .budget import full_call_plan
from .controls import CONTROL_SCHEMA, controls_commitment, load_controls
from .io import sha256_bytes, sha256_file
from .manifest import input_fingerprint, load_corpus
from .prompts import DECODING, PLANNER_TEMPLATE_ID, REALIZER_TEMPLATE_ID, planner_static_fingerprint
from .protocol import JUDGING_ARMS, JUDGING_FRAME_SIZE, JUDGING_PER_ARM, JUDGING_POOL_SIZE

FREEZE_SCHEMA = "ecm-tqag.freeze.release"
OPERATIONAL_HTTP_CAP = 550

# Source integrity is frozen by *policy*, not by a hand-maintained list: every
# implementation module reachable from the declared roots is hashed. A new module
# therefore enters the freeze automatically, and a module that disappears or
# changes after freezing is reported as drift by ``_execution_blockers``.
SOURCE_DISCOVERY_POLICY: dict[str, Any] = {
    "id": "ecm-tqag.source-discovery.v1",
    "file_roots": ("dry_run.py", "run_smoke.py", "run_paid.py"),
    "package_roots": ("ecm_tqag",),
    "pattern": "**/*.py",
    "excluded_dir_names": ("__pycache__", ".pytest_cache", "tests", "runs", ".git"),
}

# Minimum set that must always be discovered. This is a fail-closed floor against a
# broken checkout, a packaging error, or an exclusion rule that silently swallows
# implementation code. It is deliberately not the whole list.
REQUIRED_SOURCE_FILES: frozenset[str] = frozenset({
    "dry_run.py",
    "run_smoke.py",
    "ecm_tqag/__init__.py",
    "ecm_tqag/arms.py",
    "ecm_tqag/budget.py",
    "ecm_tqag/contract.py",
    "ecm_tqag/counterfactual_images.py",
    "ecm_tqag/direct.py",
    "ecm_tqag/freeze.py",
    "ecm_tqag/guards.py",
    "ecm_tqag/interfaces.py",
    "ecm_tqag/io.py",
    "ecm_tqag/item_gates.py",
    "ecm_tqag/manifest.py",
    "ecm_tqag/prompts.py",
    "ecm_tqag/protocol.py",
    "ecm_tqag/structure_reader.py",
    "ecm_tqag/run/__init__.py",
    "ecm_tqag/run/ledger.py",
    "ecm_tqag/run/transport.py",
    "ecm_tqag/stats/__init__.py",
    "ecm_tqag/stats/paired.py",
})

ROLE_ROSTER: dict[str, dict[str, Any]] = {
    "generator": {
        "provider": "openrouter",
        "model": "qwen/qwen3-vl-32b-instruct",
        "family": "qwen",
        "approval": "APPROVED",
        "vision_required": True,
    },
    "answerer_a": {
        "provider": "omniproxy",
        "model": "claude-sonnet-5",
        "family": "anthropic",
        "approval": "APPROVED",
        "vision_required": True,
    },
    "answerer_b": {
        "provider": "omniproxy",
        "model": "gpt-5.6-terra",
        "family": "openai",
        "approval": "APPROVED",
        "vision_required": True,
    },
    "image_auditor": {
        "provider": "omniproxy",
        "model": "claude-sonnet-5",
        "family": "anthropic",
        "approval": "APPROVED",
        "vision_required": True,
    },
    "model_judge_a": {
        "provider": "omniproxy",
        "model": "claude-sonnet-5",
        "family": "anthropic",
        "approval": "APPROVED",
        "vision_required": False,
    },
    "model_judge_b": {
        "provider": "omniproxy",
        "model": "gpt-5.6-terra",
        "family": "openai",
        "approval": "APPROVED",
        "vision_required": False,
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def discover_source_files() -> list[str]:
    """Return every implementation source path, relative and deterministically sorted.

    Discovery is by policy so the freezable list does not have to be guessed before
    the implementation is complete. It fails closed if any required module is absent.
    """
    root = project_root()
    excluded = set(SOURCE_DISCOVERY_POLICY["excluded_dir_names"])
    found: set[str] = set()

    for name in SOURCE_DISCOVERY_POLICY["file_roots"]:
        candidate = root / name
        if candidate.is_file():
            found.add(name)

    for package in SOURCE_DISCOVERY_POLICY["package_roots"]:
        package_dir = root / package
        if not package_dir.is_dir():
            raise ValueError(f"BLOCKED_FREEZE:source_package_missing:{package}")
        for path in package_dir.glob(SOURCE_DISCOVERY_POLICY["pattern"]):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if excluded.intersection(relative.parts[:-1]):
                continue
            found.add(relative.as_posix())

    missing = sorted(REQUIRED_SOURCE_FILES - found)
    if missing:
        raise ValueError("BLOCKED_FREEZE:source_discovery_incomplete:" + ",".join(missing))
    return sorted(found)


def _source_hashes() -> dict[str, str]:
    root = project_root()
    return {relative: sha256_file(root / relative) for relative in discover_source_files()}


def build_freeze(manifest_path: Path) -> dict[str, Any]:
    """Validate the full census and return an immutable, secret-free freeze record."""
    try:
        corpus = load_corpus(Path(manifest_path))
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("BLOCKED_"):
            raise
        raise ValueError(f"BLOCKED_INPUT_INTEGRITY:{type(exc).__name__}") from exc

    fingerprints = {row["chunk_id"]: input_fingerprint(row) for row in corpus.tlv}
    image_count = sum(len(row["evidence"]["images"]) for row in corpus.tlv)
    role_copy = deepcopy(ROLE_ROSTER)
    controls = load_controls(project_root() / "fixtures" / "sensitivity_controls.json")
    control_commitment = controls_commitment(controls)
    # Carry every HTTP attempt made before this replacement freeze. The current
    # paid census contains 54 distinct extraction task observations: 52 usable
    # interfaces and two explicit schema rejections. One additional construction
    # planner call is a verified terminal schema rejection. All 55 are one-call
    # ITT observations and must not be retried; downstream usability is separate.
    # The six v6 smoke attempts are also historical because run_paid.py changed
    # after that freeze, so the replacement freeze starts from 119 attempts.
    call_budget = full_call_plan(
        images=image_count,
        prior_attempts=119,
        satisfied_calls=55,
        operational_http_cap=OPERATIONAL_HTTP_CAP,
    )
    budget_within_cap = call_budget["worst_case_http_calls"] <= OPERATIONAL_HTTP_CAP
    return {
        "schema": FREEZE_SCHEMA,
        "state": "FROZEN_OFFLINE_EXECUTION_BLOCKED",
        "census": {"chunks": 16, "conditions": 3, "images": image_count, "packages": 48},
        "manifest_sha256": corpus.manifest_sha256,
        "input_fingerprints": fingerprints,
        "arms": [arm.name for arm in ARMS],
        "construction_call_budget": sum(arm.construction_calls_per_chunk for arm in ARMS) * 16,
        "operational_http_cap": OPERATIONAL_HTTP_CAP,
        "full_call_budget": call_budget,
        "budget_within_cap": budget_within_cap,
        "sensitivity_controls": {
            "schema": CONTROL_SCHEMA,
            "count": len(controls["controls"]),
            "commitment_sha256": control_commitment,
        },
        "judging_frame": {
            "policy": "fixed_arm_balanced_blinded_before_judging",
            "candidate_pool_size": JUDGING_POOL_SIZE,
            "frame_size": JUDGING_FRAME_SIZE,
            "per_arm": JUDGING_PER_ARM,
            "arms": list(JUDGING_ARMS),
            "selected_before_judging": True,
            "outcome_dependent": False,
        },
        "decoding": deepcopy(DECODING),
        "prompt_templates": {
            "planner": PLANNER_TEMPLATE_ID,
            "planner_static_sha256": planner_static_fingerprint(),
            "realizer": REALIZER_TEMPLATE_ID,
        },
        "source_sha256": _source_hashes(),
        "source_discovery": {
            "id": SOURCE_DISCOVERY_POLICY["id"],
            "file_roots": list(SOURCE_DISCOVERY_POLICY["file_roots"]),
            "package_roots": list(SOURCE_DISCOVERY_POLICY["package_roots"]),
            "pattern": SOURCE_DISCOVERY_POLICY["pattern"],
            "excluded_dir_names": list(SOURCE_DISCOVERY_POLICY["excluded_dir_names"]),
            "required_files": sorted(REQUIRED_SOURCE_FILES),
        },
        "roles": role_copy,
        "execution_gate": {
            "requires_execute_flag": True,
            "requires_all_roles_approved": True,
            "requires_role_specific_smoke": True,
            "fallback_forbidden": True,
            "budget_within_cap": budget_within_cap,
        },
    }


def _execution_blockers(
    freeze: dict[str, Any],
    *,
    execute: bool,
    smoke_passed_roles: set[str],
    require_smoke: bool,
) -> list[str]:
    """Return all frozen execution blockers without performing network I/O."""
    blockers: list[str] = []
    if not execute:
        blockers.append("execute_flag_missing")
    if freeze.get("schema") != FREEZE_SCHEMA:
        blockers.append("invalid_freeze_schema")
    if freeze.get("full_call_budget", {}).get("worst_case_http_calls", 10**9) > freeze.get(
        "operational_http_cap", -1
    ):
        blockers.append("call_budget_exceeds_cap")

    recorded_sources = freeze.get("source_sha256")
    if not isinstance(recorded_sources, dict):
        blockers.append("source_hashes_missing")
    else:
        current_sources = _source_hashes()
        drifted = sorted(
            path for path, digest in recorded_sources.items()
            if current_sources.get(path) != digest
        )
        missing = sorted(path for path in current_sources if path not in recorded_sources)
        if drifted or missing:
            detail = ",".join(drifted + [f"missing:{p}" for p in missing])
            blockers.append(f"source_hash_drift:{detail}")

    roles = freeze.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLE_ROSTER):
        blockers.append("invalid_role_roster")
        roles = {}
    for role_name, role in roles.items():
        if not role.get("provider") or not role.get("model") or not role.get("family"):
            blockers.append(f"role_unassigned:{role_name}")
        if role.get("approval") != "APPROVED":
            blockers.append(f"role_unapproved:{role_name}")
        if require_smoke and role_name not in smoke_passed_roles:
            blockers.append(f"smoke_missing:{role_name}")

    families = {role.get("family") for role in roles.values() if role.get("family")}
    if len(families) < 3:
        blockers.append("fewer_than_three_model_families")
    generator_family = roles.get("generator", {}).get("family")
    if any(
        roles.get(name, {}).get("family") == generator_family
        for name in ("answerer_a", "answerer_b", "image_auditor", "model_judge_a", "model_judge_b")
    ):
        blockers.append("generator_family_reused_for_evaluation")
    if roles.get("answerer_a", {}).get("family") == roles.get("answerer_b", {}).get("family"):
        blockers.append("answerer_families_not_distinct")
    return sorted(set(blockers))


def validate_pre_smoke_gate(freeze: dict[str, Any], *, execute: bool) -> None:
    """Authorize only the frozen role-smoke phase, not the paid experiment."""
    blockers = _execution_blockers(
        freeze,
        execute=execute,
        smoke_passed_roles=set(),
        require_smoke=False,
    )
    if blockers:
        raise ValueError("BLOCKED_PRE_SMOKE:" + ",".join(blockers))


def validate_execution_gate(
    freeze: dict[str, Any], *, execute: bool, smoke_passed_roles: set[str]
) -> None:
    """Reject experiment execution unless every frozen precondition is satisfied."""
    blockers = _execution_blockers(
        freeze,
        execute=execute,
        smoke_passed_roles=smoke_passed_roles,
        require_smoke=True,
    )
    if blockers:
        raise ValueError("BLOCKED_EXECUTION:" + ",".join(blockers))


def plan_ledger(freeze: dict[str, Any]) -> list[dict[str, Any]]:
    """Create 6 arms x 16 chunks without copying evidence or filesystem paths."""
    if freeze.get("schema") != FREEZE_SCHEMA or freeze.get("census", {}).get("chunks") != 16:
        raise ValueError("BLOCKED_FREEZE:invalid_freeze_record")
    fingerprints = freeze.get("input_fingerprints")
    if not isinstance(fingerprints, dict) or len(fingerprints) != 16:
        raise ValueError("BLOCKED_FREEZE:incomplete_input_fingerprints")

    rows: list[dict[str, Any]] = []
    for chunk_order, chunk_id in enumerate(sorted(fingerprints), start=1):
        chunk_token = sha256_bytes(chunk_id.encode("utf-8"))[:16]
        for arm_order, arm in enumerate(ARMS, start=1):
            rows.append({
                "cell_id": f"c{chunk_order:02d}-a{arm_order:02d}",
                "chunk_token": chunk_token,
                "input_fingerprint": fingerprints[chunk_id],
                "arm": arm.name,
                "construction_calls": arm.construction_calls_per_chunk,
                "status": "PLANNED",
            })
    if len(rows) != 96 or sum(row["construction_calls"] for row in rows) != 128:
        raise AssertionError("frozen plan violates the 6x16 design or call budget")
    return rows
