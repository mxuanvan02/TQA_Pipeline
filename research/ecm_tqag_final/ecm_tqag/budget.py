from __future__ import annotations

from dataclasses import dataclass
import math

from .protocol import JUDGING_FRAME_SIZE


@dataclass(frozen=True)
class CallPhase:
    name: str
    base_calls: int
    conditional: bool = False


def full_call_plan(
    *,
    chunks: int = 16,
    images: int = 18,
    answerer_families: int = 2,
    controls_per_family: int = 10,
    control_repeats: int = 2,
    probe_conditions: int = 5,
    maximum_eligible_full_items: int = 16,
    human_judge_sample: int = 40,
    judge_families: int = 2,
    smoke_calls: int = 6,
    prior_attempts: int = 6,
    satisfied_calls: int = 0,
    retry_fraction: float = 0.10,
    operational_http_cap: int = 550,
) -> dict:
    """Return a conservative, preregistration-level HTTP call budget.

    Construction comprises 128 planner/realizer/direct calls. Reader/caption calls
    are counted separately for full, caption-mediated, and text-assisted-reader.
    Secondary probes are budgeted only for eligible full-arm items, never all arms.
    """
    if chunks != 16 or images != 18:
        raise ValueError("BLOCKED_BUDGET:unexpected_frozen_census")
    if answerer_families < 2 or controls_per_family != 10 or control_repeats < 2:
        raise ValueError("BLOCKED_BUDGET:invalid_sensitivity_floor")
    if judge_families != 2:
        raise ValueError("BLOCKED_BUDGET:exactly_two_judge_families_required")
    if human_judge_sample != JUDGING_FRAME_SIZE:
        raise ValueError(
            f"BLOCKED_BUDGET:judge_sample_must_match_fixed_frame:"
            f"{human_judge_sample}/{JUDGING_FRAME_SIZE}"
        )
    if isinstance(prior_attempts, bool) or not isinstance(prior_attempts, int) or prior_attempts < 0:
        raise ValueError("BLOCKED_BUDGET:invalid_prior_attempts")
    if (isinstance(satisfied_calls, bool) or not isinstance(satisfied_calls, int)
            or satisfied_calls < 0):
        raise ValueError("BLOCKED_BUDGET:invalid_satisfied_calls")
    if satisfied_calls > prior_attempts:
        raise ValueError("BLOCKED_BUDGET:satisfied_calls_exceed_prior_attempts")
    if isinstance(operational_http_cap, bool) or not isinstance(operational_http_cap, int) or operational_http_cap < 1:
        raise ValueError("BLOCKED_BUDGET:invalid_operational_cap")
    if not 0 <= retry_fraction <= 0.25:
        raise ValueError("BLOCKED_BUDGET:invalid_retry_fraction")

    phases = (
        CallPhase("role_smoke_current_freeze", smoke_calls),
        # Extraction is per image, not per chunk: two census chunks declare two
        # images and silently dropping either would break the paired design.
        CallPhase("structure_and_caption_extraction", images * 3),
        CallPhase("construction", chunks * 8),
        CallPhase(
            "sensitivity_floor",
            answerer_families * controls_per_family * control_repeats,
        ),
        CallPhase(
            "full_item_secondary_probes",
            maximum_eligible_full_items * probe_conditions * answerer_families,
            conditional=True,
        ),
        CallPhase("image_audit", images),
        CallPhase("blinded_model_judging", human_judge_sample * judge_families),
    )
    gross_current_base = sum(p.base_calls for p in phases)
    if satisfied_calls > gross_current_base:
        raise ValueError("BLOCKED_BUDGET:satisfied_calls_exceed_plan")
    current_base = gross_current_base - satisfied_calls
    study_base = prior_attempts + current_base
    requested_retries = math.ceil(study_base * retry_fraction)
    # Carry every prior attempt forward without allowing the immutable hard cap
    # to grow. Accumulated attempts can only reduce the operational retry reserve.
    retries = min(requested_retries, max(0, operational_http_cap - study_base))
    study_total = study_base + retries
    current_ledger_cap = study_total - prior_attempts
    if study_total > operational_http_cap:
        raise ValueError(
            f"BLOCKED_BUDGET:study_worst_case_exceeds_cap:{study_total}/{operational_http_cap}"
        )
    return {
        "phases": [
            {"name": p.name, "base_calls": p.base_calls, "conditional": p.conditional}
            for p in phases
        ],
        "prior_attempts": prior_attempts,
        "gross_current_freeze_base_calls": gross_current_base,
        "satisfied_calls": satisfied_calls,
        "current_freeze_base_calls": current_base,
        "study_base_calls": study_base,
        # Compatibility aliases refer to the current-freeze plan only where noted.
        "base_calls": current_base,
        "retry_fraction": retry_fraction,
        "requested_retry_reserve": requested_retries,
        "retry_reserve": retries,
        "retry_reserve_constrained_by_hard_cap": retries < requested_retries,
        "study_worst_case_http_calls": study_total,
        "worst_case_http_calls": study_total,
        "current_ledger_cap": current_ledger_cap,
        "operational_http_cap": operational_http_cap,
        "secondary_scope": "eligible_full_arm_items_only",
    }
