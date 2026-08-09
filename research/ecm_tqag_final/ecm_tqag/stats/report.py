"""Release-ready structured statistics artifact.

This module composes the preregistered statistics into a single JSON-serialisable
record suitable for the ``primary statistics`` release artifact. It performs no
computation of its own beyond composition and provenance stamping, so every
number in the artifact traces back to one of the tested statistical modules.

Two properties matter for release:

1. **Determinism.** ``build_primary_statistics`` is a pure function of its
   arguments. Every bootstrap seed is required and echoed into the artifact, and
   ``artifact_digest`` is a canonical-JSON SHA-256 over the payload, so two
   replays either produce an identical digest or the difference is visible.
   There is no timestamp inside the digested payload for exactly this reason;
   the caller may attach one outside the digest.

2. **Fail-closed composition.** The artifact cannot be built with a partial
   primary table, cannot present a suppressed ``Delta_perm`` as an estimate, and
   cannot mark an exploratory judge dimension as a secondary claim.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..io import canonical, sha256_bytes
from .agreement import judge_agreement_report
from .bootstrap import DEFAULT_RESAMPLES, delta_perm_interval
from .paired import PRIMARY_N_CHUNKS, primary_contrast
from .sensitivity import sensitivity_floor
from .validate import blocked, require_alpha, require_seed

SCHEMA = "ecm-tqag.stats.v1"

# The analysis is a bounded census; these strings travel with the artifact so a
# reader cannot detach the numbers from their scope limits.
INFERENCE_SCOPE = (
    "bounded_census_of_16_image_bearing_text_sufficient_chunks;"
    "not_generalized_beyond_this_corpus_language_or_domain"
)


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise blocked(f"{name}_not_mapping")
    return dict(value)


def build_primary_statistics(*,
                             full: Sequence[bool],
                             caption_mediated: Sequence[bool],
                             chunk_ids: Sequence[str] | None = None,
                             alpha: float = 0.05,
                             expected_n: int | None = PRIMARY_N_CHUNKS,
                             sensitivity: Mapping[str, Any] | None = None,
                             delta_perm: Mapping[str, Any] | None = None,
                             judge_agreement: Mapping[str, Any] | None = None,
                             extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the primary statistics artifact.

    Only the confirmatory contrast is mandatory. The secondary blocks are
    optional because the protocol allows them to be absent (an insensitive probe
    reports no ``Delta_perm``; judging may remain exploratory), but when present
    they are validated for internal consistency.
    """
    alpha = require_alpha(alpha)
    primary = primary_contrast(full, caption_mediated, alpha=alpha,
                               chunk_ids=chunk_ids, expected_n=expected_n)

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "alpha": alpha,
        "inference_scope": INFERENCE_SCOPE,
        "unit_of_analysis": "chunk",
        "multiplicity": "single_prespecified_confirmatory_contrast_unadjusted",
        "primary": primary,
        "secondary": {},
        "exploratory": {},
    }

    if sensitivity is not None:
        floor = _require_mapping("sensitivity", sensitivity)
        payload["secondary"]["sensitivity_floor"] = floor
        if delta_perm is not None:
            reported = bool(_require_mapping("delta_perm", delta_perm).get("reported"))
            if reported and not floor.get("passed"):
                # The protocol forbids reporting Delta_perm when the floor fails.
                raise blocked("delta_perm_reported_without_sensitivity_floor")

    if delta_perm is not None:
        payload["secondary"]["delta_perm"] = _require_mapping("delta_perm", delta_perm)

    if judge_agreement is not None:
        agreement = _require_mapping("judge_agreement", judge_agreement)
        if agreement.get("secondary_claim_eligible"):
            payload["secondary"]["judge_agreement"] = agreement
        else:
            payload["exploratory"]["judge_agreement"] = agreement

    # Gate-validity failure of lexical novelty is a standing protocol fact, not a
    # per-run observation, so it is stamped into every artifact.
    payload["exploratory"]["lexical_novelty"] = {
        "status": "gate_validity_failure",
        "confirmatory": False,
        "note": "no threshold in 0.60-0.95 satisfied the discrimination criterion",
    }

    if extra is not None:
        payload["context"] = _require_mapping("extra", extra)

    payload["artifact_digest"] = sha256_bytes(canonical(payload).encode("utf-8"))
    return payload


def analyze_run(*,
                full: Sequence[bool],
                caption_mediated: Sequence[bool],
                chunk_ids: Sequence[str] | None = None,
                alpha: float = 0.05,
                expected_n: int | None = PRIMARY_N_CHUNKS,
                control_proportions: Sequence[float] | None = None,
                perturbed_proportions: Sequence[float] | None = None,
                clusters: Sequence[str] | None = None,
                bootstrap_seed: int | None = None,
                n_resamples: int = DEFAULT_RESAMPLES,
                families: Sequence[Mapping[str, Any]] | None = None,
                judge_a: Sequence[int] | None = None,
                judge_b: Sequence[int] | None = None,
                judge_categories: int | None = None,
                judge_dimension: str | None = None,
                extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """End-to-end analysis from run-level inputs to the release artifact.

    Runs the sensitivity floor first and lets its verdict decide whether a
    ``Delta_perm`` interval is computed at all, which is the ordering the
    protocol requires ("Before probing generated items ... on failure the probe
    is labeled insensitive and stops").
    """
    floor = None
    if families is not None:
        floor = sensitivity_floor([dict(f) for f in families])

    delta = None
    if control_proportions is not None or perturbed_proportions is not None:
        if control_proportions is None or perturbed_proportions is None:
            raise blocked("delta_perm_inputs_incomplete")
        if floor is None:
            # Without a floor result there is no licence to report Delta_perm.
            raise blocked("delta_perm_requires_sensitivity_floor")
        if bootstrap_seed is None:
            raise blocked("bootstrap_seed_required")
        delta = delta_perm_interval(
            control_proportions,
            perturbed_proportions,
            seed=require_seed(bootstrap_seed),
            clusters=clusters,
            alpha=alpha,
            n_resamples=n_resamples,
            sensitivity_floor_passed=bool(floor["passed"]),
        )

    agreement = None
    if judge_a is not None or judge_b is not None:
        if judge_a is None or judge_b is None:
            raise blocked("judge_ratings_incomplete")
        if judge_categories is None:
            raise blocked("judge_categories_required")
        if judge_dimension is None:
            raise blocked("judge_dimension_required")
        agreement = judge_agreement_report(
            judge_a, judge_b,
            n_categories=judge_categories,
            dimension=judge_dimension,
        )

    return build_primary_statistics(
        full=full,
        caption_mediated=caption_mediated,
        chunk_ids=chunk_ids,
        alpha=alpha,
        expected_n=expected_n,
        sensitivity=floor,
        delta_perm=delta,
        judge_agreement=agreement,
        extra=extra,
    )


def serialize(payload: Mapping[str, Any]) -> str:
    """Pretty JSON for the release file, with the digest preserved."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def verify_digest(payload: Mapping[str, Any]) -> bool:
    """Recompute the digest over the payload minus the digest field itself."""
    record = _require_mapping("payload", payload)
    claimed = record.pop("artifact_digest", None)
    if not isinstance(claimed, str) or not claimed:
        raise blocked("artifact_digest_missing")
    return sha256_bytes(canonical(record).encode("utf-8")) == claimed
