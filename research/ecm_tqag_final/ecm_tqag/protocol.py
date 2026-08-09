from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .io import canonical, sha256_bytes

# These five arms create distinct items. ``gates_off`` is only a deterministic
# rescore of ``full`` and therefore never contributes a second judging candidate.
JUDGING_ARMS = (
    "full",
    "caption_mediated",
    "text_only",
    "text_assisted_reader",
    "direct",
)
JUDGING_PER_ARM = 8
JUDGING_FRAME_SIZE = len(JUDGING_ARMS) * JUDGING_PER_ARM
# One candidate per census chunk per generating arm: 5 x 16 = 80. The judging frame
# is exactly half of this pool, allocated 8 per arm.
JUDGING_CANDIDATES_PER_ARM = 16
JUDGING_POOL_SIZE = len(JUDGING_ARMS) * JUDGING_CANDIDATES_PER_ARM


def _blocked(reason: str) -> ValueError:
    return ValueError(f"BLOCKED_PROTOCOL:{reason}")


def fixed_judging_frame(
    candidates: Iterable[Mapping[str, Any]], *, freeze_sha256: str
) -> dict[str, Any]:
    """Select the frozen 40/80 arm-balanced judging frame.

    Selection is outcome-independent: within each arm, candidates are ordered by
    SHA-256 of the freeze identity and item identifier. Exactly eight are retained.
    The private frame keeps routing fields; the blinded frame exposes only an
    opaque judge identifier and a payload commitment.
    """
    if not isinstance(freeze_sha256, str) or len(freeze_sha256) != 64:
        raise _blocked("invalid_freeze_sha256")
    try:
        int(freeze_sha256, 16)
    except ValueError as exc:
        raise _blocked("invalid_freeze_sha256") from exc

    rows = [dict(row) for row in candidates]
    ids: set[str] = set()
    by_arm: dict[str, list[dict[str, Any]]] = {arm: [] for arm in JUDGING_ARMS}
    for index, row in enumerate(rows):
        item_id = row.get("item_id")
        arm = row.get("arm")
        if not isinstance(item_id, str) or not item_id:
            raise _blocked(f"invalid_item_id:{index}")
        if item_id in ids:
            raise _blocked(f"duplicate_item_id:{item_id}")
        ids.add(item_id)
        if arm not in by_arm:
            raise _blocked(f"unexpected_judging_arm:{arm}")
        by_arm[str(arm)].append(row)

    counts = {arm: len(by_arm[arm]) for arm in JUDGING_ARMS}
    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise _blocked(f"judging_pool_must_be_arm_balanced:{counts}")
    candidates_per_arm = next(iter(unique_counts), 0)
    if not JUDGING_PER_ARM <= candidates_per_arm <= JUDGING_CANDIDATES_PER_ARM:
        raise _blocked(f"judging_pool_per_arm_out_of_range:{counts}")
    if len(rows) != len(JUDGING_ARMS) * candidates_per_arm:
        raise _blocked(f"judging_pool_size_inconsistent:{len(rows)}:{counts}")

    selected: list[dict[str, Any]] = []
    for arm in JUDGING_ARMS:
        ranked = sorted(
            by_arm[arm],
            key=lambda row: (
                hashlib.sha256(
                    f"{freeze_sha256}:{arm}:{row['item_id']}".encode("utf-8")
                ).hexdigest(),
                row["item_id"],
            ),
        )
        selected.extend(ranked[:JUDGING_PER_ARM])

    # A second stable sort removes arm blocks from the judging order while retaining
    # the exact 8-per-arm private allocation.
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"judge-order:{freeze_sha256}:{row['item_id']}".encode("utf-8")
        ).hexdigest()
    )

    private: list[dict[str, Any]] = []
    blinded: list[dict[str, str]] = []
    for ordinal, row in enumerate(selected, start=1):
        payload_commitment = row.get("item_payload_sha256")
        if not isinstance(payload_commitment, str) or len(payload_commitment) != 64:
            # During preconstruction tests no item payload exists yet; commit to the
            # complete candidate record. In a real run the runner supplies the hash
            # of the question/choices payload explicitly.
            payload_commitment = sha256_bytes(canonical(row).encode("utf-8"))
        judge_item_id = sha256_bytes(
            f"{freeze_sha256}:judge:{ordinal}:{row['item_id']}".encode("utf-8")
        )[:20]
        private_row = {
            **row,
            "judge_item_id": judge_item_id,
            "item_payload_sha256": payload_commitment,
        }
        private.append(private_row)
        blinded.append(
            {
                "judge_item_id": judge_item_id,
                "item_payload_sha256": payload_commitment,
            }
        )

    if len(private) != JUDGING_FRAME_SIZE:
        raise AssertionError("fixed judging frame must contain exactly 40 items")
    return {
        "schema": "ecm-tqag.fixed-judging-frame.v1",
        "selection": "sha256_rank_within_arm_then_frozen_blinded_order",
        "freeze_sha256": freeze_sha256,
        "candidate_pool_size": len(rows),
        "candidate_pool_per_arm": candidates_per_arm,
        "expected_candidate_pool_size": JUDGING_POOL_SIZE,
        "frame_size": JUDGING_FRAME_SIZE,
        "per_arm": JUDGING_PER_ARM,
        "blinded_before_judging": True,
        "outcome_dependent": False,
        "private_frame": private,
        "blinded_frame": blinded,
    }
