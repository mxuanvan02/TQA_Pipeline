"""Fail-closed input validation shared by every preregistered statistic.

Every rejection raises ``ValueError("BLOCKED_STATS:<reason>")`` so a caller can
never silently analyse a malformed, partially observed, or re-ordered design.
This mirrors the ``BLOCKED_<AREA>:<reason>`` convention already used by
``item_gates`` and ``run.ledger``.

The protocol is a bounded census over a fixed paired corpus, so "clean up the
input" is never an acceptable behaviour here: a missing cell, a non-boolean
outcome, or a mismatched chunk order is a protocol violation and must stop the
analysis rather than be coerced into a number.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence


def blocked(reason: str) -> ValueError:
    """Build the canonical fail-closed error for the statistics layer."""
    return ValueError(f"BLOCKED_STATS:{reason}")


def require_sequence(name: str, value: Any) -> list[Any]:
    """Materialise an ordered, non-string sequence.

    Generators are accepted (they are materialised once), but ``str``/``bytes``
    are refused: iterating a string silently yields characters, which is exactly
    the kind of accident that produces a plausible-looking wrong n.
    """
    if isinstance(value, (str, bytes)):
        raise blocked(f"{name}_not_sequence")
    if value is None:
        raise blocked(f"{name}_missing")
    if isinstance(value, (set, frozenset, dict)):
        # Unordered containers cannot carry a pairing, so they can never be a
        # valid paired-analysis input even when the element count looks right.
        raise blocked(f"{name}_unordered")
    if not isinstance(value, Sequence):
        if not isinstance(value, Iterable):
            raise blocked(f"{name}_not_sequence")
        value = list(value)
    return list(value)


def require_bool_sequence(name: str, value: Any) -> list[bool]:
    """Require a sequence of strict booleans (0/1 ints are refused).

    Accepting ``int`` here would let a partially scored cell (``None`` coerced
    to 0, or a count of successes) enter a McNemar table as a legitimate
    outcome. The paired endpoint is defined on gate-valid yield per chunk, which
    is a boolean, so anything else is a data defect.
    """
    items = require_sequence(name, value)
    if not items:
        raise blocked(f"{name}_empty")
    out: list[bool] = []
    for index, item in enumerate(items):
        if not isinstance(item, bool):
            raise blocked(f"{name}_not_boolean:{index}")
        out.append(item)
    return out


def require_unit_sequence(name: str, value: Any) -> list[float]:
    """Require a sequence of finite reals in [0, 1].

    Aggregated per-item outcomes are proportions over within-item repeats, so
    they are continuous but bounded. ``bool`` is allowed (it is a valid 0/1
    proportion); ``None`` and out-of-range values fail closed.
    """
    items = require_sequence(name, value)
    if not items:
        raise blocked(f"{name}_empty")
    out: list[float] = []
    for index, item in enumerate(items):
        if isinstance(item, bool):
            out.append(1.0 if item else 0.0)
            continue
        if not isinstance(item, (int, float)):
            raise blocked(f"{name}_not_numeric:{index}")
        item = float(item)
        if item != item or item in (float("inf"), float("-inf")):
            raise blocked(f"{name}_not_finite:{index}")
        if not 0.0 <= item <= 1.0:
            raise blocked(f"{name}_out_of_unit_range:{index}")
        out.append(item)
    return out


def require_same_length(name_a: str, seq_a: Sequence[Any],
                        name_b: str, seq_b: Sequence[Any]) -> None:
    """Refuse ragged pairs instead of truncating to the shorter arm.

    ``zip`` silently drops the tail, which would turn a missing cell into a
    smaller-but-plausible n. The protocol itemises missing cells; it never
    drops them.
    """
    if len(seq_a) != len(seq_b):
        raise blocked(f"length_mismatch:{name_a}={len(seq_a)}:{name_b}={len(seq_b)}")


def require_expected_n(observed: int, expected: int | None, *, label: str = "n") -> None:
    """Enforce the frozen unit count when one is prespecified."""
    if expected is None:
        return
    expected = require_positive_int("expected_n", expected)
    if observed != expected:
        raise blocked(f"{label}_mismatch:{observed}!={expected}")


def require_alpha(value: Any) -> float:
    """Alpha must be a real number strictly inside (0, 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise blocked("alpha_not_numeric")
    value = float(value)
    if not 0.0 < value < 1.0:
        raise blocked("alpha_out_of_range")
    return value


def require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise blocked(f"{name}_not_int")
    if value <= 0:
        raise blocked(f"{name}_not_positive")
    return value


def require_seed(value: Any) -> int:
    """Seeds are mandatory and must be non-negative integers.

    A defaulted or wall-clock seed would make an interval unreproducible, so
    there is deliberately no default anywhere in the bootstrap layer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise blocked("seed_not_int")
    if value < 0:
        raise blocked("seed_negative")
    return value


def require_unit_ids(name: str, value: Any, expected_len: int) -> list[str]:
    """Require unique, non-empty string identifiers aligned to the outcomes."""
    items = require_sequence(name, value)
    if len(items) != expected_len:
        raise blocked(f"{name}_length_mismatch:{len(items)}!={expected_len}")
    out: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise blocked(f"{name}_invalid:{index}")
        out.append(item)
    if len(set(out)) != len(out):
        raise blocked(f"{name}_duplicate")
    return out


def require_aligned_ids(name_a: str, ids_a: Sequence[str],
                        name_b: str, ids_b: Sequence[str]) -> list[str]:
    """Require identical identifier order across the two arms.

    Same-set-different-order is the dangerous case: it produces a fully
    populated table whose pairs are wrong. Order equality is required, not set
    equality.
    """
    if list(ids_a) != list(ids_b):
        raise blocked(f"pairing_misaligned:{name_a}!={name_b}")
    return list(ids_a)
