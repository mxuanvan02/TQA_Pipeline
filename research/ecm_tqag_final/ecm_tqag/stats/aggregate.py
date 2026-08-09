"""Within-item aggregation: repeats are never sample-size inflation.

The protocol is explicit: "Repeats are aggregated within item and never used as
sample-size inflation." Calls, retries and repeated answers are not independent
units. The only legitimate units are the item (for perturbation probes) and the
chunk (for the primary endpoint).

This module turns a flat list of per-repeat records into one aggregated row per
item, and reports the effective n as the number of *items*, not the number of
repeats. ``n_repeats`` is retained purely as provenance so the artifact shows how
much replication stood behind each aggregated proportion.

The aggregated per-item value is the mean of its repeats, i.e. a proportion in
[0, 1]. Downstream, ``bootstrap`` resamples items (or documents), so the repeat
count can never enter a variance calculation as if it were independent evidence.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Mapping

from .validate import blocked, require_sequence

# Aggregation rules that are meaningful for boolean repeat outcomes.
MAJORITY = "majority"
MEAN = "mean"
ALL = "all"
ANY = "any"
AGGREGATION_RULES = (MEAN, MAJORITY, ALL, ANY)


def _require_record(index: int, record: Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise blocked(f"record_not_mapping:{index}")
    return record


def _require_item_id(index: int, record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise blocked(f"{key}_invalid:{index}")
    return value


def _require_outcome(index: int, record: Mapping[str, Any], key: str) -> bool:
    if key not in record:
        raise blocked(f"outcome_missing:{index}")
    value = record[key]
    if not isinstance(value, bool):
        # A None here means an unobserved repeat. The protocol itemises missing
        # cells; it does not let them decay into False.
        raise blocked(f"outcome_not_boolean:{index}")
    return value


def aggregate_repeats(records: Iterable[Mapping[str, Any]], *,
                      item_key: str = "item_id",
                      outcome_key: str = "correct",
                      rule: str = MEAN,
                      document_key: str | None = None) -> dict[str, Any]:
    """Collapse per-repeat records into one row per item.

    Returns ``items`` in first-seen order (deterministic, no dict-hash
    dependence) with, per item, the repeat count, the success count, the
    aggregated proportion, and the rule-derived boolean outcome.

    ``rule`` controls the boolean projection only; ``proportion`` is always the
    mean so an interval on a continuous per-item value stays available.
    ``majority`` on an even split resolves to ``False`` (a tie is not evidence of
    success), which is recorded per item as ``tie: True``.
    """
    if rule not in AGGREGATION_RULES:
        raise blocked(f"unknown_rule:{rule}")

    rows = require_sequence("records", records)
    if not rows:
        raise blocked("records_empty")

    buckets: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for index, raw in enumerate(rows):
        record = _require_record(index, raw)
        item_id = _require_item_id(index, record, item_key)
        outcome = _require_outcome(index, record, outcome_key)

        document_id = None
        if document_key is not None:
            document_id = record.get(document_key)
            if not isinstance(document_id, str) or not document_id.strip():
                raise blocked(f"{document_key}_invalid:{index}")

        bucket = buckets.get(item_id)
        if bucket is None:
            buckets[item_id] = {
                "item_id": item_id,
                "document_id": document_id,
                "n_repeats": 1,
                "n_success": int(outcome),
            }
            continue

        if document_key is not None and bucket["document_id"] != document_id:
            # One item cannot belong to two documents; that would corrupt the
            # cluster structure the bootstrap relies on.
            raise blocked(f"document_conflict:{item_id}")
        bucket["n_repeats"] += 1
        bucket["n_success"] += int(outcome)

    items: list[dict[str, Any]] = []
    for bucket in buckets.values():
        n_repeats = bucket["n_repeats"]
        n_success = bucket["n_success"]
        proportion = n_success / float(n_repeats)
        tie = (2 * n_success == n_repeats)
        if rule == MEAN:
            outcome = proportion >= 0.5 and not tie
        elif rule == MAJORITY:
            outcome = (2 * n_success) > n_repeats
        elif rule == ALL:
            outcome = n_success == n_repeats
        else:  # ANY
            outcome = n_success > 0
        items.append({
            "item_id": bucket["item_id"],
            "document_id": bucket["document_id"],
            "n_repeats": n_repeats,
            "n_success": n_success,
            "proportion": proportion,
            "outcome": bool(outcome),
            "tie": tie,
        })

    n_repeat_records = len(rows)
    return {
        "rule": rule,
        "n_items": len(items),
        "effective_n": len(items),
        "n_repeat_records": n_repeat_records,
        # The whole point of this module, stated in the artifact:
        "sample_size_inflation": False,
        "unit": "item",
        "items": items,
        "item_ids": [i["item_id"] for i in items],
        "outcomes": [i["outcome"] for i in items],
        "proportions": [i["proportion"] for i in items],
        "document_ids": ([i["document_id"] for i in items]
                         if document_key is not None else None),
    }


def paired_aggregate(control_records: Iterable[Mapping[str, Any]],
                     perturbed_records: Iterable[Mapping[str, Any]], *,
                     item_key: str = "item_id",
                     outcome_key: str = "correct",
                     rule: str = MEAN,
                     document_key: str | None = None) -> dict[str, Any]:
    """Aggregate both conditions and require an identical item set and order.

    Reordering is the failure mode that produces a complete-looking but wrongly
    paired analysis, so this reindexes the perturbed arm onto the control arm's
    item order rather than trusting input order, and fails closed when the sets
    differ at all.
    """
    control = aggregate_repeats(control_records, item_key=item_key,
                               outcome_key=outcome_key, rule=rule,
                               document_key=document_key)
    perturbed = aggregate_repeats(perturbed_records, item_key=item_key,
                                 outcome_key=outcome_key, rule=rule,
                                 document_key=document_key)

    control_ids = control["item_ids"]
    perturbed_index = {i["item_id"]: i for i in perturbed["items"]}

    missing = [i for i in control_ids if i not in perturbed_index]
    if missing:
        raise blocked(f"perturbed_missing_items:{sorted(missing)[:5]}")
    extra = [i for i in perturbed_index if i not in set(control_ids)]
    if extra:
        raise blocked(f"perturbed_extra_items:{sorted(extra)[:5]}")

    ordered_perturbed = [perturbed_index[i] for i in control_ids]
    return {
        "rule": rule,
        "unit": "item",
        "n_items": control["n_items"],
        "effective_n": control["n_items"],
        "sample_size_inflation": False,
        "item_ids": control_ids,
        "document_ids": control["document_ids"],
        "control_outcomes": control["outcomes"],
        "perturbed_outcomes": [i["outcome"] for i in ordered_perturbed],
        "control_proportions": control["proportions"],
        "perturbed_proportions": [i["proportion"] for i in ordered_perturbed],
        "control_repeat_records": control["n_repeat_records"],
        "perturbed_repeat_records": perturbed["n_repeat_records"],
        "control": control,
        "perturbed": {**perturbed, "items": ordered_perturbed},
    }
