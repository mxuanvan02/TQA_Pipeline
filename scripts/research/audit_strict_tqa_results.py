#!/usr/bin/env python3
"""Mechanically replay ECM-TQAG graph-program ledgers against frozen evidence.

The audit verifies only record/provenance properties that can be checked without
re-asking a model: package binding, MCQ shape, literal text/structure anchors,
graph validity, motif matching, program execution, atom provenance, and
planner-to-realizer bindings. It does not judge legal correctness,
unique-best-answer quality, pedagogy, difficulty, or image-grounding truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ecm_graph_program import (  # noqa: E402
    CONSTRUCTION_FIELDS,
    anchor_errors,
    atom_choice_binding,
    validate_construction,
)

MANIFEST = ROOT / "research/artifacts/ecm_inputs_8chunks_v3.json"
METHODS = {"direct", "answer_first", "ecm"}
CONDITIONS = {"T", "TL_struct", "TLV"}


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evidence(manifest_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
    """Return {(chunk_id, condition): {"evidence": ..., "doc_id": ...}} and digest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "ecm-tqag.multimodal-inputs.v3":
        raise ValueError("unexpected_manifest_schema")
    evidence_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    for package in manifest.get("packages", []):
        key = (package.get("chunk_id"), package.get("condition"))
        if not isinstance(key[0], str) or key[1] not in CONDITIONS or key in evidence_by_cell:
            raise ValueError("invalid_manifest_package")
        evidence = package.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("invalid_manifest_evidence")
        evidence_by_cell[key] = {"evidence": evidence, "doc_id": package.get("doc_id")}
    if len(evidence_by_cell) != 24:
        raise ValueError("expected_24_evidence_packages")
    return evidence_by_cell, sha256_file(manifest_path)


def allowed_sources(condition: str) -> set[str]:
    return {"T": {"text"}, "TL_struct": {"text", "structure"}, "TLV": {"text", "structure", "image"}}[condition]


# anchor_errors is imported from src.ecm_graph_program so that the runner and
# this audit enforce byte-identical anchor rules.


def ecm_errors(output: dict[str, Any], evidence: dict[str, Any], condition: str,
               source: dict[str, Any] | None = None) -> list[str]:
    """Replay the v5 graph/rho/motif/program/atom/trace contract and realizer binding."""
    if not isinstance(output, dict) or not CONSTRUCTION_FIELDS.issubset(output):
        return ["ecm_construction_missing"]
    construction = {key: output[key] for key in CONSTRUCTION_FIELDS}
    errors = [
        f"ecm_construction:{code}"
        for code in validate_construction(construction, evidence, condition, source)
    ]
    if errors:
        return errors

    atoms = construction["answer_atoms"]
    expected_ids = [atom["id"] for atom in atoms]
    expected_values = [atom["value"] for atom in atoms]
    expected_trace = list(dict.fromkeys(
        step_id for atom in atoms for step_id in atom["support"]["step_ids"]
    ))
    if output.get("answer_atom_ids") != expected_ids:
        errors.append("ecm_answer_atom_ids_not_locked")
    if output.get("answer_parts") != expected_values:
        errors.append("ecm_answer_parts_not_executor_atoms")
    if output.get("program_trace") != expected_trace:
        errors.append("ecm_program_trace_not_locked")

    choices, answer_index = output.get("choices"), output.get("answer_index")
    if (len(expected_values) != 1 or not isinstance(choices, list)
            or isinstance(answer_index, bool) or not isinstance(answer_index, int)
            or answer_index not in range(len(choices))):
        errors.append("ecm_selected_choice_not_checkable")
    else:
        binding = atom_choice_binding(expected_values[0], choices[answer_index])
        if not binding["ok"]:
            errors.append("ecm_selected_choice_not_bound_to_atom:" + str(binding["relation"]))
        elif output.get("atom_choice_binding") not in (None, binding):
            errors.append("ecm_atom_choice_binding_not_replayable")

    graph_nodes = {node["id"]: node for node in construction["document_graph"]["nodes"]}
    required_ids = construction["matched_motif"]["bindings"]["evidence_node_ids"]
    anchors = output.get("evidence_anchor")
    if not isinstance(anchors, list) or len(anchors) != len(required_ids):
        return errors + ["ecm_locked_anchor_missing_or_wrong_count"]
    seen: set[str] = set()
    for position, anchor in enumerate(anchors, 1):
        where = f"ecm_anchor:{position}"
        if not isinstance(anchor, dict) or set(anchor) != {"evidence_id", "source", "support"}:
            errors.append(f"{where}:schema")
            continue
        evidence_id = anchor.get("evidence_id")
        if (not isinstance(evidence_id, str) or evidence_id not in required_ids or evidence_id in seen
                or anchor.get("source") != graph_nodes[evidence_id]["source"]
                or anchor.get("support") != graph_nodes[evidence_id]["value"]):
            errors.append(f"{where}:not_locked_node")
            continue
        seen.add(evidence_id)
    if seen != set(required_ids):
        errors.append("ecm_locked_anchor_incomplete")
    return errors


def audit_row(row: dict[str, Any], evidence_by_cell: dict[tuple[str, str], dict[str, Any]], manifest_hash: str) -> list[str]:
    errors: list[str] = []
    method, condition, chunk_id = row.get("method"), row.get("condition"), row.get("chunk_id")
    if method not in METHODS or condition not in CONDITIONS or not isinstance(chunk_id, str):
        return ["invalid_method_or_condition"]
    cell = evidence_by_cell.get((chunk_id, condition))
    if cell is None:
        return ["evidence_missing"]
    evidence = cell["evidence"]
    source = {
        "doc_id": cell["doc_id"],
        "chunk_id": chunk_id,
        "condition": condition,
        "manifest_sha256": manifest_hash,
    }
    if row.get("source_hash") != manifest_hash:
        errors.append("manifest_hash_mismatch")
    output = row.get("parsed_response")
    if row.get("status") != "PARSED" or not isinstance(output, dict):
        return errors + ["not_parsed"]

    choices, answer_index = output.get("choices"), output.get("answer_index")
    if (not isinstance(output.get("question"), str) or not output["question"].strip()
            or not isinstance(choices, list) or len(choices) != 4
            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
            or isinstance(answer_index, bool) or not isinstance(answer_index, int) or answer_index not in range(4)):
        return errors + ["invalid_core_item"]
    if len({normalize(choice) for choice in choices}) != 4:
        errors.append("duplicate_options")
    if method != "ecm":
        errors.extend(anchor_errors(output.get("evidence_anchor"), evidence, condition, "anchor"))
    if method == "answer_first" and output.get("answer") != choices[answer_index]:
        errors.append("answer_first_not_selected_option")
    if method == "ecm":
        errors.extend(ecm_errors(output, evidence, condition, source))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay graph--motif--program TQA records against frozen evidence."
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, required=True, help="Must not already exist.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    evidence_by_cell, manifest_hash = load_evidence(args.manifest)
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("results file is empty")

    item_reports: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for row in rows:
        errors = audit_row(row, evidence_by_cell, manifest_hash)
        item_reports.append({
            "run_id": row.get("run_id"), "chunk_id": row.get("chunk_id"),
            "condition": row.get("condition"), "method": row.get("method"),
            "status": "PASS" if not errors else "FAIL", "errors": errors,
        })
        all_errors.extend(errors)
    result = {
        "schema": "ecm-tqag.graph-program-mechanical-audit.v5",
        "status": "PASS" if not all_errors else "FAIL",
        "scope": "Mechanical/provenance audit only: this does not establish semantic correctness, unique-best-answer validity, legal validity, pedagogical quality, or image-grounding truth.",
        "source_results": str(args.results),
        "source_results_sha256": sha256_file(args.results),
        "evidence_manifest": str(args.manifest),
        "evidence_manifest_sha256": manifest_hash,
        "record_count": len(rows),
        "pass_count": sum(item["status"] == "PASS" for item in item_reports),
        "fail_count": sum(item["status"] == "FAIL" for item in item_reports),
        "error_counts": dict(Counter(all_errors)),
        "items": item_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "record_count", "pass_count", "fail_count", "error_counts")}, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
