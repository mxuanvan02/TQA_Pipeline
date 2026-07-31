#!/usr/bin/env python3
"""Mechanically audit ECM-TQAG ledgers against frozen evidence.

The audit verifies only record/provenance properties that can be checked without
re-asking a model: package binding, MCQ shape, literal text/structure anchors,
and ECM planner-to-realizer bindings. It does not judge legal correctness,
unique-best-answer quality, pedagogy, difficulty, or image-grounding truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "research/artifacts/ecm_inputs_8chunks_v3.json"
METHODS = {"direct", "answer_first", "ecm"}
CONDITIONS = {"T", "TL_struct", "TLV"}
ROLES = {"premise", "mapping", "constraint"}


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_evidence(manifest_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], str]:
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
        evidence_by_cell[key] = evidence
    if len(evidence_by_cell) != 24:
        raise ValueError("expected_24_evidence_packages")
    return evidence_by_cell, sha256_file(manifest_path)


def allowed_sources(condition: str) -> set[str]:
    return {"T": {"text"}, "TL_struct": {"text", "structure"}, "TLV": {"text", "structure", "image"}}[condition]


def anchor_errors(anchors: Any, evidence: dict[str, Any], condition: str, label: str) -> list[str]:
    """Validate non-ECM anchors directly against the frozen package."""
    errors: list[str] = []
    allowed = allowed_sources(condition)
    if not isinstance(anchors, list) or not anchors:
        return [f"{label}:missing"]
    for position, entry in enumerate(anchors, 1):
        where = f"{label}:{position}"
        if not isinstance(entry, dict) or set(entry) != {"source", "support"}:
            errors.append(f"{where}:schema")
            continue
        source, support = entry.get("source"), entry.get("support")
        if source not in allowed or not isinstance(support, str) or not support.strip():
            errors.append(f"{where}:invalid_source_or_support")
            continue
        if source == "text" and normalize(support) not in normalize(evidence["text"]):
            errors.append(f"{where}:text_not_literal")
        elif source == "structure":
            title = str(evidence.get("document_structure", {}).get("section_title", ""))
            if normalize(support) != normalize(title):
                errors.append(f"{where}:structure_not_section_title")
        elif source == "image":
            match = re.search(r"(?:ảnh|image)\s*(\d+)", support, flags=re.I)
            image_count = len(evidence.get("images", []))
            if not match or int(match.group(1)) not in range(1, image_count + 1):
                errors.append(f"{where}:image_order_not_identified")
    return errors


def ecm_errors(output: dict[str, Any], evidence: dict[str, Any], condition: str) -> list[str]:
    """Check the v3 locked planner/realizer provenance contract."""
    errors: list[str] = []
    units = output.get("evidence_units")
    plan = output.get("evidence_plan")
    atoms = output.get("answer_atoms")
    answer_parts = output.get("answer_parts")
    if not isinstance(units, list) or len(units) < 2:
        return ["ecm_units_missing_or_too_short"]
    if not isinstance(plan, dict):
        return ["ecm_plan_missing"]
    if (not isinstance(atoms, list) or not atoms
            or any(not isinstance(atom, str) or not atom.strip() for atom in atoms)
            or len({normalize(atom) for atom in atoms}) != len(atoms)):
        errors.append("ecm_atoms_invalid")
    if answer_parts != atoms:
        errors.append("ecm_answer_parts_not_locked_atoms")

    allowed = allowed_sources(condition)
    structure_title = str(evidence.get("document_structure", {}).get("section_title", ""))
    image_count = len(evidence.get("images", []))
    unit_by_id: dict[str, dict[str, str]] = {}
    for position, unit in enumerate(units, 1):
        where = f"ecm_unit:{position}"
        if not isinstance(unit, dict) or set(unit) != {"id", "source", "role", "content"}:
            errors.append(f"{where}:schema")
            continue
        unit_id, source, role, content = unit.get("id"), unit.get("source"), unit.get("role"), unit.get("content")
        if not isinstance(source, str) or not isinstance(role, str):
            errors.append(f"{where}:invalid")
            continue
        prefix = {"text": "T", "structure": "S", "image": "I"}.get(source)
        if (not isinstance(unit_id, str) or not prefix or not re.fullmatch(prefix + r"[1-9][0-9]*", unit_id)
                or unit_id in unit_by_id or source not in allowed or role not in ROLES
                or not isinstance(content, str) or not content.strip()):
            errors.append(f"{where}:invalid")
            continue
        unit_by_id[unit_id] = {"source": source, "role": role, "content": content}
        if source == "text" and normalize(content) not in normalize(evidence["text"]):
            errors.append(f"{where}:text_not_literal")
        elif source == "structure" and normalize(content) != normalize(structure_title):
            errors.append(f"{where}:structure_not_section_title")
        elif source == "image":
            match = re.search(r"(?:ảnh|image)\s*(\d+)", content, flags=re.I)
            if not match or int(match.group(1)) not in range(1, image_count + 1):
                errors.append(f"{where}:image_order_not_identified")

    if set(plan) != {"target", "required_units", "chain", "image_necessary"}:
        return errors + ["ecm_plan_schema"]
    required = plan.get("required_units")
    chain = plan.get("chain")
    if (not isinstance(plan.get("target"), str) or not plan["target"].strip()
            or not isinstance(required, list) or len(required) < 2 or len(set(required)) != len(required)
            or any(not isinstance(unit_id, str) or unit_id not in unit_by_id for unit_id in required)
            or not isinstance(chain, list) or len(chain) < 2 or not isinstance(plan.get("image_necessary"), bool)):
        return errors + ["ecm_plan_invalid"]
    required_ids = set(required)
    if len({unit_by_id[unit_id]["role"] for unit_id in required_ids}) < 2:
        errors.append("ecm_required_units_same_role")

    referenced: set[str] = set()
    chain_outputs: set[str] = set()
    for position, step in enumerate(chain, 1):
        where = f"ecm_chain:{position}"
        if (not isinstance(step, dict) or set(step) != {"from", "to"}
                or not isinstance(step.get("from"), list) or not step["from"]
                or not isinstance(step.get("to"), str) or not step["to"].strip()
                or any(not isinstance(unit_id, str) or unit_id not in required_ids for unit_id in step["from"])):
            errors.append(f"{where}:invalid")
            continue
        referenced.update(step["from"])
        chain_outputs.add(normalize(step["to"]))
    if referenced != required_ids:
        errors.append("ecm_required_units_not_covered_by_chain")
    if isinstance(atoms, list) and not all(isinstance(atom, str) and normalize(atom) in chain_outputs for atom in atoms):
        errors.append("ecm_atoms_not_derived_from_chain")

    if condition == "TLV":
        if not plan["image_necessary"] or not any(unit_by_id[unit_id]["source"] == "image" for unit_id in required_ids):
            errors.append("ecm_tlv_image_not_required")
    elif plan["image_necessary"]:
        errors.append("ecm_non_tlv_claims_image")

    trace = output.get("trace")
    if not isinstance(trace, list) or not trace:
        errors.append("ecm_trace_missing")
    else:
        trace_ids: set[str] = set()
        for position, entry in enumerate(trace, 1):
            where = f"ecm_trace:{position}"
            if not isinstance(entry, dict) or set(entry) != {"evidence_id", "supports"}:
                errors.append(f"{where}:schema")
                continue
            evidence_id, support = entry.get("evidence_id"), entry.get("supports")
            if (not isinstance(evidence_id, str) or evidence_id not in required_ids or evidence_id in trace_ids
                    or not isinstance(support, str) or not support.strip()):
                errors.append(f"{where}:invalid")
                continue
            trace_ids.add(evidence_id)
        if trace_ids != required_ids:
            errors.append("ecm_trace_does_not_cover_required_units")

    anchors = output.get("evidence_anchor")
    if not isinstance(anchors, list) or len(anchors) != len(required_ids):
        errors.append("ecm_locked_anchor_missing_or_wrong_count")
        return errors
    anchor_ids: set[str] = set()
    for position, anchor in enumerate(anchors, 1):
        where = f"ecm_anchor:{position}"
        if not isinstance(anchor, dict) or set(anchor) != {"evidence_id", "source", "support"}:
            errors.append(f"{where}:schema")
            continue
        evidence_id = anchor.get("evidence_id")
        if (not isinstance(evidence_id, str) or evidence_id not in required_ids or evidence_id in anchor_ids
                or anchor.get("source") != unit_by_id[evidence_id]["source"]
                or anchor.get("support") != unit_by_id[evidence_id]["content"]):
            errors.append(f"{where}:not_locked_unit")
            continue
        anchor_ids.add(evidence_id)
    if anchor_ids != required_ids:
        errors.append("ecm_locked_anchor_incomplete")
    return errors


def audit_row(row: dict[str, Any], evidence_by_cell: dict[tuple[str, str], dict[str, Any]], manifest_hash: str) -> list[str]:
    errors: list[str] = []
    method, condition, chunk_id = row.get("method"), row.get("condition"), row.get("chunk_id")
    if method not in METHODS or condition not in CONDITIONS or not isinstance(chunk_id, str):
        return ["invalid_method_or_condition"]
    evidence = evidence_by_cell.get((chunk_id, condition))
    if evidence is None:
        return ["evidence_missing"]
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
        errors.extend(ecm_errors(output, evidence, condition))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evidence-chain TQA records against frozen evidence.")
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
        "schema": "ecm-tqag.evidencechain-mechanical-audit.v3",
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
