"""Executable ECM construction contract: graph -> motif -> program -> atoms.

Implements the published protocol stages:

    G = (N, E, rho)        typed document graph with node provenance
    G_m subset G           motif-matched subgraph (closed catalog)
    z = compile(G_m)       restricted program (closed opcode set)
    A = Exec(z, G_m)       executor-derived answer atoms
    Gamma = Trace(z, G_m)  atom-to-source provenance trace

The model may propose only ``(G, motif_request)``. Matching, compilation,
execution, provenance (rho) and trace (Gamma) are computed by this module.

This module deliberately validates only structural/provenance properties. It
cannot establish that a model-proposed relation or claim is semantically true.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

GRAPH_SCHEMA = "ecm-tqag.document-graph.v1"
PROVENANCE_SCHEMA = "ecm-tqag.node-provenance.v1"
TRACE_SCHEMA = "ecm-tqag.atom-trace.v1"
CONSTRUCTION_SCHEMA = "ecm-tqag.graph-program-construction.v5"
PROGRAM_VERSION = "ecm-rpl.v1"
EVIDENCE_SOURCES = {"text", "structure", "image"}
EVIDENCE_ROLES = {"premise", "mapping", "constraint"}
EDGE_RELATIONS = {"supports", "maps_to", "constrains"}
ROLE_RELATION = {"premise": "supports", "mapping": "maps_to", "constraint": "constrains"}
MOTIF_CATALOG: dict[str, frozenset[str]] = {
    "premise_mapping_to_claim": frozenset({"premise", "mapping"}),
    "premise_constraint_to_claim": frozenset({"premise", "constraint"}),
    "mapping_constraint_to_claim": frozenset({"mapping", "constraint"}),
    "premise_mapping_constraint_to_claim": frozenset({"premise", "mapping", "constraint"}),
}
SOURCE_DESCRIPTOR_FIELDS = ("doc_id", "chunk_id", "condition", "manifest_sha256")


class ContractError(ValueError):
    """Fail-closed contract error with a stable machine-readable code."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def receipt(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def allowed_sources(condition: str) -> set[str]:
    try:
        return {
            "T": {"text"},
            "TL_struct": {"text", "structure"},
            "TLV": {"text", "structure", "image"},
        }[condition]
    except KeyError as exc:
        raise ContractError("invalid_condition") from exc


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def _source_fields(source: Any) -> dict[str, Any]:
    """Normalize the optional package descriptor into fixed provenance keys."""
    if source is None:
        return {key: None for key in SOURCE_DESCRIPTOR_FIELDS}
    _require(isinstance(source, dict), "source_descriptor_not_object")
    return {key: source.get(key) for key in SOURCE_DESCRIPTOR_FIELDS}


def _page_from_path(path: Any) -> int | None:
    if not isinstance(path, str):
        return None
    match = re.search(r"_page_(\d+)", path)
    return int(match.group(1)) if match else None


def _image_order(value: str) -> int:
    match = re.search(r"(?:ảnh|image)\s*(\d+)", value, flags=re.I)
    _require(bool(match), "graph_image_order_missing")
    assert match is not None
    return int(match.group(1))


def _validate_source_node(node: dict[str, Any], evidence: dict[str, Any], condition: str) -> None:
    source, node_id, value = node["source"], node["id"], node["value"]
    _require(source in allowed_sources(condition), "graph_source_not_available")
    prefix = {"text": "T", "structure": "S", "image": "I"}[source]
    _require(bool(re.fullmatch(prefix + r"[1-9][0-9]*", node_id)), "graph_evidence_id_prefix")
    if source == "text":
        _require(normalize(value) in normalize(str(evidence.get("text", ""))), "graph_text_not_literal")
    elif source == "structure":
        title = str(evidence.get("document_structure", {}).get("section_title", ""))
        _require(bool(title) and normalize(value) == normalize(title), "graph_structure_not_title")
    else:
        image_count = len(evidence.get("images", []))
        _require(_image_order(value) in range(1, image_count + 1), "graph_image_order_invalid")


def validate_document_graph(graph: Any, evidence: dict[str, Any], condition: str) -> None:
    """Validate typed nodes/edges and literal package binding; raise on failure."""
    _require(isinstance(graph, dict), "graph_not_object")
    _require(set(graph) == {"schema", "nodes", "edges"}, "graph_schema_fields")
    _require(graph["schema"] == GRAPH_SCHEMA, "graph_schema_version")
    nodes, edges = graph["nodes"], graph["edges"]
    _require(isinstance(nodes, list) and len(nodes) >= 3, "graph_nodes_too_short")
    _require(isinstance(edges, list) and len(edges) >= 2, "graph_edges_too_short")

    node_by_id: dict[str, dict[str, Any]] = {}
    claim_ids: list[str] = []
    for node in nodes:
        _require(isinstance(node, dict), "graph_node_not_object")
        _require(set(node) == {"id", "kind", "source", "role", "value"}, "graph_node_schema")
        node_id, kind, source, role, value = (node.get(key) for key in ("id", "kind", "source", "role", "value"))
        _require(isinstance(node_id, str) and node_id not in node_by_id, "graph_node_id")
        _require(isinstance(value, str) and bool(value.strip()), "graph_node_value")
        if kind == "evidence":
            _require(source in EVIDENCE_SOURCES and role in EVIDENCE_ROLES, "graph_evidence_type")
            _validate_source_node(node, evidence, condition)
        elif kind == "claim":
            _require(source == "derived" and role == "conclusion", "graph_claim_type")
            _require(bool(re.fullmatch(r"C[1-9][0-9]*", node_id)), "graph_claim_id")
            claim_ids.append(node_id)
        else:
            raise ContractError("graph_node_kind")
        node_by_id[node_id] = node
    _require(len(claim_ids) == 1, "graph_requires_one_claim")

    edge_ids: set[str] = set()
    semantic_edges: set[tuple[str, str, str]] = set()
    incoming_to_claim: set[str] = set()
    for edge in edges:
        _require(isinstance(edge, dict), "graph_edge_not_object")
        _require(set(edge) == {"id", "source", "target", "relation", "directed"}, "graph_edge_schema")
        edge_id, source, target, relation = (edge.get(key) for key in ("id", "source", "target", "relation"))
        _require(isinstance(edge_id, str) and bool(re.fullmatch(r"E[1-9][0-9]*", edge_id)) and edge_id not in edge_ids,
                 "graph_edge_id")
        _require(source in node_by_id and target in node_by_id, "graph_dangling_edge")
        _require(edge.get("directed") is True and relation in EDGE_RELATIONS, "graph_edge_type")
        _require(node_by_id[source]["kind"] == "evidence" and node_by_id[target]["kind"] == "claim",
                 "graph_edge_direction")
        semantic = (str(source), str(target), str(relation))
        _require(semantic not in semantic_edges, "graph_duplicate_semantic_edge")
        semantic_edges.add(semantic)
        edge_ids.add(edge_id)
        incoming_to_claim.add(str(source))
    _require(len(incoming_to_claim) >= 2, "graph_claim_needs_multi_evidence")


def node_provenance(graph: dict[str, Any], evidence: dict[str, Any], condition: str,
                    source: Any = None) -> dict[str, Any]:
    """Compute rho: every node -> a deterministic source descriptor."""
    base = _source_fields(source)
    text = str(evidence.get("text", ""))
    structure = evidence.get("document_structure") or {}
    images = evidence.get("images") or []
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
    mapping: dict[str, Any] = {}
    for node in graph["nodes"]:
        descriptor: dict[str, Any] = {
            "schema": PROVENANCE_SCHEMA, "kind": node["kind"], "source": node["source"], **base,
        }
        if node["kind"] == "claim":
            descriptor["derived_from"] = sorted(
                edge["source"] for edge in graph["edges"] if edge["target"] == node["id"]
            )
        elif node["source"] == "text":
            start = text.find(node["value"])
            descriptor.update({
                "text_sha256": text_digest,
                "char_span": [start, start + len(node["value"])] if start >= 0 else None,
                "match": "exact" if start >= 0 else "normalized",
            })
        elif node["source"] == "structure":
            descriptor.update({
                "section_title": structure.get("section_title"),
                "header_level": structure.get("header_level"),
                "declared_image_order": structure.get("declared_image_order"),
            })
        else:
            order = _image_order(node["value"])
            image = images[order - 1] if order - 1 < len(images) else {}
            descriptor.update({
                "declared_order": image.get("declared_order", order),
                "image_sha256": image.get("sha256"),
                "image_bytes": image.get("bytes"),
                "page": _page_from_path(image.get("path")),
            })
        mapping[node["id"]] = descriptor
    return mapping


def match_motif(graph: dict[str, Any], request: Any, condition: str) -> dict[str, Any]:
    """Replay a closed motif predicate against a validated graph."""
    _require(isinstance(request, dict), "motif_request_not_object")
    _require(set(request) == {"motif_id", "evidence_node_ids", "target_node_id", "image_necessary"},
             "motif_request_schema")
    motif_id = request.get("motif_id")
    required_roles = MOTIF_CATALOG.get(str(motif_id))
    _require(required_roles is not None, "motif_unknown")
    evidence_ids, target_id = request.get("evidence_node_ids"), request.get("target_node_id")
    _require(isinstance(evidence_ids, list) and len(evidence_ids) >= 2
             and all(isinstance(item, str) for item in evidence_ids)
             and len(set(evidence_ids)) == len(evidence_ids), "motif_evidence_ids")
    _require(isinstance(target_id, str), "motif_target_id")
    _require(isinstance(request.get("image_necessary"), bool), "motif_image_necessary_type")

    node_by_id = {node["id"]: node for node in graph["nodes"]}
    _require(target_id in node_by_id and node_by_id[target_id]["kind"] == "claim", "motif_target_not_claim")
    _require(all(item in node_by_id and node_by_id[item]["kind"] == "evidence" for item in evidence_ids),
             "motif_node_not_evidence")
    roles = frozenset(node_by_id[item]["role"] for item in evidence_ids)
    _require(roles == required_roles, "motif_role_mismatch")

    matched_edges: list[str] = []
    for node_id in evidence_ids:
        candidates = [edge for edge in graph["edges"] if edge["source"] == node_id and edge["target"] == target_id]
        _require(len(candidates) == 1, "motif_requires_one_edge_per_evidence")
        _require(candidates[0]["relation"] == ROLE_RELATION[node_by_id[node_id]["role"]],
                 "motif_role_relation_mismatch")
        matched_edges.append(candidates[0]["id"])
    extra = [edge for edge in graph["edges"] if edge["target"] == target_id and edge["source"] not in evidence_ids]
    _require(not extra, "motif_excludes_unbound_incoming_edge")

    has_image = any(node_by_id[item]["source"] == "image" for item in evidence_ids)
    if condition == "TLV":
        _require(request["image_necessary"] is True and has_image, "motif_tlv_image_required")
    else:
        _require(request["image_necessary"] is False and not has_image, "motif_non_tlv_image_forbidden")
    return {
        "catalog_version": "ecm-motif-catalog.v1",
        "motif_id": motif_id,
        "bindings": {
            "evidence_node_ids": list(evidence_ids),
            "target_node_id": target_id,
            "image_necessary": request["image_necessary"],
        },
        "matched_node_ids": [*evidence_ids, target_id],
        "matched_edge_ids": matched_edges,
    }


def compile_restricted_program(matched_motif: dict[str, Any]) -> dict[str, Any]:
    """Compile a motif match to the only allowed typed program shape."""
    bindings = matched_motif["bindings"]
    return {
        "version": PROGRAM_VERSION,
        "steps": [
            {"id": "S1", "op": "select_nodes", "args": {"node_ids": bindings["evidence_node_ids"]},
             "output": "evidence_nodes", "output_type": "NodeSet"},
            {"id": "S2", "op": "traverse_to_claim",
             "args": {"input": "evidence_nodes", "edge_ids": matched_motif["matched_edge_ids"],
                      "target_node_id": bindings["target_node_id"], "cardinality": "expect_exactly_one"},
             "output": "claim_nodes", "output_type": "NodeSet"},
            {"id": "S3", "op": "read_value", "args": {"input": "claim_nodes"},
             "output": "claim_values", "output_type": "ValueList"},
            {"id": "S4", "op": "return_atoms", "args": {"input": "claim_values", "order": "graph_id"},
             "output": "answer_atoms", "output_type": "AtomList"},
        ],
    }


def execute_program(graph: dict[str, Any], matched_motif: dict[str, Any], program: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute the closed program and compute authoritative atom provenance."""
    expected = compile_restricted_program(matched_motif)
    _require(program == expected, "program_not_compiler_output")
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    edge_by_id = {edge["id"]: edge for edge in graph["edges"]}
    evidence_ids = matched_motif["bindings"]["evidence_node_ids"]
    target_id = matched_motif["bindings"]["target_node_id"]
    edge_ids = matched_motif["matched_edge_ids"]
    _require(all(edge_by_id[edge_id]["source"] in evidence_ids and edge_by_id[edge_id]["target"] == target_id
                 for edge_id in edge_ids), "program_traversal_mismatch")
    target_ids = sorted({edge_by_id[edge_id]["target"] for edge_id in edge_ids})
    _require(target_ids == [target_id], "program_cardinality_not_one")
    value = node_by_id[target_id]["value"]
    _require(isinstance(value, str) and bool(value.strip()), "program_empty_atom")
    return [{
        "id": "A1",
        "type": "string",
        "value": value,
        "support": {
            "node_ids": [*evidence_ids, target_id],
            "edge_ids": list(edge_ids),
            "step_ids": ["S1", "S2", "S3", "S4"],
        },
    }]


def anchor_errors(anchors: Any, evidence: dict[str, Any], condition: str,
                  label: str = "anchor") -> list[str]:
    """Validate non-ECM evidence anchors against the frozen package.

    This is the single definition used both online by the runner and offline by
    the audit, so a record accepted at generation time cannot fail replay for a
    check that only one side implemented.
    """
    errors: list[str] = []
    try:
        allowed = allowed_sources(condition)
    except ContractError:
        return [f"{label}:invalid_condition"]
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
        if source == "text":
            if normalize(support) not in normalize(str(evidence.get("text", ""))):
                errors.append(f"{where}:text_not_literal")
        elif source == "structure":
            title = str(evidence.get("document_structure", {}).get("section_title", ""))
            if not title or normalize(support) != normalize(title):
                errors.append(f"{where}:structure_not_section_title")
        else:
            image_count = len(evidence.get("images", []))
            try:
                order = _image_order(support)
            except ContractError:
                errors.append(f"{where}:image_order_not_identified")
                continue
            if order not in range(1, image_count + 1):
                errors.append(f"{where}:image_order_not_identified")
    return errors


ATOM_CHOICE_MIN_TOKEN_RATIO = 0.5


def atom_choice_binding(atom_value: str, choice: str) -> dict[str, Any]:
    """Check that a realized option is a faithful surface form of an atom.

    Exact string equality would force the correct option to be a verbatim copy
    of the executor atom, which is both unnatural for MCQ authoring and a
    surface-form giveaway. Instead the binding is deterministic and replayable:
    after whitespace/case normalization one string must be a contiguous
    substring of the other, and the shorter must retain at least
    ``ATOM_CHOICE_MIN_TOKEN_RATIO`` of the longer one's tokens. The ratio is a
    declared contract parameter, not an estimated quantity; the measured value
    is returned so an auditor can recompute it.
    """
    atom_norm, choice_norm = normalize(atom_value), normalize(choice)
    atom_tokens, choice_tokens = atom_norm.split(), choice_norm.split()
    if not atom_tokens or not choice_tokens:
        return {"ok": False, "relation": "empty", "token_ratio": 0.0}
    if atom_norm == choice_norm:
        relation = "identical"
    elif choice_norm in atom_norm:
        relation = "choice_within_atom"
    elif atom_norm in choice_norm:
        relation = "atom_within_choice"
    else:
        return {"ok": False, "relation": "disjoint", "token_ratio": 0.0}
    shorter, longer = sorted((len(atom_tokens), len(choice_tokens)))
    ratio = round(shorter / longer, 4)
    return {
        "ok": ratio >= ATOM_CHOICE_MIN_TOKEN_RATIO,
        "relation": relation,
        "token_ratio": ratio,
        "min_token_ratio": ATOM_CHOICE_MIN_TOKEN_RATIO,
    }


def trace_program(matched_motif: dict[str, Any], atoms: list[dict[str, Any]],
                  provenance: dict[str, Any]) -> list[dict[str, Any]]:
    """Compute Gamma: each atom -> its source nodes/regions via rho."""
    trace: list[dict[str, Any]] = []
    for atom in atoms:
        support = atom["support"]
        trace.append({
            "schema": TRACE_SCHEMA,
            "atom_id": atom["id"],
            "edge_ids": list(support["edge_ids"]),
            "step_ids": list(support["step_ids"]),
            "regions": [
                {"node_id": node_id, "provenance": provenance[node_id]}
                for node_id in support["node_ids"]
            ],
        })
    return trace


def build_construction(proposal: Any, evidence: dict[str, Any], condition: str,
                       source: Any = None) -> dict[str, Any]:
    """Validate a model proposal, then match/compile/execute/trace locally."""
    _require(isinstance(proposal, dict), "planner_not_object")
    _require(set(proposal) == {"document_graph", "motif_request"}, "planner_top_level")
    graph = proposal["document_graph"]
    validate_document_graph(graph, evidence, condition)
    provenance = node_provenance(graph, evidence, condition, source)
    matched = match_motif(graph, proposal["motif_request"], condition)
    program = compile_restricted_program(matched)
    atoms = execute_program(graph, matched, program)
    trace = trace_program(matched, atoms, provenance)
    core = {
        "schema": CONSTRUCTION_SCHEMA,
        "document_graph": graph,
        "node_provenance": provenance,
        "matched_motif": matched,
        "restricted_program": program,
        "answer_atoms": atoms,
        "atom_trace": trace,
    }
    return {**core, "construction_sha256": receipt(core)}


CONSTRUCTION_FIELDS = {
    "schema", "document_graph", "node_provenance", "matched_motif",
    "restricted_program", "answer_atoms", "atom_trace", "construction_sha256",
}


def validate_construction(construction: Any, evidence: dict[str, Any], condition: str,
                          source: Any = None) -> list[str]:
    """Replay every deterministic stage; return stable error codes."""
    try:
        _require(isinstance(construction, dict), "construction_not_object")
        _require(set(construction) == CONSTRUCTION_FIELDS, "construction_fields")
        _require(construction["schema"] == CONSTRUCTION_SCHEMA, "construction_schema")
        graph = construction["document_graph"]
        validate_document_graph(graph, evidence, condition)
        expected_provenance = node_provenance(graph, evidence, condition, source)
        _require(construction["node_provenance"] == expected_provenance, "node_provenance_mismatch")
        matched = construction["matched_motif"]
        _require(isinstance(matched, dict) and set(matched) == {"catalog_version", "motif_id", "bindings",
                                                               "matched_node_ids", "matched_edge_ids"},
                 "matched_motif_schema")
        bindings = matched["bindings"]
        _require(isinstance(bindings, dict), "matched_motif_bindings")
        request = {
            "motif_id": matched["motif_id"],
            "evidence_node_ids": bindings.get("evidence_node_ids"),
            "target_node_id": bindings.get("target_node_id"),
            "image_necessary": bindings.get("image_necessary"),
        }
        replayed_match = match_motif(graph, request, condition)
        _require(matched == replayed_match, "matched_motif_not_replayable")
        expected_program = compile_restricted_program(matched)
        _require(construction["restricted_program"] == expected_program, "restricted_program_mismatch")
        expected_atoms = execute_program(graph, matched, expected_program)
        _require(construction["answer_atoms"] == expected_atoms, "answer_atoms_not_executor_output")
        expected_trace = trace_program(matched, expected_atoms, expected_provenance)
        _require(construction["atom_trace"] == expected_trace, "atom_trace_not_tracer_output")
        core = {key: construction[key] for key in CONSTRUCTION_FIELDS if key != "construction_sha256"}
        _require(construction["construction_sha256"] == receipt(core), "construction_receipt_mismatch")
        return []
    except (ContractError, KeyError, TypeError, IndexError) as exc:
        return [str(exc) or type(exc).__name__]
