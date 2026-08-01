from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "research" / "run_qwen37_tqa_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_qwen37_tqa_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AUDIT_SCRIPT = ROOT / "scripts" / "research" / "audit_strict_tqa_results.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_strict_tqa_results", AUDIT_SCRIPT)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT_MODULE = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT_MODULE)

from src.ecm_graph_program import (  # noqa: E402
    ContractError,
    anchor_errors,
    atom_choice_binding,
    build_construction,
    validate_construction,
)


class Qwen37GraphProgramContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tlv_evidence = {
            "text": "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh.",
            "document_structure": {
                "section_title": "Chủ thể trong giao dịch bảo lãnh",
                "header_level": 1,
            },
            "images": [{"declared_order": 1}],
        }
        self.proposal: dict[str, Any] = {
            "document_graph": {
                "schema": "ecm-tqag.document-graph.v1",
                "nodes": [
                    {
                        "id": "T1",
                        "kind": "evidence",
                        "source": "text",
                        "role": "premise",
                        "value": "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh.",
                    },
                    {
                        "id": "I1",
                        "kind": "evidence",
                        "source": "image",
                        "role": "mapping",
                        "value": "Ảnh 1: quan hệ (2) nối bên bảo lãnh với bên được bảo lãnh.",
                    },
                    {
                        "id": "C1",
                        "kind": "claim",
                        "source": "derived",
                        "role": "conclusion",
                        "value": "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (2).",
                    },
                ],
                "edges": [
                    {
                        "id": "E1",
                        "source": "T1",
                        "target": "C1",
                        "relation": "supports",
                        "directed": True,
                    },
                    {
                        "id": "E2",
                        "source": "I1",
                        "target": "C1",
                        "relation": "maps_to",
                        "directed": True,
                    },
                ],
            },
            "motif_request": {
                "motif_id": "premise_mapping_to_claim",
                "evidence_node_ids": ["T1", "I1"],
                "target_node_id": "C1",
                "image_necessary": True,
            },
        }
        self.construction = build_construction(self.proposal, self.tlv_evidence, "TLV")

    def valid_ecm_realization(self) -> dict[str, Any]:
        return {
            "answer_atom_ids": ["A1"],
            "answer_parts": ["Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (2)."],
            "program_trace": ["S1", "S2", "S3", "S4"],
            "question": "Nhận định nào ánh xạ đúng hợp đồng dịch vụ bảo lãnh vào sơ đồ?",
            "choices": [
                "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (2).",
                "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (1).",
                "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (3).",
                "Không có quan hệ nào trong sơ đồ tương ứng với hợp đồng dịch vụ bảo lãnh.",
            ],
            "answer_index": 0,
            "rationale": "Node T1 xác định cặp chủ thể và node I1 ánh xạ cặp đó vào quan hệ (2).",
            "evidence_anchor": [
                {
                    "evidence_id": "T1",
                    "source": "text",
                    "support": "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh.",
                },
                {
                    "evidence_id": "I1",
                    "source": "image",
                    "support": "Ảnh 1: quan hệ (2) nối bên bảo lãnh với bên được bảo lãnh.",
                },
            ],
        }

    def test_direct_contract_accepts_nonduplicate_choices_and_literal_anchor(self) -> None:
        raw = json.dumps({
            "question": "Ai là các bên của hợp đồng dịch vụ bảo lãnh?",
            "choices": [
                "Bên bảo lãnh và bên được bảo lãnh",
                "Chỉ bên bảo lãnh",
                "Chỉ bên được bảo lãnh",
                "Cơ quan nhà nước",
            ],
            "answer_index": 0,
            "rationale": "Evidence nêu hai bên của hợp đồng.",
            "evidence_anchor": [
                {"source": "text", "support": "bên bảo lãnh và bên được bảo lãnh"},
            ],
        })
        result = MODULE.validate_response("direct", raw, {"text"}, self.tlv_evidence, "TLV")
        self.assertEqual(result["status"], "PARSED")

    def test_duplicate_choices_are_rejected(self) -> None:
        raw = json.dumps({
            "question": "Q?",
            "choices": ["A", "B", "A", "D"],
            "answer_index": 0,
            "rationale": "R",
            "evidence_anchor": [{"source": "text", "support": "A"}],
        })
        result = MODULE.validate_response("direct", raw, {"text"}, self.tlv_evidence, "TLV")
        self.assertEqual(result["reason"], "duplicate_choices")

    def test_answer_first_requires_answer_to_match_selected_choice(self) -> None:
        raw = json.dumps({
            "evidence_excerpt": "x",
            "answer": "B",
            "question": "Q?",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "rationale": "R",
            "evidence_anchor": [
                {"source": "text", "support": "bên bảo lãnh và bên được bảo lãnh"},
            ],
        })
        result = MODULE.validate_response(
            "answer_first", raw, {"text"}, self.tlv_evidence, "TLV",
        )
        self.assertEqual(result["reason"], "answer_first_inconsistent")

    def test_planner_builds_graph_motif_program_and_executor_atoms(self) -> None:
        result = MODULE.validate_ecm_plan(json.dumps(self.proposal), self.tlv_evidence, "TLV")
        self.assertEqual(result["status"], "PARSED")
        construction = result["parsed_response"]
        self.assertEqual(construction["matched_motif"]["motif_id"], "premise_mapping_to_claim")
        self.assertEqual(
            [step["op"] for step in construction["restricted_program"]["steps"]],
            ["select_nodes", "traverse_to_claim", "read_value", "return_atoms"],
        )
        self.assertEqual(construction["answer_atoms"], self.construction["answer_atoms"])
        self.assertEqual(validate_construction(construction, self.tlv_evidence, "TLV"), [])

    def test_planner_cannot_supply_answer_atoms_or_program(self) -> None:
        injected = deepcopy(self.proposal)
        injected["answer_atoms"] = [{"id": "A1", "value": "forged"}]
        result = MODULE.validate_ecm_plan(json.dumps(injected), self.tlv_evidence, "TLV")
        self.assertEqual(result["reason"], "construction_planner_top_level")

    def test_motif_matcher_rejects_role_relation_mismatch(self) -> None:
        invalid = deepcopy(self.proposal)
        invalid["document_graph"]["edges"][1]["relation"] = "supports"
        with self.assertRaisesRegex(ContractError, "motif_role_relation_mismatch"):
            build_construction(invalid, self.tlv_evidence, "TLV")

    def test_tlv_motif_requires_a_bound_image_node(self) -> None:
        invalid = deepcopy(self.proposal)
        invalid["motif_request"]["image_necessary"] = False
        with self.assertRaisesRegex(ContractError, "motif_tlv_image_required"):
            build_construction(invalid, self.tlv_evidence, "TLV")

    def test_replay_rejects_tampered_restricted_program(self) -> None:
        tampered = deepcopy(self.construction)
        tampered["restricted_program"]["steps"][2]["op"] = "invent_value"
        self.assertIn("restricted_program_mismatch", validate_construction(tampered, self.tlv_evidence, "TLV"))

    def test_replay_rejects_tampered_executor_atom(self) -> None:
        tampered = deepcopy(self.construction)
        tampered["answer_atoms"][0]["value"] = "Giá trị giả mạo."
        self.assertIn("answer_atoms_not_executor_output", validate_construction(tampered, self.tlv_evidence, "TLV"))

    def test_replay_rejects_tampered_receipt(self) -> None:
        tampered = deepcopy(self.construction)
        tampered["construction_sha256"] = "0" * 64
        self.assertIn("construction_receipt_mismatch", validate_construction(tampered, self.tlv_evidence, "TLV"))

    def test_realizer_must_preserve_executor_atoms_and_trace(self) -> None:
        realization = self.valid_ecm_realization()
        valid = MODULE.validate_response(
            "ecm",
            json.dumps(realization),
            {"text", "structure", "image"},
            self.tlv_evidence,
            "TLV",
            self.construction,
        )
        self.assertEqual(valid["status"], "PARSED")

        realization["answer_parts"] = ["Đã thay đổi atom."]
        changed = MODULE.validate_response(
            "ecm",
            json.dumps(realization),
            {"text", "structure", "image"},
            self.tlv_evidence,
            "TLV",
            self.construction,
        )
        self.assertEqual(changed["reason"], "ecm_answer_atoms_changed")

    def test_realizer_selected_choice_must_be_bound_to_executor_atom(self) -> None:
        realization = self.valid_ecm_realization()
        realization["answer_index"] = 1
        result = MODULE.validate_response(
            "ecm",
            json.dumps(realization),
            {"text", "structure", "image"},
            self.tlv_evidence,
            "TLV",
            self.construction,
        )
        self.assertEqual(result["reason"], "ecm_selected_choice_not_bound_to_atom:disjoint")

    def test_realizer_rejects_anchor_not_equal_to_locked_graph_node(self) -> None:
        realization = self.valid_ecm_realization()
        realization["evidence_anchor"][1]["support"] = "Ảnh 1: nội dung khác."
        result = MODULE.validate_response(
            "ecm",
            json.dumps(realization),
            {"text", "structure", "image"},
            self.tlv_evidence,
            "TLV",
            self.construction,
        )
        self.assertEqual(result["reason"], "ecm_evidence_anchor_not_locked")

    def test_realizer_prompt_exposes_only_locked_construction(self) -> None:
        prompt, fields = MODULE.ecm_realizer_prompt(self.construction)
        self.assertEqual(fields, ["locked_construction"])
        self.assertNotIn("EVIDENCE TEXT:", prompt)
        self.assertIn('"document_graph"', prompt)
        self.assertIn('"matched_motif"', prompt)
        self.assertIn('"restricted_program"', prompt)
        self.assertIn('"answer_atoms"', prompt)

    def test_mechanical_audit_replays_v5_and_detects_tampering(self) -> None:
        output = {**self.construction, **self.valid_ecm_realization()}
        self.assertEqual(AUDIT_MODULE.ecm_errors(output, self.tlv_evidence, "TLV"), [])
        output["restricted_program"]["steps"][0]["op"] = "forged_op"
        errors = AUDIT_MODULE.ecm_errors(output, self.tlv_evidence, "TLV")
        self.assertIn("ecm_construction:restricted_program_mismatch", errors)

    def test_rho_maps_every_node_to_a_source_descriptor(self) -> None:
        provenance = self.construction["node_provenance"]
        self.assertEqual(set(provenance), {"T1", "I1", "C1"})
        text_descriptor = provenance["T1"]
        self.assertEqual(text_descriptor["match"], "exact")
        self.assertEqual(
            text_descriptor["char_span"],
            [0, len(self.tlv_evidence["text"])],
        )
        self.assertEqual(
            text_descriptor["text_sha256"],
            hashlib.sha256(self.tlv_evidence["text"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(provenance["I1"]["declared_order"], 1)
        self.assertEqual(provenance["C1"]["derived_from"], ["I1", "T1"])

    def test_rho_records_the_package_source_descriptor(self) -> None:
        source = {
            "doc_id": "17,18. LUAT HIEN PHAP VIET NAM",
            "chunk_id": "chunk_79",
            "condition": "TLV",
            "manifest_sha256": "a" * 64,
        }
        construction = build_construction(self.proposal, self.tlv_evidence, "TLV", source)
        self.assertEqual(
            construction["node_provenance"]["T1"]["doc_id"],
            "17,18. LUAT HIEN PHAP VIET NAM",
        )
        self.assertEqual(construction["node_provenance"]["I1"]["manifest_sha256"], "a" * 64)
        self.assertEqual(validate_construction(construction, self.tlv_evidence, "TLV", source), [])
        wrong_source = {**source, "doc_id": "another document"}
        self.assertIn(
            "node_provenance_mismatch",
            validate_construction(construction, self.tlv_evidence, "TLV", wrong_source),
        )

    def test_replay_rejects_tampered_rho(self) -> None:
        tampered = deepcopy(self.construction)
        tampered["node_provenance"]["T1"]["char_span"] = [0, 5]
        self.assertIn(
            "node_provenance_mismatch",
            validate_construction(tampered, self.tlv_evidence, "TLV"),
        )

    def test_gamma_links_each_atom_to_its_source_regions(self) -> None:
        trace = self.construction["atom_trace"]
        self.assertEqual(len(trace), len(self.construction["answer_atoms"]))
        entry = trace[0]
        self.assertEqual(entry["atom_id"], "A1")
        self.assertEqual(entry["step_ids"], ["S1", "S2", "S3", "S4"])
        self.assertEqual([region["node_id"] for region in entry["regions"]], ["T1", "I1", "C1"])
        self.assertEqual(
            entry["regions"][0]["provenance"],
            self.construction["node_provenance"]["T1"],
        )

    def test_replay_rejects_tampered_gamma(self) -> None:
        tampered = deepcopy(self.construction)
        tampered["atom_trace"][0]["regions"] = tampered["atom_trace"][0]["regions"][:1]
        self.assertIn(
            "atom_trace_not_tracer_output",
            validate_construction(tampered, self.tlv_evidence, "TLV"),
        )

    def test_markdown_fence_is_stripped_and_recorded(self) -> None:
        fenced = "```json\n" + json.dumps(self.proposal) + "\n```"
        result = MODULE.validate_ecm_plan(fenced, self.tlv_evidence, "TLV")
        self.assertEqual(result["status"], "PARSED")
        self.assertIn("stripped_markdown_code_fence", result["raw_json_normalizations"])

    def test_non_json_fence_language_is_not_stripped(self) -> None:
        fenced = "```python\nnot json\n```"
        value, normalizations, error = MODULE.parse_response_json(fenced)
        self.assertIsNone(value)
        self.assertEqual(normalizations, [])
        self.assertTrue(str(error).startswith("invalid_json:"))

    def test_atom_choice_binding_accepts_faithful_paraphrase_length(self) -> None:
        atom = self.construction["answer_atoms"][0]["value"]
        shortened = atom.split(" tương ứng")[0]
        binding = atom_choice_binding(atom, shortened)
        self.assertEqual(binding["relation"], "choice_within_atom")
        self.assertTrue(binding["ok"])

    def test_atom_choice_binding_rejects_disjoint_or_too_short(self) -> None:
        atom = self.construction["answer_atoms"][0]["value"]
        self.assertFalse(atom_choice_binding(atom, "Một đáp án khác hẳn.")["ok"])
        truncated = " ".join(atom.split()[:2])
        stub = atom_choice_binding(atom, truncated)
        self.assertEqual(stub["relation"], "choice_within_atom")
        self.assertFalse(stub["ok"])

    def test_realizer_choice_need_not_copy_atom_verbatim(self) -> None:
        realization = self.valid_ecm_realization()
        atom = self.construction["answer_atoms"][0]["value"]
        realization["choices"][realization["answer_index"]] = atom.split(" tương ứng")[0]
        result = MODULE.validate_response(
            "ecm",
            json.dumps(realization),
            {"text", "structure", "image"},
            self.tlv_evidence,
            "TLV",
            self.construction,
        )
        self.assertEqual(result["status"], "PARSED")
        self.assertEqual(result["parsed_response"]["atom_choice_binding"]["relation"], "choice_within_atom")

    def test_audit_replays_atom_choice_binding(self) -> None:
        realization = self.valid_ecm_realization()
        atom = self.construction["answer_atoms"][0]["value"]
        realization["choices"][realization["answer_index"]] = atom.split(" tương ứng")[0]
        output = {**self.construction, **realization}
        output["atom_choice_binding"] = atom_choice_binding(atom, output["choices"][output["answer_index"]])
        self.assertEqual(AUDIT_MODULE.ecm_errors(output, self.tlv_evidence, "TLV"), [])
        output["choices"][output["answer_index"]] = "Nội dung không liên quan."
        errors = AUDIT_MODULE.ecm_errors(output, self.tlv_evidence, "TLV")
        self.assertIn("ecm_selected_choice_not_bound_to_atom:disjoint", errors)

    def test_runner_and_audit_reject_the_same_nonliteral_anchor(self) -> None:
        """A record the runner accepts must never fail the offline replay."""
        forged = json.dumps({
            "question": "Cơ quan nào là cơ quan chấp hành?",
            "choices": ["Chính phủ", "Quốc hội", "Tòa án", "Viện kiểm sát"],
            "answer_index": 0,
            "rationale": "Evidence nêu cơ quan chấp hành.",
            "evidence_anchor": [
                {"source": "text", "support": "Một câu không có trong package."},
            ],
        })
        runner = MODULE.validate_response(
            "direct", forged, {"text"}, self.tlv_evidence, "TLV",
        )
        self.assertEqual(runner["status"], "REJECTED")
        self.assertEqual(runner["reason"], "invalid_evidence_anchor:anchor:1:text_not_literal")
        self.assertEqual(
            anchor_errors(
                json.loads(forged)["evidence_anchor"], self.tlv_evidence, "TLV",
            ),
            ["anchor:1:text_not_literal"],
        )

    def test_runner_accepts_a_literal_anchor_the_audit_also_accepts(self) -> None:
        accepted = json.dumps({
            "question": "Nội dung nào được nêu trong evidence?",
            "choices": [
                "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh.",
                "Hợp đồng chỉ do một bên đơn phương xác lập.",
                "Hợp đồng không cần bên được bảo lãnh.",
                "Hợp đồng do cơ quan nhà nước ban hành.",
            ],
            "answer_index": 0,
            "rationale": "Evidence nêu rõ hai bên của hợp đồng.",
            "evidence_anchor": [
                {"source": "text", "support": self.tlv_evidence["text"]},
            ],
        })
        runner = MODULE.validate_response(
            "direct", accepted, {"text"}, self.tlv_evidence, "TLV",
        )
        self.assertEqual(runner["status"], "PARSED")
        self.assertEqual(
            anchor_errors(
                json.loads(accepted)["evidence_anchor"], self.tlv_evidence, "TLV",
            ),
            [],
        )

    def test_manifest_builds_12_cell_tlv_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "image.jpeg"
            image.write_bytes(b"test-image")
            image_record = {
                "path": str(image),
                "bytes": image.stat().st_size,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                "declared_order": 1,
            }
            packages = []
            for chunk_index in range(8):
                chunk_id = f"chunk-{chunk_index}"
                packages.extend([
                    {"chunk_id": chunk_id, "condition": "T", "evidence": {"text": "Evidence."}},
                    {
                        "chunk_id": chunk_id,
                        "condition": "TL_struct",
                        "evidence": {"text": "Evidence.", "document_structure": {}},
                    },
                    {
                        "chunk_id": chunk_id,
                        "condition": "TLV",
                        "evidence": {"text": "Evidence.", "document_structure": {}, "images": [image_record]},
                    },
                ])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps({"schema": MODULE.SCHEMA, "packages": packages}),
                encoding="utf-8",
            )
            manifest, digest = MODULE.load_manifest(manifest_path)
            plan = MODULE.build_plan(
                manifest,
                digest,
                "unit",
                MODULE.DEFAULT_MODEL,
                {"chunk-0", "chunk-1", "chunk-2", "chunk-3"},
                ("TLV",),
                7,
            )
            self.assertEqual(len(plan), 12)
            self.assertEqual({row["condition"] for row in plan}, {"TLV"})
            self.assertEqual({row["method"] for row in plan}, set(MODULE.METHODS))


if __name__ == "__main__":
    unittest.main()
