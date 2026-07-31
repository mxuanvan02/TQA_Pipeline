from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research" / "run_qwen37_tqa_pilot.py"
SPEC = importlib.util.spec_from_file_location("run_qwen37_tqa_pilot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research" / "audit_strict_tqa_results.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_strict_tqa_results", AUDIT_SCRIPT)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
AUDIT_MODULE = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT_MODULE)


class Qwen37EvidenceChainContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tlv_evidence = {
            "text": "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh.",
            "document_structure": {"section_title": "Chủ thể trong giao dịch bảo lãnh", "header_level": 1},
            "images": [{"declared_order": 1}],
        }
        self.locked_plan = {
            "evidence_units": [
                {
                    "id": "T1",
                    "source": "text",
                    "role": "premise",
                    "content": "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh.",
                },
                {
                    "id": "I1",
                    "source": "image",
                    "role": "mapping",
                    "content": "Ảnh 1: quan hệ (2) nối người bảo lãnh với người được bảo lãnh.",
                },
            ],
            "evidence_plan": {
                "target": "Ánh xạ loại hợp đồng vào quan hệ trong sơ đồ.",
                "required_units": ["T1", "I1"],
                "chain": [
                    {"from": ["T1"], "to": "Cặp chủ thể của hợp đồng dịch vụ là bên bảo lãnh và bên được bảo lãnh."},
                    {"from": ["I1"], "to": "Quan hệ (2) là quan hệ của bên bảo lãnh với bên được bảo lãnh."},
                    {"from": ["T1", "I1"], "to": "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (2)."},
                ],
                "image_necessary": True,
            },
            "answer_atoms": ["Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (2)."],
        }

    def valid_ecm_realization(self) -> dict[str, Any]:
        return {
            "motif": "Ánh xạ văn bản–sơ đồ",
            "derivation": "Kết hợp cặp chủ thể trong văn bản với quan hệ (2) trong sơ đồ.",
            "answer_parts": list(self.locked_plan["answer_atoms"]),
            "trace": [
                {"evidence_id": "T1", "supports": "Xác định cặp chủ thể của hợp đồng dịch vụ."},
                {"evidence_id": "I1", "supports": "Ánh xạ cặp chủ thể vào quan hệ (2)."},
            ],
            "question": "Nhận định nào ánh xạ đúng hợp đồng dịch vụ bảo lãnh vào sơ đồ?",
            "choices": [
                "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (2).",
                "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (1).",
                "Hợp đồng dịch vụ bảo lãnh tương ứng với quan hệ (3).",
                "Không có quan hệ nào trong sơ đồ tương ứng với hợp đồng dịch vụ bảo lãnh.",
            ],
            "answer_index": 0,
            "rationale": "Văn bản xác định cặp chủ thể, còn sơ đồ ánh xạ cặp đó vào quan hệ (2).",
            "evidence_anchor": [
                {"evidence_id": "T1", "source": "text", "support": "Hợp đồng dịch vụ bảo lãnh được ký giữa bên bảo lãnh và bên được bảo lãnh."},
                {"evidence_id": "I1", "source": "image", "support": "Ảnh 1: quan hệ (2) nối người bảo lãnh với người được bảo lãnh."},
            ],
        }

    def test_direct_contract_accepts_nonduplicate_choices_and_anchor(self) -> None:
        raw = json.dumps({
            "question": "Cơ quan nào được nêu trong evidence?",
            "choices": ["Quốc hội", "Chính phủ", "Tòa án", "Ủy ban"],
            "answer_index": 0,
            "rationale": "Evidence nêu Quốc hội.",
            "evidence_anchor": [{"source": "text", "support": "Quốc hội"}],
        })
        result = MODULE.validate_response("direct", raw, {"text"})
        self.assertEqual(result["status"], "PARSED")

    def test_duplicate_choices_are_rejected(self) -> None:
        raw = json.dumps({
            "question": "Q?", "choices": ["A", "B", "A", "D"], "answer_index": 0,
            "rationale": "R", "evidence_anchor": [{"source": "text", "support": "A"}],
        })
        result = MODULE.validate_response("direct", raw, {"text"})
        self.assertEqual(result["reason"], "duplicate_choices")

    def test_answer_first_requires_answer_to_match_selected_choice(self) -> None:
        raw = json.dumps({
            "evidence_excerpt": "x", "answer": "B", "question": "Q?",
            "choices": ["A", "B", "C", "D"], "answer_index": 0, "rationale": "R",
            "evidence_anchor": [{"source": "text", "support": "x"}],
        })
        result = MODULE.validate_response("answer_first", raw, {"text"})
        self.assertEqual(result["reason"], "answer_first_inconsistent")

    def test_ecm_tlv_planner_accepts_source_bound_plan_and_derived_atom(self) -> None:
        result = MODULE.validate_ecm_plan(json.dumps(self.locked_plan), self.tlv_evidence, "TLV")
        self.assertEqual(result["status"], "PARSED")

    def test_ecm_planner_rejects_atom_not_derived_from_chain(self) -> None:
        invalid = json.loads(json.dumps(self.locked_plan))
        invalid["answer_atoms"] = ["Một kết luận không có trong chain."]
        result = MODULE.validate_ecm_plan(json.dumps(invalid), self.tlv_evidence, "TLV")
        self.assertEqual(result["reason"], "ecm_answer_atoms_not_derived")

    def test_ecm_planner_rejects_required_units_with_same_role(self) -> None:
        invalid = json.loads(json.dumps(self.locked_plan))
        invalid["evidence_units"][1]["role"] = "premise"
        result = MODULE.validate_ecm_plan(json.dumps(invalid), self.tlv_evidence, "TLV")
        self.assertEqual(result["reason"], "ecm_required_units_same_role")

    def test_ecm_tlv_planner_rejects_plan_without_required_image(self) -> None:
        invalid = json.loads(json.dumps(self.locked_plan))
        invalid["evidence_plan"]["image_necessary"] = False
        result = MODULE.validate_ecm_plan(json.dumps(invalid), self.tlv_evidence, "TLV")
        self.assertEqual(result["reason"], "ecm_tlv_image_not_required")

    def test_ecm_realizer_must_preserve_locked_answer_atoms(self) -> None:
        realization = self.valid_ecm_realization()
        result = MODULE.validate_response(
            "ecm", json.dumps(realization), {"text", "structure", "image"},
            self.tlv_evidence, "TLV", self.locked_plan,
        )
        self.assertEqual(result["status"], "PARSED")
        realization["answer_parts"] = ["Đã thay đổi atom."]
        changed = MODULE.validate_response(
            "ecm", json.dumps(realization), {"text", "structure", "image"},
            self.tlv_evidence, "TLV", self.locked_plan,
        )
        self.assertEqual(changed["reason"], "ecm_answer_atoms_changed")

    def test_ecm_realizer_rejects_anchor_that_is_not_locked_unit(self) -> None:
        realization = self.valid_ecm_realization()
        realization["evidence_anchor"][1]["support"] = "Ảnh 1: nội dung khác."
        result = MODULE.validate_response(
            "ecm", json.dumps(realization), {"text", "structure", "image"},
            self.tlv_evidence, "TLV", self.locked_plan,
        )
        self.assertEqual(result["reason"], "ecm_evidence_anchor_not_locked")

    def test_realizer_prompt_exposes_only_locked_plan(self) -> None:
        prompt, fields = MODULE.ecm_realizer_prompt(self.locked_plan)
        self.assertEqual(fields, ["locked_plan"])
        self.assertNotIn("EVIDENCE TEXT:", prompt)
        self.assertIn('"answer_atoms"', prompt)

    def test_mechanical_audit_enforces_v3_locked_anchor_contract(self) -> None:
        output = {**self.locked_plan, **self.valid_ecm_realization()}
        self.assertEqual(AUDIT_MODULE.ecm_errors(output, self.tlv_evidence, "TLV"), [])
        output["evidence_anchor"][1]["support"] = "Ảnh 1: nội dung khác."
        errors = AUDIT_MODULE.ecm_errors(output, self.tlv_evidence, "TLV")
        self.assertIn("ecm_anchor:2:not_locked_unit", errors)

    def test_manifest_builds_12_cell_tlv_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "image.jpeg"
            image.write_bytes(b"test-image")
            image_record = {
                "path": str(image), "bytes": image.stat().st_size,
                "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "declared_order": 1,
            }
            packages = []
            for chunk_index in range(8):
                chunk_id = f"chunk-{chunk_index}"
                packages.extend([
                    {"chunk_id": chunk_id, "condition": "T", "evidence": {"text": "Evidence."}},
                    {"chunk_id": chunk_id, "condition": "TL_struct", "evidence": {"text": "Evidence.", "document_structure": {}}},
                    {"chunk_id": chunk_id, "condition": "TLV", "evidence": {"text": "Evidence.", "document_structure": {}, "images": [image_record]}},
                ])
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"schema": MODULE.SCHEMA, "packages": packages}), encoding="utf-8")
            manifest, digest = MODULE.load_manifest(manifest_path)
            plan = MODULE.build_plan(manifest, digest, "unit", MODULE.DEFAULT_MODEL, {"chunk-0", "chunk-1", "chunk-2", "chunk-3"}, ("TLV",), 7)
            self.assertEqual(len(plan), 12)
            self.assertEqual({row["condition"] for row in plan}, {"TLV"})
            self.assertEqual({row["method"] for row in plan}, set(MODULE.METHODS))


if __name__ == "__main__":
    unittest.main()
