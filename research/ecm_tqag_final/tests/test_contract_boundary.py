import json

import pytest

from ecm_tqag.prompts import realizer_payload, realizer_prompt


def internal_sealed() -> dict:
    return {
        "motif": "figure_condition_apply",
        "condition": "TLV",
        "atoms": [{"atom_id": "a0", "op": "locate", "value": "x"}],
        "anchors": [{"atom_id": "a0", "excerpt": "quy tắc nguồn", "sentence_index": 0}],
        "visual_nodes": ["image-1"],
        "visual_observations": [
            {"atom_id": "a0", "visual_node": 1, "bbox": [0.1, 0.2, 0.3, 0.4],
             "anchor": "n1", "observation": "nút trên nối với nút dưới"}
        ],
        "construction_hash": "abc123",
    }


def test_internal_seal_is_projected_to_public_realizer_contract() -> None:
    public = realizer_payload(internal_sealed())
    assert set(public) == {"motif", "atoms", "anchors", "visual_observations", "trace_hash"}
    assert public["trace_hash"] == "abc123"
    rendered = realizer_prompt(internal_sealed())
    payload = json.loads(rendered.split("SEALED=", 1)[1])
    assert payload == public
    assert '"condition"' not in rendered
    assert '"visual_nodes"' not in rendered
    assert '"construction_hash"' not in rendered


def test_unknown_internal_metadata_fails_closed() -> None:
    sealed = internal_sealed() | {"chunk_id": "must-not-leak"}
    with pytest.raises(ValueError, match="forbidden keys"):
        realizer_payload(sealed)
