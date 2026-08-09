from __future__ import annotations

import pytest

from ecm_tqag.interface_merge import merge_caption_interfaces, merge_graph_interfaces


def _graph(label: str) -> dict:
    return {"graph_type": "TREE", "nodes": [
        {"id": "n1", "label": label, "level": 0, "bbox": [0.1, 0.1, 0.4, 0.3]},
        {"id": "n2", "label": label + "-2", "level": 1, "bbox": [0.2, 0.5, 0.5, 0.7]},
    ], "edges": [{"src": "n1", "dst": "n2", "kind": "ARROW", "label": ""}]}


def test_merge_three_images_is_ordered_and_collision_free() -> None:
    out = merge_graph_interfaces([
        {"declared_order": 3, "interface": _graph("c")},
        {"declared_order": 1, "interface": _graph("a")},
        {"declared_order": 2, "interface": _graph("b")},
    ])
    assert out["image_count"] == 3
    assert [x["image_index"] for x in out["images"]] == [1, 2, 3]
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["i1:n1", "i1:n2", "i2:n1", "i2:n2", "i3:n1", "i3:n2"]
    assert out["edges"][2]["src"] == "i3:n1"
    assert out["edges"][2]["dst"] == "i3:n2"


def test_merge_rejects_missing_order_and_out_of_range_bbox() -> None:
    with pytest.raises(ValueError, match="orders_must_be_contiguous"):
        merge_graph_interfaces([
            {"declared_order": 1, "interface": _graph("a")},
            {"declared_order": 3, "interface": _graph("b")},
        ])
    bad = _graph("a")
    bad["nodes"][0]["bbox"] = [-0.1, 0, 0.2, 0.3]
    with pytest.raises(ValueError, match="bbox_out_of_unit_square"):
        merge_graph_interfaces([{"declared_order": 1, "interface": bad}])


def test_caption_merge_keeps_image_boundary() -> None:
    out = merge_caption_interfaces([
        {"declared_order": 2, "interface": {"caption": "ảnh hai", "relations": ["a nối b"]}},
        {"declared_order": 1, "interface": {"caption": "ảnh một", "relations": ["x chứa y"]}},
    ])
    assert [x["caption"] for x in out["images"]] == ["ảnh một", "ảnh hai"]
    assert [x["image_index"] for x in out["images"]] == [1, 2]
