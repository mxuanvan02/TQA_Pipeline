"""Deterministic multi-image interface composition.

Each image is kept as a separate, numbered visual source.  Graph node IDs are
namespaced by that source so edges cannot accidentally cross images or collide.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _ordered(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("interface_merge:images_must_be_nonempty")
    orders = [r.get("declared_order") for r in records]
    int_orders: list[int] = []
    for order in orders:
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("interface_merge:invalid_declared_order")
        int_orders.append(order)
    if len(set(int_orders)) != len(int_orders) or sorted(int_orders) != list(range(1, len(records) + 1)):
        raise ValueError("interface_merge:orders_must_be_contiguous")
    return sorted(records, key=lambda r: int(r["declared_order"]))


def _bbox(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("interface_merge:bbox_out_of_unit_square")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
        raise ValueError("interface_merge:bbox_out_of_unit_square")
    vals = [float(v) for v in value]
    x0, y0, x1, y1 = vals
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("interface_merge:bbox_out_of_unit_square")
    return [round(v, 6) for v in vals]


def merge_graph_interfaces(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = _ordered(records)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for image_index, record in enumerate(ordered, 1):
        graph = record.get("interface")
        if not isinstance(graph, dict):
            raise ValueError(f"interface_merge:graph_not_object:{image_index}")
        raw_nodes = graph.get("nodes")
        raw_edges = graph.get("edges")
        if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
            raise ValueError(f"interface_merge:graph_lists_missing:{image_index}")
        local_ids = {n.get("id") for n in raw_nodes if isinstance(n, dict)}
        if len(local_ids) != len(raw_nodes) or None in local_ids:
            raise ValueError(f"interface_merge:duplicate_or_invalid_node_id:{image_index}")
        prefix = f"i{image_index}:"
        for node in raw_nodes:
            if not isinstance(node, dict) or not isinstance(node.get("id"), str):
                raise ValueError(f"interface_merge:invalid_node:{image_index}")
            copied = dict(node)
            copied["id"] = prefix + node["id"]
            copied["bbox"] = _bbox(node.get("bbox"))
            copied["image_index"] = image_index
            nodes.append(copied)
        for edge in raw_edges:
            if not isinstance(edge, dict):
                raise ValueError(f"interface_merge:invalid_edge:{image_index}")
            if edge.get("src") not in local_ids or edge.get("dst") not in local_ids:
                raise ValueError(f"interface_merge:dangling_edge:{image_index}")
            copied = dict(edge)
            copied["src"] = prefix + edge["src"]
            copied["dst"] = prefix + edge["dst"]
            copied["image_index"] = image_index
            edges.append(copied)
        images.append({"image_index": image_index, "declared_order": record["declared_order"]})
    return {"interface_kind": "merged_closed_graph", "image_count": len(ordered),
            "images": images, "nodes": nodes, "edges": edges}


def merge_caption_interfaces(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = _ordered(records)
    images: list[dict[str, Any]] = []
    all_relations: list[str] = []
    for image_index, record in enumerate(ordered, 1):
        interface = record.get("interface")
        if not isinstance(interface, dict) or not isinstance(interface.get("caption"), str):
            raise ValueError(f"interface_merge:caption_not_object:{image_index}")
        relations = interface.get("relations")
        if not isinstance(relations, list) or any(not isinstance(x, str) for x in relations):
            raise ValueError(f"interface_merge:relations_invalid:{image_index}")
        images.append({"image_index": image_index, "declared_order": record["declared_order"],
                       "caption": interface["caption"], "relations": list(relations)})
        all_relations.extend(f"[image {image_index}] {r}" for r in relations)
    return {"interface_kind": "merged_caption", "image_count": len(images),
            "images": images, "relations": all_relations,
            "caption": " ".join(x["caption"] for x in images)}


__all__ = ["merge_graph_interfaces", "merge_caption_interfaces"]
