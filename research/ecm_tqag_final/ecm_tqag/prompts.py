from __future__ import annotations

import hashlib
import json
from typing import Any

SYSTEM_JSON = "Return one valid JSON object only."
DECODING = {"temperature": 0, "max_tokens": 1200}
PLANNER_TEMPLATE_ID = "ecm-tqag.answer-first-planner"
REALIZER_TEMPLATE_ID = "ecm-tqag.sealed-realizer"

# The item object every construction path must emit. Both the sealed realizer
# (planner_realizer arms) and the direct generators (text_only, direct) render
# this exact block, so an item is the same object regardless of which arm made
# it. If the arms declared different item schemas, an item-level comparison
# between them would be comparing formats, not construction quality.
ITEM_KEYS = frozenset({"question", "choices", "answer_index", "rationale", "distractor_faults"})
ITEM_SCHEMA_BLOCK = (
    "{\"question\":\"...\",\"choices\":[\"...\",\"...\",\"...\",\"...\"],"
    "\"answer_index\":0,\"rationale\":\"...\",\"distractor_faults\":[\"...\",\"...\",\"...\"]}"
)
N_CHOICES = 4
N_DISTRACTOR_FAULTS = 3


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def planner_prompt(text: str, structure: dict[str, Any], interface: dict[str, Any],
                   *, interface_kind: str) -> str:
    """One downstream template for graph and caption interfaces.

    The interface slot changes by arm; instructions, text, structure, output schema,
    decoding and call path do not. No package identifier or path is accepted.
    """
    if interface_kind not in {"closed_graph", "caption"}:
        raise ValueError("interface_kind must be closed_graph or caption")
    return (
        "Bạn là bộ LẬP KẾ HOẠCH ANSWER-FIRST. Không viết câu hỏi. "
        "Chọn đúng một quan hệ có trong INTERFACE, trích đúng một quy tắc/điều kiện "
        "10--25 từ từ TEXT, rồi khóa đáp án phụ thuộc vào cả hai nguồn. Nếu không có "
        "quan hệ phù hợp, trả {\"abstain\":\"no_cross_modal_dependency\"}.\n"
        "Chỉ trả JSON theo một trong hai typed schema sau. "
        "Motif diagram_text_reconcile dùng ops locate, locate, reconcile; motif "
        "figure_condition_apply dùng ops locate, locate, apply. Bước 0 là visual: "
        "anchor và result đọc từ INTERFACE, visual_node là chỉ số ảnh 1-based, bbox "
        "chuẩn hóa [x0,y0,x1,y1]. Bước 1 là text: anchor phải là trích dẫn nguyên văn "
        "10--25 từ trong TEXT, không có visual_node/bbox. Bước 2 là derived: anchor=null.\n"
        "{\"motif\":\"diagram_text_reconcile\",\"steps\":["
        "{\"op\":\"locate\",\"anchor\":\"...\",\"result\":\"...\","
        "\"visual_node\":1,\"bbox\":[0.1,0.1,0.4,0.3]},"
        "{\"op\":\"locate\",\"anchor\":\"...\",\"result\":\"...\"},"
        "{\"op\":\"reconcile\",\"anchor\":null,\"result\":\"...\"}]}\n"
        f"INTERFACE_KIND={interface_kind}\n"
        f"DOCUMENT_STRUCTURE={canonical(structure)}\n"
        f"INTERFACE={canonical(interface)}\n"
        f"TEXT={text}"
    )


def planner_static_fingerprint() -> str:
    sentinel = planner_prompt("<TEXT>", {"slot": "<STRUCTURE>"}, {"slot": "<INTERFACE>"},
                              interface_kind="closed_graph")
    # Normalize only the declared interface-kind value so graph/caption arms can
    # prove that the surrounding template is byte-identical.
    sentinel = sentinel.replace("INTERFACE_KIND=closed_graph", "INTERFACE_KIND=<KIND>")
    return hashlib.sha256(sentinel.encode("utf-8")).hexdigest()


_PLANNER_MOTIFS = {
    "diagram_text_reconcile",
    "figure_condition_apply",
}
_PLANNER_STEP_KEYS = {"op", "anchor", "result", "visual_node", "bbox"}


def planner_program(raw: str) -> dict[str, Any]:
    """Parse the planner's closed typed-program contract without repairing it."""
    try:
        obj = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"planner_schema:invalid_json:{type(exc).__name__}") from exc
    if not isinstance(obj, dict):
        raise ValueError("planner_schema:top_level_not_object")
    if obj == {"abstain": "no_cross_modal_dependency"}:
        return obj
    if set(obj) != {"motif", "steps"}:
        raise ValueError("planner_schema:expected_motif_and_steps")
    if obj["motif"] not in _PLANNER_MOTIFS:
        raise ValueError("planner_schema:unknown_motif")
    steps = obj["steps"]
    if not isinstance(steps, list) or len(steps) != 3:
        raise ValueError("planner_schema:steps_must_have_three_entries")
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or not set(step) <= _PLANNER_STEP_KEYS:
            raise ValueError(f"planner_schema:invalid_step:{i}")
        if not isinstance(step.get("op"), str) or not isinstance(step.get("result"), str) or not step["result"].strip():
            raise ValueError(f"planner_schema:invalid_step_fields:{i}")
        if i < 2 and (not isinstance(step.get("anchor"), str) or not step["anchor"].strip()):
            raise ValueError(f"planner_schema:missing_anchor:{i}")
        if i == 2 and step.get("anchor") is not None:
            raise ValueError("planner_schema:derived_step_must_not_have_anchor")
    return obj


_REALIZER_PUBLIC_KEYS = {"motif", "atoms", "anchors", "visual_observations", "trace_hash"}
_REALIZER_INTERNAL_KEYS = {
    "motif", "condition", "atoms", "anchors", "visual_nodes",
    "visual_observations", "construction_hash",
}


def realizer_payload(sealed: dict[str, Any]) -> dict[str, Any]:
    """Project the internal construction record onto the public realizer contract.

    Internal routing metadata is deliberately discarded. Unknown keys fail closed so
    identifiers, paths, conditions, or run metadata cannot silently enter a prompt.
    """
    extra = set(sealed) - _REALIZER_INTERNAL_KEYS
    if extra:
        raise ValueError(f"sealed payload contains forbidden keys: {sorted(extra)}")
    required = {"motif", "atoms", "anchors", "visual_observations", "construction_hash"}
    missing = required - set(sealed)
    if missing:
        raise ValueError(f"sealed payload missing required keys: {sorted(missing)}")
    public = {
        "motif": sealed["motif"],
        "atoms": sealed["atoms"],
        "anchors": sealed["anchors"],
        "visual_observations": sealed["visual_observations"],
        "trace_hash": sealed["construction_hash"],
    }
    if set(public) != _REALIZER_PUBLIC_KEYS:
        raise AssertionError("realizer projection violated its closed schema")
    return public


def realizer_prompt(sealed: dict[str, Any]) -> str:
    public = realizer_payload(sealed)
    return (
        "Bạn là bộ DIỄN ĐẠT. Không được xem tài liệu gốc. Giữ nguyên answer atoms; "
        "viết một câu hỏi trắc nghiệm tiếng Việt có đúng bốn lựa chọn. Chỉ trả JSON "
        f"{ITEM_SCHEMA_BLOCK}.\n"
        f"SEALED={canonical(public)}"
    )
