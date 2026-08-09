"""ECM v2: closed motif catalog, deterministic compiler, executor, and hard guards.

Design contract (pre-registered in prereg_ecm_v2.json):
  motif  : must be a member of MOTIF_CATALOG (closed set)  -> matcher
  program: compiled from the motif's typed step template     -> compiler
  exec   : each step is checked against the evidence         -> executor
  guard  : min_steps>=2, min_distinct_anchors>=2, anti-verbatim

Nothing here calls a model. All functions are deterministic and unit-testable,
so a reviewer can rerun the compiler/executor without API access.
"""
from __future__ import annotations
import re, unicodedata, math
from collections import Counter
from typing import Any

# ---------------------------------------------------------------- text utils
def norm(s: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).lower()).strip()

def words(s: Any) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ỹ0-9]+", norm(s))

def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;:!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 25]

def longest_common_run(a: list[str], b_text: str) -> int:
    """Longest run of consecutive words of `a` that appears verbatim in b_text."""
    if not a:
        return 0
    bstr = " " + " ".join(words(b_text)) + " "
    for L in range(min(len(a), 30), 0, -1):
        for i in range(len(a) - L + 1):
            if " " + " ".join(a[i:i + L]) + " " in bstr:
                return L
    return 0

# ---------------------------------------------------------------- motif catalog
# Each motif declares typed steps. `op` is the reasoning operation the step
# performs; `needs` is the number of DISTINCT evidence anchors it consumes.
MOTIF_CATALOG: dict[str, dict[str, Any]] = {
    "compare_two_entities": {
        "steps": [
            {"op": "locate", "needs": 1, "desc": "xác định thuộc tính của thực thể thứ nhất"},
            {"op": "locate", "needs": 1, "desc": "xác định thuộc tính của thực thể thứ hai"},
            {"op": "compare", "needs": 0, "desc": "so sánh hai thuộc tính để rút ra khác biệt"},
        ],
        "min_anchors": 2,
    },
    "condition_chain": {
        "steps": [
            {"op": "locate", "needs": 1, "desc": "xác định điều kiện áp dụng"},
            {"op": "locate", "needs": 1, "desc": "xác định hệ quả pháp lý của điều kiện"},
            {"op": "chain", "needs": 0, "desc": "nối điều kiện với hệ quả"},
        ],
        "min_anchors": 2,
    },
    "exception_detect": {
        "steps": [
            {"op": "locate", "needs": 1, "desc": "xác định quy tắc chung"},
            {"op": "locate", "needs": 1, "desc": "xác định trường hợp loại trừ / ngoại lệ"},
            {"op": "subtract", "needs": 0, "desc": "loại ngoại lệ khỏi quy tắc chung"},
        ],
        "min_anchors": 2,
    },
    # NOTE (v3 fix): `traverse` is a DERIVED reasoning step, not a quotable one.
    # A parent-child relation in Vietnamese legal prose is normally spread over
    # two sentences, so requiring the traversal step itself to carry a verbatim
    # anchor was an internal contradiction: the pilot rejected 5/5 of these with
    # step1_op_mismatch while the model's own decomposition (locate, locate,
    # traverse) was correct. Aligned with the other four motifs: two anchored
    # locate steps feed one derived step. Anchor count and distinctness are
    # UNCHANGED (min_anchors=2), so this is not a relaxation of the guard.
    "hierarchy_traverse": {
        "steps": [
            {"op": "locate", "needs": 1, "desc": "xác định phần tử khởi đầu trong cấu trúc"},
            {"op": "locate", "needs": 1, "desc": "xác định phần tử/cấp quan hệ liên quan"},
            {"op": "traverse", "needs": 0, "desc": "đi lên/xuống ít nhất một cấp và nêu phần tử đích"},
        ],
        "min_anchors": 2,
    },
    "numeric_derive": {
        "steps": [
            {"op": "locate", "needs": 1, "desc": "xác định giá trị/mốc số thứ nhất"},
            {"op": "locate", "needs": 1, "desc": "xác định giá trị/thời hạn thứ hai"},
            {"op": "compute", "needs": 0, "desc": "thực hiện phép tính trên hai giá trị"},
        ],
        "min_anchors": 2,
    },
    # ------------------------------------------------------------------ v4:
    # CROSS-MODAL MOTIFS.
    #
    # Why these exist. The five motifs above are text-only by construction:
    # every anchored step quotes prose, and `visual_node` was an OPTIONAL tag a
    # planner could bolt onto any step. Consequence measured in full_v3e: TLV
    # was effectively `TL_struct` + a label, 4/16 TLV cells self-abstained with
    # no_visual_evidence because the chosen motif gave the image no role, and
    # only 2/16 TLV cells reached PARSED. "Multimodal" was a condition name,
    # not a mechanism.
    #
    # These motifs make the image STRUCTURALLY REQUIRED: one anchored step must
    # read the image (`modality="visual"`), another must quote prose
    # (`modality="text"`), and the derived step joins the two. A planner cannot
    # satisfy the step template without genuinely using both modalities, so
    # "no visual binding" becomes a compile failure of the motif rather than a
    # missing self-declared tag. `min_visual_anchors`/`min_text_anchors` are
    # enforced in compile_program(), so the requirement is deterministic and
    # replayable without API access.
    "diagram_text_reconcile": {
        "steps": [
            {"op": "locate", "needs": 1, "modality": "visual",
             "desc": "đọc một quan hệ/nhãn cụ thể trên sơ đồ hoặc hình trong ảnh trang"},
            {"op": "locate", "needs": 1, "modality": "text",
             "desc": "trích quy định trong văn bản nói về quan hệ đó"},
            {"op": "reconcile", "needs": 0,
             "desc": "đối chiếu quan hệ đọc từ hình với quy định trong văn bản để rút ra kết luận"},
        ],
        "min_anchors": 2, "min_visual_anchors": 1, "min_text_anchors": 1,
    },
    "figure_condition_apply": {
        "steps": [
            {"op": "locate", "needs": 1, "modality": "visual",
             "desc": "định vị một giá trị/ô/phần tử cụ thể trong bảng hoặc hình ở ảnh trang"},
            {"op": "locate", "needs": 1, "modality": "text",
             "desc": "trích điều kiện hoặc ngưỡng áp dụng nêu trong văn bản"},
            {"op": "apply", "needs": 0,
             "desc": "áp điều kiện của văn bản lên giá trị đọc từ hình để suy ra kết quả"},
        ],
        "min_anchors": 2, "min_visual_anchors": 1, "min_text_anchors": 1,
    },
}
VALID_OPS = {s["op"] for m in MOTIF_CATALOG.values() for s in m["steps"]}

# Motifs whose step template structurally requires an image. Used by the runner
# to offer only these under TLV, and to refuse them outside TLV.
CROSS_MODAL_MOTIFS = frozenset(
    name for name, spec in MOTIF_CATALOG.items()
    if spec.get("min_visual_anchors", 0) > 0)


def step_modality(motif: str, index: int) -> str:
    """Declared modality of an anchored step: 'visual', 'text', or 'any'.

    Text-only motifs predate the modality field, so they report 'any' and keep
    their original behaviour (no visual anchor is required or forbidden).
    """
    spec = MOTIF_CATALOG.get(motif)
    if not spec or not 0 <= index < len(spec["steps"]):
        return "any"
    return spec["steps"][index].get("modality", "any")

# ---------------------------------------------------------------- compiler
class CompileError(ValueError):
    pass

def compile_program(motif: Any, steps: Any) -> dict[str, Any]:
    """Match motif against the closed catalog and compile a typed program.

    Raises CompileError with a machine-readable reason on any contract breach.
    """
    if not isinstance(motif, str) or motif not in MOTIF_CATALOG:
        raise CompileError(f"motif_not_in_catalog:{motif!r}")
    spec = MOTIF_CATALOG[motif]
    if not isinstance(steps, list) or len(steps) != len(spec["steps"]):
        raise CompileError(f"step_count_mismatch:expected={len(spec['steps'])}:got={len(steps) if isinstance(steps,list) else type(steps).__name__}")
    compiled = []
    for i, (got, want) in enumerate(zip(steps, spec["steps"])):
        if not isinstance(got, dict):
            raise CompileError(f"step{i}_not_object")
        op = got.get("op")
        if op != want["op"]:
            raise CompileError(f"step{i}_op_mismatch:expected={want['op']}:got={op!r}")
        anchor = got.get("anchor")
        modality = want.get("modality", "any")
        if want["needs"] > 0:
            # A text anchor must be a replayable quote, hence the original
            # 10-character floor. A visual anchor may legitimately be a
            # one-token label read from a figure (e.g. a map label "Đạo");
            # its evidential burden is enforced by bbox/result/visual checks.
            min_anchor_len = 1 if modality == "visual" else 10
            if not isinstance(anchor, str) or len(anchor.strip()) < min_anchor_len:
                raise CompileError(f"step{i}_anchor_missing_or_too_short")
        result = got.get("result")
        if not isinstance(result, str) or not result.strip():
            raise CompileError(f"step{i}_result_missing")

        # --- modality contract (v4) ---------------------------------------
        # A step whose template declares modality="visual" MUST carry a
        # visual_node (1-based image index) and MUST NOT be satisfiable by
        # prose alone; a step declaring "text" MUST NOT claim a visual_node.
        # This is what turns "multimodal" from a self-declared tag into a
        # compile-time requirement: an image-requiring motif cannot compile
        # unless the planner actually read the image.
        vnode = got.get("visual_node")
        if modality == "visual":
            if vnode is None:
                raise CompileError(f"step{i}_visual_step_missing_visual_node")
            if isinstance(vnode, bool) or not isinstance(vnode, int) or vnode < 1:
                raise CompileError(f"step{i}_visual_node_not_positive_int:{vnode!r}")
        elif modality == "text" and vnode is not None:
            raise CompileError(f"step{i}_text_step_must_not_bind_visual:{vnode!r}")

        compiled.append({"index": i, "op": op, "needs": want["needs"],
                         "modality": modality,
                         "visual_node": vnode if modality == "visual" else None,
                         "anchor": anchor.strip() if isinstance(anchor, str) else None,
                         "result": result.strip(), "desc": want["desc"]})
    n_anchor_steps = sum(1 for s in compiled if s["needs"] > 0)
    if n_anchor_steps < spec["min_anchors"]:
        raise CompileError(f"insufficient_anchor_steps:{n_anchor_steps}<{spec['min_anchors']}")

    # Cross-modal motifs must satisfy BOTH per-modality floors. Text-only
    # motifs declare neither floor, so these checks are no-ops for them and the
    # frozen v2/v3 behaviour is preserved exactly.
    n_visual = sum(1 for s in compiled if s["needs"] > 0 and s["modality"] == "visual")
    n_text = sum(1 for s in compiled if s["needs"] > 0 and s["modality"] == "text")
    min_visual = spec.get("min_visual_anchors", 0)
    min_text = spec.get("min_text_anchors", 0)
    if n_visual < min_visual:
        raise CompileError(f"insufficient_visual_anchors:{n_visual}<{min_visual}")
    if n_text < min_text:
        raise CompileError(f"insufficient_text_anchors:{n_text}<{min_text}")

    # A cross-modal program is only useful if its dependency path is explicit:
    # visual value -> textual rule/condition -> derived answer.  Record the
    # indices so downstream gates and audits can replay the path without
    # inferring it from free-form descriptions.
    visual_steps = [s["index"] for s in compiled
                    if s["needs"] > 0 and s.get("modality") == "visual"]
    text_steps = [s["index"] for s in compiled
                  if s["needs"] > 0 and s.get("modality") == "text"]
    derived_steps = [s["index"] for s in compiled if s["needs"] == 0]
    dependency_path = None
    if visual_steps and text_steps and derived_steps:
        visual_i, text_i = visual_steps[0], text_steps[0]
        derived_i = next((i for i in derived_steps if i > max(visual_i, text_i)),
                         None)
        if derived_i is not None:
            dependency_path = {
                "visual_step": visual_i,
                "text_step": text_i,
                "derived_step": derived_i,
                "closed": True,
            }

    return {"motif": motif, "steps": compiled, "n_steps": len(compiled),
            "n_anchor_steps": n_anchor_steps, "min_anchors": spec["min_anchors"],
            "n_visual_anchors": n_visual, "n_text_anchors": n_text,
            "min_visual_anchors": min_visual, "min_text_anchors": min_text,
            "is_cross_modal": min_visual > 0,
            "visual_dependency_path": dependency_path}

# ---------------------------------------------------------------- executor
def execute_program(program: dict[str, Any], evidence_text: str,
                    idf_cut: float | None = None,
                    idf_table: dict[str, float] | None = None) -> dict[str, Any]:
    """Verify each anchor step really lands on the evidence, and that the
    anchors are DISTINCT sentences. Returns an execution report."""
    sents = sentences(evidence_text)
    if not sents:
        return {"ok": False, "reason": "evidence_has_no_sentences", "matched": []}
    matched, per_step = [], []
    for s in program["steps"]:
        if s["needs"] == 0:
            per_step.append({"index": s["index"], "op": s["op"], "matched_sentence": None,
                             "match_score": None, "status": "DERIVED"})
            continue
        # A visual anchor describes what was READ OFF THE IMAGE, so by design it
        # is NOT a span of evidence_text. Scoring it against prose would mark
        # every cross-modal step UNGROUNDED (that is exactly what the text-only
        # executor did). It is verified separately by the visual verifier
        # (ecm_v3_core.verify_visual_anchors) against the declared image list,
        # and it is deliberately EXCLUDED from `matched` so it cannot be counted
        # toward distinct_anchor_sentences -- the text-side distinctness floor
        # stays as strict as before.
        if s.get("modality") == "visual":
            per_step.append({"index": s["index"], "op": s["op"], "matched_sentence": None,
                             "match_score": None, "status": "VISUAL",
                             "visual_node": s.get("visual_node")})
            continue
        aw = set(words(s["anchor"]))
        if idf_table is not None and idf_cut is not None:
            aw = {w for w in aw if idf_table.get(w, 99.0) >= idf_cut}
        best_i, best_score = -1, 0.0
        for i, sent in enumerate(sents):
            sw = set(words(sent))
            if idf_table is not None and idf_cut is not None:
                sw = {w for w in sw if idf_table.get(w, 99.0) >= idf_cut}
            if not aw:
                continue
            score = len(aw & sw) / len(aw)
            if score > best_score:
                best_i, best_score = i, score
        status = "MATCHED" if best_score >= 0.5 else "UNGROUNDED"
        if status == "MATCHED":
            matched.append(best_i)
        per_step.append({"index": s["index"], "op": s["op"], "matched_sentence": best_i,
                         "match_score": round(best_score, 3), "status": status})
    n_ungrounded = sum(1 for p in per_step if p["status"] == "UNGROUNDED")
    distinct = len(set(matched))

    # Distinctness floor applies to TEXT anchors only. A cross-modal motif has
    # one text anchor and one visual anchor, so demanding 2 distinct SENTENCES
    # would reject every cross-modal program by construction (the visual anchor
    # has no sentence). The floor is therefore the number of text-anchor steps
    # the compiled program actually declares -- for the five text-only motifs
    # that is still 2, identical to the frozen v2/v3 behaviour.
    n_text_anchor_steps = sum(
        1 for s in program["steps"]
        if s["needs"] > 0 and s.get("modality") != "visual")
    text_floor = min(program["min_anchors"], n_text_anchor_steps)

    ok = n_ungrounded == 0 and distinct >= text_floor
    reason = None
    if n_ungrounded:
        reason = f"ungrounded_steps:{n_ungrounded}"
    elif distinct < text_floor:
        reason = f"anchors_not_distinct:{distinct}<{text_floor}"
    return {"ok": ok, "reason": reason, "per_step": per_step,
            "distinct_anchor_sentences": distinct, "n_evidence_sentences": len(sents),
            "text_anchor_floor": text_floor,
            "n_visual_steps": sum(1 for p in per_step if p["status"] == "VISUAL"),
            "matched": sorted(set(matched))}

# ---------------------------------------------------------------- hard guards
MAX_VERBATIM_RUN = 7   # reject question containing >=8 consecutive source words

def guard_ecm_v2(parsed: dict[str, Any], evidence_text: str,
                 idf_cut: float | None = None,
                 idf_table: dict[str, float] | None = None) -> dict[str, Any]:
    """Full ECM-v2 gate. Returns {status, reason, program, execution}."""
    motif = parsed.get("motif")
    steps = parsed.get("steps")
    try:
        program = compile_program(motif, steps)
    except CompileError as exc:
        return {"status": "REJECTED", "reason": f"compile:{exc}", "program": None, "execution": None}
    if program["n_steps"] < 2:
        return {"status": "REJECTED", "reason": "min_steps:<2", "program": program, "execution": None}
    execution = execute_program(program, evidence_text, idf_cut, idf_table)
    if not execution["ok"]:
        return {"status": "REJECTED", "reason": f"exec:{execution['reason']}",
                "program": program, "execution": execution}
    run = longest_common_run(words(parsed.get("question", "")), evidence_text)
    if run > MAX_VERBATIM_RUN:
        return {"status": "REJECTED", "reason": f"anti_verbatim:run={run}>{MAX_VERBATIM_RUN}",
                "program": program, "execution": execution, "verbatim_run": run}
    return {"status": "PARSED", "reason": None, "program": program,
            "execution": execution, "verbatim_run": run}

def build_idf(corpus_texts: list[str], drop_frac: float = 0.40) -> tuple[dict[str, float], float]:
    all_s = []
    for t in corpus_texts:
        all_s.extend(sentences(t))
    N = len(all_s)
    df = Counter()
    for s in all_s:
        for w in set(words(s)):
            df[w] += 1
    table = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
    vals = sorted(table.values())
    cut = vals[int(len(vals) * drop_frac)] if vals else 0.0
    return table, cut
