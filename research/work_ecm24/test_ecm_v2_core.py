"""Self-test for ecm_v2_core: asserts BOTH the accept path and every reject path.

A green run proves the INSTRUMENT (compiler/executor/guards), not the method.
Run: python3 test_ecm_v2_core.py
"""
from __future__ import annotations
import json, sys
from ecm_v2_core import (
    MOTIF_CATALOG, VALID_OPS, CROSS_MODAL_MOTIFS,
    compile_program, execute_program, guard_ecm_v2,
    CompileError, build_idf, longest_common_run, words, sentences,
)

# ---------------------------------------------------------------- fixture
# Two clearly distinct sentences so a 2-anchor motif can ground on both.
EVIDENCE = (
    "Quốc hội là cơ quan đại biểu cao nhất của Nhân dân, thực hiện quyền lập hiến và quyền lập pháp. "
    "Uỷ ban Thường vụ Quốc hội là cơ quan thường trực của Quốc hội, ban hành pháp lệnh theo thẩm quyền. "
    "Chính phủ là cơ quan hành chính nhà nước cao nhất, thực hiện quyền hành pháp và tổ chức thi hành pháp luật. "
    "Thời hiệu khởi kiện được tính là hai năm kể từ ngày phát sinh tranh chấp dân sự tại toà án. "
    "Trường hợp bất khả kháng thì thời gian xảy ra sự kiện đó không tính vào thời hiệu khởi kiện."
)

def good_steps():
    return [
        {"op": "locate", "needs": 1,
         "anchor": "Quốc hội là cơ quan đại biểu cao nhất của Nhân dân thực hiện quyền lập hiến",
         "result": "Quốc hội giữ quyền lập hiến và lập pháp"},
        {"op": "locate", "needs": 1,
         "anchor": "Uỷ ban Thường vụ Quốc hội là cơ quan thường trực ban hành pháp lệnh",
         "result": "UBTVQH là cơ quan thường trực, ban hành pháp lệnh"},
        {"op": "compare", "needs": 0, "anchor": None,
         "result": "Khác biệt: lập hiến/lập pháp so với pháp lệnh thường trực"},
    ]

RESULTS: list[dict] = []

def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append({"case": name, "pass": bool(cond), "detail": detail})
    print(("  PASS  " if cond else "  FAIL  ") + name + ((" :: " + detail) if detail else ""))

IDF_TABLE, IDF_CUT = build_idf([EVIDENCE], drop_frac=0.40)

# ================================================================ 1. catalog sanity
print("\n[1] catalog integrity")
# v4 added two cross-modal motifs (diagram_text_reconcile,
# figure_condition_apply) whose templates structurally require an image, so the
# catalog is 5 text-only + 2 cross-modal = 7. Asserting the split rather than a
# bare total keeps this test meaningful if either family grows again.
check("catalog_has_7_motifs", len(MOTIF_CATALOG) == 7, f"n={len(MOTIF_CATALOG)}")
check("catalog_has_2_cross_modal", len(CROSS_MODAL_MOTIFS) == 2,
      str(sorted(CROSS_MODAL_MOTIFS)))
check("catalog_has_5_text_only",
      len(set(MOTIF_CATALOG) - CROSS_MODAL_MOTIFS) == 5)
check("every_motif_min_2_anchor_steps",
      all(sum(1 for s in m["steps"] if s["needs"] > 0) >= 2 for m in MOTIF_CATALOG.values()))
check("every_motif_min_3_steps", all(len(m["steps"]) >= 3 for m in MOTIF_CATALOG.values()))
# VALID_OPS is DERIVED from the catalog, so hard-coding the exact set made this
# assert break whenever a motif was edited (the v3 hierarchy_traverse fix
# dropped `identify`, leaving 6 ops, and this test failed even though every
# guard still worked). Assert the invariant instead: every op the catalog uses
# must come from the closed vocabulary, and no op may be free-form.
# v4: `reconcile` / `apply` are the cross-modal join ops -- the derived step that
# consumes one visual anchor and one text anchor.
_OP_VOCAB = {"locate", "compare", "chain", "subtract", "traverse", "identify",
             "compute", "reconcile", "apply"}
check("ops_are_typed", VALID_OPS <= _OP_VOCAB and len(VALID_OPS) >= 5,
      str(sorted(VALID_OPS)))
check("every_step_op_in_vocab",
      all(s["op"] in _OP_VOCAB for m in MOTIF_CATALOG.values() for s in m["steps"]))

# ================================================================ 2. accept path
print("\n[2] accept path (green)")
prog = compile_program("compare_two_entities", good_steps())
check("compiles", prog["n_steps"] == 3 and prog["n_anchor_steps"] == 2)
ex = execute_program(prog, EVIDENCE, IDF_CUT, IDF_TABLE)
check("executes_ok", ex["ok"], json.dumps(ex.get("reason")))
check("distinct_anchors_ge_2", ex["distinct_anchor_sentences"] >= 2,
      f"distinct={ex['distinct_anchor_sentences']}")
ok_parsed = {"motif": "compare_two_entities", "steps": good_steps(),
             "question": "Điểm khác biệt cơ bản nào phân định vai trò của hai thiết chế nêu trên?",
             "choices": ["a", "b", "c", "d"], "answer_index": 0, "rationale": "r"}
g = guard_ecm_v2(ok_parsed, EVIDENCE, IDF_CUT, IDF_TABLE)
check("guard_accepts_good_record", g["status"] == "PARSED", str(g["reason"]))

# ================================================================ 3. reject paths
print("\n[3] reject paths (guard MUST block)")

# 3.1 motif outside the closed catalog  -> this is what v1 allowed
bad = dict(ok_parsed, motif="tra cứu thông tin cơ bản")
r = guard_ecm_v2(bad, EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_free_form_motif", r["status"] == "REJECTED" and "motif_not_in_catalog" in r["reason"],
      str(r["reason"]))

# 3.2 wrong step count
bad = dict(ok_parsed, steps=good_steps()[:2])
r = guard_ecm_v2(bad, EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_step_count_mismatch", r["status"] == "REJECTED" and "step_count_mismatch" in r["reason"],
      str(r["reason"]))

# 3.3 op sequence violated
st = good_steps(); st[1]["op"] = "compare"
r = guard_ecm_v2(dict(ok_parsed, steps=st), EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_op_mismatch", r["status"] == "REJECTED" and "op_mismatch" in r["reason"], str(r["reason"]))

# 3.4 anchor missing on a step that needs one
st = good_steps(); st[1]["anchor"] = "ngắn"
r = guard_ecm_v2(dict(ok_parsed, steps=st), EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_missing_anchor", r["status"] == "REJECTED" and "anchor_missing" in r["reason"], str(r["reason"]))

# 3.5 BOTH anchors point at the SAME sentence -> single-hop disguised as 2-hop
st = good_steps()
st[1]["anchor"] = "Quốc hội là cơ quan đại biểu cao nhất của Nhân dân thực hiện quyền lập pháp"
r = guard_ecm_v2(dict(ok_parsed, steps=st), EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_anchors_not_distinct",
      r["status"] == "REJECTED" and "anchors_not_distinct" in str(r["reason"]), str(r["reason"]))

# 3.6 anchor not present in evidence at all
st = good_steps()
st[1]["anchor"] = "Toà án Hiến pháp Liên bang có thẩm quyền giải thích điều ước quốc tế đa phương"
r = guard_ecm_v2(dict(ok_parsed, steps=st), EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_ungrounded_anchor",
      r["status"] == "REJECTED" and "ungrounded" in str(r["reason"]), str(r["reason"]))

# 3.7 anti-verbatim: question copies >=8 consecutive source words
copied = "Uỷ ban Thường vụ Quốc hội là cơ quan thường trực của Quốc hội là gì?"
r = guard_ecm_v2(dict(ok_parsed, question=copied), EVIDENCE, IDF_CUT, IDF_TABLE)
check("reject_verbatim_question",
      r["status"] == "REJECTED" and "anti_verbatim" in str(r["reason"]), str(r["reason"]))

# ================================================================ 4. util correctness
print("\n[4] utility correctness (hand-computed)")
check("longest_common_run_detects_copy",
      longest_common_run(words(copied), EVIDENCE) >= 8,
      f"run={longest_common_run(words(copied), EVIDENCE)}")
check("longest_common_run_zero_on_novel",
      longest_common_run(words("Con mèo nhảy qua hàng rào gỗ màu xanh"), EVIDENCE) <= 2,
      f"run={longest_common_run(words('Con mèo nhảy qua hàng rào gỗ màu xanh'), EVIDENCE)}")
check("sentences_splits_fixture", len(sentences(EVIDENCE)) == 5, f"n={len(sentences(EVIDENCE))}")

# ================================================================ 5. all motifs compile
print("\n[5] every catalog motif compiles from its own template")
for mname, spec in MOTIF_CATALOG.items():
    st = []
    for s in spec["steps"]:
        step = {"op": s["op"], "needs": s["needs"],
                "anchor": ("x" * 15) if s["needs"] > 0 else None,
                "result": "r"}
        # v4: a step whose template declares modality="visual" must carry a
        # visual_node (the compiler's modality contract) and a localising bbox.
        # Synthesising text-only steps for every motif made this loop assert the
        # OPPOSITE of the contract: cross-modal motifs correctly refused to
        # compile, and the test read that as a failure.
        if s.get("modality") == "visual":
            step["visual_node"] = 1
            step["bbox"] = [0.10, 0.20, 0.40, 0.50]
        st.append(step)
    try:
        p = compile_program(mname, st)
        check(f"compiles::{mname}", p["n_steps"] == len(spec["steps"]))
    except CompileError as exc:
        check(f"compiles::{mname}", False, str(exc))

# v4: the contract must also FAIL closed -- a cross-modal motif offered
# text-only steps has to be rejected, or "multimodal" is unenforced.
print("\n[5b] cross-modal motifs refuse text-only steps")
for mname in sorted(CROSS_MODAL_MOTIFS):
    spec = MOTIF_CATALOG[mname]
    st = [{"op": s["op"], "needs": s["needs"],
           "anchor": ("x" * 15) if s["needs"] > 0 else None, "result": "r"}
          for s in spec["steps"]]
    try:
        compile_program(mname, st)
        check(f"refuses_text_only::{mname}", False, "compiled without visual")
    except CompileError as exc:
        check(f"refuses_text_only::{mname}",
              "visual_step_missing_visual_node" in str(exc), str(exc))

# ================================================================ summary
npass = sum(1 for r in RESULTS if r["pass"])
ntot = len(RESULTS)
print(f"\n{'='*60}\nSELF-TEST: {npass}/{ntot} passed")
print("Scope: proves the deterministic instrument only, NOT question quality.")
with open("audit/ecm_v2_selftest.json", "w", encoding="utf-8") as fh:
    json.dump({"passed": npass, "total": ntot, "cases": RESULTS,
               "scope": "instrument_only_not_method_evidence"}, fh, ensure_ascii=False, indent=2)
print("saved audit/ecm_v2_selftest.json")
sys.exit(0 if npass == ntot else 1)
