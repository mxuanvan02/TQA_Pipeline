#!/usr/bin/env python3
"""Self-test for ecm_v3_core: the three manuscript-mandated mechanisms.

Scope: proves the DETERMINISTIC INSTRUMENT only (trace, sealed realizer,
visual binding). It says nothing about question quality, which requires
blinded judging.

Design rule: every mechanism is tested on BOTH paths --
  * one green case that must be accepted, and
  * explicit negative cases that must be REJECTED with the right reason.
A guard that only passes green cases proves nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import ecm_v3_core as v3  # noqa: E402

PASS = 0
FAIL = 0
RESULTS: list[dict] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" :: {detail}" if detail else ""))
    RESULTS.append({"name": name, "ok": bool(cond), "detail": detail})


# --------------------------------------------------------------- fixtures
# NOTE ON FIXTURE SIZE (do not shrink):
# check_sealed_payload() enforces excerpt_leak_ratio <= 0.35, i.e. the anchors
# handed to the realizer must be a small slice of the chunk. That threshold is
# calibrated against REAL chunks, which run 195-669 words (median 638). A toy
# 5-sentence fixture makes the denominator artificially tiny (two ~20-word
# anchors -> ratio 0.42) and would fail a guard that is correct. The fixture
# below is therefore sized to the real corpus (~20 sentences) so the ratio test
# exercises the same regime as production. Lowering the threshold to make a
# small fixture pass would be tuning the instrument to the test.
EVIDENCE = (
    "Điều 15 quy định thời hiệu khởi kiện vụ án dân sự là hai năm kể từ ngày "
    "quyền lợi hợp pháp bị xâm phạm. "
    "Trường hợp bất khả kháng hoặc trở ngại khách quan thì thời gian đó không "
    "tính vào thời hiệu khởi kiện. "
    "Cơ quan tiến hành tố tụng có trách nhiệm giải thích quyền và nghĩa vụ cho "
    "đương sự trước phiên tòa sơ thẩm. "
    "Bản án sơ thẩm chưa có hiệu lực pháp luật ngay mà phải chờ hết thời hạn "
    "kháng cáo mười lăm ngày. "
    "Viện kiểm sát cùng cấp có quyền kháng nghị bản án theo thủ tục phúc thẩm "
    "trong thời hạn luật định. "
    "Đương sự có nghĩa vụ cung cấp chứng cứ và chứng minh cho yêu cầu của mình "
    "là có căn cứ và hợp pháp. "
    "Tòa án chỉ thu thập chứng cứ trong những trường hợp do bộ luật này quy "
    "định khi đương sự không thể tự mình thực hiện. "
    "Người làm chứng phải khai báo trung thực những tình tiết mà mình biết "
    "liên quan đến vụ việc dân sự đang giải quyết. "
    "Phiên tòa phúc thẩm chỉ xem xét lại phần bản án bị kháng cáo hoặc bị "
    "kháng nghị theo quy định của pháp luật tố tụng. "
    "Quyết định công nhận sự thỏa thuận của các đương sự có hiệu lực thi hành "
    "ngay và không bị kháng cáo theo thủ tục phúc thẩm. "
    "Thẩm phán phải từ chối tiến hành tố tụng nếu có căn cứ cho rằng họ không "
    "vô tư khi làm nhiệm vụ được giao. "
    "Chi phí giám định do bên yêu cầu giám định nộp tạm ứng trừ trường hợp các "
    "bên có thỏa thuận khác về nghĩa vụ này. "
    "Bản án đã có hiệu lực pháp luật có thể bị kháng nghị theo thủ tục giám "
    "đốc thẩm khi phát hiện vi phạm nghiêm trọng thủ tục. "
    "Thời hạn kháng nghị giám đốc thẩm là ba năm kể từ ngày bản án của tòa án "
    "có hiệu lực pháp luật thi hành. "
    "Tòa án cấp sơ thẩm phải gửi hồ sơ vụ án cùng kháng cáo cho tòa án cấp "
    "phúc thẩm trong thời hạn luật định. "
    "Việc hòa giải được tiến hành theo nguyên tắc tôn trọng sự tự nguyện thỏa "
    "thuận của các đương sự tham gia tố tụng. "
    "Người bảo vệ quyền lợi hợp pháp của đương sự được tham gia phiên tòa và "
    "trình bày ý kiến tranh luận trước hội đồng. "
    "Án phí sơ thẩm do bên có yêu cầu không được chấp nhận chịu theo mức mà "
    "pháp luật về án phí đã ấn định."
)

IDF_TABLE, IDF_CUT = v3.build_idf([EVIDENCE])


def good_plan(visual: bool = False) -> dict:
    """A plan that should pass stage-1: two distinct grounded anchors."""
    s0: dict = {"op": "locate",
          "anchor": "thời hiệu khởi kiện vụ án dân sự là hai năm kể từ ngày quyền lợi hợp pháp bị xâm phạm",
          "result": "thời hiệu khởi kiện là hai năm"}
    s1 = {"op": "locate",
          "anchor": "Bản án sơ thẩm chưa có hiệu lực pháp luật ngay mà phải chờ hết thời hạn kháng cáo mười lăm ngày",
          "result": "thời hạn kháng cáo mười lăm ngày"}
    if visual:
        s1["visual_node"] = 1
    return {"motif": "numeric_derive",
            "steps": [s0, s1,
                      {"op": "compute", "result": "chênh lệch giữa hai mốc thời gian"}]}


print("[1] stage-1 green path (must be accepted)")
plan = v3.guard_plan_v3(good_plan(), EVIDENCE, "T", None, IDF_CUT, IDF_TABLE)
check("plan_ok", plan["status"] == "PLAN_OK", str(plan.get("reason")))
check("trace_built", plan.get("trace") is not None)
if plan.get("trace"):
    check("n_atoms_eq_3", plan["trace"]["n_atoms"] == 3,
          f'n={plan["trace"]["n_atoms"]}')
    check("two_text_grounded", plan["trace"]["n_text_grounded"] == 2,
          f'n={plan["trace"]["n_text_grounded"]}')
    check("trace_hash_present", len(plan["trace"]["trace_hash"]) == 16)

print("\n[2] trace Gamma: spans must slice back to real source characters")
tr = plan["trace"]
ver = v3.verify_trace(tr, EVIDENCE)
check("verify_trace_ok", ver["ok"], str(ver["problems"]))
# hand-check: each text gamma slices to the recorded excerpt
sliced_ok = True
for g in tr["gamma"]:
    if g["kind"] != "text":
        continue
    a, b = g["char_span"]
    if v3.norm(EVIDENCE[a:b])[:60] != v3.norm(g["source_excerpt"])[:60]:
        sliced_ok = False
check("char_span_slices_to_source", sliced_ok)
check("derived_atom_links_to_anchors",
      any(g["kind"] == "derived" and g["derived_from"] for g in tr["gamma"]))

# determinism: same input -> same trace hash
tr2 = v3.build_trace(plan["program"], plan["execution"], EVIDENCE)
check("trace_hash_deterministic", tr2["trace_hash"] == tr["trace_hash"],
      tr["trace_hash"])

print("\n[3] trace: corrupted spans MUST be rejected")
bad = json.loads(json.dumps(tr))
bad["gamma"][0]["char_span"] = [999999, 1000000]
check("reject_span_out_of_range",
      not v3.verify_trace(bad, EVIDENCE)["ok"],
      str(v3.verify_trace(bad, EVIDENCE)["problems"][:1]))

bad2 = json.loads(json.dumps(tr))
bad2["gamma"][0]["source_excerpt"] = "văn bản hoàn toàn không có trong bằng chứng gốc"
check("reject_span_text_mismatch",
      not v3.verify_trace(bad2, EVIDENCE)["ok"],
      str(v3.verify_trace(bad2, EVIDENCE)["problems"][:1]))

bad3 = json.loads(json.dumps(tr))
bad3["gamma"].pop()
check("reject_atom_gamma_mismatch",
      not v3.verify_trace(bad3, EVIDENCE)["ok"],
      str(v3.verify_trace(bad3, EVIDENCE)["problems"][:1]))

print("\n[4] sealed realizer: payload must NOT carry the full chunk")
sealed = v3.seal_construction(plan["program"], tr, "T", None)
blind = v3.assert_realizer_blind(sealed, EVIDENCE)
check("sealed_is_blind", blind["ok"], str(blind["problems"]))
check("sealed_has_no_evidence_key", "evidence" not in sealed and "text" not in sealed)
check("sealed_keys_allowed",
      set(sealed) <= v3._REALIZER_ALLOWED_KEYS, str(sorted(sealed)))
check("leak_ratio_below_035", blind["leak_ratio"] <= v3.MAX_LEAK_RATIO,
      f'ratio={blind["leak_ratio"]}')
check("sealed_carries_atoms", len(sealed["atoms"]) == 3)

# --- INVARIANT: the sealer must never emit a payload its own guard refuses.
# The v3 pilot lost 2/9 rows to seal:a0:excerpt_too_long because the sealer
# truncated by characters (200) while the guard measured words (40). Assert the
# round-trip property directly, and stress it with an absurdly long anchor.
check("sealer_output_passes_own_guard", blind["ok"], str(blind["problems"]))
check("every_sealed_excerpt_within_budget",
      all(len(v3.words(a["excerpt"])) <= v3.MAX_EXCERPT_WORDS
          for a in sealed["anchors"]),
      str([len(v3.words(a["excerpt"])) for a in sealed["anchors"]]))

_stress = json.loads(json.dumps(tr))
for _g in _stress["gamma"]:
    if _g.get("kind") == "text":
        _g["source_excerpt"] = " ".join(["dai"] * 500)
_sealed_stress = v3.seal_construction(plan["program"], _stress, "T", None)
_blind_stress = v3.assert_realizer_blind(_sealed_stress, EVIDENCE)
check("sealer_truncates_absurd_anchor",
      all(len(v3.words(a["excerpt"])) <= v3.MAX_EXCERPT_WORDS
          for a in _sealed_stress["anchors"]),
      str([len(v3.words(a["excerpt"])) for a in _sealed_stress["anchors"]]))
check("stress_sealed_no_excerpt_too_long",
      not any("excerpt_too_long" in p for p in _blind_stress["problems"]),
      str(_blind_stress["problems"]))

# --- REGRESSION: full_v3 hit seal:excerpt_leak_ratio:0.405>0.35 on the
# corpus's shortest chunk (195 words, 2 anchors x 40 words = 79 = 40.5%).
# excerpt_word_budget() must shrink the per-anchor cap so 2 anchors at the
# cap can never exceed MAX_LEAK_RATIO of a short chunk, and seal_construction
# must actually apply that shrunk cap when evidence_text is passed through.
_short_evidence = " ".join(["từ"] * 195)
_budget = v3.excerpt_word_budget(_short_evidence, n_anchors=2)
check("excerpt_budget_shrinks_on_short_chunk", _budget < v3.MAX_EXCERPT_WORDS,
      f"budget={_budget}")
check("excerpt_budget_keeps_ratio_safe",
      (2 * _budget) / len(v3.words(_short_evidence)) <= v3.MAX_LEAK_RATIO,
      f"budget={_budget} ratio={(2 * _budget) / len(v3.words(_short_evidence)):.3f}")

_sealed_short = v3.seal_construction(plan["program"], tr, "T", None,
                                     evidence_text=_short_evidence)
_cap_short = v3.excerpt_word_budget(_short_evidence, len(_sealed_short["anchors"]))
_blind_short = v3.assert_realizer_blind(_sealed_short, _short_evidence,
                                        max_excerpt_words=_cap_short)
check("seal_construction_applies_shrunk_budget_on_short_chunk", _blind_short["ok"],
      str(_blind_short["problems"]))

print("\n[5] sealed realizer: leaks MUST be rejected")
leaky = json.loads(json.dumps(sealed))
leaky["evidence_text"] = EVIDENCE
check("reject_unexpected_key",
      not v3.assert_realizer_blind(leaky, EVIDENCE)["ok"],
      str(v3.assert_realizer_blind(leaky, EVIDENCE)["problems"][:1]))

leaky2 = json.loads(json.dumps(sealed))
leaky2["anchors"] = [{"atom_id": "a0", "excerpt": EVIDENCE, "sentence_index": 0}]
r2 = v3.assert_realizer_blind(leaky2, EVIDENCE)
check("reject_full_chunk_in_excerpt", not r2["ok"], str(r2["problems"][:1]))

print("\n[6] TLV visual binding")
prog_novis = v3.guard_plan_v3(good_plan(), EVIDENCE, "T", None,
                              IDF_CUT, IDF_TABLE)["program"]
# TLV with no binding -> reject
r = v3.require_visual_binding(prog_novis, "TLV", ["img1"])
check("reject_TLV_without_binding", not r["ok"], str(r["reason"]))
# TLV with no images in package -> reject
r = v3.require_visual_binding(prog_novis, "TLV", [])
check("reject_TLV_package_without_images", not r["ok"], str(r["reason"]))
# TLV with a valid binding -> accept
plan_vis = v3.guard_plan_v3(good_plan(visual=True), EVIDENCE, "TLV", ["img1"],
                            IDF_CUT, IDF_TABLE)
check("accept_TLV_with_binding", plan_vis["status"] == "PLAN_OK",
      str(plan_vis.get("reason")))
check("TLV_binding_counted", plan_vis.get("visual", {}).get("n_bound") == 1)
# visual node out of range -> reject
bad_vis = good_plan(visual=True)
bad_vis["steps"][1]["visual_node"] = 5
r = v3.guard_plan_v3(bad_vis, EVIDENCE, "TLV", ["img1"], IDF_CUT, IDF_TABLE)
check("reject_visual_node_out_of_range", r["status"] == "REJECTED",
      str(r.get("reason")))
# binding declared under non-TLV -> reject
r = v3.guard_plan_v3(good_plan(visual=True), EVIDENCE, "T", None,
                     IDF_CUT, IDF_TABLE)
check("reject_visual_binding_outside_TLV", r["status"] == "REJECTED",
      str(r.get("reason")))

print("\n[7] stage-1 inherits every v2 rejection path")
neg = [
    ("free_form_motif",
     {"motif": "tra cứu thông tin", "steps": good_plan()["steps"]}),
    ("step_count_mismatch",
     {"motif": "numeric_derive", "steps": good_plan()["steps"][:2]}),
    ("op_mismatch",
     {"motif": "numeric_derive",
      "steps": [{**good_plan()["steps"][0], "op": "compare"},
                good_plan()["steps"][1], good_plan()["steps"][2]]}),
    ("anchor_too_short",
     {"motif": "numeric_derive",
      "steps": [{**good_plan()["steps"][0], "anchor": "ngắn"},
                good_plan()["steps"][1], good_plan()["steps"][2]]}),
    ("ungrounded_anchor",
     {"motif": "numeric_derive",
      "steps": [{**good_plan()["steps"][0],
                 "anchor": "quy định về hợp đồng thương mại quốc tế và trọng tài"},
                good_plan()["steps"][1], good_plan()["steps"][2]]}),
]
for name, p in neg:
    r = v3.guard_plan_v3(p, EVIDENCE, "T", None, IDF_CUT, IDF_TABLE)
    check(f"reject_{name}", r["status"] == "REJECTED", str(r.get("reason"))[:60])

# anchors not distinct: both anchors point at the same sentence
same = {"motif": "numeric_derive",
        "steps": [good_plan()["steps"][0],
                  {"op": "locate",
                   "anchor": "thời hiệu khởi kiện vụ án dân sự là hai năm",
                   "result": "hai năm"},
                  good_plan()["steps"][2]]}
r = v3.guard_plan_v3(same, EVIDENCE, "T", None, IDF_CUT, IDF_TABLE)
check("reject_anchors_not_distinct", r["status"] == "REJECTED",
      str(r.get("reason"))[:60])

print("\n[8] stage-2 realizer gate (check set C)")
good_mcq = {"question": "Khoảng cách giữa hai mốc thời gian nêu trên là bao lâu?",
            "choices": ["Hai năm và mười lăm ngày", "Một năm", "Ba tháng", "Mười ngày"],
            "answer_index": 0}
r = v3.guard_realized_v3(good_mcq, sealed, EVIDENCE)
check("accept_valid_mcq", r["status"] == "PARSED", str(r.get("reason")))

c_neg = [
    ("missing_question", {**good_mcq, "question": ""}),
    ("only_three_choices", {**good_mcq, "choices": ["a", "b", "c"]}),
    ("duplicate_choices",
     {**good_mcq, "choices": ["Hai năm", "Hai năm", "Ba tháng", "Mười ngày"]}),
    ("answer_index_out_of_range", {**good_mcq, "answer_index": 7}),
    ("answer_index_bool", {**good_mcq, "answer_index": True}),
]
for name, m in c_neg:
    r = v3.guard_realized_v3(m, sealed, EVIDENCE)
    check(f"reject_{name}", r["status"] == "REJECTED", str(r.get("reason"))[:50])

# anti-verbatim at stage 2
copied = {**good_mcq,
          "question": "Điều 15 quy định thời hiệu khởi kiện vụ án dân sự là hai năm kể từ ngày nào?"}
r = v3.guard_realized_v3(copied, sealed, EVIDENCE)
check("reject_verbatim_question", r["status"] == "REJECTED",
      str(r.get("reason"))[:50])

print("\n[9] every catalog motif still compiles under v3")
for motif, spec in v3.MOTIF_CATALOG.items():
    steps = []
    for st in spec["steps"]:
        s = {"op": st["op"], "result": f"kết quả {st['op']}"}
        if st["needs"] > 0:
            s["anchor"] = "văn bản neo đủ dài để vượt ngưỡng mười ký tự"
        # v4 modality contract: visual steps need a node + a localising bbox.
        if st.get("modality") == "visual":
            s["visual_node"] = 1
            s["bbox"] = [0.10, 0.20, 0.40, 0.50]
        steps.append(s)
    try:
        prog = v3.compile_program(motif, steps)
        check(f"compiles::{motif}", prog["motif"] == motif)
    except v3.CompileError as exc:
        check(f"compiles::{motif}", False, str(exc))

print("\n" + "=" * 60)
print(f"SELF-TEST v3: {PASS}/{PASS + FAIL} passed")
print("Scope: deterministic instrument only, NOT question quality.")
out = ROOT / "audit" / "ecm_v3_selftest.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(
    {"passed": PASS, "failed": FAIL, "results": RESULTS},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved {out.relative_to(ROOT)}")
sys.exit(1 if FAIL else 0)
