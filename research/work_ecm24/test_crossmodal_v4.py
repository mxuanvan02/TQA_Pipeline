
import sys, json
sys.path.insert(0, ".")
import ecm_v2_core as v2
import ecm_v3_core as core

EV = ("Bộ máy nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam bao gồm Quốc hội, Chủ tịch nước, "
      "Chính phủ, Tòa án nhân dân và Viện kiểm sát nhân dân. "
      "Quốc hội là cơ quan quyền lực nhà nước cao nhất của nước Cộng hòa xã hội chủ nghĩa Việt Nam. "
      "Chính phủ là cơ quan hành chính nhà nước cao nhất, thực hiện quyền hành pháp theo quy định. "
      "Tòa án nhân dân thực hiện quyền tư pháp và xét xử độc lập chỉ tuân theo pháp luật.")
IMGS = [{"path": "p.jpg", "declared_order": 1}]

R = []
def chk(name, cond, extra=""):
    R.append((name, bool(cond), extra))

# ---- 1. cross-modal motifs registered
chk("catalog has 7 motifs", len(v2.MOTIF_CATALOG) == 7, str(len(v2.MOTIF_CATALOG)))
chk("CROSS_MODAL_MOTIFS = 2", len(core.CROSS_MODAL_MOTIFS) == 2, str(sorted(core.CROSS_MODAL_MOTIFS)))

# ---- 2. HAPPY PATH: valid cross-modal plan under TLV
good = {"motif": "diagram_text_reconcile", "steps": [
    {"op":"locate","visual_node":1,"bbox":[0.12,0.30,0.55,0.52],
     "anchor":"khối Quốc hội nằm trên khối Chính phủ trong sơ đồ tổ chức",
     "result":"Sơ đồ đặt Quốc hội ở cấp cao hơn Chính phủ"},
    {"op":"locate","anchor":"Quốc hội là cơ quan quyền lực nhà nước cao nhất",
     "result":"Văn bản xác định Quốc hội là cơ quan quyền lực cao nhất"},
    {"op":"reconcile","result":"Vị trí trên sơ đồ khớp với quy định trong văn bản về thứ bậc"}]}
p = core.guard_plan_v3(good, EV, "TLV", declared_images=IMGS, require_cross_modal=True)
chk("TLV valid cross-modal -> PLAN_OK", p["status"]=="PLAN_OK", p.get("reason") or "")
if p["status"]=="PLAN_OK":
    chk("program.is_cross_modal", p["program"]["is_cross_modal"])
    chk("n_visual_anchors==1", p["program"]["n_visual_anchors"]==1)
    chk("n_text_anchors==1", p["program"]["n_text_anchors"]==1)
    kinds=[g["kind"] for g in p["trace"]["gamma"]]
    chk("gamma kinds visual+text+derived", kinds==["visual","text","derived"], str(kinds))
    chk("visual verifier n_checked==1", p["visual_anchor_verify"]["n_checked"]==1)
    s = core.seal_construction(p["program"], p["trace"], "TLV", ["img1"], evidence_text=EV)
    chk("sealed has visual_observations", len(s["visual_observations"])==1)
    b = core.assert_realizer_blind(s, EV, max_excerpt_words=core.excerpt_word_budget(EV,1))
    chk("realizer blind ok (obs not counted)", b["ok"], str(b["problems"]))

def rej(name, plan, want):
    ok = plan["status"]=="REJECTED" and want in (plan.get("reason") or "")
    chk(name, ok, plan.get("reason") or plan["status"])

# ---- 3. loophole: no bbox
import copy
d=copy.deepcopy(good); del d["steps"][0]["bbox"]
rej("no bbox -> rejected", core.guard_plan_v3(d,EV,"TLV",declared_images=IMGS,require_cross_modal=True), "bbox_missing")

# ---- 4. loophole: whole-page bbox
d=copy.deepcopy(good); d["steps"][0]["bbox"]=[0.0,0.0,1.0,1.0]
rej("whole-page bbox -> rejected", core.guard_plan_v3(d,EV,"TLV",declared_images=IMGS,require_cross_modal=True), "covers_whole_page")

# ---- 5. loophole: placeholder observation
d=copy.deepcopy(good); d["steps"][0]["result"]="Theo hình"
rej("placeholder obs -> rejected", core.guard_plan_v3(d,EV,"TLV",declared_images=IMGS,require_cross_modal=True), "observation")

# ---- 6. loophole: no visual_node on visual step
d=copy.deepcopy(good); del d["steps"][0]["visual_node"]
rej("visual step w/o visual_node -> compile fail", core.guard_plan_v3(d,EV,"TLV",declared_images=IMGS,require_cross_modal=True), "visual_step_missing_visual_node")

# ---- 7. loophole: text-only motif under TLV (the OLD degenerate case)
textonly = {"motif":"compare_two_entities","steps":[
    {"op":"locate","anchor":"Quốc hội là cơ quan quyền lực nhà nước cao nhất","result":"Quốc hội: quyền lực cao nhất"},
    {"op":"locate","anchor":"Chính phủ là cơ quan hành chính nhà nước cao nhất","result":"Chính phủ: hành chính cao nhất"},
    {"op":"compare","result":"Khác nhau về loại quyền lực"}]}
rej("text-only motif under TLV -> rejected", core.guard_plan_v3(textonly,EV,"TLV",declared_images=IMGS,require_cross_modal=True), "not_cross_modal")

# ---- 8. cross-modal motif OUTSIDE TLV must be refused
rej("cross-modal outside TLV -> rejected", core.guard_plan_v3(good,EV,"TL_struct",declared_images=None,require_cross_modal=False), "outside_TLV")

# ---- 9. REGRESSION: text-only motif under T still works exactly as before
pt = core.guard_plan_v3(textonly, EV, "T", declared_images=None)
chk("REGRESSION text-only under T -> PLAN_OK", pt["status"]=="PLAN_OK", pt.get("reason") or "")
if pt["status"]=="PLAN_OK":
    chk("REGRESSION 2 distinct text anchors", pt["execution"]["distinct_anchor_sentences"]==2)
    chk("REGRESSION visual verifier no-op", pt["visual_anchor_verify"]["n_checked"]==0)

# ---- 10. visual_node out of range
d=copy.deepcopy(good); d["steps"][0]["visual_node"]=5
rej("visual_node out of range -> rejected", core.guard_plan_v3(d,EV,"TLV",declared_images=IMGS,require_cross_modal=True), "out_of_range")

# ---- 11. prompt only offers cross-modal motifs under TLV
import run_ecm_v3 as R3
tlv_cat = R3.catalog_spec_text("TLV"); t_cat = R3.catalog_spec_text("T")
chk("TLV prompt offers only cross-modal", "diagram_text_reconcile" in tlv_cat and "compare_two_entities" not in tlv_cat)
chk("T prompt offers only text-only", "compare_two_entities" in t_cat and "diagram_text_reconcile" not in t_cat)
pr,_ = R3.planner_prompt({"text":EV,"images":IMGS}, "TLV")
chk("TLV prompt demands bbox", "bbox" in pr)

npass=sum(1 for _,o,_ in R if o)
print(f"\n=== {npass}/{len(R)} PASS ===\n")
for n,o,e in R:
    print(("  PASS  " if o else "  FAIL  ")+n+(f"   [{e}]" if e and not o else ""))
sys.exit(0 if npass==len(R) else 1)
