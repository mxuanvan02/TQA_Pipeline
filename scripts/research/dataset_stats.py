#!/usr/bin/env python3
"""
dataset_stats.py -- Comprehensive dataset analysis for paper Section 4.
=======================================================================
Who:    Run after Stage 4+5 complete.
Where:  Root of TQA_Pipeline project.
How:    python scripts/research/dataset_stats.py
Output:
  - research/artifacts/dataset_stats.json     (machine-readable)
  - research/artifacts/dataset_stats.tex      (LaTeX table snippets -> paste into paper)
  - research/artifacts/bloom_dist.csv         (for figure generation)
  - research/artifacts/domain_dist.csv        (for figure generation)

Paper usage:
  In main.tex preamble: \\input{research/artifacts/dataset_stats.tex}
"""


from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ─── Paths (auto-detect data layout) ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
_candidate_bases = [ROOT / "data" / "output", ROOT / "data"]
DATA_BASE = next(
    (b for b in _candidate_bases if (b / "interim" / "raw_qa_pairs.json").exists()),
    _candidate_bases[0],
)
INTERIM   = DATA_BASE / "interim"
PROCESSED = DATA_BASE / "processed"
ARTIFACTS = ROOT / "research" / "artifacts"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pct(num: int | float, denom: int | float, decimals: int = 1) -> str:
    if denom == 0:
        return "—"
    return f"{100 * num / denom:.{decimals}f}\\%"


def _fmt(n: int | float) -> str:
    if isinstance(n, int):
        return f"{n:,}"
    return f"{n:.2f}"


# ─── Domain extractor ─────────────────────────────────────────────────────────
# Maps doc_id prefix (book number from filename) → domain label
_DOMAIN_MAP: dict[str, str] = {
    "4":  "History of Law",
    "10": "Legislative Drafting",
    "11": "Legal Theory",
    "12": "Legal Theory",
    "13": "Civil Law I",
    "14": "Civil Law II",
    "15": "Commercial Law I",
    "16": "Commercial Law II",
    "17": "Constitutional Law VN",
    "18": "Constitutional Law VN",
    "19": "Comparative Constitutional",
    "20": "Comparative Law",
    "21": "Administrative Law",
    "22": "Admin. Procedure",
    "23": "Environmental Law",
    "24": "Civil Procedure",
    "25": "Family Law",
    "26": "Criminal Law (General)",
    "27": "Criminal Law (Specific)",
    "28": "Criminal Procedure",
    "29": "Tax / Budget Law",
    "30": "Banking Law",
    "31": "International Law",
    "32": "International Law",
    "33": "Private International Law",
    "34": "Intl. Commercial Law",
    "36": "Intl. Economic Law",
    "37": "Land Law",
    "38": "Labor Law",
    "39": "IP Law",
    "40": "Forensic Psychology",
    "41": "Competition Law",
    "42": "Civil Registration",
    "43": "Legal Practice",
    "46": "Criminology",
    "47": "Criminal Qualification",
    "48": "Social Security Law",
    "49": "Notarization Law",
    "50": "Inheritance Law",
    "51": "Real Estate Law",
    "52": "Land Dispute Law",
    "53": "Contract Skills",
}


def _infer_domain(qa: dict) -> str:
    doc_id = str(qa.get("doc_id", ""))
    # Extract leading digits
    m = re.match(r"^(\d+)", doc_id)
    if m:
        return _DOMAIN_MAP.get(m.group(1), "Other")
    qa_id = str(qa.get("qa_id", ""))
    m = re.match(r"^(\d+)", qa_id)
    if m:
        return _DOMAIN_MAP.get(m.group(1), "Other")
    # Fallback: use context filename if present
    ctx = str(qa.get("context_text", ""))[:60]
    return "Other"


# ─── Core Analysis ────────────────────────────────────────────────────────────
def analyze(raw_path: Path, filtered_path: Path, dataset_path: Path) -> dict:
    print(f"[INFO] Loading raw QA pairs from: {raw_path}")
    raw_pairs: list[dict] = _load_json(raw_path) if raw_path.exists() else []
    if not isinstance(raw_pairs, list):
        raw_pairs = []

    print(f"[INFO] Loading filtered QA pairs from: {filtered_path}")
    filtered: list[dict] = _load_json(filtered_path) if filtered_path.exists() else []
    if not isinstance(filtered, list):
        filtered = []

    print(f"[INFO] Loading final dataset from: {dataset_path}")
    final: list[dict] = _load_jsonl(dataset_path) if dataset_path.exists() else []

    n_raw      = len(raw_pairs)
    n_filtered = len(filtered)
    n_final    = len(final)

    # ── Pass rate ──────────────────────────────────────────────────────────
    pass_rate = n_filtered / n_raw if n_raw else 0.0
    reject_n  = n_raw - n_filtered

    # ── Bloom distribution (from final dataset) ────────────────────────────
    bloom_counter: Counter[str] = Counter()
    mm_count = 0
    domain_counter: Counter[str] = Counter()
    q_lengths: list[int] = []
    ctx_lengths: list[int] = []
    ans_counts: list[int] = []

    # Describe the current public release whenever available.
    source = final or filtered or raw_pairs
    for qa in source:
        bloom_counter[str(qa.get("bloom_level", "Unknown"))] += 1
        is_mm = bool(qa.get("is_multimodal", False))
        if not is_mm and isinstance(qa.get("context_payload"), dict):
            is_mm = bool(qa["context_payload"].get("visuals"))
        if is_mm:
            mm_count += 1
        domain_counter[_infer_domain(qa)] += 1

        q_len = len(str(qa.get("question_content", "")))
        if q_len:
            q_lengths.append(q_len)

        ctx_text = qa.get("context_text", "")
        if not ctx_text and isinstance(qa.get("context_payload"), dict):
            ctx_text = qa["context_payload"].get("text", "")
        ctx_len = len(str(ctx_text))
        if ctx_len:
            ctx_lengths.append(ctx_len)

        cands = qa.get("candidate_answers", [])
        if isinstance(cands, list):
            ans_counts.append(len(cands))

    # ── Eval score distributions (Stage 4 scores) ─────────────────────────
    gs_values: list[float] = []
    lf_values: list[float] = []
    mm_align_values: list[float] = []
    overall_values: list[float] = []

    for qa in filtered:
        scores = qa.get("eval_scores", {})
        if scores:
            gs_values.append(float(scores.get("groundedness", 0)))
            lf_values.append(float(scores.get("legal_fluency", 0)))
            mm_align_values.append(float(scores.get("multimodal_alignment", 0)))
            overall_values.append(float(scores.get("overall", 0)))

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    # ── Unique documents ───────────────────────────────────────────────────
    doc_ids = set()
    for qa in source:
        doc_id = str(qa.get("doc_id", "")).strip()
        if doc_id:
            doc_ids.add(doc_id)
            continue
        qa_id = str(qa.get("qa_id", ""))
        if "_chunk_" in qa_id:
            doc_ids.add(qa_id.split("_chunk_", 1)[0])

    stats = {
        "scale": {
            "n_raw_qa_pairs": n_raw,
            "n_filtered_qa_pairs": n_filtered,
            "n_final_dataset_records": n_final,
            "n_rejected": reject_n,
            "pass_rate": round(pass_rate, 4),
            "n_source_documents": len(doc_ids),
            "n_multimodal_qa": mm_count,
            "multimodal_ratio": round(mm_count / max(len(source), 1), 4),
        },
        "bloom_distribution": dict(bloom_counter.most_common()),
        "domain_distribution": dict(domain_counter.most_common()),
        "avg_question_length_chars": round(sum(q_lengths) / max(len(q_lengths), 1), 1),
        "avg_context_length_chars": round(sum(ctx_lengths) / max(len(ctx_lengths), 1), 1),
        "avg_candidate_answers": round(sum(ans_counts) / max(len(ans_counts), 1), 1),
        "eval_scores_summary": {
            "n_scored": len(gs_values),
            "avg_groundedness": round(_avg(gs_values), 4),
            "avg_legal_fluency": round(_avg(lf_values), 4),
            "avg_multimodal_alignment": round(_avg(mm_align_values), 4),
            "avg_overall": round(_avg(overall_values), 4),
        },
    }

    return stats


# ─── LaTeX Snippet Generator ──────────────────────────────────────────────────
def build_latex(stats: dict) -> str:
    s = stats["scale"]
    es = stats["eval_scores_summary"]
    bd = stats["bloom_distribution"]
    lines = [
        "% ============================================================",
        "% AUTO-GENERATED by scripts/research/dataset_stats.py",
        "% Paste this file into your paper preamble with \\input{...}",
        "% ============================================================",
        "",
        "% ── Dataset Scale Macros ──────────────────────────────────",
        f"\\newcommand{{\\TotalRawQA}}{{{s['n_raw_qa_pairs']:,}}}",
        f"\\newcommand{{\\TotalFilteredQA}}{{{s['n_filtered_qa_pairs']:,}}}",
        f"\\newcommand{{\\TotalFinalQA}}{{{s['n_final_dataset_records']:,}}}",
        f"\\newcommand{{\\TotalRejected}}{{{s['n_rejected']:,}}}",
        f"\\newcommand{{\\PassRate}}{{{s['pass_rate']*100:.1f}\\%}}",
        f"\\newcommand{{\\TotalSourceDocs}}{{{s['n_source_documents']}}}",
        f"\\newcommand{{\\MultimodalQA}}{{{s['n_multimodal_qa']:,}}}",
        f"\\newcommand{{\\MultimodalRatio}}{{{s['multimodal_ratio']*100:.1f}\\%}}",
        "",
        "% ── Average Length Macros ─────────────────────────────────",
        f"\\newcommand{{\\AvgQLen}}{{{stats['avg_question_length_chars']:.0f}}}",
        f"\\newcommand{{\\AvgCtxLen}}{{{stats['avg_context_length_chars']:.0f}}}",
        f"\\newcommand{{\\AvgCandidates}}{{{stats['avg_candidate_answers']:.1f}}}",
        "",
        "% ── Eval Score Macros ────────────────────────────────────",
        f"\\newcommand{{\\AvgGroundedness}}{{{es['avg_groundedness']*100:.1f}\\%}}",
        f"\\newcommand{{\\AvgLegalFluency}}{{{es['avg_legal_fluency']*100:.1f}\\%}}",
        f"\\newcommand{{\\AvgOverall}}{{{es['avg_overall']*100:.1f}\\%}}",
        "",
        "% ── Bloom Distribution Macros ────────────────────────────",
    ]
    for level, cnt in bd.items():
        macro = "BloomN" + level.replace(" ", "").replace("'", "")
        lines.append(f"\\newcommand{{\\{macro}}}{{{cnt:,}}}")
    lines += [
        "",
        "% ── Summary Table (Table 1 in paper) ────────────────────",
        "% Copy the tabular environment below into your Section 4:",
        "%",
        "% \\begin{table}[h]",
        "% \\centering",
        "% \\caption{DHH-LegalQA Dataset Statistics}",
        "% \\label{tab:dataset_stats}",
        "% \\begin{tabular}{lr}",
        "% \\toprule",
        "% \\textbf{Statistic} & \\textbf{Value} \\\\",
        "% \\midrule",
        f"% Raw QA pairs (Stage 3 output) & \\TotalRawQA \\\\",
        f"% Filtered QA pairs (after Stage 4) & \\TotalFilteredQA \\\\",
        f"% Pass rate & \\PassRate \\\\",
        f"% Source textbooks & \\TotalSourceDocs \\\\",
        f"% Multimodal QA pairs & \\MultimodalQA\\ (\\MultimodalRatio) \\\\",
        f"% Avg. question length (chars) & \\AvgQLen \\\\",
        f"% Avg. context length (chars) & \\AvgCtxLen \\\\",
        f"% Avg. candidate answers per QA & \\AvgCandidates \\\\",
        "% \\midrule",
        "% \\multicolumn{2}{l}{\\textit{Judge scores (cross-family: Gemma-2-2B)}} \\\\",
        f"% Avg. Groundedness & \\AvgGroundedness \\\\",
        f"% Avg. Legal Fluency & \\AvgLegalFluency \\\\",
        f"% Avg. Overall (Pass) & \\AvgOverall \\\\",
        "% \\bottomrule",
        "% \\end{tabular}",
        "% \\end{table}",
        "",
    ]
    return "\n".join(lines)


# ─── CSV Writers ─────────────────────────────────────────────────────────────
def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    raw_path      = INTERIM / "raw_qa_pairs.json"
    filtered_path = INTERIM / "filtered_qa_pairs.json"
    dataset_path  = PROCESSED / "dataset.jsonl"

    missing = [p for p in [raw_path, filtered_path, dataset_path] if not p.exists()]
    if missing:
        print("[WARN] Some files not found (may be on Drive in Colab):")
        for p in missing:
            print(f"  - {p}")
        if raw_path not in missing:
            print("[INFO] Proceeding with available files...")
        else:
            print("[ERROR] raw_qa_pairs.json not found. Run Stage 3 first.")
            sys.exit(1)

    stats = analyze(raw_path, filtered_path, dataset_path)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # JSON
    json_out = ARTIFACTS / "dataset_stats.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] JSON  → {json_out}")

    # LaTeX
    tex_out = ARTIFACTS / "dataset_stats.tex"
    with open(tex_out, "w", encoding="utf-8") as f:
        f.write(build_latex(stats))
    print(f"[OK] LaTeX → {tex_out}")

    # Bloom CSV
    bloom_rows = [
        {"bloom_level": k, "count": v, "pct": round(100 * v / max(sum(stats["bloom_distribution"].values()), 1), 1)}
        for k, v in stats["bloom_distribution"].items()
    ]
    _write_csv(ARTIFACTS / "bloom_dist.csv", bloom_rows)

    # Domain CSV
    domain_rows = [
        {"domain": k, "count": v, "pct": round(100 * v / max(sum(stats["domain_distribution"].values()), 1), 1)}
        for k, v in stats["domain_distribution"].items()
    ]
    _write_csv(ARTIFACTS / "domain_dist.csv", domain_rows)

    # Console summary
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    scale = stats["scale"]
    print(f"  Raw QA pairs          : {scale['n_raw_qa_pairs']:,}")
    print(f"  Filtered QA pairs     : {scale['n_filtered_qa_pairs']:,}")
    print(f"  Final dataset records : {scale['n_final_dataset_records']:,}")
    print(f"  Pass rate             : {scale['pass_rate']*100:.1f}%")
    print(f"  Source documents      : {scale['n_source_documents']}")
    print(f"  Multimodal QA         : {scale['n_multimodal_qa']:,} ({scale['multimodal_ratio']*100:.1f}%)")
    print(f"  Avg question length   : {stats['avg_question_length_chars']:.0f} chars")
    print(f"  Avg context length    : {stats['avg_context_length_chars']:.0f} chars")
    print(f"\n  Bloom distribution:")
    for lvl, cnt in stats["bloom_distribution"].items():
        total = sum(stats["bloom_distribution"].values())
        print(f"    {lvl:<20} {cnt:>6,}  ({100*cnt/max(total,1):.1f}%)")
    es = stats["eval_scores_summary"]
    if es["n_scored"] > 0:
        print(f"\n  Judge scores (n={es['n_scored']:,}):")
        print(f"    Groundedness          : {es['avg_groundedness']*100:.1f}%")
        print(f"    Legal Fluency         : {es['avg_legal_fluency']*100:.1f}%")
        print(f"    Overall Pass          : {es['avg_overall']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
