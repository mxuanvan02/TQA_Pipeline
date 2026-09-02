#!/usr/bin/env python3
"""
human_visual_qc.py
Machine-level Visual QC for DHH2026 TQA Pipeline.

Tasks:
1. Inspect image_audit_results.csv — flag images to DISCARD, KEEP, or REVIEW.
2. Inspect 10 high-risk Markdown files for table classification:
   TOC / front-matter / OCR-artifact  vs.  QA-ready substantive tables.
3. Cross-reference raw_pdf_missing_visual_candidates.csv for gap candidates.
4. Output:
   - data/output/interim/HUMAN_VISUAL_QC_REPORT.md
   - data/output/interim/human_visual_qc_actions.csv
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/output/interim"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper patterns
# ─────────────────────────────────────────────────────────────────────────────

TOC_PATTERNS = [
    re.compile(r"^\s*mục\s*lục", re.IGNORECASE),
    re.compile(r"^\s*table\s+of\s+contents", re.IGNORECASE),
    re.compile(r"^\s*danh\s+mục", re.IGNORECASE),
    re.compile(r"^\s*lời\s+(nói\s+)?đầu", re.IGNORECASE),
    re.compile(r"^\s*chương\s+trình", re.IGNORECASE),
    re.compile(r"^\s*tài\s+liệu\s+tham\s+khảo", re.IGNORECASE),
    re.compile(r"^\s*bibliography", re.IGNORECASE),
    re.compile(r"^\s*index\b", re.IGNORECASE),
]

FRONT_MATTER_PATTERNS = [
    re.compile(r"^\s*nhà\s+xuất\s+bản", re.IGNORECASE),
    re.compile(r"^\s*isbn\b", re.IGNORECASE),
    re.compile(r"bìa\s+sách", re.IGNORECASE),
    re.compile(r"tên\s+giáo\s+trình", re.IGNORECASE),
    re.compile(r"xuất\s+bản\s+lần\s+thứ", re.IGNORECASE),
    re.compile(r"^\s*in\s+lần\s+thứ", re.IGNORECASE),
]

OCR_ARTIFACT_PATTERNS = [
    re.compile(r"[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]{5,}"),  # clusters of non-Latin/Vietnamese chars
    re.compile(r"[□▪■]{3,}"),
    re.compile(r"\.{5,}"),           # long dotted rows (OCR dot-leaders)
    re.compile(r"\s{10,}"),          # massive whitespace runs
    re.compile(r"([A-Z0-9]{1,3}\s*){8,}"),  # header/footer number sequences
]

SUBSTANTIVE_LEGAL_KEYWORDS = [
    "điều", "khoản", "điểm", "nghị định", "thông tư", "luật",
    "quyết định", "bộ luật", "tội phạm", "hợp đồng", "thừa kế",
    "đất đai", "thuế", "hôn nhân", "tố tụng", "hình sự",
    "dân sự", "hành chính", "kinh tế",
]

# High-risk files requested by Hermes
HIGH_RISK_FILES = [
    "38.LUAT LAO DONG VIET NAM.md",
    "13. LUAT DAN SU VIET NAM-TAP1.md",
    "4. LICH SU NNUOC & PLUAT VIET NAM.md",
    "37. LUAT DAT DAI.md",
    "21. LUAT HANH CHINH VIET NAM.md",
    "23. LUAT MOI TRUONG.md",
    "28. LUAT TO TUNG HINH SU VIET NAM.md",
    "31.32. LUAT QUOC TE.md",
    "26.LUAT HINH SU VN (PHAN CHUNG).md",
    "14. LUAT DAN SU VIET NAM-TAP2.md",
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Image audit analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_image_audit() -> list[dict]:
    """Read image_audit_results.csv and apply QC classification."""
    records = []
    csv_path = INTERIM / "image_audit_results.csv"
    if not csv_path.exists():
        return records

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classification = row.get("classification", "").lower()
            legal_relevance = int(row.get("legal_relevance", 0) or 0)
            confidence = int(row.get("confidence", 0) or 0)
            action_required = row.get("action_required", "").upper()
            notes = row.get("notes", "")
            page = int(row.get("page_number", 0) or 0)
            source = row.get("source_textbook", "")

            # Classify
            if action_required == "DISCARD" or classification in ("noise", "duplicate"):
                issue_type = "irrelevant_or_noise"
                severity = "high"
                recommended_action = "DISCARD"
            elif page == 0 and classification in ("noise", "diagram"):
                issue_type = "front_matter"
                severity = "medium"
                recommended_action = "DISCARD"
            elif legal_relevance >= 3 and confidence >= 3:
                issue_type = "qa_candidate"
                severity = "none"
                recommended_action = "KEEP_AS_IS"
            elif legal_relevance <= 2:
                issue_type = "low_legal_relevance"
                severity = "medium"
                recommended_action = "REVIEW"
            else:
                issue_type = "borderline"
                severity = "low"
                recommended_action = "REVIEW"

            records.append({
                "artifact_type": "image",
                "path_or_candidate_id": row.get("image_id", ""),
                "source_textbook": source,
                "page": page,
                "issue_type": issue_type,
                "severity": severity,
                "recommended_action": recommended_action,
                "notes": notes[:200] if notes else "",
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 3. Table QC in high-risk markdown files
# ─────────────────────────────────────────────────────────────────────────────

def classify_table_block(lines: list[str], context_before: list[str]) -> str:
    """Return 'toc_or_front_matter', 'ocr_artifact', or 'qa_candidate'."""
    block_text = "\n".join(lines)
    context_text = "\n".join(context_before[-10:])

    # Check context for TOC/front-matter signals
    for pat in TOC_PATTERNS + FRONT_MATTER_PATTERNS:
        if pat.search(context_text):
            return "toc_or_front_matter"

    # Check block itself for OCR artifacts
    ocr_hits = sum(1 for pat in OCR_ARTIFACT_PATTERNS if pat.search(block_text))
    if ocr_hits >= 2:
        return "ocr_artifact"

    # Check for substantive legal content
    lower = block_text.lower()
    kw_hits = sum(1 for kw in SUBSTANTIVE_LEGAL_KEYWORDS if kw in lower)
    if kw_hits >= 2:
        return "qa_candidate"

    if kw_hits == 1:
        return "borderline_review"

    return "ocr_artifact"


def extract_table_page(content: str, table_start_line: int) -> int:
    """Attempt to find the most recent page marker before this line."""
    lines = content.splitlines()
    for i in range(min(table_start_line, len(lines) - 1), -1, -1):
        m = re.search(r"---\s*page\s*(\d+)\s*---", lines[i], re.IGNORECASE)
        if m:
            return int(m.group(1))
    return -1


def analyse_high_risk_markdown() -> list[dict]:
    """Parse each high-risk file and classify each markdown table block."""
    records = []

    for fname in HIGH_RISK_FILES:
        fpath = INTERIM / fname
        if not fpath.exists():
            # Try alternative naming (& vs &)
            alt = fname.replace(" & ", " \u0026 ")
            fpath = INTERIM / alt
        if not fpath.exists():
            records.append({
                "artifact_type": "markdown_table",
                "path_or_candidate_id": fname,
                "source_textbook": fname.replace(".md", ""),
                "page": -1,
                "issue_type": "file_not_found",
                "severity": "high",
                "recommended_action": "INVESTIGATE",
                "notes": "File not found on disk.",
            })
            continue

        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            records.append({
                "artifact_type": "markdown_table",
                "path_or_candidate_id": fname,
                "source_textbook": fname.replace(".md", ""),
                "page": -1,
                "issue_type": "read_error",
                "severity": "high",
                "recommended_action": "INVESTIGATE",
                "notes": str(e)[:200],
            })
            continue

        lines = content.splitlines()
        in_table = False
        table_lines: list[str] = []
        table_start = 0
        context_before: list[str] = []
        table_count = 0

        for i, line in enumerate(lines):
            is_table_row = re.match(r"^\s*\|", line) is not None

            if is_table_row and not in_table:
                in_table = True
                table_lines = [line]
                table_start = i
                context_before = lines[max(0, i - 15):i]
            elif is_table_row and in_table:
                table_lines.append(line)
            elif not is_table_row and in_table:
                # End of table block — classify it
                in_table = False
                table_count += 1
                if len(table_lines) < 2:
                    table_lines = []
                    continue

                classification = classify_table_block(table_lines, context_before)
                page = extract_table_page(content, table_start)

                if classification == "toc_or_front_matter":
                    severity = "high"
                    rec_action = "DISCARD"
                elif classification == "ocr_artifact":
                    severity = "medium"
                    rec_action = "DISCARD"
                elif classification == "qa_candidate":
                    severity = "none"
                    rec_action = "KEEP_AS_IS"
                else:
                    severity = "low"
                    rec_action = "REVIEW"

                records.append({
                    "artifact_type": "markdown_table",
                    "path_or_candidate_id": f"{fname}::table_{table_count:03d}@line{table_start}",
                    "source_textbook": fname.replace(".md", ""),
                    "page": page,
                    "issue_type": classification,
                    "severity": severity,
                    "recommended_action": rec_action,
                    "notes": f"Block rows={len(table_lines)}; context={'; '.join(context_before[-2:])[:150]}",
                })
                table_lines = []
                context_before.append(line)
            else:
                if not in_table:
                    context_before.append(line)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gap candidate QC pass
# ─────────────────────────────────────────────────────────────────────────────

def analyse_gap_candidates() -> list[dict]:
    """Apply QC pass on raw_pdf_missing_visual_candidates.csv."""
    records = []
    csv_path = INTERIM / "raw_pdf_missing_visual_candidates.csv"
    if not csv_path.exists():
        return records

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf = row.get("pdf", "")
            page = row.get("page", "")
            visual_type = row.get("visual_type", "")
            reason = row.get("reason", "")
            confidence = int(row.get("confidence", 3) or 3)
            action = row.get("action", "KEEP_AS_IS")

            # Rasterized text-page guard
            if "rasterized" in reason.lower() or "text-page" in reason.lower():
                issue_type = "rasterized_text_page"
                severity = "high"
                recommended_action = "DISCARD"
            elif confidence >= 4 and visual_type == "table":
                issue_type = "structured_table_candidate"
                severity = "none"
                recommended_action = "KEEP_AS_IS"
            elif confidence <= 2:
                issue_type = "low_confidence_gap"
                severity = "medium"
                recommended_action = "REVIEW"
            else:
                issue_type = "gap_candidate"
                severity = "low"
                recommended_action = action

            records.append({
                "artifact_type": "gap_candidate",
                "path_or_candidate_id": f"{pdf}::p{page}",
                "source_textbook": pdf.replace(".pdf", ""),
                "page": page,
                "issue_type": issue_type,
                "severity": severity,
                "recommended_action": recommended_action,
                "notes": reason[:200],
            })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 5. Write outputs
# ─────────────────────────────────────────────────────────────────────────────

def write_csv(records: list[dict], out_path: Path) -> None:
    fieldnames = [
        "artifact_type", "path_or_candidate_id", "source_textbook",
        "page", "issue_type", "severity", "recommended_action", "notes",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"  Written: {out_path.name}  ({len(records)} rows)")


def write_report(records: list[dict], out_path: Path) -> None:
    # Aggregate stats
    total = len(records)
    by_action: dict[str, int] = {}
    by_type: dict[str, int] = {}
    high_sev: list[dict] = []

    for r in records:
        by_action[r["recommended_action"]] = by_action.get(r["recommended_action"], 0) + 1
        by_type[r["artifact_type"]] = by_type.get(r["artifact_type"], 0) + 1
        if r["severity"] == "high":
            high_sev.append(r)

    keep = by_action.get("KEEP_AS_IS", 0)
    discard = by_action.get("DISCARD", 0)
    review = by_action.get("REVIEW", 0)

    lines = [
        "# HUMAN VISUAL QC REPORT",
        "",
        "## Scope",
        f"- **Total artifacts reviewed**: {total}",
        f"- **Images (84 original)**: {by_type.get('image', 0)}",
        f"- **Markdown tables (high-risk files)**: {by_type.get('markdown_table', 0)}",
        f"- **Gap candidates (raw PDF)**: {by_type.get('gap_candidate', 0)}",
        "",
        "## Recommended Actions Summary",
        f"| Action | Count |",
        f"|--------|-------|",
        f"| KEEP_AS_IS | {keep} |",
        f"| DISCARD | {discard} |",
        f"| REVIEW | {review} |",
        f"| INVESTIGATE | {by_action.get('INVESTIGATE', 0)} |",
        "",
        "## High-Severity Issues (require immediate action)",
        "",
    ]
    if high_sev:
        for r in high_sev[:30]:  # cap to keep report readable
            lines.append(
                f"- **[{r['artifact_type']}]** `{r['path_or_candidate_id'][:80]}` "
                f"— {r['issue_type']} → **{r['recommended_action']}**"
            )
            if r["notes"]:
                lines.append(f"  - _{r['notes'][:120]}_")
    else:
        lines.append("_None detected._")

    lines += [
        "",
        "## Methodology",
        "- Image QC: applied `legal_relevance`, `confidence`, `classification` from "
        "`image_audit_results.csv`. Front-matter pages (page=0) with noise classification auto-discarded.",
        "- Markdown table QC: each table block in 10 high-risk files classified by:",
        "  - TOC/front-matter pattern match in surrounding context",
        "  - OCR artifact detection (non-Latin clusters, dot-leaders, whitespace runs)",
        "  - Substantive legal keyword density (≥2 keywords → QA-ready)",
        "- Gap candidate QC: confidence threshold (<3 → REVIEW) applied; rasterized "
        "text-pages auto-discarded.",
        "",
        "## Next Steps for Hermes",
        "1. Use `human_visual_qc_actions.csv` as the authoritative filter input.",
        "2. Merge KEEP_AS_IS gap candidates into `multimodal_candidate_pool_merged.csv`.",
        "3. Discard all DISCARD entries from active candidate pools.",
        "4. REVIEW entries require a second pass with rendered page images.",
        "",
        "_Generated by `scripts/human_visual_qc.py` (Antigravity machine-level QC)_",
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Written: {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== HUMAN VISUAL QC ===")
    print("Phase 1: Image audit QC …")
    img_records = analyse_image_audit()
    print(f"  → {len(img_records)} image records processed.")

    print("Phase 2: High-risk Markdown table QC …")
    md_records = analyse_high_risk_markdown()
    print(f"  → {len(md_records)} table records processed.")

    print("Phase 3: Gap candidate QC …")
    gap_records = analyse_gap_candidates()
    print(f"  → {len(gap_records)} gap records processed.")

    all_records = img_records + md_records + gap_records
    print(f"\nTotal: {len(all_records)} records.")

    csv_out = INTERIM / "human_visual_qc_actions.csv"
    report_out = INTERIM / "HUMAN_VISUAL_QC_REPORT.md"

    write_csv(all_records, csv_out)
    write_report(all_records, report_out)

    # Summary to stdout
    by_action: dict[str, int] = {}
    for r in all_records:
        by_action[r["recommended_action"]] = by_action.get(r["recommended_action"], 0) + 1
    print("\n--- Summary ---")
    for action, count in sorted(by_action.items()):
        print(f"  {action}: {count}")
    print("=== DONE ===")


if __name__ == "__main__":
    main()
