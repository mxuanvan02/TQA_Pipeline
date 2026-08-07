#!/usr/bin/env python3
"""Runner for the visual-dependence ablation (ECM-TQAG, v4 step 6).

WHAT THIS ESTABLISHES THAT THE GUARDS CANNOT
--------------------------------------------
`guard_plan_v3` + `verify_visual_anchors` prove the PLANNER used the image: a
cross-modal motif cannot compile without a visual anchor naming a declared image
and committing to a bbox. That is a claim about CONSTRUCTION.

It says nothing about the QUESTION. A model can emit a structurally valid visual
anchor -- even a habitual, copy-pasted bbox -- and still produce an item that is
fully answerable from prose. Calling such an item "multimodal" is unfalsifiable.

This runner supplies the missing half: an independent ANSWERER (a different model
family from the generator, to avoid self-preference) attempts each frozen item
twice, differing in exactly one factor:

    arm "with_image"    : evidence.text + document_structure + page image
    arm "without_image" : evidence.text + document_structure           (no image)

Question string, choice strings, choice order, evidence text, decoding and model
are byte-identical across arms; `visual_dependence.assert_pairing_valid` refuses
to score any pair where that is not true. Verdicts are assigned by
`visual_dependence.classify_item` (majority over k repeats), never here: this
module only OBSERVES answers, so every verdict is replayable from the ledger.

The answerer is instructed to return {"answer_index": N} and nothing else. It
never sees the gold answer, the rationale, the plan, the motif, or the condition
label -- so it cannot infer the intended answer from construction metadata.

Dry-run by default. `--execute` is required to spend money.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_ecm24_matrix as v1  # noqa: E402  (image_part / load_manifest)
import run_ecm_v3 as v3  # noqa: E402  (call_with_retry / parse_model_json)
import visual_dependence as vd  # noqa: E402

SCHEMA = "ecm-tqag.visual-dependence-run.v1"

# The answerer MUST NOT be the generator family. The generator is
# qwen/qwen3-vl-32b-instruct; scoring its own items with a Qwen model would let
# self-preference inflate the with-image arm.
DEFAULT_ANSWERER = "google/gemini-2.5-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

ANSWERER_SYSTEM = "Return valid JSON only."

ANSWERER_INSTRUCTION = (
    "Bạn là người TRẢ LỜI câu hỏi trắc nghiệm. Chỉ dựa vào tư liệu được cung cấp "
    "bên dưới.\n"
    "Nếu tư liệu không đủ để xác định đáp án, hãy chọn phương án bạn cho là hợp lý "
    "nhất — nhưng KHÔNG được suy đoán từ kiến thức ngoài tư liệu.\n"
    'Chỉ in ra một đối tượng JSON duy nhất: {"answer_index": N} với N là số nguyên '
    "0..3 ứng với vị trí phương án đúng trong danh sách.\n"
    "Không giải thích, không thêm chữ nào khác.\n"
)


def answerer_parts(item: dict[str, Any], attach_image: bool) -> list[dict[str, Any]]:
    """Build the message content. The ONLY difference between arms is the image.

    `document_structure` is included in BOTH arms: removing it alongside the
    image would confound two factors and make a positive result uninterpretable.
    """
    lines = [ANSWERER_INSTRUCTION, "\n--- TƯ LIỆU ---\n", item["evidence_text"]]

    structure = item.get("document_structure")
    if isinstance(structure, dict):
        lines.append(
            f"\n\nCấu trúc tài liệu:\n"
            f"- Tiêu đề mục: {structure.get('section_title')}\n"
            f"- Cấp tiêu đề: {structure.get('header_level')}\n"
            f"- Thứ tự ảnh đã khai báo: {structure.get('declared_image_order')}\n")

    if attach_image:
        n = len(item.get("images") or [])
        lines.append(f"\nCó {n} ảnh trang gốc được đính kèm, đánh số 1..{n}.\n")

    lines.append("\n\n--- CÂU HỎI ---\n")
    lines.append(item["question"])
    lines.append("\n\n--- CÁC PHƯƠNG ÁN ---\n")
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{i}. {choice}\n")

    parts: list[dict[str, Any]] = [{"type": "text", "text": "".join(lines)}]
    if attach_image:
        for image in item.get("images") or []:
            part, _audit = v1.image_part(image)
            parts.append(part)
    return parts


def parse_answer_index(raw: str, n_choices: int) -> tuple[int | None, str | None]:
    """Accept ONLY {"answer_index": int-in-range}. Anything else is a failure.

    A failed parse yields is_correct=None, which `_majority_correct` excludes
    from the vote -- a provider/contract failure is not evidence about vision.
    """
    parsed, reason = v3.parse_model_json(raw)
    if parsed is None:
        return None, reason or "unparseable"
    value = parsed.get("answer_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return None, "answer_index_not_int"
    if not 0 <= value < n_choices:
        return None, "answer_index_out_of_range"
    return value, None


def load_items(results_path: Path, manifest_path: Path,
               condition: str) -> list[dict[str, Any]]:
    """Collect PARSED items of one condition, joined to their manifest evidence.

    The manifest is the ONLY source of image bytes and evidence text, so the
    ablation cannot silently score against a re-generated or drifted payload:
    `image_part` re-verifies size+sha256 for every attached image.
    """
    manifest, _hash = v1.load_manifest(manifest_path)
    by_key = {(p["chunk_id"], p["condition"]): p for p in manifest["packages"]}

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "PARSED" or row.get("condition") != condition:
            continue
        response = row.get("parsed_response") or {}
        question = response.get("question")
        choices = response.get("choices")
        gold = response.get("answer_index")
        if not isinstance(question, str) or not isinstance(choices, list):
            continue
        if isinstance(gold, bool) or not isinstance(gold, int):
            continue

        package = by_key.get((row["chunk_id"], condition))
        if package is None:
            raise ValueError(f"BLOCKED_PAIRING:no_manifest_package:{row['chunk_id']}")
        evidence = package["evidence"]

        fingerprint = vd.item_fingerprint(question, choices, evidence["text"])
        if fingerprint in seen:
            raise ValueError(f"BLOCKED_PAIRING:duplicate_item:{fingerprint[:16]}")
        seen.add(fingerprint)

        items.append({
            "chunk_id": row["chunk_id"],
            "doc_id": row.get("doc_id"),
            "run_id": row.get("run_id"),
            "question": question,
            "choices": choices,
            "gold_index": gold,
            "evidence_text": evidence["text"],
            "document_structure": evidence.get("document_structure"),
            "images": evidence.get("images") or [],
            "item_fingerprint": fingerprint,
        })
    return items


def run(results_path: Path, manifest_path: Path, out_dir: Path, *,
        condition: str, execute: bool, base_url: str, model: str,
        repeats: int, timeout_sec: int, retries: int,
        api_key: str | None) -> dict[str, Any]:
    if execute and not api_key:
        raise ValueError("BLOCKED_AUTH:execute_requires_api_key")
    items = load_items(results_path, manifest_path, condition)

    # An item with no image cannot have a with_image arm; scoring it would make
    # the two arms identical and guarantee TEXT_SUFFICIENT for a reason that has
    # nothing to do with vision.
    usable = [it for it in items if it["images"]]
    skipped = [it["chunk_id"] for it in items if not it["images"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    ledger_path = out_dir / "verdicts.json"

    planned = len(usable) * len(vd.ARMS) * repeats
    if not execute:
        return {"schema": SCHEMA, "status": "DRY_RUN", "n_items": len(usable),
                "n_items_skipped_no_image": len(skipped),
                "skipped_chunk_ids": skipped,
                "planned_api_calls": planned, "answerer_model": model,
                "repeats_per_arm": repeats, "condition": condition}

    all_trials: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    n_calls = 0
    usage_total = {"prompt": 0, "completion": 0}

    with trials_path.open("w", encoding="utf-8") as handle:
        for item in usable:
            item_trials: list[dict[str, Any]] = []
            for arm in vd.ARMS:
                attach = arm == vd.ARM_WITH
                parts = answerer_parts(item, attach)
                for repeat in range(repeats):
                    payload = {
                        "model": model, "temperature": 0, "max_tokens": 64,
                        "messages": [
                            {"role": "system", "content": ANSWERER_SYSTEM},
                            {"role": "user", "content": parts}]}
                    result = v3.call_with_retry(base_url, payload, timeout_sec,
                                                retries, api_key)
                    n_calls += 1
                    answer, parse_reason = (None, result["reason"])
                    if result["ok"]:
                        answer, parse_reason = parse_answer_index(
                            result["raw"], len(item["choices"]))
                    usage = result.get("usage") or {}
                    usage_total["prompt"] += usage.get("prompt_tokens") or 0
                    usage_total["completion"] += usage.get("completion_tokens") or 0

                    trial = {
                        "chunk_id": item["chunk_id"],
                        "item_fingerprint": item["item_fingerprint"],
                        "arm": arm,
                        "repeat": repeat,
                        "image_attached": attach,
                        "answer_index": answer,
                        "gold_index": item["gold_index"],
                        "is_correct": None if answer is None
                                      else answer == item["gold_index"],
                        "failure_reason": parse_reason,
                        "answerer_model": model,
                        "elapsed_sec": result.get("elapsed_sec"),
                        "retry_count": result.get("retry_count"),
                    }
                    item_trials.append(trial)
                    handle.write(json.dumps(trial, ensure_ascii=False) + "\n")
                    handle.flush()
                    time.sleep(0.2)

            verdict = vd.classify_item(item_trials)
            verdict["chunk_id"] = item["chunk_id"]
            verdicts.append(verdict)
            all_trials.extend(item_trials)

    summary = vd.aggregate(verdicts)
    summary.update({
        "run_schema": SCHEMA,
        "status": "EXECUTED",
        "condition": condition,
        "source_results": str(results_path),
        "answerer_model": model,
        "generator_model_note": "answerer is a DIFFERENT family from the "
                                "generator (qwen/qwen3-vl-32b-instruct) to "
                                "avoid self-preference",
        "repeats_per_arm": repeats,
        "api_call_count": n_calls,
        "tokens": usage_total,
        "n_items_skipped_no_image": len(skipped),
        "skipped_chunk_ids": skipped,
        "verdicts": verdicts,
    })
    ledger_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual-dependence ablation; dry-run by default.")
    parser.add_argument("--results", type=Path, required=True,
                        help="results.jsonl from a run_ecm_v3 execution")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "ecm_inputs_final_v2.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition", default="TLV")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_ANSWERER)
    parser.add_argument("--repeats", type=int, default=vd.DEFAULT_REPEATS)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--api-key-file")
    args = parser.parse_args()

    api_key = None
    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()

    summary = run(args.results, args.manifest, args.out_dir,
                  condition=args.condition, execute=args.execute,
                  base_url=args.base_url, model=args.model,
                  repeats=args.repeats, timeout_sec=args.timeout_sec,
                  retries=args.retries, api_key=api_key)
    printable = {k: v for k, v in summary.items() if k != "verdicts"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
