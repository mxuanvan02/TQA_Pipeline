"""
Stage 4 — Evaluation (LLM-as-a-Judge) + Stage 5 Final Output
==============================================================
Who:    Qwen2.5-0.5B-Instruct (same LLM, strict judge system prompt).
Where:  Reads data/interim/raw_qa_pairs.json
        Writes data/interim/filtered_qa_pairs.json (Stage 4)
        Writes data/processed/dataset.jsonl       (Stage 5)
How:    1. Score each QA pair on Groundedness, Multimodal Alignment,
           and Legal Fluency using a structured rubric.
        2. Drop samples below threshold.
        3. Format surviving records into the TQARecord JSONL schema.
Input:  raw_qa_pairs.json (from Stage 3).
Output: filtered_qa_pairs.json + dataset.jsonl.

GPU Optimization (L4 — 22.5 GB VRAM):
    - Batched evaluation: 32 QA pairs per batch (short 256-token output)
    - Left-padding for causal LM batch generation
    - Drive checkpointing per-batch for Colab resilience
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import (
    CFG,
    ContextPayload,
    DriveBackupConfig,
    EvalConfig,
    GPUOptConfig,
    LLMConfig,
    PathConfig,
    TQARecord,
)
from src.utils import (
    batched,
    get_logger,
    load_drive_checkpoint,
    load_json,
    log_gpu_memory,
    save_json,
    save_jsonl,
    setup_tokenizer_for_batch,
    sync_json_to_drive,
    sync_to_drive,
    verify_drive_mount,
)

log = get_logger("04_evaluate")


# ─────────────────────────────────────────────
# Evaluation Prompt
# ─────────────────────────────────────────────
JUDGE_SYSTEM_PROMPT = """\
You are a strict, impartial evaluation judge for Vietnamese Civil Law exam questions.
You MUST evaluate ONLY based on the criteria below. Do NOT add commentary.
Respond using strict XML tags.
"""

JUDGE_TEMPLATE = """\
Evaluate the following QA pair on a Pass (1) or Fail (0) basis for each criterion:

## Context:
{context}

## Question:
{question}

## Ground Truth Answer:
{ground_truth}

## Legal Rationale:
{rationale}

## Evaluation Criteria (Reply with '1' for Pass, '0' for Fail):
1. **Groundedness**: Is the question and answer completely supported by the context? (1 = Yes, 0 = No)
2. **Multimodal Alignment**: If visual info exists, does the QA properly reference it? (1 = Yes, 0 = No/Ignored. If no visuals, reply '1')
3. **Legal Fluency**: Is the legal reasoning (syllogism) correct and logical? (1 = Yes, 0 = No)

## Output Format (strict XML tags):
<evaluation>
<groundedness>1 or 0</groundedness>
<multimodal_alignment>1 or 0</multimodal_alignment>
<legal_fluency>1 or 0</legal_fluency>
<justification>one-sentence explanation</justification>
</evaluation>

Respond with ONLY the XML tags.
"""


# ─────────────────────────────────────────────
# Judge LLM (with Batch Support)
# ─────────────────────────────────────────────
class QAJudge:
    """
    LLM-as-a-judge for QA pair quality evaluation.

    Who:    Qwen2.5-0.5B-Instruct (4-bit, deterministic temperature).
    How:    Scores each QA pair on 3 criteria, supports batch inference.
    """

    def __init__(self, cfg: LLMConfig, eval_cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the LLM."""
        if self._model is not None:
            return

        log.info("Loading Judge LLM: %s", self.cfg.model_name)
        start = time.perf_counter()

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if self.cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.cfg.torch_dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            quantization_config=quantization_config,
            device_map=self.cfg.device_map,
            torch_dtype=getattr(torch, self.cfg.torch_dtype),
            trust_remote_code=True,
        )

        # Configure tokenizer for batch inference
        setup_tokenizer_for_batch(self._tokenizer)

        elapsed = time.perf_counter() - start
        log.info("  Judge LLM loaded in %.1fs", elapsed)
        log_gpu_memory("QAJudge loaded")

    def _build_prompt(self, qa: dict[str, Any]) -> str:
        """Build a fully-formatted evaluation prompt for one QA pair."""
        visual_info = ""
        if qa.get("visual_descriptions"):
            descs = [d.get("summary", "") for d in qa["visual_descriptions"]]
            visual_info = "\n[Visual context: " + "; ".join(d for d in descs if d) + "]"

        user_prompt = JUDGE_TEMPLATE.format(
            context=qa.get("context_text", "")[:1500] + visual_info,
            question=qa.get("question_content", ""),
            ground_truth=qa.get("ground_truth", ""),
            rationale=qa.get("legal_rationale", ""),
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def evaluate(self, qa: dict[str, Any]) -> dict[str, Any] | None:
        """
        Score a single QA pair (backward compatible, uses batch-of-1).

        Input:  Raw QA dict from Stage 3.
        Output: QA dict enriched with evaluation scores, or None if fails.
        """
        results = self.evaluate_batch([qa])
        return results[0] if results[0] is not None else None

    def evaluate_batch(
        self, qa_pairs: list[dict[str, Any]]
    ) -> list[dict[str, Any] | None]:
        """
        Batch-evaluate multiple QA pairs.

        Who:    The loaded Judge LLM (left-padded batch inference).
        Input:  List of QA dicts from Stage 3.
        Output: List of QA dicts enriched with eval_scores, or None per item.

        GPU Optimization:
            - Tokenizes all prompts together with padding
            - Short max_new_tokens (256) allows large batch sizes (32+)
            - Deterministic (do_sample=False) for consistent evaluation
        """
        self._load()

        # Build all prompt strings
        prompt_texts = []
        for qa in qa_pairs:
            try:
                prompt_texts.append(self._build_prompt(qa))
            except Exception as exc:
                log.warning("  Prompt build failed for %s: %s",
                            qa.get("qa_id", "?"), exc)
                prompt_texts.append(None)

        # Filter out failed prompts
        valid_indices = [i for i, p in enumerate(prompt_texts) if p is not None]
        valid_prompts = [prompt_texts[i] for i in valid_indices]

        results: list[dict[str, Any] | None] = [None] * len(qa_pairs)

        if not valid_prompts:
            return results

        try:
            # Batch tokenize with left-padding
            inputs = self._tokenizer(
                valid_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self._model.device)

            prompt_len = inputs.input_ids.shape[1]

            # Batch generate (deterministic for evaluation)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.cfg.eval_temperature,
                    do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

            # Decode and parse each output
            for batch_idx, valid_idx in enumerate(valid_indices):
                response = self._tokenizer.decode(
                    outputs[batch_idx][prompt_len:],
                    skip_special_tokens=True,
                )

                scores = _parse_eval_xml(response)
                if scores:
                    qa_pairs[valid_idx]["eval_scores"] = scores
                    results[valid_idx] = qa_pairs[valid_idx]
                else:
                    log.debug("  Could not parse eval for %s",
                              qa_pairs[valid_idx].get("qa_id", "?"))

            return results

        except Exception as exc:
            log.warning("  Batch evaluation failed: %s", exc)
            log.info("  Falling back to sequential evaluation...")
            return self._evaluate_sequential_fallback(
                qa_pairs, valid_indices, valid_prompts
            )

    def _evaluate_sequential_fallback(
        self,
        qa_pairs: list[dict[str, Any]],
        valid_indices: list[int],
        valid_prompts: list[str],
    ) -> list[dict[str, Any] | None]:
        """Fallback: evaluate one-by-one if batch fails (e.g., OOM)."""
        results: list[dict[str, Any] | None] = [None] * len(qa_pairs)

        for batch_idx, valid_idx in enumerate(valid_indices):
            try:
                inputs = self._tokenizer(
                    valid_prompts[batch_idx],
                    return_tensors="pt",
                ).to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=self.cfg.eval_temperature,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                response = self._tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                scores = _parse_eval_xml(response)
                if scores:
                    qa_pairs[valid_idx]["eval_scores"] = scores
                    results[valid_idx] = qa_pairs[valid_idx]

            except Exception as exc:
                log.warning("  Sequential eval failed for %s: %s",
                            qa_pairs[valid_idx].get("qa_id", "?"), exc)

        return results

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("  Judge LLM unloaded")


def _parse_eval_xml(text: str) -> dict[str, Any] | None:
    """Parse evaluation scores from LLM XML output."""
    scores: dict[str, Any] = {
        "groundedness": 0.0,
        "multimodal_alignment": 0.0,
        "legal_fluency": 0.0,
        "justification": "",
        "overall": 0.0,
    }

    g_match = re.search(r"<groundedness>([\s\S]*?)</groundedness>", text, re.IGNORECASE)
    m_match = re.search(r"<multimodal_alignment>([\s\S]*?)</multimodal_alignment>", text, re.IGNORECASE)
    l_match = re.search(r"<legal_fluency>([\s\S]*?)</legal_fluency>", text, re.IGNORECASE)
    j_match = re.search(r"<justification>([\s\S]*?)</justification>", text, re.IGNORECASE)

    if not (g_match and m_match and l_match):
        return None

    try:
        def _parse_binary(val: str) -> float:
            return 1.0 if "1" in val else 0.0

        scores["groundedness"] = _parse_binary(g_match.group(1))
        scores["multimodal_alignment"] = _parse_binary(m_match.group(1))
        scores["legal_fluency"] = _parse_binary(l_match.group(1))

        if j_match:
            scores["justification"] = j_match.group(1).strip()

        scores["overall"] = 1.0 if (scores["groundedness"] == 1.0 and scores["legal_fluency"] == 1.0) else 0.0

        return scores

    except Exception:
        return None


# ─────────────────────────────────────────────
# Filtering Logic
# ─────────────────────────────────────────────
def filter_qa_pairs(
    qa_pairs: list[dict[str, Any]],
    cfg: EvalConfig,
) -> list[dict[str, Any]]:
    """
    Drop QA pairs that fall below quality thresholds.

    Who:    Pure Python threshold comparison.
    How:    Check each criterion against its threshold.
    Input:  QA pairs with eval_scores attached.
    Output: Filtered list of quality QA pairs.
    """
    filtered: list[dict[str, Any]] = []

    for qa in qa_pairs:
        scores = qa.get("eval_scores", {})

        if not scores:
            continue

        passes = (
            scores.get("groundedness", 0) >= cfg.groundedness_threshold
            and scores.get("legal_fluency", 0) >= cfg.legal_fluency_threshold
            and scores.get("overall", 0) >= cfg.overall_threshold
        )

        # Only check multimodal alignment if the QA is multimodal
        if qa.get("is_multimodal", False):
            passes = passes and (
                scores.get("multimodal_alignment", 0)
                >= cfg.multimodal_alignment_threshold
            )

        if passes:
            filtered.append(qa)

    log.info(
        "Filtering: %d / %d QA pairs passed (%.0f%% pass rate)",
        len(filtered),
        len(qa_pairs),
        100 * len(filtered) / max(len(qa_pairs), 1),
    )
    return filtered


# ─────────────────────────────────────────────
# Stage 5: Final Output Formatting
# ─────────────────────────────────────────────
def format_to_tqa_records(
    qa_pairs: list[dict[str, Any]],
) -> list[TQARecord]:
    """
    Convert filtered QA dicts into the final TQARecord Pydantic schema.

    Who:    Pydantic TQARecord model (defined in config.py).
    How:    Maps raw dict fields to the strict JSONL schema.
    Input:  Filtered QA pairs from Stage 4.
    Output: List of validated TQARecord instances.
    """
    records: list[TQARecord] = []

    for qa in qa_pairs:
        try:
            record = TQARecord(
                qa_id=qa.get("qa_id", ""),
                domain_tag=qa.get("domain_tag", "civil_law"),
                context_payload=ContextPayload(
                    text=qa.get("context_text", ""),
                    visuals=qa.get("context_visuals", []),
                ),
                question_content=qa.get("question_content", ""),
                is_multimodal=qa.get("is_multimodal", False),
                candidate_answers=qa.get("candidate_answers", []),
                ground_truth=qa.get("ground_truth", ""),
                legal_rationale=qa.get("legal_rationale", ""),
            )
            records.append(record)
        except Exception as exc:
            log.warning("  Schema validation failed for %s: %s",
                        qa.get("qa_id", "?"), exc)

    log.info("Formatted %d TQARecords", len(records))
    return records


# ─────────────────────────────────────────────
# Pipeline Orchestration (Batched)
# ─────────────────────────────────────────────
def run_evaluation(
    paths: PathConfig | None = None,
    llm_cfg: LLMConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    drive_cfg: DriveBackupConfig | None = None,
    gpu_cfg: GPUOptConfig | None = None,
    limit: int | None = None,
) -> list[TQARecord]:
    """
    End-to-end Stage 4 + 5: evaluate, filter, format (BATCHED).

    Who:    QAJudge (batched) + filter + TQARecord formatter.
    Where:  data/interim/raw_qa_pairs.json -> data/processed/dataset.jsonl
    How:    Batch score -> filter -> format -> save.
    Input:  raw_qa_pairs.json from Stage 3.
    Output: Final TQARecord list saved as JSONL.

    GPU Optimization:
        - eval_cfg.batch_size QA pairs evaluated per GPU batch (default 32)
        - Short output (256 tokens) allows larger batches than Stage 3
        - Drive checkpointing per-batch for resilience
    """
    paths = paths or CFG.paths
    llm_cfg = llm_cfg or CFG.llm
    eval_cfg = eval_cfg or CFG.evaluation
    drive_cfg = drive_cfg or CFG.drive_backup
    gpu_cfg = gpu_cfg or CFG.gpu

    # Pre-flight: verify Drive mount
    verify_drive_mount(drive_cfg)

    # Load raw QA pairs
    raw_pairs = load_json(paths.raw_qa_pairs)
    if limit:
        raw_pairs = raw_pairs[:limit]
        log.info("Limited to %d QA pairs", limit)

    # ── Separate cached vs. new QA pairs ──
    new_pairs = []
    evaluated: list[dict[str, Any]] = []

    for qa in raw_pairs:
        qa_id = qa.get("qa_id", "unknown")
        drive_qa_eval = drive_cfg.evaluated_qa_dir / f"{qa_id}.json"

        saved_eval = load_drive_checkpoint(drive_qa_eval, drive_cfg)
        if saved_eval is not None:
            evaluated.append(saved_eval)
        else:
            new_pairs.append(qa)

    if not new_pairs:
        log.info("All %d QA pairs already evaluated (loaded from Drive cache)", len(raw_pairs))
    else:
        log.info(
            "Evaluating %d new QA pairs (batch_size=%d), %d cached",
            len(new_pairs), eval_cfg.batch_size, len(raw_pairs) - len(new_pairs),
        )

        # ── Stage 4: Batched Evaluate ──
        judge = QAJudge(llm_cfg, eval_cfg)
        batch_count = 0
        total_batches = (len(new_pairs) + eval_cfg.batch_size - 1) // eval_cfg.batch_size

        for batch_qa in tqdm(
            batched(new_pairs, eval_cfg.batch_size),
            total=total_batches,
            desc="Eval Batches",
        ):
            batch_count += 1

            # Log GPU usage periodically
            if batch_count % gpu_cfg.log_gpu_interval == 0:
                log_gpu_memory(f"Eval batch {batch_count}/{total_batches}")

            # Batch evaluate
            batch_results = judge.evaluate_batch(batch_qa)

            # Collect results & checkpoint to Drive
            for qa_result in batch_results:
                if qa_result is not None:
                    evaluated.append(qa_result)

                    # Per-QA Drive checkpoint
                    qa_id = qa_result.get("qa_id", "unknown")
                    drive_qa_eval = drive_cfg.evaluated_qa_dir / f"{qa_id}.json"
                    sync_json_to_drive(qa_result, drive_qa_eval, drive_cfg)

            # Periodic VRAM cleanup
            if batch_count % gpu_cfg.empty_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        judge.unload()

    log.info("  Successfully evaluated %d / %d pairs", len(evaluated), len(raw_pairs))

    # Filter below-threshold samples
    filtered = filter_qa_pairs(evaluated, eval_cfg)

    # Save filtered pairs (intermediate artifact)
    save_json(filtered, paths.filtered_qa_pairs)

    # ── Stage 5: Format to JSONL ──
    records = format_to_tqa_records(filtered)

    # Save final dataset
    save_jsonl(records, paths.dataset_jsonl)

    # ── Final Backups to Drive ──
    sync_to_drive(
        paths.filtered_qa_pairs,
        drive_cfg.backup_base / "interim/filtered_qa_pairs.json",
        drive_cfg,
        label="filtered_qa_pairs.json",
    )
    sync_to_drive(
        paths.dataset_jsonl,
        drive_cfg.processed_dir / "dataset.jsonl",
        drive_cfg,
        label="dataset.jsonl (final output)",
    )

    log.info(
        "Pipeline complete! %d records -> %s (batch_size=%d)",
        len(records),
        paths.dataset_jsonl,
        eval_cfg.batch_size,
    )
    return records


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4-5: Evaluate QA pairs (LLM-as-judge) + format JSONL output"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to N QA pairs")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override eval batch size (default: from config)",
    )
    args = parser.parse_args()

    log.info("Stage 4 -- Evaluation (LLM-as-a-Judge, Batched)")

    if args.batch_size:
        from dataclasses import replace
        eval_cfg = replace(CFG.evaluation, batch_size=args.batch_size)
        results = run_evaluation(limit=args.limit, eval_cfg=eval_cfg)
    else:
        results = run_evaluation(limit=args.limit)

    if not results:
        log.warning("No records survived evaluation. Check thresholds or raw QA quality.")
        sys.exit(1)

    log.info("Final dataset: %d records", len(results))


if __name__ == "__main__":
    main()
