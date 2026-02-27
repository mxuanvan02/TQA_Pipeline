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
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from typing import Any

import torch
from tqdm import tqdm

from src.config import (
    CFG,
    ContextPayload,
    EvalConfig,
    LLMConfig,
    PathConfig,
    TQARecord,
)
from src.utils import get_logger, load_json, save_json, save_jsonl

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
# Judge LLM
# ─────────────────────────────────────────────
class QAJudge:
    """
    LLM-as-a-judge for QA pair quality evaluation.

    Who:    Qwen2.5-0.5B-Instruct (4-bit, deterministic temperature).
    How:    Scores each QA pair on 3 criteria, returns normalized scores.
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

        log.info("⚖️  Loading Judge LLM: %s", self.cfg.model_name)
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

        elapsed = time.perf_counter() - start
        log.info("  ✅ Judge LLM loaded in %.1fs", elapsed)

    def evaluate(self, qa: dict[str, Any]) -> dict[str, Any] | None:
        """
        Score a single QA pair.

        Input:  Raw QA dict from Stage 3.
        Output: QA dict enriched with evaluation scores, or None if parsing fails.
        """
        self._load()

        # Build visual context string
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

        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.cfg.eval_temperature,
                    do_sample=False,  # deterministic for evaluation
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            scores = _parse_eval_xml(response)
            if scores:
                qa["eval_scores"] = scores
                return qa

            log.debug("  ⚠️  Could not parse eval for %s", qa.get("qa_id", "?"))
            return None

        except Exception as exc:
            log.warning("  ⚠️  Evaluation failed for %s: %s", qa.get("qa_id", "?"), exc)
            return None

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("🗑️  Judge LLM unloaded")


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
        # Extract binary score 1 or 0
        def _parse_binary(val: str) -> float:
            return 1.0 if "1" in val else 0.0

        scores["groundedness"] = _parse_binary(g_match.group(1))
        scores["multimodal_alignment"] = _parse_binary(m_match.group(1))
        scores["legal_fluency"] = _parse_binary(l_match.group(1))
        
        if j_match:
            scores["justification"] = j_match.group(1).strip()
            
        # Overall is pass only if Groundedness and Legal Fluency pass.
        # Multimodal is checked later in filter_qa_pairs based on `is_multimodal`
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
        "🔍 Filtering: %d / %d QA pairs passed (%.0f%% pass rate)",
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
            log.warning("  ⚠️  Schema validation failed for %s: %s",
                        qa.get("qa_id", "?"), exc)

    log.info("📋 Formatted %d TQARecords", len(records))
    return records


# ─────────────────────────────────────────────
# Pipeline Orchestration
# ─────────────────────────────────────────────
def run_evaluation(
    paths: PathConfig | None = None,
    llm_cfg: LLMConfig | None = None,
    eval_cfg: EvalConfig | None = None,
    limit: int | None = None,
) -> list[TQARecord]:
    """
    End-to-end Stage 4 + 5: evaluate, filter, format.

    Who:    QAJudge + filter + TQARecord formatter.
    Where:  data/interim/raw_qa_pairs.json → data/processed/dataset.jsonl
    How:    Score → filter → format → save.
    Input:  raw_qa_pairs.json from Stage 3.
    Output: Final TQARecord list saved as JSONL.
    """
    paths = paths or CFG.paths
    llm_cfg = llm_cfg or CFG.llm
    eval_cfg = eval_cfg or CFG.evaluation

    # Load raw QA pairs
    raw_pairs = load_json(paths.raw_qa_pairs)
    if limit:
        raw_pairs = raw_pairs[:limit]
        log.info("🔒 Limited to %d QA pairs", limit)

    log.info("📊 Evaluating %d raw QA pairs...", len(raw_pairs))

    # ── Stage 4: Evaluate ──
    judge = QAJudge(llm_cfg, eval_cfg)
    evaluated: list[dict[str, Any]] = []

    for qa in tqdm(raw_pairs, desc="Evaluating QA pairs"):
        result = judge.evaluate(qa)
        if result is not None:
            evaluated.append(result)

    judge.unload()

    log.info("  ✅ Successfully evaluated %d / %d pairs", len(evaluated), len(raw_pairs))

    # Filter below-threshold samples
    filtered = filter_qa_pairs(evaluated, eval_cfg)

    # Save filtered pairs (intermediate artifact)
    save_json(filtered, paths.filtered_qa_pairs)

    # ── Stage 5: Format to JSONL ──
    records = format_to_tqa_records(filtered)

    # Save final dataset
    save_jsonl(records, paths.dataset_jsonl)

    log.info(
        "🏁 Pipeline complete! %d records → %s",
        len(records),
        paths.dataset_jsonl,
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
    args = parser.parse_args()

    log.info("🚀 Stage 4 — Evaluation (LLM-as-a-Judge)")
    results = run_evaluation(limit=args.limit)

    if not results:
        log.warning("No records survived evaluation. Check thresholds or raw QA quality.")
        sys.exit(1)

    log.info("✅ Final dataset: %d records", len(results))


if __name__ == "__main__":
    main()
