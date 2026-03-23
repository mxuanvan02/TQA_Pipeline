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
    AsyncDriveWriter,
    batched,
    batched_by_length,
    get_logger,
    load_drive_checkpoint,
    load_json,
    log_gpu_memory,
    save_json,
    save_jsonl,
    setup_tokenizer_for_batch,
    sync_json_to_drive,
    sync_to_drive,
    try_torch_compile,
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
Evaluate the following QA pair on a Pass (1) or Fail (0) basis for each criterion, and categorize its cognitive depth.

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

## Taxonomy Classification:
Categorize the question into: 
- **Remember**: Direct lookup of definitions/clauses.
- **Understand**: Explanation or summarization.
- **Apply**: Scenario-based reasoning.

## Output Format (strict XML tags):
<evaluation>
<groundedness>1 or 0</groundedness>
<multimodal_alignment>1 or 0</multimodal_alignment>
<legal_fluency>1 or 0</legal_fluency>
<taxonomy_level>Remember/Understand/Apply</taxonomy_level>
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

    Primary:  vLLM offline inference engine (PagedAttention, optimal GPU sat).
    Fallback: HuggingFace generate() with batched evaluation.
    """

    def __init__(self, cfg: LLMConfig, eval_cfg: EvalConfig, gpu_cfg: GPUOptConfig | None = None) -> None:
        self.cfg = cfg
        self.eval_cfg = eval_cfg
        self.gpu_cfg = gpu_cfg or CFG.gpu
        self._model = None
        self._tokenizer = None
        self._vllm_engine = None
        self._vllm_sampling = None

    def _load(self) -> None:
        """
        Lazy-load the LLM on first use.

        # [Paper Note — Cross-Family Judge Loading Strategy]
        # Primary judge: google/gemma-2-2b-it (Google/Gemma family)
        # Fallback judge: microsoft/Phi-3-mini-4k-instruct (Microsoft family)
        # Both are DIFFERENT families from the generator (Qwen) to avoid
        # same-family self-reinforcement bias in LLM-as-a-judge evaluation.
        # See §4.1 of the paper for experimental validation.
        """
        if self._model is not None or self._vllm_engine is not None:
            return

        if getattr(self.cfg, "use_vllm", False):
            try:
                self._load_vllm()
                return
            except Exception as e:
                log.warning("Failed to load vLLM (%s). Falling back to HuggingFace.", e)
                self.cfg = type(self.cfg)(**{**self.cfg.__dict__, "use_vllm": False})

        # Try primary judge model, then fallback if download/load fails.
        try:
            self._load_hf()
        except Exception as primary_err:
            fallback = getattr(self.eval_cfg, "fallback_judge_model_name", None)
            if fallback and fallback != self.cfg.model_name:
                log.warning(
                    "Primary judge model '%s' failed to load (%s). "
                    "Trying fallback judge: '%s'",
                    self.cfg.model_name, primary_err, fallback,
                )
                from dataclasses import replace as dc_replace
                self.cfg = dc_replace(self.cfg, model_name=fallback)
                self._load_hf()
            else:
                raise

    def _load_vllm(self) -> None:
        """Load the model using vLLM for high-throughput inference."""
        log.info("Loading Judge LLM via vLLM: %s (4-bit=%s)", self.cfg.model_name, self.cfg.load_in_4bit)
        start = time.perf_counter()

        from vllm import LLM, SamplingParams

        # Autodetect AWQ: vLLM only supports 'awq' if it is pre-quantized in the repo
        model_is_awq = "awq" in self.cfg.model_name.lower()
        quantization = "awq" if model_is_awq else None
        
        self._vllm_engine = LLM(
            model=self.cfg.model_name,
            quantization=quantization,
            max_model_len=3072,
            gpu_memory_utilization=getattr(self.gpu_cfg, "vllm_gpu_utilization", 0.90),
            max_num_seqs=getattr(self.gpu_cfg, "vllm_max_num_seqs", 1024),
            trust_remote_code=True,
            enforce_eager=False,
        )
        
        self._tokenizer = self._vllm_engine.get_tokenizer()
        
        self._vllm_sampling = SamplingParams(
            max_tokens=256,
            temperature=self.cfg.eval_temperature,
            top_p=1.0,
        )

        elapsed = time.perf_counter() - start
        log.info("  vLLM engine loaded in %.1fs", elapsed)

    def _load_hf(self) -> None:
        """Fallback: load via HuggingFace transformers."""
        log.info("Loading Judge LLM via HF: %s", self.cfg.model_name)
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

        # Apply torch.compile for faster inference
        self._model = try_torch_compile(self._model, self.gpu_cfg)

        elapsed = time.perf_counter() - start
        log.info("  HF Judge LLM loaded in %.1fs", elapsed)
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
            
        if self._vllm_engine is not None:
            # --- vLLM Engine Path ---
            try:
                outputs = self._vllm_engine.generate(valid_prompts, self._vllm_sampling, use_tqdm=False)
                for batch_idx, valid_idx in enumerate(valid_indices):
                    response = outputs[batch_idx].outputs[0].text
                    scores = _parse_eval_xml(response)
                    if scores:
                        result = {**qa_pairs[valid_idx], "eval_scores": scores}
                        results[valid_idx] = result
                    else:
                        log.debug("  Could not parse eval for %s", qa_pairs[valid_idx].get("qa_id", "?"))
                return results
            except Exception as exc:
                log.warning("  vLLM evaluation failed: %s", exc)
                return results

        # --- HuggingFace Fallback Path ---
        try:
            # Group by length
            length_fn = lambda x: len(x)
            batches = batched_by_length(valid_prompts, length_fn, batch_size=32)
            
            for orig_indices, batch_prompts in batches:
                # Batch tokenize with left-padding
                inputs = self._tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(self._model.device)

                prompt_len = inputs.input_ids.shape[1]

                # Batch generate (deterministic for evaluation)
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=self.cfg.eval_temperature,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                        use_cache=False,  # Fixes DynamicCache error for certain models
                    )

                # Decode and parse each output
                for batch_idx, orig_idx in enumerate(orig_indices):
                    valid_idx = valid_indices[orig_idx]
                    response = self._tokenizer.decode(
                        outputs[batch_idx][prompt_len:],
                        skip_special_tokens=True,
                    )

                    scores = _parse_eval_xml(response)
                    if scores:
                        result = {**qa_pairs[valid_idx], "eval_scores": scores}
                        results[valid_idx] = result
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

                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=self.cfg.eval_temperature,
                        do_sample=False,
                        pad_token_id=self._tokenizer.pad_token_id,
                        use_cache=False,  # Fixes DynamicCache error
                    )

                response = self._tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                scores = _parse_eval_xml(response)
                if scores:
                    result = {**qa_pairs[valid_idx], "eval_scores": scores}
                    results[valid_idx] = result

            except Exception as exc:
                log.warning("  Sequential eval failed for %s: %s",
                            qa_pairs[valid_idx].get("qa_id", "?"), exc)

        return results

    def unload(self) -> None:
        """Release GPU memory."""
        if self._vllm_engine is not None:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            del self._vllm_engine
            self._vllm_engine = None
			
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
        "taxonomy_level": "Understand",  # Default fallback
        "justification": "",
        "overall": 0.0,
    }

    g_match = re.search(r"<groundedness>([\s\S]*?)</groundedness>", text, re.IGNORECASE)
    m_match = re.search(r"<multimodal_alignment>([\s\S]*?)</multimodal_alignment>", text, re.IGNORECASE)
    l_match = re.search(r"<legal_fluency>([\s\S]*?)</legal_fluency>", text, re.IGNORECASE)
    t_match = re.search(r"<taxonomy_level>([\s\S]*?)</taxonomy_level>", text, re.IGNORECASE)
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
            
        if t_match:
            scores["taxonomy_level"] = t_match.group(1).strip().capitalize()

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
                bloom_level=qa.get("bloom_level", ""),
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
    no_filter: bool = False,
    skip_judge: bool = False,
    allow_same_model: bool = False,
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
    if skip_judge and not no_filter:
        log.warning("--skip-judge implies --no-filter; enabling no_filter automatically.")
        no_filter = True

    # [Paper Note — Anti-bias Guard]
    # Warn (not raise) if generator and judge are from the same family,
    # because in a low-resource setting the user may intentionally accept
    # same-family evaluation as a baseline ablation.
    # The cross-family judge (gemma-2-2b-it) is the DEFAULT and recommended path.
    if not skip_judge and not allow_same_model:
        gen_family = llm_cfg.model_name.split("/")[0].lower() if "/" in llm_cfg.model_name else ""
        judge_family = (llm_cfg.model_name if allow_same_model else CFG.evaluation.judge_model_name).split("/")[0].lower()
        # Use the actual judge model being loaded
        active_judge = getattr(llm_cfg, "model_name", CFG.evaluation.judge_model_name)
        active_gen   = CFG.llm.model_name
        active_gen_family   = active_gen.split("/")[0].lower()   if "/" in active_gen   else active_gen.lower()
        active_judge_family = active_judge.split("/")[0].lower() if "/" in active_judge else active_judge.lower()
        if active_gen_family == active_judge_family:
            log.warning(
                "[Paper Note] Generator ('%s', family='%s') and Judge ('%s', family='%s') "
                "appear to be from the SAME model family. This may inflate evaluation scores "
                "due to self-reinforcement bias. Use --allow-same-model to suppress this warning "
                "or set judge_model_name to a different family (e.g., google/gemma-2-2b-it).",
                active_gen, active_gen_family, active_judge, active_judge_family,
            )

    # Load raw QA pairs
    raw_pairs = load_json(paths.raw_qa_pairs)
    if limit:
        raw_pairs = raw_pairs[:limit]
        log.info("Limited to %d QA pairs", limit)

    # ── Separate cached vs. new QA pairs ──
    new_pairs = []
    evaluated: list[dict[str, Any]] = []

    if skip_judge:
        log.info("Skipping judge evaluation (--skip-judge). Using raw QA pairs directly.")
        evaluated = raw_pairs
    else:
        for qa in raw_pairs:
            qa_id = qa.get("qa_id", "unknown")
            drive_qa_eval = drive_cfg.evaluated_qa_dir / f"{qa_id}.json"

            saved_eval = load_drive_checkpoint(drive_qa_eval, drive_cfg)
            if saved_eval is not None:
                evaluated.append(saved_eval)
            else:
                new_pairs.append(qa)

    if not skip_judge:
        if not new_pairs:
            log.info("All %d QA pairs already evaluated (loaded from Drive cache)", len(raw_pairs))
        else:
            log.info(
                "Evaluating %d new QA pairs (batch_size=%d), %d cached",
                len(new_pairs), eval_cfg.batch_size, len(raw_pairs) - len(new_pairs),
            )

            # Prepare async writer if configured
            async_writer = None
            if getattr(gpu_cfg, "async_drive_io", False):
                async_writer = AsyncDriveWriter(max_workers=2)

            # ── Stage 4: Batched Evaluate ──
            judge = QAJudge(llm_cfg, eval_cfg, gpu_cfg)
            judge._load()
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
                        if async_writer:
                            async_writer.submit(qa_result, drive_qa_eval, drive_cfg)
                        else:
                            sync_json_to_drive(qa_result, drive_qa_eval, drive_cfg)

                # Periodic VRAM cleanup
                if batch_count % gpu_cfg.empty_cache_interval == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if async_writer:
                async_writer.flush()

            judge.unload()

        log.info("  Successfully evaluated %d / %d pairs", len(evaluated), len(raw_pairs))

    # Filter below-threshold samples
    filtered = evaluated if no_filter else filter_qa_pairs(evaluated, eval_cfg)
    if no_filter:
        log.info("Skipping filter (--no-filter). Keeping all %d records.", len(filtered))

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

    # ── Stage 6: Academic Summary Report ──
    _print_academic_summary(evaluated, filtered)

    return records


def _print_academic_summary(evaluated: list[dict[str, Any]], filtered: list[dict[str, Any]]) -> None:
    """Print a structured report to populate paper tables (Table 1, Table 2)."""
    if not evaluated:
        return

    total = len(evaluated)
    passed = len(filtered)
    
    # 1. Main Metrics (Table 1)
    avg_g = sum(q.get("eval_scores", {}).get("groundedness", 0) for q in evaluated) / total
    avg_f = sum(q.get("eval_scores", {}).get("legal_fluency", 0) for q in evaluated) / total
    avg_a = sum(q.get("eval_scores", {}).get("multimodal_alignment", 0) for q in evaluated) / total
    
    # 2. Taxonomy Distribution
    bloom_counts = {"Remember": 0, "Understand": 0, "Apply": 0}
    for q in evaluated:
        lvl = q.get("eval_scores", {}).get("taxonomy_level", "Understand")
        if lvl in bloom_counts:
            bloom_counts[lvl] += 1
        else:
            bloom_counts["Understand"] += 1 # fallback

    # 3. Error Taxonomy (Table 2)
    rejected = [q for q in evaluated if q not in filtered]
    error_counts = {
        "Legal Hallucination": 0,
        "Visual Misalignment": 0,
        "Logical Fallacy": 0,
        "Semantic Overlap": 0,
        "Other": 0
    }
    
    for q in rejected:
        scores = q.get("eval_scores", {})
        just = scores.get("justification", "").lower()
        
        if scores.get("groundedness", 1.0) < 1.0:
            error_counts["Legal Hallucination"] += 1
        elif scores.get("multimodal_alignment", 1.0) < 1.0:
            error_counts["Visual Misalignment"] += 1
        elif scores.get("legal_fluency", 1.0) < 1.0:
            error_counts["Logical Fallacy"] += 1
        else:
            error_counts["Other"] += 1

    print("\n" + "="*60)
    print("           ACADEMIC SUMMARY REPORT (FOR PAPER)")
    print("="*60)
    print(f"Total Evaluated: {total}")
    print(f"Final Passed:    {passed} ({passed/total*100:.1f}%)")
    print("-" * 60)
    print(f"TABLE 1: MAIN METRICS (Averages)")
    print(f"  - Groundedness:      {avg_g*100:.1f}%")
    print(f"  - Legal Fluency:     {avg_f*100:.1f}%")
    print(f"  - MM Alignment:      {avg_a*100:.1f}%")
    print("-" * 60)
    print(f"BLOOM TAXONOMY DISTRIBUTION")
    for k, v in bloom_counts.items():
        print(f"  - {k:12}: {v:4} ({v/total*100:.1f}%)")
    print("-" * 60)
    print(f"TABLE 2: ERROR TAXONOMY (Rejected Samples: {len(rejected)})")
    if len(rejected) > 0:
        for k, v in error_counts.items():
            print(f"  - {k:20}: {v:4} ({v/len(rejected)*100:.1f}%)")
    print("="*60 + "\n")


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
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override judge model name (for cross-judge experiments)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Ablation/baseline: skip threshold filtering",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Baseline: skip judge evaluation and format raw QA directly",
    )
    parser.add_argument(
        "--disable-multimodal-alignment",
        action="store_true",
        help="Ablation: disable multimodal alignment criterion in filtering",
    )
    parser.add_argument(
        "--allow-same-model",
        action="store_true",
        help="Allow using the same model for generation and judge (not recommended).",
    )
    parser.add_argument("--groundedness-threshold", type=float, default=None)
    parser.add_argument("--multimodal-threshold", type=float, default=None)
    parser.add_argument("--legal-fluency-threshold", type=float, default=None)
    parser.add_argument("--overall-threshold", type=float, default=None)
    args = parser.parse_args()

    log.info("Stage 4 -- Evaluation (LLM-as-a-Judge, Batched)")

    from dataclasses import replace

    llm_cfg = replace(CFG.llm, model_name=CFG.evaluation.judge_model_name)
    eval_cfg = CFG.evaluation

    if args.model_name:
        llm_cfg = replace(llm_cfg, model_name=args.model_name)
    if args.batch_size:
        eval_cfg = replace(eval_cfg, batch_size=args.batch_size)
    if args.disable_multimodal_alignment:
        eval_cfg = replace(eval_cfg, multimodal_alignment_threshold=0.0)
    if args.groundedness_threshold is not None:
        eval_cfg = replace(eval_cfg, groundedness_threshold=args.groundedness_threshold)
    if args.multimodal_threshold is not None:
        eval_cfg = replace(eval_cfg, multimodal_alignment_threshold=args.multimodal_threshold)
    if args.legal_fluency_threshold is not None:
        eval_cfg = replace(eval_cfg, legal_fluency_threshold=args.legal_fluency_threshold)
    if args.overall_threshold is not None:
        eval_cfg = replace(eval_cfg, overall_threshold=args.overall_threshold)

    results = run_evaluation(
        limit=args.limit,
        llm_cfg=llm_cfg,
        eval_cfg=eval_cfg,
        no_filter=args.no_filter,
        skip_judge=args.skip_judge,
        allow_same_model=args.allow_same_model,
    )

    if not results:
        log.warning("No records survived evaluation. Check thresholds or raw QA quality.")
        sys.exit(1)

    log.info("Final dataset: %d records", len(results))


if __name__ == "__main__":
    main()
