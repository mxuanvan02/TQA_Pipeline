"""
Stage 3 — Synthetic Question-Answer Generation (QAG)
=====================================================
Who:    Qwen2.5-0.5B-Instruct (or 1.5B) loaded in 4-bit.
Where:  Reads data/interim/multimodal_contexts.json
        Writes data/interim/raw_qa_pairs.json
How:    For each context chunk, prompts the LLM to generate QA pairs at
        three Bloom's Taxonomy levels using Legal Syllogism reasoning.
        Uses BATCHED inference to maximize GPU L4 utilization.
Input:  multimodal_contexts.json (list of chunk records from Stage 2).
Output: raw_qa_pairs.json (list of QA records with rationale).

GPU Optimization (L4 — 22.5 GB VRAM):
    - 4-bit Qwen-0.5B uses ~400 MB VRAM → massive headroom for batching
    - Batch size 16: tokenize + generate 16 prompts simultaneously
    - Left-padding for causal LM batched generation
    - ThreadPool prefetch for I/O-bound prompt preparation
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from src.config import CFG, DriveBackupConfig, GPUOptConfig, LLMConfig, PathConfig, QAGConfig
from src.utils import (
    AsyncDriveWriter,
    batched,
    batched_by_length,
    get_logger,
    load_drive_checkpoint,
    load_json,
    log_gpu_memory,
    save_json,
    setup_tokenizer_for_batch,
    sync_json_to_drive,
    sync_to_drive,
    try_torch_compile,
    verify_drive_mount,
)

log = get_logger("03_qag_generator")


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert Vietnamese Civil Law examiner and professor.
Your task is to create high-quality exam questions from legal textbook content.
You MUST respond using predefined XML tags. Do NOT use markdown code blocks or JSON.
"""

QA_GENERATION_TEMPLATE = """\
Based on the following legal textbook content, generate {n_questions} question(s) \
at the Bloom's Taxonomy level: **{bloom_level}**.

## Bloom's Taxonomy Level Definitions:
- **Remember**: Factual recall — who, what, when, where. Direct retrieval from text.
- **Understand**: Explain, summarize, compare concepts. Demonstrate comprehension.
- **Apply**: Use legal knowledge in a new scenario or case study.

## Context:
{context_text}

{visual_context}

## Requirements:
1. Each question MUST be answerable from the given context.
2. Provide exactly 4 candidate answers (A, B, C, D) — one correct.
3. Structure the rationale using **Legal Syllogism**:
   - **Major Premise**: The general legal rule/article.
   - **Minor Premise**: The specific facts of the question scenario.
   - **Conclusion**: The logical deduction from the premises.
4. Questions and answers must be in Vietnamese.

## Output Format (strict XML tags):
For each question, wrap the output EXACTLY like this:

<qa_pair>
<question>Câu hỏi thi ...</question>
<candidate_answers>
A. ...
B. ...
C. ...
D. ...
</candidate_answers>
<ground_truth>A. ... (full correct answer text)</ground_truth>
<legal_rationale>Đại tiền đề: ... Tiểu tiền đề: ... Kết luận: ...</legal_rationale>
</qa_pair>

Respond with ONLY the XML tags. No explanations before or after.
"""


# ─────────────────────────────────────────────
# LLM Loader (with Batch Support)
# ─────────────────────────────────────────────
class QAGenerator:
    """
    Lazy-loaded LLM for generating QA pairs.

    Primary:  vLLM offline inference engine (PagedAttention, optimal GPU sat).
    Fallback: HuggingFace generate() with batched evaluation.
    """

    def __init__(self, cfg: LLMConfig, gpu_cfg: GPUOptConfig | None = None) -> None:
        self.cfg = cfg
        self.gpu_cfg = gpu_cfg or CFG.gpu
        self._model = None
        self._tokenizer = None
        self._vllm_engine = None
        self._vllm_sampling = None

    def _load(self) -> None:
        """Lazy-load the LLM on first use."""
        if self._model is not None or self._vllm_engine is not None:
            return

        if getattr(self.cfg, "use_vllm", False):
            try:
                self._load_vllm()
                return
            except Exception as e:
                log.warning("Failed to load vLLM (%s). Falling back to HuggingFace.", e)
                self.cfg = type(self.cfg)(**{**self.cfg.__dict__, "use_vllm": False})

        self._load_hf()

    def _load_vllm(self) -> None:
        """Load the model using vLLM for high-throughput inference."""
        log.info("Loading LLM via vLLM: %s (4-bit=%s)", self.cfg.model_name, self.cfg.load_in_4bit)
        start = time.perf_counter()

        from vllm import LLM, SamplingParams

        quantization = "awq" if self.cfg.load_in_4bit else None
        
        self._vllm_engine = LLM(
            model=self.cfg.model_name,
            quantization=quantization,
            max_model_len=3072,  # 2048 input + 1024 output
            gpu_memory_utilization=getattr(self.gpu_cfg, "vllm_gpu_utilization", 0.90),
            trust_remote_code=True,
            enforce_eager=False,
        )
        
        # We still need tokenizer for building prompts
        self._tokenizer = self._vllm_engine.get_tokenizer()
        
        self._vllm_sampling = SamplingParams(
            max_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            repetition_penalty=self.cfg.repetition_penalty,
        )

        elapsed = time.perf_counter() - start
        log.info("  vLLM engine loaded in %.1fs", elapsed)

    def _load_hf(self) -> None:
        """Fallback: load via HuggingFace transformers."""
        log.info("Loading LLM via HF: %s (4-bit=%s)", self.cfg.model_name, self.cfg.load_in_4bit)
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

        # Configure tokenizer for batch inference (left-padding for causal LM)
        setup_tokenizer_for_batch(self._tokenizer)

        # Apply torch.compile for faster inference
        self._model = try_torch_compile(self._model, self.gpu_cfg)

        elapsed = time.perf_counter() - start
        log.info("  HF LLM loaded in %.1fs", elapsed)
        log_gpu_memory("QAGenerator loaded")

    def _build_prompt(
        self, context: dict[str, Any], bloom_level: str, n_questions: int = 1
    ) -> str:
        """Build a fully-formatted prompt string for one context + bloom level."""
        visual_ctx = ""
        if context.get("visual_descriptions"):
            descs = [d.get("summary", "") for d in context["visual_descriptions"]]
            visual_ctx = "## Visual Information:\n" + "\n".join(
                f"- Image: {d}" for d in descs if d
            )

        user_prompt = QA_GENERATION_TEMPLATE.format(
            n_questions=n_questions,
            bloom_level=bloom_level,
            context_text=context["text"][:2000],
            visual_context=visual_ctx,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_qa(
        self,
        context: dict[str, Any],
        bloom_level: str,
        n_questions: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Generate QA pairs for a single context chunk at a given Bloom level.
        (Kept for backward compatibility, uses batch-of-1 internally.)

        Input:  Context dict + Bloom level.
        Output: List of QA dicts.
        """
        results = self.generate_qa_batch([context], bloom_level, n_questions)
        return results[0] if results else []

    def generate_qa_batch(
        self,
        contexts: list[dict[str, Any]],
        bloom_level: str,
        n_questions: int = 1,
    ) -> list[list[dict[str, Any]]]:
        """
        Batch-generate QA pairs for multiple contexts at a given Bloom level.

        Who:    The loaded LLM (with left-padding for batch inference).
        Input:  List of context dicts + single Bloom level.
        Output: List of lists — one list of QA dicts per context.

        GPU Optimization:
            - Tokenizes all prompts together with padding
            - Runs model.generate() once for the entire batch
            - L4 with 4-bit Qwen-0.5B can handle batch_size=16-32 easily
        """
        self._load()

        # Build all prompt strings
        prompt_texts = []
        for ctx in contexts:
            try:
                prompt_texts.append(self._build_prompt(ctx, bloom_level, n_questions))
            except Exception as exc:
                log.warning("  Prompt build failed for %s: %s",
                            ctx.get("chunk_id", "?"), exc)
                prompt_texts.append(None)

        # Filter out failed prompts, remember indices
        valid_indices = [i for i, p in enumerate(prompt_texts) if p is not None]
        valid_prompts = [prompt_texts[i] for i in valid_indices]

        if not valid_prompts:
            return [[] for _ in contexts]
            
        if self._vllm_engine is not None:
            # --- vLLM Engine Path ---
            try:
                outputs = self._vllm_engine.generate(valid_prompts, self._vllm_sampling, use_tqdm=False)
                
                all_results: list[list[dict[str, Any]]] = [[] for _ in contexts]
                for batch_idx, valid_idx in enumerate(valid_indices):
                    response = outputs[batch_idx].outputs[0].text
                    all_results[valid_idx] = _parse_qa_xml(response)
                    
                return all_results
            except Exception as exc:
                log.warning("  vLLM generation failed: %s", exc)
                return [[] for _ in contexts]

        # --- HuggingFace Fallback Path ---
        try:
            # Group by length to reduce padding waste
            length_fn = lambda x: len(x)
            batches = batched_by_length(valid_prompts, length_fn, batch_size=16)
            
            all_results: list[list[dict[str, Any]]] = [[] for _ in contexts]
            
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

                # Batch generate
                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.max_new_tokens,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        do_sample=True,
                        repetition_penalty=self.cfg.repetition_penalty,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                # Decode each output in the batch
                for batch_idx, orig_idx in enumerate(orig_indices):
                    valid_idx = valid_indices[orig_idx]
                    response = self._tokenizer.decode(
                        outputs[batch_idx][prompt_len:],
                        skip_special_tokens=True,
                    )
                    all_results[valid_idx] = _parse_qa_xml(response)

            return all_results

        except Exception as exc:
            log.warning("  Batch QA generation failed: %s", exc)
            # Fallback: try one-by-one
            log.info("  Falling back to sequential generation...")
            return self._generate_sequential_fallback(
                contexts, valid_indices, valid_prompts, bloom_level
            )

    def _generate_sequential_fallback(
        self,
        contexts: list[dict[str, Any]],
        valid_indices: list[int],
        valid_prompts: list[str],
        bloom_level: str,
    ) -> list[list[dict[str, Any]]]:
        """Fallback: generate one-by-one if batch fails (e.g., OOM)."""
        all_results: list[list[dict[str, Any]]] = [[] for _ in contexts]

        for batch_idx, valid_idx in enumerate(valid_indices):
            try:
                inputs = self._tokenizer(
                    valid_prompts[batch_idx],
                    return_tensors="pt",
                ).to(self._model.device)

                with torch.inference_mode():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.max_new_tokens,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                        do_sample=True,
                        repetition_penalty=self.cfg.repetition_penalty,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                response = self._tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )
                all_results[valid_idx] = _parse_qa_xml(response)

            except Exception as exc:
                log.warning("  Sequential fallback failed for %s / %s: %s",
                            contexts[valid_idx].get("chunk_id", "?"), bloom_level, exc)

        return all_results

    def unload(self) -> None:
        """Release GPU memory."""
        if self._vllm_engine is not None:
            # vLLM occupies memory differently
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
        log.info("  LLM unloaded, GPU memory freed")


# ─────────────────────────────────────────────
# XML Parsing
# ─────────────────────────────────────────────
def _parse_qa_xml(text: str) -> list[dict[str, Any]]:
    """
    Extract QA pairs from LLM XML output via Regex.

    How:  Finds all <qa_pair> blocks and extracts nested tags.
          More stable than JSON parsing for 0.5B models.
    """
    pairs: list[dict[str, Any]] = []

    # Extract blocks
    blocks = re.findall(r"<qa_pair>(.*?)</qa_pair>", text, re.DOTALL | re.IGNORECASE)

    # If no <qa_pair> tags exist, perhaps it just spit out the tags directly
    if not blocks:
        blocks = [text]

    for block in blocks:
        q_match = re.search(r"<question>(.*?)</question>", block, re.DOTALL | re.IGNORECASE)
        ca_match = re.search(r"<candidate_answers>(.*?)</candidate_answers>", block, re.DOTALL | re.IGNORECASE)
        gt_match = re.search(r"<ground_truth>(.*?)</ground_truth>", block, re.DOTALL | re.IGNORECASE)
        lr_match = re.search(r"<legal_rationale>(.*?)</legal_rationale>", block, re.DOTALL | re.IGNORECASE)

        if q_match and ca_match and gt_match and lr_match:
            # Parse candidate answers string into a list
            ca_text = ca_match.group(1).strip()
            lines = [ln.strip() for ln in ca_text.split("\n") if ln.strip()]
            candidates = [ln for ln in lines if re.match(r"^[A-E][\.).]", ln)]

            # Fallback if candidates failed to parse as A. B. C. D.
            if len(candidates) < 2:
                candidates = lines

            pairs.append({
                "question": q_match.group(1).strip(),
                "candidate_answers": candidates,
                "ground_truth": gt_match.group(1).strip(),
                "legal_rationale": lr_match.group(1).strip(),
            })

    if not pairs:
        log.debug("  Could not parse QA XML: %s...", text[:100])

    return pairs


# ─────────────────────────────────────────────
# Pipeline Orchestration (Batched)
# ─────────────────────────────────────────────
def run_qag(
    paths: PathConfig | None = None,
    llm_cfg: LLMConfig | None = None,
    qag_cfg: QAGConfig | None = None,
    drive_cfg: DriveBackupConfig | None = None,
    gpu_cfg: GPUOptConfig | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate QA pairs for all contexts across all Bloom levels.

    Who:    QAGenerator (batched LLM wrapper).
    Where:  data/interim/multimodal_contexts.json -> data/interim/raw_qa_pairs.json
    How:    Batches contexts together, generates N questions per batch per Bloom level.
    Input:  multimodal_contexts.json from Stage 2.
    Output: raw_qa_pairs.json — enriched QA records.

    GPU Optimization:
        - Processes contexts in batches of qag_cfg.batch_size (default 16)
        - Each batch generates for one Bloom level at a time
        - Drive checkpointing per-chunk (survives Colab disconnects)
    """
    paths = paths or CFG.paths
    llm_cfg = llm_cfg or CFG.llm
    qag_cfg = qag_cfg or CFG.qag
    drive_cfg = drive_cfg or CFG.drive_backup
    gpu_cfg = gpu_cfg or CFG.gpu

    # Pre-flight: verify Drive mount
    verify_drive_mount(drive_cfg)

    # Load contexts
    contexts = load_json(paths.multimodal_contexts)
    if limit:
        contexts = contexts[:limit]
        log.info("Limited to %d contexts", limit)

    # ── Separate cached vs. new contexts ──
    new_contexts = []
    all_qa_pairs: list[dict[str, Any]] = []

    for ctx in contexts:
        chunk_id = ctx.get("chunk_id", "unknown")
        drive_chunk_qa = drive_cfg.qa_chunks_dir / f"{chunk_id}.json"

        saved_qa = load_drive_checkpoint(drive_chunk_qa, drive_cfg)
        if saved_qa is not None:
            all_qa_pairs.extend(saved_qa)
            log.info("  Loaded cached QA pairs from Drive: %s", chunk_id)
        else:
            new_contexts.append(ctx)

    if not new_contexts:
        log.info("All %d contexts already processed (loaded from Drive cache)", len(contexts))
        save_json(all_qa_pairs, paths.raw_qa_pairs)
        return all_qa_pairs

    log.info(
        "Processing %d new contexts (batch_size=%d), %d cached",
        len(new_contexts), qag_cfg.batch_size, len(contexts) - len(new_contexts),
    )

    generator = QAGenerator(llm_cfg, gpu_cfg)
    generator._load()  # Pre-load LLM + tokenizer (needed by _build_prompt)
    batch_count = 0
    total_batches = (len(new_contexts) + qag_cfg.batch_size - 1) // qag_cfg.batch_size

    # Prepare async writer if configured
    async_writer = None
    if getattr(gpu_cfg, "async_drive_io", False):
        async_writer = AsyncDriveWriter(max_workers=2)

    # Index for O(1) checkpoint lookups (replaces O(N*K) list scan)
    qa_by_chunk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa in all_qa_pairs:
        qa_by_chunk[qa.get("chunk_id", "unknown")].append(qa)

    # ── Process in batches ──
    for batch_contexts in tqdm(
        batched(new_contexts, qag_cfg.batch_size),
        total=total_batches,
        desc="QAG Batches",
    ):
        batch_count += 1

        # Log GPU usage periodically
        if batch_count % gpu_cfg.log_gpu_interval == 0:
            log_gpu_memory(f"QAG batch {batch_count}/{total_batches}")

        # ── Merged Bloom: build ALL prompts (contexts × levels) in one batch ──
        if qag_cfg.merge_bloom_levels:
            # Build all prompts upfront: N contexts × 3 levels = 3N prompts
            all_prompts: list[str | None] = []
            prompt_map: list[tuple[int, str]] = []  # (ctx_idx, bloom_level)

            for ctx_idx, ctx in enumerate(batch_contexts):
                for bloom_level in qag_cfg.bloom_levels:
                    try:
                        all_prompts.append(
                            generator._build_prompt(ctx, bloom_level, qag_cfg.questions_per_level)
                        )
                    except Exception:
                        all_prompts.append(None)
                    prompt_map.append((ctx_idx, bloom_level))

            # Filter valid prompts
            valid_indices = [i for i, p in enumerate(all_prompts) if p is not None]
            valid_prompts = [all_prompts[i] for i in valid_indices]

            if valid_prompts:
                try:
                    if generator._vllm_engine is not None:
                        # vLLM generation
                        outputs = generator._vllm_engine.generate(valid_prompts, generator._vllm_sampling, use_tqdm=False)
                        for out_idx, valid_idx in enumerate(valid_indices):
                            ctx_idx, bloom_level = prompt_map[valid_idx]
                            response = outputs[out_idx].outputs[0].text
                            qa_list = _parse_qa_xml(response)
                            ctx = batch_contexts[ctx_idx]
                            chunk_id = ctx.get("chunk_id", "unknown")
                            doc_id = ctx.get("doc_id", "unknown")

                            for i, qa in enumerate(qa_list):
                                qa_id = f"{chunk_id}_{bloom_level.lower()}_{i}"
                                enriched = {
                                    "qa_id": qa_id,
                                    "doc_id": doc_id,
                                    "chunk_id": chunk_id,
                                    "domain_tag": "civil_law",
                                    "bloom_level": bloom_level,
                                    "context_text": ctx.get("text", ""),
                                    "context_visuals": ctx.get("image_paths", []),
                                    "visual_descriptions": ctx.get("visual_descriptions", []),
                                    "is_multimodal": ctx.get("is_multimodal", False),
                                    "question_content": qa.get("question", ""),
                                    "candidate_answers": qa.get("candidate_answers", []),
                                    "ground_truth": qa.get("ground_truth", ""),
                                    "legal_rationale": qa.get("legal_rationale", ""),
                                }
                                all_qa_pairs.append(enriched)
                                qa_by_chunk[chunk_id].append(enriched)
                    else:
                        # HF Generation (with grouping by length)
                        length_fn = lambda x: len(x)
                        batches = batched_by_length(valid_prompts, length_fn, batch_size=16)
                        
                        for orig_indices, batch_prompts in batches:
                            inputs = generator._tokenizer(
                                batch_prompts,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=2048,
                            ).to(generator._model.device)

                            prompt_len = inputs.input_ids.shape[1]

                            with torch.inference_mode():
                                outputs = generator._model.generate(
                                    **inputs,
                                    max_new_tokens=llm_cfg.max_new_tokens,
                                    temperature=llm_cfg.temperature,
                                    top_p=llm_cfg.top_p,
                                    do_sample=True,
                                    repetition_penalty=llm_cfg.repetition_penalty,
                                    pad_token_id=generator._tokenizer.pad_token_id,
                                )

                            # Decode
                            for batch_idx, orig_idx in enumerate(orig_indices):
                                valid_idx = valid_indices[orig_idx]
                                ctx_idx, bloom_level = prompt_map[valid_idx]
                                
                                response = generator._tokenizer.decode(
                                    outputs[batch_idx][prompt_len:],
                                    skip_special_tokens=True,
                                )
                                qa_list = _parse_qa_xml(response)
                                ctx = batch_contexts[ctx_idx]
                                chunk_id = ctx.get("chunk_id", "unknown")
                                doc_id = ctx.get("doc_id", "unknown")

                                for i, qa in enumerate(qa_list):
                                    qa_id = f"{chunk_id}_{bloom_level.lower()}_{i}"
                                    enriched = {
                                        "qa_id": qa_id,
                                        "doc_id": doc_id,
                                        "chunk_id": chunk_id,
                                        "domain_tag": "civil_law",
                                        "bloom_level": bloom_level,
                                        "context_text": ctx.get("text", ""),
                                        "context_visuals": ctx.get("image_paths", []),
                                        "visual_descriptions": ctx.get("visual_descriptions", []),
                                        "is_multimodal": ctx.get("is_multimodal", False),
                                        "question_content": qa.get("question", ""),
                                        "candidate_answers": qa.get("candidate_answers", []),
                                        "ground_truth": qa.get("ground_truth", ""),
                                        "legal_rationale": qa.get("legal_rationale", ""),
                                    }
                                    all_qa_pairs.append(enriched)
                                    qa_by_chunk[chunk_id].append(enriched)

                except Exception as exc:
                    log.warning("  Merged bloom batch failed: %s, falling back to per-level", exc)
                    # Fallback to per-level generation
                    for bloom_level in qag_cfg.bloom_levels:
                        batch_results = generator.generate_qa_batch(
                            contexts=batch_contexts,
                            bloom_level=bloom_level,
                            n_questions=qag_cfg.questions_per_level,
                        )
                        for ctx_idx, (ctx, qa_list) in enumerate(zip(batch_contexts, batch_results)):
                            chunk_id = ctx.get("chunk_id", "unknown")
                            doc_id = ctx.get("doc_id", "unknown")
                            for i, qa in enumerate(qa_list):
                                qa_id = f"{chunk_id}_{bloom_level.lower()}_{i}"
                                enriched = {
                                    "qa_id": qa_id, "doc_id": doc_id,
                                    "chunk_id": chunk_id, "domain_tag": "civil_law",
                                    "bloom_level": bloom_level,
                                    "context_text": ctx.get("text", ""),
                                    "context_visuals": ctx.get("image_paths", []),
                                    "visual_descriptions": ctx.get("visual_descriptions", []),
                                    "is_multimodal": ctx.get("is_multimodal", False),
                                    "question_content": qa.get("question", ""),
                                    "candidate_answers": qa.get("candidate_answers", []),
                                    "ground_truth": qa.get("ground_truth", ""),
                                    "legal_rationale": qa.get("legal_rationale", ""),
                                }
                                all_qa_pairs.append(enriched)
                                qa_by_chunk[chunk_id].append(enriched)
        else:
            # Original per-level generation (merge_bloom_levels=False)
            for bloom_level in qag_cfg.bloom_levels:
                batch_results = generator.generate_qa_batch(
                    contexts=batch_contexts,
                    bloom_level=bloom_level,
                    n_questions=qag_cfg.questions_per_level,
                )
                for ctx_idx, (ctx, qa_list) in enumerate(zip(batch_contexts, batch_results)):
                    chunk_id = ctx.get("chunk_id", "unknown")
                    doc_id = ctx.get("doc_id", "unknown")
                    for i, qa in enumerate(qa_list):
                        qa_id = f"{chunk_id}_{bloom_level.lower()}_{i}"
                        enriched = {
                            "qa_id": qa_id, "doc_id": doc_id,
                            "chunk_id": chunk_id, "domain_tag": "civil_law",
                            "bloom_level": bloom_level,
                            "context_text": ctx.get("text", ""),
                            "context_visuals": ctx.get("image_paths", []),
                            "visual_descriptions": ctx.get("visual_descriptions", []),
                            "is_multimodal": ctx.get("is_multimodal", False),
                            "question_content": qa.get("question", ""),
                            "candidate_answers": qa.get("candidate_answers", []),
                            "ground_truth": qa.get("ground_truth", ""),
                            "legal_rationale": qa.get("legal_rationale", ""),
                        }
                        all_qa_pairs.append(enriched)
                        qa_by_chunk[chunk_id].append(enriched)

        # ── Per-chunk Drive checkpointing (O(1) lookup via dict) ──
        for ctx in batch_contexts:
            chunk_id = ctx.get("chunk_id", "unknown")
            chunk_qa = qa_by_chunk.get(chunk_id, [])
            if chunk_qa:
                drive_chunk_qa = drive_cfg.qa_chunks_dir / f"{chunk_id}.json"
                if async_writer:
                    async_writer.submit(
                        chunk_qa, drive_chunk_qa, drive_cfg,
                        label=f"QA pairs for {chunk_id}",
                    )
                else:
                    sync_json_to_drive(
                        chunk_qa, drive_chunk_qa, drive_cfg,
                        label=f"QA pairs for {chunk_id}",
                    )

        # Periodic VRAM cleanup
        if batch_count % gpu_cfg.empty_cache_interval == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Wait for pending checkpoints
    if async_writer:
        async_writer.flush()
        
    # Free GPU
    generator.unload()

    # Save output
    save_json(all_qa_pairs, paths.raw_qa_pairs)

    # Final Backup of aggregated file
    sync_to_drive(
        paths.raw_qa_pairs,
        drive_cfg.backup_base / "interim/raw_qa_pairs.json",
        drive_cfg,
        label="raw_qa_pairs.json",
    )

    log.info(
        "Stage 3 complete: %d QA pairs from %d contexts (batch_size=%d)",
        len(all_qa_pairs),
        len(contexts),
        qag_cfg.batch_size,
    )
    return all_qa_pairs


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate QA pairs via Bloom's Taxonomy + Legal Syllogism"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to N contexts")
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (default: from config)",
    )
    args = parser.parse_args()

    log.info("Stage 3 -- Synthetic QAG Pipeline (Batched)")

    # Override batch size from CLI if provided
    if args.batch_size:
        from dataclasses import replace
        qag_cfg = replace(CFG.qag, batch_size=args.batch_size)
        results = run_qag(limit=args.limit, qag_cfg=qag_cfg)
    else:
        results = run_qag(limit=args.limit)

    if not results:
        log.warning("No QA pairs generated. Check multimodal_contexts.json.")
        sys.exit(1)

    log.info("Generated %d raw QA pairs", len(results))


if __name__ == "__main__":
    main()
