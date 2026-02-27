"""
Stage 3 — Synthetic Question-Answer Generation (QAG)
=====================================================
Who:    Qwen2.5-1.5B-Instruct (or Gemma-2-2B-it) loaded in 4-bit.
Where:  Reads data/interim/multimodal_contexts.json
        Writes data/interim/raw_qa_pairs.json
How:    For each context chunk, prompts the LLM to generate QA pairs at
        three Bloom's Taxonomy levels using Legal Syllogism reasoning.
        Structured JSON output is enforced via prompt engineering.
Input:  multimodal_contexts.json (list of chunk records from Stage 2).
Output: raw_qa_pairs.json (list of QA records with rationale).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import uuid
from typing import Any

import torch
from tqdm import tqdm

from src.config import CFG, LLMConfig, PathConfig, QAGConfig
from src.utils import get_logger, load_json, save_json

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
# LLM Loader
# ─────────────────────────────────────────────
class QAGenerator:
    """
    Lazy-loaded LLM for generating QA pairs.

    Who:    Qwen2.5-1.5B-Instruct (4-bit quantized).
    How:    Loaded once, reused across all chunks. Uses chat template
            with system prompt for consistent formatting.
    """

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the LLM on first use."""
        if self._model is not None:
            return

        log.info("🧠 Loading LLM: %s (4-bit=%s)", self.cfg.model_name, self.cfg.load_in_4bit)
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
        log.info("  ✅ LLM loaded in %.1fs", elapsed)

    def generate_qa(
        self,
        context: dict[str, Any],
        bloom_level: str,
        n_questions: int = 1,
    ) -> list[dict[str, Any]]:
        """
        Generate QA pairs for a single context chunk at a given Bloom level.

        Who:    The loaded LLM.
        Input:  Context dict (from multimodal_contexts.json) + Bloom level.
        Output: List of QA dicts with question, answers, ground_truth, rationale.
        """
        self._load()

        # Build visual context string
        visual_ctx = ""
        if context.get("visual_descriptions"):
            descs = [
                d.get("summary", "") for d in context["visual_descriptions"]
            ]
            visual_ctx = "## Visual Information:\n" + "\n".join(
                f"- Image: {d}" for d in descs if d
            )

        # Fill prompt template
        user_prompt = QA_GENERATION_TEMPLATE.format(
            n_questions=n_questions,
            bloom_level=bloom_level,
            context_text=context["text"][:2000],  # Truncate for context window
            visual_context=visual_ctx,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
                    max_new_tokens=self.cfg.max_new_tokens,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    do_sample=True,
                    repetition_penalty=self.cfg.repetition_penalty,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            return _parse_qa_xml(response)

        except Exception as exc:
            log.warning("  ⚠️  QA generation failed for %s / %s: %s",
                        context.get("chunk_id", "?"), bloom_level, exc)
            return []

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model, self._tokenizer
        self._model = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("🗑️  LLM unloaded, GPU memory freed")


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
            # Split by A., B., C., D. or similar newline bullets
            lines = [ln.strip() for ln in ca_text.split("\n") if ln.strip()]
            candidates = [ln for ln in lines if re.match(r"^[A-E][\.\)]", ln)]
            
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
        log.debug("  ⚠️  Could not parse QA XML: %s...", text[:100])
        
    return pairs


# ─────────────────────────────────────────────
# Pipeline Orchestration
# ─────────────────────────────────────────────
def run_qag(
    paths: PathConfig | None = None,
    llm_cfg: LLMConfig | None = None,
    qag_cfg: QAGConfig | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    Generate QA pairs for all contexts across all Bloom levels.

    Who:    QAGenerator (LLM wrapper).
    Where:  data/interim/multimodal_contexts.json → data/interim/raw_qa_pairs.json
    How:    For each context × Bloom level, generate N questions.
    Input:  multimodal_contexts.json from Stage 2.
    Output: raw_qa_pairs.json — enriched QA records.
    """
    paths = paths or CFG.paths
    llm_cfg = llm_cfg or CFG.llm
    qag_cfg = qag_cfg or CFG.qag

    # Load contexts
    contexts = load_json(paths.multimodal_contexts)
    if limit:
        contexts = contexts[:limit]
        log.info("🔒 Limited to %d contexts", limit)

    generator = QAGenerator(llm_cfg)
    all_qa_pairs: list[dict[str, Any]] = []

    for ctx in tqdm(contexts, desc="Generating QA pairs"):
        chunk_id = ctx.get("chunk_id", "unknown")
        doc_id = ctx.get("doc_id", "unknown")

        for bloom_level in qag_cfg.bloom_levels:
            qa_list = generator.generate_qa(
                context=ctx,
                bloom_level=bloom_level,
                n_questions=qag_cfg.questions_per_level,
            )

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

    generator.unload()

    # Save output
    save_json(all_qa_pairs, paths.raw_qa_pairs)
    log.info(
        "🏁 Stage 3 complete: %d QA pairs from %d contexts",
        len(all_qa_pairs),
        len(contexts),
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
    args = parser.parse_args()

    log.info("🚀 Stage 3 — Synthetic QAG Pipeline")
    results = run_qag(limit=args.limit)

    if not results:
        log.warning("No QA pairs generated. Check multimodal_contexts.json.")
        sys.exit(1)

    log.info("✅ Generated %d raw QA pairs", len(results))


if __name__ == "__main__":
    main()
