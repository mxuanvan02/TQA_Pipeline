"""
Stage 2 — Context Structuring & Representation
================================================
Who:    1) Python regex parser (Markdown chunking).
        2) Chunk quality cleaner (noise/garbage filter).
        3) VLM — Vintern-1B-v3 (image description for Vietnamese content).
Where:  Reads from data/interim/*.md + data/interim/images/
        Writes to  data/interim/multimodal_contexts.json
How:    1. Parse each .md file and split on headers (#, ##, ###) into
           semantic chunks with metadata (doc_id, section title, level).
        2. **Clean chunks**: strip noise lines, filter garbage/TOC chunks.
        3. For each image referenced in a chunk, invoke the VLM to
           produce a JSON description {entities, relationships, summary}.
        4. Fuse text chunk + VLM output into a unified context record.
Input:  .md files + .jpg/.png images from Stage 1.
Output: multimodal_contexts.json — list of context records.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from src.config import CFG, ChunkCleaningConfig, ChunkingConfig, PathConfig, VLMConfig
from src.utils import get_files, get_logger, save_json

log = get_logger("02_structuring")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────
@dataclass
class Chunk:
    """A semantic chunk extracted from a Markdown document."""

    doc_id: str
    chunk_id: str
    section_title: str
    header_level: int
    text: str
    image_paths: list[str] = field(default_factory=list)
    visual_descriptions: list[dict[str, Any]] = field(default_factory=list)
    is_multimodal: bool = False


# ─────────────────────────────────────────────
# 1 · Markdown Chunking
# ─────────────────────────────────────────────
# Regex: find image references in markdown  ![alt](path)
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _build_splitter_regex(cfg: ChunkingConfig) -> re.Pattern[str]:
    """Combine Markdown headers and Vietnamese legal splits into one pattern."""
    parts = []
    # Markdown headers (e.g., ^#{1,6}\s+...)
    parts.append(r"^(#{1,6})\s+(.*)")
    
    # Add legal patterns
    for p in cfg.legal_split_patterns:
        # Note: Legal patterns are matched cleanly
        # Extract the entire match as the 'title' group
        parts.append(f"({p})([\\s\\S]*?)$")
        
    pattern_str = "|".join(parts)
    return re.compile(pattern_str, re.MULTILINE)


def parse_markdown_to_chunks(
    md_text: str,
    doc_id: str,
    image_dir: Path,
    cfg: ChunkingConfig,
) -> list[Chunk]:
    """
    Split Markdown text into semantic chunks based on headers.

    Who:    Pure Python parser (no ML model needed).
    How:    Finds all header positions, slices text between them.
            Discovers image references within each chunk.
    Input:  Raw Markdown string + doc identifier.
    Output: List of Chunk dataclass instances.
    """
    # Find chunk boundaries using combined splitters
    splitter_re = _build_splitter_regex(cfg)
    headers = list(splitter_re.finditer(md_text))

    if not headers:
        # No headers → treat entire document as one chunk
        return [
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_chunk_0",
                section_title="Full Document",
                header_level=0,
                text=md_text.strip(),
                image_paths=_find_images(md_text, image_dir),
                is_multimodal=len(_find_images(md_text, image_dir)) > 0
            )
        ]

    chunks: list[Chunk] = []

    # Include any text before the first header
    if headers[0].start() > 0:
        preamble = md_text[: headers[0].start()].strip()
        img_paths = _find_images(preamble, image_dir)
        if len(preamble) >= cfg.min_chunk_chars or img_paths:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=f"{doc_id}_chunk_0",
                    section_title="Preamble",
                    header_level=0,
                    text=preamble,
                    image_paths=img_paths,
                    is_multimodal=bool(img_paths)
                )
            )

    for i, match in enumerate(headers):
        # We need to figure out which part matched (Markdown vs Legal)
        if match.group(1):  # Matched markdown header
            level = len(match.group(1))
            title = match.group(2).strip()
            # Only split on configured header levels
            if f"{'#' * level}" not in cfg.split_headers:
                continue
        else: # Matched legal pattern
            level = 2 # Treat legal sections like ## (H2) equivalent
            # Find which non-None group matched
            title = "Legal Subsection"
            for g in range(3, match.lastindex + 1):
                if match.group(g):
                    title = match.group(g).split("\n")[0][:50]
                    break

        # Determine chunk boundaries
        start = match.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md_text)
        text = md_text[start:end].strip()

        img_paths = _find_images(text, image_dir)

        # Apply length filters, but exempt multimodal chunks
        if len(text) < cfg.min_chunk_chars and not img_paths:
            continue
        if len(text) > cfg.max_chunk_chars:
            text = text[: cfg.max_chunk_chars] + "\n\n[... truncated ...]"

        chunk_id = f"{doc_id}_chunk_{len(chunks)}"

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                section_title=title,
                header_level=level,
                text=text,
                image_paths=img_paths,
                is_multimodal=len(img_paths) > 0,
            )
        )

    log.info(
        "  📝 %s: %d chunks (%d multimodal)",
        doc_id,
        len(chunks),
        sum(1 for c in chunks if c.is_multimodal),
    )
    return chunks


# ─────────────────────────────────────────────
# 1.5 · Chunk Quality Cleaning
# ─────────────────────────────────────────────
def clean_chunks(
    chunks: list[Chunk],
    cfg: ChunkCleaningConfig,
) -> list[Chunk]:
    """
    Filter and clean chunks to remove noise before VLM/LLM processing.

    Who:    Pure Python heuristics (no ML model).
    Where:  Called between chunking and VLM in the pipeline.
    How:    1. Strip noise lines (page numbers, dot leaders, rules).
            2. Validate minimum word count, special char ratio, sentence count.
            3. Drop TOC-style chunks.
    Input:  Raw chunks from parse_markdown_to_chunks.
    Output: Filtered list of cleaned chunks.
    """
    import re as _re

    noise_matchers = [_re.compile(p) for p in cfg.noise_patterns]
    toc_ref_pattern = _re.compile(r"\.\.+\s*\d+|\d+\s*$")
    cleaned: list[Chunk] = []
    dropped = 0

    for chunk in chunks:
        # --- Step 1: Strip noise lines ---
        lines = chunk.text.splitlines()
        clean_lines: list[str] = []
        for line in lines:
            is_noise = any(m.match(line.strip()) for m in noise_matchers)
            if not is_noise:
                clean_lines.append(line)

        cleaned_text = "\n".join(clean_lines).strip()

        # --- Step 2: Check TOC indicator ---
        toc_matches = sum(1 for ln in clean_lines if toc_ref_pattern.search(ln))
        if toc_matches > cfg.toc_indicator_threshold:
            log.debug("  🗑️  Dropped TOC chunk: %s", chunk.chunk_id)
            dropped += 1
            continue

        # --- Step 3: Validate meaningful content (EXEMPT Multimodal) ---
        if not chunk.is_multimodal:
            # Word count (split on whitespace)
            words = cleaned_text.split()
            if len(words) < cfg.min_meaningful_words:
                log.debug("  🗑️  Dropped short chunk (%d words): %s",
                          len(words), chunk.chunk_id)
                dropped += 1
                continue

            # Special character ratio
            if cleaned_text:
                alpha_chars = sum(1 for c in cleaned_text if c.isalnum() or c.isspace())
                special_ratio = 1.0 - (alpha_chars / len(cleaned_text))
                if special_ratio > cfg.max_special_char_ratio:
                    log.debug("  🗑️  Dropped noisy chunk (%.0f%% special chars): %s",
                              special_ratio * 100, chunk.chunk_id)
                    dropped += 1
                    continue

            # Sentence count
            sentence_endings = len(_re.findall(r"[.?!]\s", cleaned_text + " "))
            if sentence_endings < cfg.min_sentence_count:
                log.debug("  🗑️  Dropped chunk (no complete sentences): %s",
                          chunk.chunk_id)
                dropped += 1
                continue

        # --- Step 4: Update chunk with cleaned text ---
        chunk.text = cleaned_text
        cleaned.append(chunk)

    if dropped:
        log.info("  🧹 Cleaned: kept %d / %d chunks (dropped %d noise/garbage)",
                 len(cleaned), len(chunks), dropped)

    return cleaned


def _find_images(text: str, image_dir: Path) -> list[str]:
    """Extract image paths from Markdown text and resolve against image_dir."""
    refs = _IMAGE_RE.findall(text)
    resolved: list[str] = []

    for ref in refs:
        # Try as absolute, then relative to image_dir
        p = Path(ref)
        if p.exists():
            resolved.append(str(p))
        else:
            candidate = image_dir / p.name
            if candidate.exists():
                resolved.append(str(candidate))

    # Also check for any images in the doc-specific image dir
    if image_dir.exists() and not resolved:
        for img in sorted(image_dir.iterdir()):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                resolved.append(str(img))

    return resolved


# ─────────────────────────────────────────────
# 2 · VLM Image Description
# ─────────────────────────────────────────────
class VLMDescriber:
    """
    Lazy-loaded VLM for generating structured image descriptions.

    Who:    InternVL2-1B (or Qwen2-VL-2B-Instruct).
    How:    Loads model in 4-bit quantization, processes images one by one,
            returns JSON with {entities, relationships, summary}.
    """

    def __init__(self, cfg: VLMConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._tokenizer = None

    def _load(self) -> None:
        """Lazy-load the VLM on first use to save memory."""
        if self._model is not None:
            return

        log.info("🧠 Loading VLM: %s (4-bit=%s)", self.cfg.model_name, self.cfg.load_in_4bit)
        start = time.perf_counter()

        from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

        quantization_config = None
        if self.cfg.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._model = AutoModel.from_pretrained(
            self.cfg.model_name,
            quantization_config=quantization_config,
            device_map=self.cfg.device_map,
            torch_dtype=getattr(torch, self.cfg.torch_dtype),
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, trust_remote_code=True
        )

        elapsed = time.perf_counter() - start
        log.info("  ✅ VLM loaded in %.1fs", elapsed)

    def describe_image(self, image_path: str) -> dict[str, Any]:
        """
        Generate a structured description for a single image.

        Input:  Path to an image file.
        Output: Dict with keys: entities, relationships, summary.
        """
        self._load()

        try:
            image = Image.open(image_path).convert("RGB")

            prompt = (
                "Analyze this image from a Vietnamese Civil Law textbook. "
                "Return a JSON object with these keys:\n"
                '- "entities": list of named entities (articles, legal concepts, people, organizations)\n'
                '- "relationships": list of relationships between entities\n'
                '- "summary": one-sentence description of what the image shows\n'
                "Respond ONLY with valid JSON, no extra text."
            )

            # Build conversation for InternVL2 / Qwen2-VL style
            if "InternVL" in self.cfg.model_name or "Vintern" in self.cfg.model_name:
                return self._describe_internvl(image, prompt)
            else:
                return self._describe_qwen_vl(image, prompt)

        except Exception as exc:
            log.warning("  ⚠️  VLM failed on %s: %s", image_path, exc)
            return {
                "entities": [],
                "relationships": [],
                "summary": f"[Image at {Path(image_path).name}]",
            }

    def _describe_internvl(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        """InternVL2-style inference."""
        pixel_values = _load_internvl_image(image, self._model)

        generation_config = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "temperature": self.cfg.temperature,
            "do_sample": self.cfg.temperature > 0,
        }

        response = self._model.chat(
            self._tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )

        return _parse_vlm_json(response)

    def _describe_qwen_vl(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        """Qwen2-VL-style inference."""
        from transformers import AutoProcessor

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.model_name, trust_remote_code=True
            )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self._model.device)

        ids = self._model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            do_sample=self.cfg.temperature > 0,
        )
        output = self._processor.batch_decode(
            ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0]

        return _parse_vlm_json(output)

    def unload(self) -> None:
        """Release GPU memory."""
        del self._model, self._processor, self._tokenizer
        self._model = self._processor = self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("🗑️  VLM unloaded, GPU memory freed")


def _load_internvl_image(image: Image.Image, model: Any) -> torch.Tensor:
    """
    Prepare an image for InternVL2 input.
    Falls back to simple resize if the model-specific transform is unavailable.
    """
    try:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        return transform(image).unsqueeze(0).to(model.device).to(torch.float16)
    except Exception:
        return image


def _parse_vlm_json(text: str) -> dict[str, Any]:
    """Attempt to parse JSON from VLM output, with fallback."""
    # Try to extract JSON from possible markdown code block
    json_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    json_str = json_match.group(1) if json_match else text

    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError:
        return {
            "entities": [],
            "relationships": [],
            "summary": text.strip()[:200],
        }


# ─────────────────────────────────────────────
# 3 · Fusion Pipeline
# ─────────────────────────────────────────────
def process_documents(
    paths: PathConfig | None = None,
    vlm_cfg: VLMConfig | None = None,
    chunking_cfg: ChunkingConfig | None = None,
    cleaning_cfg: ChunkCleaningConfig | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    End-to-end Stage 2: chunk all MDs and describe images.

    Who:    Parser + VLM pipeline.
    Where:  data/interim/*.md → data/interim/multimodal_contexts.json
    How:    Iterates over each .md, chunks it, runs VLM on images, fuses.
    Input:  Markdown + image files from Stage 1.
    Output: List of context dicts saved as JSON.
    """
    paths = paths or CFG.paths
    vlm_cfg = vlm_cfg or CFG.vlm
    chunking_cfg = chunking_cfg or CFG.chunking
    cleaning_cfg = cleaning_cfg or CFG.chunk_cleaning

    md_files = get_files(paths.interim, extension=".md")
    if not md_files:
        log.warning("⚠️  No .md files found in %s", paths.interim)
        return []

    if limit:
        md_files = md_files[:limit]

    # Initialize VLM (lazy — only loads if images exist)
    vlm = VLMDescriber(vlm_cfg)
    has_images = False

    all_contexts: list[dict[str, Any]] = []

    for md_path in tqdm(md_files, desc="Processing documents"):
        doc_id = md_path.stem
        md_text = md_path.read_text(encoding="utf-8")

        # Resolve per-document image directory
        doc_image_dir = paths.interim_images / doc_id

        # Parse into chunks
        chunks = parse_markdown_to_chunks(
            md_text=md_text,
            doc_id=doc_id,
            image_dir=doc_image_dir,
            cfg=chunking_cfg,
        )

        # ── Clean chunks (filter noise/garbage BEFORE VLM) ──
        chunks = clean_chunks(chunks, cleaning_cfg)

        # Process images with VLM (only on cleaned chunks)
        for chunk in chunks:
            if chunk.image_paths:
                has_images = True
                for img_path in chunk.image_paths:
                    desc = vlm.describe_image(img_path)
                    chunk.visual_descriptions.append(desc)
                chunk.is_multimodal = True

            all_contexts.append(asdict(chunk))

    # Free GPU
    if has_images:
        vlm.unload()

    # Save output
    save_json(all_contexts, paths.multimodal_contexts)
    log.info(
        "🏁 Stage 2 complete: %d contexts from %d documents",
        len(all_contexts),
        len(md_files),
    )
    return all_contexts


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Chunk Markdown + VLM image description → multimodal contexts"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit to N documents")
    args = parser.parse_args()

    log.info("🚀 Stage 2 — Context Structuring & Representation")
    results = process_documents(limit=args.limit)

    if not results:
        log.warning("No contexts were generated. Check data/interim/ for .md files.")
        sys.exit(1)

    log.info("✅ Generated %d multimodal contexts", len(results))


if __name__ == "__main__":
    main()
