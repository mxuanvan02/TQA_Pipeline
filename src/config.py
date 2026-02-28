"""
TQA Pipeline — Central Configuration Module
============================================
Who:    Imported by every pipeline stage (01–04) and utils.
Where:  /content/TQA_Pipeline/src/config.py
How:    Defines all paths, model identifiers, generation hyper-parameters,
        chunking rules, evaluation thresholds, and the final output schema
        as frozen dataclasses / Pydantic models.
Input:  None (pure configuration).
Output: Importable constants & dataclass singletons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("tqa_pipeline")


# ─────────────────────────────────────────────
# 1 · Path Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class PathConfig:
    """Immutable directory layout for the entire pipeline."""

    root: Path = Path("/content/TQA_Pipeline")

    # Data directories
    raw: Path = field(default=Path("/content/TQA_Pipeline/data/raw"))
    interim: Path = field(default=Path("/content/TQA_Pipeline/data/interim"))
    processed: Path = field(default=Path("/content/TQA_Pipeline/data/processed"))

    # Interim sub-paths (created lazily by each stage)
    interim_images: Path = field(
        default=Path("/content/TQA_Pipeline/data/interim/images")
    )
    multimodal_contexts: Path = field(
        default=Path("/content/TQA_Pipeline/data/interim/multimodal_contexts.json")
    )
    raw_qa_pairs: Path = field(
        default=Path("/content/TQA_Pipeline/data/interim/raw_qa_pairs.json")
    )
    filtered_qa_pairs: Path = field(
        default=Path("/content/TQA_Pipeline/data/interim/filtered_qa_pairs.json")
    )
    dataset_jsonl: Path = field(
        default=Path("/content/TQA_Pipeline/data/processed/dataset.jsonl")
    )

    def ensure_dirs(self) -> None:
        """Create every directory if it does not already exist."""
        for d in (self.raw, self.interim, self.interim_images, self.processed):
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 2 · Model Configuration
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class VLMConfig:
    """Vision-Language Model settings (Stage 2 — image description)."""

    model_name: str = "5Rii/Vintern-1B-v3"  # Best lightweight VLM for Vietnamese
    # Fallback: "Qwen/Qwen2-VL-2B-Instruct"
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.3
    device_map: str = "auto"


@dataclass(frozen=True)
class LLMConfig:
    """Text-only LLM settings (Stage 3 — QAG, Stage 4 — Evaluation)."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"  # Lightest instruction-tuned LLM
    # Fallback: "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    max_new_tokens: int = 1024
    temperature: float = 0.7          # for creative QA generation
    eval_temperature: float = 0.1     # deterministic evaluation
    top_p: float = 0.9
    device_map: str = "auto"
    repetition_penalty: float = 1.15


# ─────────────────────────────────────────────
# 3 · Pipeline Hyper-parameters
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class ChunkingConfig:
    """Rules for Markdown → semantic chunks (Stage 2)."""

    split_headers: tuple[str, ...] = ("#", "##", "###")
    # Tách chunk theo cấu trúc văn bản pháp luật VN bổ sung nếu marker không nhận diện #
    legal_split_patterns: tuple[str, ...] = (
        r"^(?i)(?:Điều|Khoản)\s+\d+(?:[\.\:])?\s*",
        r"^(?i)Chương\s+[IVXLCDM]+(?:[\.\:])?\s*",
        r"^(?i)Mục\s+\d+(?:[\.\:])?\s*",
    )
    min_chunk_chars: int = 100        # skip trivially short chunks
    max_chunk_chars: int = 3000       # hard ceiling per context window
    overlap_chars: int = 0            # no overlap for header-based split


@dataclass(frozen=True)
class ChunkCleaningConfig:
    """Rules for filtering noise/garbage chunks before VLM/LLM processing."""

    min_meaningful_words: int = 15      # minimum Vietnamese words in chunk
    max_special_char_ratio: float = 0.5 # reject if >50% non-alphanumeric
    min_sentence_count: int = 1         # at least 1 sentence (ends with .?!)
    # Regex patterns for noise lines to strip (page numbers, headers, footers)
    noise_patterns: tuple[str, ...] = (
        r"^\s*\d+\s*$",                  # lone page numbers
        r"^\s*trang\s+\d+",              # "trang 42"
        r"^\s*[-_=]{3,}\s*$",            # horizontal rules
        r"^\s*\.\.\.\.+\s*$",             # dot leaders
    )
    # Drop chunks that are purely table-of-contents style
    toc_indicator_threshold: int = 5    # if >5 lines match page-ref pattern


@dataclass(frozen=True)
class QAGConfig:
    """Question-Answer Generation hyper-parameters (Stage 3)."""

    # Bloom's Taxonomy levels to generate
    bloom_levels: tuple[str, ...] = (
        "Remember",      # Level 1 — factual recall
        "Understand",    # Level 2 — explain / summarize
        "Apply",         # Level 3 — apply to new scenario
    )
    questions_per_level: int = 1      # per chunk, per Bloom level
    batch_size: int = 4               # chunks processed in one batch


@dataclass(frozen=True)
class EvalConfig:
    """LLM-as-a-judge binary scoring (Stage 4, optimized for 0.5B model)."""

    groundedness_threshold: float = 1.0     # 1 (Pass) or 0 (Fail)
    multimodal_alignment_threshold: float = 1.0 # 1 (Pass) or 0 (Fail)
    legal_fluency_threshold: float = 1.0    # 1 (Pass) or 0 (Fail)
    overall_threshold: float = 1.0          # Must pass all to be included
    score_scale: int = 1                    # Binary indicator


# ─────────────────────────────────────────────
# 4 · Marker-PDF Configuration (Stage 1)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class MarkerConfig:
    """CLI / API flag mapping for marker-pdf."""

    force_ocr: bool = True
    extract_images: bool = True
    paginate_output: bool = True
    output_format: str = "markdown"
    batch_multiplier: int = 2         # marker batch multiplier


# ─────────────────────────────────────────────
# 5 · Google Drive Backup Config (Session Resilience)
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class DriveBackupConfig:
    """Centralized Google Drive backup paths for Colab session resilience.

    Who:    All pipeline stages (01–04).
    How:    Provides a single source of truth for Drive backup locations,
            replaces hardcoded paths scattered across multiple files.
    """

    drive_root: Path = Path("/content/drive/MyDrive")
    backup_base: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup")
    )

    # Stage 1 — Digitization
    interim_md: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup/interim")
    )
    interim_images: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup/interim/images")
    )

    # Stage 2 — Structuring
    contexts_dir: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup/interim/contexts")
    )

    # Stage 3 — QAG
    qa_chunks_dir: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup/interim/qa_chunks")
    )

    # Stage 4 — Evaluation
    evaluated_qa_dir: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup/interim/evaluated_qa")
    )

    # Final outputs
    processed_dir: Path = field(
        default=Path("/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup/processed")
    )

    def is_drive_mounted(self) -> bool:
        """Check if Google Drive is actually mounted (not just a local dir).

        Verifies that /content/drive/MyDrive is a real mount point,
        not a directory accidentally created by mkdir(parents=True).
        """
        import os

        drive_path = self.drive_root
        if not drive_path.exists():
            return False

        # Check if it's a mount point (reliable on Linux/Colab)
        if os.path.ismount(str(drive_path)):
            return True

        # Fallback: check if parent /content/drive is a mount point
        if os.path.ismount("/content/drive"):
            return True

        # Fallback: check for typical Drive marker files/dirs
        # Google Drive always has 'My Drive' content when mounted
        if (drive_path / ".shortcut-targets-by-id").exists():
            return True

        # If the directory exists but is suspiciously empty, it's likely local
        try:
            contents = list(drive_path.iterdir())
            return len(contents) > 0
        except PermissionError:
            return False

    def ensure_dirs(self) -> None:
        """Create all backup directories (only if Drive is mounted)."""
        if not self.is_drive_mounted():
            logger.warning(
                "⚠️  Google Drive NOT mounted at %s — backups will be DISABLED",
                self.drive_root,
            )
            return

        for d in (
            self.interim_md,
            self.interim_images,
            self.contexts_dir,
            self.qa_chunks_dir,
            self.evaluated_qa_dir,
            self.processed_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 6 · Output Schema (Stage 5 — JSONL record)
# ─────────────────────────────────────────────
class ContextPayload(BaseModel):
    """Nested object inside each dataset record."""

    text: str = Field(..., description="Markdown text chunk used as context")
    visuals: list[str] = Field(
        default_factory=list,
        description="List of file paths to associated images",
    )


class TQARecord(BaseModel):
    """Final JSONL schema definition — one record per QA pair."""

    qa_id: str = Field(..., description="Unique identifier: <doc>_<chunk>_<bloom>_<seq>")
    domain_tag: str = Field(default="civil_law", description="Subject domain")
    context_payload: ContextPayload
    question_content: str
    is_multimodal: bool = Field(
        default=False,
        description="True if context_payload.visuals is non-empty",
    )
    candidate_answers: list[str] = Field(
        default_factory=list,
        description="Distractor + correct answer (shuffled)",
    )
    ground_truth: str = Field(..., description="Correct answer string")
    legal_rationale: str = Field(
        ...,
        description="Legal Syllogism: Major Premise → Minor Premise → Conclusion",
    )


# ─────────────────────────────────────────────
# 7 · Aggregate Config Singleton
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration object passed to every stage."""

    paths: PathConfig = field(default_factory=PathConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    chunk_cleaning: ChunkCleaningConfig = field(default_factory=ChunkCleaningConfig)
    qag: QAGConfig = field(default_factory=QAGConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    marker: MarkerConfig = field(default_factory=MarkerConfig)
    drive_backup: DriveBackupConfig = field(default_factory=DriveBackupConfig)


# Instantiate the global config
CFG = PipelineConfig()
