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
import os
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

    root: Path = field(
        default_factory=lambda: Path(os.environ.get("TQA_ROOT", str(Path(__file__).resolve().parent.parent)))
    )

    # Data directories (auto-resolved in __post_init__)
    raw: Path = field(default=Path("data/raw"))
    interim: Path = field(default=Path("data/interim"))
    processed: Path = field(default=Path("data/processed"))

    # Interim sub-paths (auto-resolved in __post_init__)
    interim_images: Path = field(default=Path("data/interim/images"))
    multimodal_contexts: Path = field(default=Path("data/interim/multimodal_contexts.json"))
    raw_qa_pairs: Path = field(default=Path("data/interim/raw_qa_pairs.json"))
    filtered_qa_pairs: Path = field(default=Path("data/interim/filtered_qa_pairs.json"))
    dataset_jsonl: Path = field(default=Path("data/processed/dataset.jsonl"))

    def __post_init__(self) -> None:
        root = Path(os.environ.get("TQA_ROOT", str(self.root))).expanduser()

        # Prefer explicit override; otherwise auto-detect old/new dataset layout.
        data_dir_override = os.environ.get("TQA_DATA_DIR")
        if data_dir_override:
            data_base = Path(data_dir_override).expanduser()
        else:
            default_data_base = root / "data"
            legacy_output_base = default_data_base / "output"
            # Score candidate layouts and pick the one that actually has
            # the most pipeline artifacts (helps resume from Stage 3/4).
            def _score(base: Path) -> int:
                score = 0
                if (base / "raw").exists():
                    score += 1
                    try:
                        if any((base / "raw").glob("*.pdf")) or any((base / "raw").glob("*.PDF")):
                            score += 1
                    except Exception:
                        pass

                if (base / "interim").exists():
                    score += 1
                if (base / "interim" / "multimodal_contexts.json").exists():
                    score += 4
                try:
                    if any((base / "interim" / "contexts").glob("*.json")):
                        score += 3
                except Exception:
                    pass
                try:
                    if any((base / "interim" / "qa_chunks").glob("*.json")):
                        score += 2
                except Exception:
                    pass
                if (base / "processed" / "dataset.jsonl").exists():
                    score += 4
                return score

            default_score = _score(default_data_base)
            legacy_score = _score(legacy_output_base)
            data_base = legacy_output_base if legacy_score > default_score else default_data_base

        object.__setattr__(self, "root", root)
        object.__setattr__(self, "raw", data_base / "raw")
        object.__setattr__(self, "interim", data_base / "interim")
        object.__setattr__(self, "processed", data_base / "processed")
        object.__setattr__(self, "interim_images", data_base / "interim" / "images")
        object.__setattr__(self, "multimodal_contexts", data_base / "interim" / "multimodal_contexts.json")
        object.__setattr__(self, "raw_qa_pairs", data_base / "interim" / "raw_qa_pairs.json")
        object.__setattr__(self, "filtered_qa_pairs", data_base / "interim" / "filtered_qa_pairs.json")
        object.__setattr__(self, "dataset_jsonl", data_base / "processed" / "dataset.jsonl")

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

    model_name: str = "5CD-AI/Vintern-1B-v3_5"  # Best lightweight VLM for Vietnamese
    # Fallback: "Qwen/Qwen2-VL-2B-Instruct"
    torch_dtype: str = "float16"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.3
    device_map: str = "auto"
    batch_size: int = 16             # images per VLM batch (L4: true batched inference)


@dataclass(frozen=True)
class LLMConfig:
    """Text-only LLM settings (Stage 3 — QAG, Stage 4 — Evaluation)."""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # Optimized for vLLM & T4 GPUs
    torch_dtype: str = "bfloat16"
    load_in_4bit: bool = True
    use_vllm: bool = True             # Primary offline inference engine
    max_new_tokens: int = 512         # Reduced from 1024 for faster generation
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
        r"^(?:Điều|Khoản)\s+\d+(?:[\.\:])?\s*",
        r"^Chương\s+[IVXLCDM]+(?:[\.\:])?\s*",
        r"^Mục\s+\d+(?:[\.\:])?\s*",
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
    batch_size: int = 64              # Tối ưu hóa cho NVIDIA L4 GPU (24GB VRAM)
    merge_bloom_levels: bool = True   # merge all bloom levels into one GPU call (3× speedup)


@dataclass(frozen=True)
class EvalConfig:
    """
    LLM-as-a-judge binary scoring (Stage 4).

    # [Paper Note — Judge Model Design Choice]
    # To mitigate *same-family bias* (where a judge from the same model
    # family as the generator tends to over-approve generated outputs due
    # to shared pre-training data and alignment methodology), we deliberately
    # select a judge from a DIFFERENT model family than the generator.
    #
    # Generator : Qwen/Qwen2.5-0.5B-Instruct  (Alibaba Cloud / Qwen family)
    # Judge      : google/gemma-2-2b-it         (Google / Gemma family)
    #
    # Rationale for gemma-2-2b-it in LOW-RESOURCE setting:
    #   - 2B parameters → fits on Colab T4/L4 even with 4-bit quantisation
    #   - Supported by BitsAndBytes 4-bit (NF4) → ~1.0 GB VRAM footprint
    #   - Multilingual pre-training includes Vietnamese text
    #   - Open weights, no API key required (reproducible research)
    #   - DIFFERENT architecture family from Qwen → reduces self-reinforcement
    #
    # Alternative lightweight cross-family judges (if gemma-2 unavailable):
    #   - microsoft/Phi-3-mini-4k-instruct (3.8B, Microsoft family)
    #   - meta-llama/Llama-3.2-3B-Instruct (3B, Meta family — needs HF token)
    #
    # Cross-model validation (for paper ablation):
    #   Run evaluate.py with --model-name <cross_validation_model_name>
    #   to verify score distributions are consistent across judge families.
    """

    # Judge from a different model family than the generator (anti-bias design).
    # [Paper Note] Cited in §4 Quality Analysis as cross-family judge strategy.
    judge_model_name: str = "google/gemma-2-2b-it"

    # Fallback judge if gemma-2 download fails (same-size, different family).
    # Usage: python -m src.04_evaluate --model-name microsoft/Phi-3-mini-4k-instruct
    fallback_judge_model_name: str = "microsoft/Phi-3-mini-4k-instruct"

    # Binary pass/fail thresholds (scale: 0 = Fail, 1 = Pass).
    # [Paper Note] §4.1: "A QA pair is retained if and only if all three
    # criteria receive a binary Pass score from the cross-family judge."
    groundedness_threshold: float = 1.0         # Context grounding: must be fully supported
    multimodal_alignment_threshold: float = 1.0 # Visual reference: must correctly cite visuals
    legal_fluency_threshold: float = 1.0        # Syllogism: Major→Minor→Conclusion must be valid
    overall_threshold: float = 1.0              # Composite: all criteria must pass
    score_scale: int = 1                        # Binary (0/1) — see [Paper Note] §4.1
    batch_size: int = 64                        # Tối ưu hóa cho NVIDIA L4 GPU (24GB VRAM)

    # --- Augmented Evaluation Prompts ---
    # [Paper Note] §5.2: "To ensure granular quality control, we supplement the binary
    # pass/fail scoring with secondary checks for Legal Grounding and Bloom Taxonomy classification."
    legal_grounding_template: str = """
[Bối cảnh pháp lý]: {context}
[Câu hỏi]: {question}
[Câu trả lời candidate]: {answer}

NHIỆM VỤ: Bạn là một Thẩm phán nghiêm khắc. Hãy kiểm tra xem Câu trả lời có sử dụng bất kỳ thông tin nào KHÔNG nằm trong [Bối cảnh pháp lý] ở trên không? 
Đặc biệt chú ý đến: Tên văn bản, Số hiệu điều luật, và các mốc thời gian.

CHỈ TRẢ VỀ JSON:
{{
  "is_grounded": true/false,
  "unsupported_facts": ["danh sách các ý kiến bịa đặt"],
  "score": 0-1
}}
"""

    bloom_classifier_template: str = """
Câu hỏi: {question}

NHIỆM VỤ: Phân loại câu hỏi này vào một trong 3 cấp độ Bloom:
1. 'Remember': Hỏi trực tiếp về định nghĩa, số liệu trong luật.
2. 'Understand': Yêu cầu giải thích ý nghĩa hoặc tóm tắt nội dung điều luật.
3. 'Apply': Đưa ra tình huống thực tế và hỏi cách áp dụng điều luật này.

CHỈ TRẢ VỀ MỘT TỪ DUY NHẤT: [Remember|Understand|Apply]
"""


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
    batch_multiplier: int = 12        # marker batch multiplier (L4 optimized — higher fills VRAM)
    parallel_workers: int = 3          # L4 có ~24GB VRAM, 3 workers (3x7=21GB) là mức tối đa an toàn. Không nên lên 4.

    # Explicit surya batch sizes (override auto-detection for L4 GPU)
    # These are set as env vars BEFORE surya imports
    recognition_batch_size: int = 512  # bottleneck step — L4 has headroom
    detector_batch_size: int = 128     # bbox detection
    layout_batch_size: int = 128       # layout recognition
    order_batch_size: int = 64         # reading order
    table_rec_batch_size: int = 64     # table recognition
    equation_batch_size: int = 64      # equation detection
    dataset_num_workers: int = 4       # parallel data loading threads


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

    drive_root: Path = field(
        default_factory=lambda: Path(os.environ.get("TQA_DRIVE_ROOT", "/content/drive/MyDrive"))
    )
    backup_base: Path = field(
        default=Path("Colab_Workspaces/TQA_Pipeline_Backup_7B")
    )

    # Stage 1 — Digitization (auto-resolved in __post_init__)
    interim_md: Path = field(default=Path("interim"))
    interim_images: Path = field(default=Path("interim/images"))

    # Stage 2 — Structuring (auto-resolved in __post_init__)
    contexts_dir: Path = field(default=Path("interim/contexts"))

    # Stage 3 — QAG (auto-resolved in __post_init__)
    qa_chunks_dir: Path = field(default=Path("interim/qa_chunks"))

    # Stage 4 — Evaluation (auto-resolved in __post_init__)
    evaluated_qa_dir: Path = field(default=Path("interim/evaluated_qa"))

    # Final outputs (auto-resolved in __post_init__)
    processed_dir: Path = field(default=Path("processed"))

    def __post_init__(self) -> None:
        drive_root = Path(os.environ.get("TQA_DRIVE_ROOT", str(self.drive_root))).expanduser()
        backup_override = os.environ.get("TQA_BACKUP_BASE")
        backup_base = Path(backup_override).expanduser() if backup_override else drive_root / "Colab_Workspaces" / "TQA_Pipeline_Backup_7B"

        object.__setattr__(self, "drive_root", drive_root)
        object.__setattr__(self, "backup_base", backup_base)
        object.__setattr__(self, "interim_md", backup_base / "interim")
        object.__setattr__(self, "interim_images", backup_base / "interim" / "images")
        object.__setattr__(self, "contexts_dir", backup_base / "interim" / "contexts")
        object.__setattr__(self, "qa_chunks_dir", backup_base / "interim" / "qa_chunks")
        object.__setattr__(self, "evaluated_qa_dir", backup_base / "interim" / "evaluated_qa")
        object.__setattr__(self, "processed_dir", backup_base / "processed")

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
# 6 · GPU Optimization Config
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class GPUOptConfig:
    """GPU optimization settings tuned for NVIDIA L4 (22.5 GB VRAM).

    Who:    All pipeline stages.
    How:    Controls batch sizes, data prefetching, and memory management.
    Why:    4-bit quantized models (Vintern-1B ~700MB, Qwen-0.5B ~400MB)
            leave >20 GB VRAM free. Batching fills the GPU pipeline,
            raising utilization from ~5% to 60-80%.
    """

    # Data loading
    prefetch_workers: int = 4          # threads for async I/O (image load, tokenize)
    pin_memory: bool = True            # faster host→device transfer (DataLoader)
    prefetch_factor: int = 2           # batches to prefetch ahead

    # Memory management
    empty_cache_interval: int = 50     # torch.cuda.empty_cache() every N batches
    log_gpu_interval: int = 10         # log VRAM usage every N batches

    # torch.compile for text LLMs (Qwen — ~20-40% inference speedup)
    use_torch_compile: bool = True     # enable for Qwen2.5 on PyTorch 2.x
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", "max-autotune"

    # torch.backends optimizations for inference
    cudnn_benchmark: bool = True       # auto-tune convolution algorithms
    matmul_precision: str = "medium"   # trade precision for speed (float32 matmul)

    # VLM batching (InternVL2/Vintern models)
    enable_vlm_batch: bool = True      # attempt batched VLM inference (fallback if fail)

    # vLLM optimization
    vllm_gpu_utilization: float = 0.90 # NVIDIA L4 có 24GB VRAM, 0.90 đảm bảo tận dụng tối đa mà không bị OOM hệ thống
    vllm_max_num_seqs: int = 1024      # Max concurrent sequences (pushes GPU scheduler to 100%)
    async_drive_io: bool = True        # run checkpointing in background threads


# ─────────────────────────────────────────────
# 7 · Output Schema (Stage 5 — JSONL record)
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
    bloom_level: str = Field(default="", description="Bloom's Taxonomy level: Remember/Understand/Apply")
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
# 8 · Aggregate Config Singleton
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
    gpu: GPUOptConfig = field(default_factory=GPUOptConfig)


# Instantiate the global config
CFG = PipelineConfig()
