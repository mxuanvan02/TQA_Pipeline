"""
Stage 1 — Multimodal Document Digitization
============================================
Who:    marker-pdf library (OCR + layout analysis engine).
Where:  Reads from data/raw/*.pdf → writes to data/interim/<doc>/ (MD + images).
How:    Wraps the marker Python API with force_ocr, extract_images, and
        paginate_output flags. Processes each PDF independently so partial
        failures don't block the entire batch.
Input:  .pdf files in  /content/TQA_Pipeline/data/raw/
Output: .md files   in /content/TQA_Pipeline/data/interim/<doc_stem>.md
        .jpg/.png   in /content/TQA_Pipeline/data/interim/images/<doc_stem>/
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# ⚡ GPU Optimization: MUST be set BEFORE surya/marker imports
# These env vars control surya's internal batch sizes.
# Default auto-detection is very conservative; explicit values
# fill the L4's 22.5 GB VRAM properly.
# ─────────────────────────────────────────────
import os
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "128")  # OCR recognition (bottleneck)
os.environ.setdefault("DETECTOR_BATCH_SIZE", "36")       # bbox detection
os.environ.setdefault("LAYOUT_BATCH_SIZE", "36")         # layout analysis
os.environ.setdefault("ORDER_BATCH_SIZE", "16")          # reading order

import argparse
import shutil
import sys
import time
from pathlib import Path

from src.config import CFG, DriveBackupConfig, MarkerConfig, PathConfig
from src.utils import (
    get_files,
    get_logger,
    restore_from_drive,
    sync_to_drive,
    verify_drive_mount,
)

log = get_logger("01_digitize")


# ─────────────────────────────────────────────
# ⚡ Torch Inference Optimizations
# ─────────────────────────────────────────────
def _apply_torch_optimizations() -> None:
    """Apply torch.backends settings for faster GPU inference."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # auto-tune conv algorithms
            torch.set_float32_matmul_precision("medium")  # speed > precision
            log.info(
                "⚡ Torch optimizations applied: cudnn.benchmark=True, "
                "matmul_precision=medium, GPU=%s (%.1f GB free)",
                torch.cuda.get_device_name(),
                torch.cuda.mem_get_info()[0] / 1024**3,
            )
    except Exception as exc:
        log.warning("⚠️ Could not apply torch optimizations: %s", exc)


# ─────────────────────────────────────────────
# Core Digitization
# ─────────────────────────────────────────────
def digitize_pdf(
    pdf_path: Path,
    output_dir: Path,
    image_dir: Path,
    marker_cfg: MarkerConfig,
    drive_cfg: DriveBackupConfig,
) -> Path | None:
    """
    Convert a single PDF to Markdown + extracted images using marker-pdf.

    Who:    marker-pdf converter (GPU-accelerated OCR).
    Where:  Outputs to *output_dir* / *image_dir*.
    How:    Uses marker's Python API (`marker.converters.pdf`) for
            fine-grained control. Falls back to CLI if API is unavailable.
    Input:  Single .pdf file path.
    Output: Path to the generated .md file, or None on failure.
    """
    doc_stem = pdf_path.stem
    md_output = output_dir / f"{doc_stem}.md"
    doc_image_dir = image_dir / doc_stem

    # ── Google Drive Backup Paths (from centralized config) ──
    drive_md = drive_cfg.interim_md / f"{doc_stem}.md"
    drive_img_dir = drive_cfg.interim_images / doc_stem

    # Skip if already processed locally OR backed up on Drive
    if md_output.exists() or drive_md.exists():
        log.info("⏭️  Already processed: %s", doc_stem)

        # Restore from Drive if local file is missing (new session)
        if not md_output.exists() and drive_md.exists():
            restore_from_drive(
                drive_md, md_output, drive_cfg,
                label=f"MD for {doc_stem}",
            )

            # Restore images too
            if drive_img_dir.exists():
                doc_image_dir.mkdir(parents=True, exist_ok=True)
                for img in drive_img_dir.glob("*"):
                    restore_from_drive(
                        img, doc_image_dir / img.name, drive_cfg,
                    )

        return md_output

    log.info("📄 Digitizing: %s", pdf_path.name)
    start = time.perf_counter()

    try:
        # ── Attempt 1: marker Python API ──
        md_text, images = _convert_with_api(pdf_path, marker_cfg)

        # Save markdown
        output_dir.mkdir(parents=True, exist_ok=True)
        md_output.write_text(md_text, encoding="utf-8")

        # Save extracted images
        if images:
            doc_image_dir.mkdir(parents=True, exist_ok=True)
            for img_name, img_data in images.items():
                img_path = doc_image_dir / img_name
                if hasattr(img_data, "save"):
                    # PIL Image
                    img_data.save(str(img_path))
                elif isinstance(img_data, bytes):
                    img_path.write_bytes(img_data)

            log.info("  🖼️  Extracted %d images → %s", len(images), doc_image_dir)

        elapsed = time.perf_counter() - start
        log.info("  ✅ Done: %s (%.1fs, %d chars)", doc_stem, elapsed, len(md_text))

        # ── Backup to Drive (using centralized utilities) ──
        sync_to_drive(md_output, drive_md, drive_cfg, label=doc_stem)

        if images:
            for img_name in images.keys():
                local_img = doc_image_dir / img_name
                if local_img.exists():
                    sync_to_drive(
                        local_img,
                        drive_img_dir / img_name,
                        drive_cfg,
                    )

        return md_output

    except Exception as exc:
        log.error("  ❌ Failed to digitize %s: %s", pdf_path.name, exc)
        return None


def _convert_with_api(pdf_path: Path, cfg: MarkerConfig) -> tuple[str, dict]:
    """
    Internal: invoke marker-pdf Python API.

    Who:    marker.converters.pdf.PdfConverter (or equivalent).
    How:    Instantiates a converter with configuration flags and runs it
            on the target PDF. Returns (markdown_text, images_dict).
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        # Create model dictionary (marker manages its own model loading)
        model_dict = create_model_dict()

        converter = PdfConverter(
            artifact_dict=model_dict,
            config={
                "force_ocr": cfg.force_ocr,
                "extract_images": cfg.extract_images,
                "paginate_output": cfg.paginate_output,
                "output_format": cfg.output_format,
                "batch_multiplier": cfg.batch_multiplier,  # ⚡ scale surya batch sizes
            },
        )

        result = converter(str(pdf_path))

        # marker ≥1.0 returns a ConverterResult with .markdown and .images
        md_text = result.markdown if hasattr(result, "markdown") else str(result)
        images = result.images if hasattr(result, "images") else {}

        return md_text, images

    except ImportError:
        log.warning("marker Python API not available, falling back to CLI")
        return _convert_with_cli(pdf_path, cfg)


def _convert_with_cli(pdf_path: Path, cfg: MarkerConfig) -> tuple[str, dict]:
    """
    Fallback: invoke marker via CLI subprocess.

    Who:    `marker_single` CLI command.
    How:    Builds CLI args from MarkerConfig and captures output.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            sys.executable, "-m", "marker",
            str(pdf_path),
            "--output_dir", tmpdir,
        ]
        if cfg.force_ocr:
            cmd.append("--force_ocr")

        log.info("  🔧 CLI: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Find generated files in tmpdir
        md_files = list(Path(tmpdir).rglob("*.md"))
        if not md_files:
            raise RuntimeError(f"marker CLI produced no MD output for {pdf_path.name}")

        md_text = md_files[0].read_text(encoding="utf-8")

        # Collect images
        images: dict[str, bytes] = {}
        for img_file in Path(tmpdir).rglob("*.jpg"):
            images[img_file.name] = img_file.read_bytes()
        for img_file in Path(tmpdir).rglob("*.png"):
            images[img_file.name] = img_file.read_bytes()

        return md_text, images


# ─────────────────────────────────────────────
# Batch Runner
# ─────────────────────────────────────────────
def run_digitization(
    paths: PathConfig | None = None,
    marker_cfg: MarkerConfig | None = None,
    drive_cfg: DriveBackupConfig | None = None,
    limit: int | None = None,
) -> list[Path]:
    """
    Digitize all PDFs in data/raw/.

    Who:    Orchestration function — loops over PDFs.
    Where:  data/raw/ → data/interim/
    How:    Calls digitize_pdf() for each discovered PDF.
    Input:  PathConfig (default: global CFG.paths).
    Output: List of successfully generated .md file paths.
    """
    paths = paths or CFG.paths
    marker_cfg = marker_cfg or CFG.marker
    drive_cfg = drive_cfg or CFG.drive_backup

    paths.ensure_dirs()

    # Pre-flight: verify Drive mount
    verify_drive_mount(drive_cfg)

    pdf_files = get_files(paths.raw, extension=".pdf")
    if not pdf_files:
        log.warning("⚠️  No PDF files found in %s", paths.raw)
        return []

    if limit:
        pdf_files = pdf_files[:limit]
        log.info("🔒 Limited to %d PDF(s)", limit)

    results: list[Path] = []
    for i, pdf in enumerate(pdf_files, 1):
        log.info("━━━ [%d/%d] ━━━", i, len(pdf_files))
        md_path = digitize_pdf(
            pdf_path=pdf,
            output_dir=paths.interim,
            image_dir=paths.interim_images,
            marker_cfg=marker_cfg,
            drive_cfg=drive_cfg,
        )
        if md_path is not None:
            results.append(md_path)

    log.info(
        "🏁 Digitization complete: %d/%d PDFs succeeded",
        len(results),
        len(pdf_files),
    )
    return results


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    """CLI wrapper with --limit flag for dry-run testing."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Digitize PDF textbooks to Markdown + images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N PDFs (for testing)",
    )
    args = parser.parse_args()

    log.info("🚀 Stage 1 — Multimodal Document Digitization")
    _apply_torch_optimizations()
    log.info(
        "⚡ Surya batch sizes: RECOGNITION=%s, DETECTOR=%s, LAYOUT=%s",
        os.environ.get("RECOGNITION_BATCH_SIZE"),
        os.environ.get("DETECTOR_BATCH_SIZE"),
        os.environ.get("LAYOUT_BATCH_SIZE"),
    )
    results = run_digitization(limit=args.limit)

    if not results:
        log.warning("No documents were digitized. Check data/raw/ for PDF files.")
        sys.exit(1)


if __name__ == "__main__":
    main()

