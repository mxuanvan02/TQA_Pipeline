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

import os

from src.config import CFG, DriveBackupConfig, MarkerConfig, PathConfig

# ─────────────────────────────────────────────
# ⚡ GPU Optimization: MUST be set BEFORE surya/marker imports
# These env vars control surya's internal batch sizes.
# Default auto-detection is very conservative; explicit values
# fill the L4's 22.5 GB VRAM properly.
# ─────────────────────────────────────────────
os.environ.setdefault("RECOGNITION_BATCH_SIZE", str(CFG.marker.recognition_batch_size))
os.environ.setdefault("DETECTOR_BATCH_SIZE", str(CFG.marker.detector_batch_size))
os.environ.setdefault("LAYOUT_BATCH_SIZE", str(CFG.marker.layout_batch_size))
os.environ.setdefault("ORDER_BATCH_SIZE", str(CFG.marker.order_batch_size))
os.environ.setdefault("TABLE_REC_BATCH_SIZE", str(CFG.marker.table_rec_batch_size))
os.environ.setdefault("EQUATION_BATCH_SIZE", str(CFG.marker.equation_batch_size))
os.environ.setdefault("DATASET_NUM_WORKERS", str(CFG.marker.dataset_num_workers))

import argparse
import multiprocessing as mp
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
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
            torch.backends.cudnn.benchmark = CFG.gpu.cudnn_benchmark
            torch.set_float32_matmul_precision(CFG.gpu.matmul_precision)
            # Pre-allocate CUDA memory pool for fewer fragmentation pauses
            torch.cuda.empty_cache()
            log.info(
                "⚡ Torch optimizations applied: cudnn.benchmark=%s, "
                "matmul_precision=%s, GPU=%s (%.1f GB free)",
                CFG.gpu.cudnn_benchmark,
                CFG.gpu.matmul_precision,
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
    model_dict: dict | None = None,
) -> Path | None:
    """
    Convert a single PDF to Markdown + extracted images using marker-pdf.

    Who:    marker-pdf converter (GPU-accelerated OCR).
    Where:  Outputs to *output_dir* / *image_dir*.
    How:    Uses marker's Python API with a SHARED model_dict to avoid
            reloading models for every PDF. Falls back to CLI if API is unavailable.
    Input:  Single .pdf file path + shared model_dict (created once).
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
        # ── Attempt 1: marker Python API (with shared models) ──
        md_text, images = _convert_with_api(pdf_path, marker_cfg, model_dict)

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


def _convert_with_api(
    pdf_path: Path, cfg: MarkerConfig, model_dict: dict | None = None,
) -> tuple[str, dict]:
    """
    Internal: invoke marker-pdf Python API.

    Who:    marker.converters.pdf.PdfConverter (or equivalent).
    How:    Reuses a SHARED model_dict across all PDFs to avoid
            reloading models (~3 GB, 10-30s) for each PDF.
    GPU:    Eliminates model reload overhead → massive speedup for multi-PDF batches.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        # Reuse shared models or create new ones (fallback)
        if model_dict is None:
            log.info("  Creating model dict (no shared models provided)")
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
# ─────────────────────────────────────────────
# Multi-Process Worker (module-level for pickling)
# ─────────────────────────────────────────────
def _parallel_worker(args: tuple) -> list[str | None]:
    """
    Worker process: load own models and digitize a chunk of PDFs.

    Who:    Spawned by ProcessPoolExecutor (separate CUDA context).
    How:    Each worker loads its own marker models (~7 GB VRAM),
            then processes its assigned PDFs sequentially.
    GPU:    2 workers × ~7 GB = ~14 GB → fills L4's 22.5 GB.
    """
    pdf_paths, output_dir, image_dir, marker_cfg, drive_cfg, worker_id = args

    # Initialize logging for this worker
    wlog = get_logger(f"01_digitize_w{worker_id}")

    # Apply torch optimizations in this process
    _apply_torch_optimizations()

    # Load models for THIS worker (own CUDA context, own ~7 GB)
    model_dict = None
    try:
        from marker.models import create_model_dict

        wlog.info("Worker %d: loading marker models...", worker_id)
        t0 = time.perf_counter()
        model_dict = create_model_dict()
        wlog.info("Worker %d: models loaded in %.1fs", worker_id, time.perf_counter() - t0)
    except Exception as exc:
        wlog.warning("Worker %d: model load failed: %s", worker_id, exc)

    results: list[str | None] = []
    for i, pdf_path in enumerate(pdf_paths, 1):
        wlog.info("Worker %d: [%d/%d] %s", worker_id, i, len(pdf_paths), pdf_path.stem)
        md_path = digitize_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            image_dir=image_dir,
            marker_cfg=marker_cfg,
            drive_cfg=drive_cfg,
            model_dict=model_dict,
        )
        results.append(str(md_path) if md_path else None)

    wlog.info("Worker %d: done (%d/%d succeeded)", worker_id,
              sum(1 for r in results if r), len(results))
    return results


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
    How:    If parallel_workers > 1, spawns separate processes that each
            load their own marker models (~7 GB each), truly filling the GPU.
            Otherwise, creates shared model_dict ONCE for sequential processing.
    Input:  PathConfig (default: global CFG.paths).
    Output: List of successfully generated .md file paths.

    GPU Optimization:
        - parallel_workers=2: 2 processes × ~7 GB = ~14 GB on L4 (vs 7 GB single)
        - Each process has own CUDA context → true GPU memory parallelism
        - Falls back to sequential if parallel fails (e.g., OOM)
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

    n_workers = marker_cfg.parallel_workers
    results: list[Path] = []

    # ── ⚡ PARALLEL MODE: spawn N processes, each with own models ──
    if n_workers > 1 and len(pdf_files) > 1:
        n_workers = min(n_workers, len(pdf_files))
        log.info(
            "⚡ Parallel digitization: %d workers × ~7 GB each ≈ %.0f GB VRAM",
            n_workers, 7.3 * n_workers,
        )

        # Split PDFs across workers (interleaved for balanced workload)
        chunks: list[list[Path]] = [[] for _ in range(n_workers)]
        for i, pdf in enumerate(pdf_files):
            chunks[i % n_workers].append(pdf)

        worker_args = [
            (chunk, paths.interim, paths.interim_images, marker_cfg, drive_cfg, wid)
            for wid, chunk in enumerate(chunks)
        ]

        try:
            # Use 'spawn' context to safely initialize CUDA in child processes
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=ctx,
            ) as executor:
                all_results = list(executor.map(_parallel_worker, worker_args))

            # Collect results from all workers
            for worker_results in all_results:
                for r in worker_results:
                    if r is not None:
                        results.append(Path(r))

            log.info(
                "🏁 Parallel digitization complete: %d/%d PDFs succeeded (%d workers)",
                len(results), len(pdf_files), n_workers,
            )
            return results

        except Exception as exc:
            log.warning(
                "⚠️  Parallel processing failed (%s) — falling back to sequential",
                exc,
            )
            results = []  # Reset — will retry below

    # ── SEQUENTIAL MODE: single process, shared models ──
    model_dict = None
    try:
        from marker.models import create_model_dict

        log.info("⚡ Loading marker models (shared across all PDFs)...")
        model_load_start = time.perf_counter()
        model_dict = create_model_dict()
        model_load_elapsed = time.perf_counter() - model_load_start
        log.info(
            "  ✅ Models loaded in %.1fs (will be reused for %d PDFs)",
            model_load_elapsed, len(pdf_files),
        )

        # GPU warmup
        import torch
        if torch.cuda.is_available():
            _ = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()

    except ImportError:
        log.warning("⚠️  marker.models not available — will use CLI fallback")
    except Exception as exc:
        log.warning("⚠️  Failed to pre-load models: %s — will load per-PDF", exc)

    for i, pdf in enumerate(pdf_files, 1):
        log.info("━━━ [%d/%d] ━━━", i, len(pdf_files))
        md_path = digitize_pdf(
            pdf_path=pdf,
            output_dir=paths.interim,
            image_dir=paths.interim_images,
            marker_cfg=marker_cfg,
            drive_cfg=drive_cfg,
            model_dict=model_dict,
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
    """CLI wrapper with --limit and --workers flags."""
    parser = argparse.ArgumentParser(
        description="Stage 1: Digitize PDF textbooks to Markdown + images"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N PDFs (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override parallel workers (default: from config, 0=sequential)",
    )
    args = parser.parse_args()

    log.info("🚀 Stage 1 — Multimodal Document Digitization")
    _apply_torch_optimizations()
    log.info(
        "⚡ Surya batch sizes: RECOGNITION=%s, DETECTOR=%s, LAYOUT=%s, TABLE=%s",
        os.environ.get("RECOGNITION_BATCH_SIZE"),
        os.environ.get("DETECTOR_BATCH_SIZE"),
        os.environ.get("LAYOUT_BATCH_SIZE"),
        os.environ.get("TABLE_REC_BATCH_SIZE"),
    )

    # Override workers from CLI if provided
    if args.workers is not None:
        from dataclasses import replace
        marker_cfg = replace(CFG.marker, parallel_workers=args.workers)
        results = run_digitization(limit=args.limit, marker_cfg=marker_cfg)
    else:
        results = run_digitization(limit=args.limit)

    if not results:
        log.warning("No documents were digitized. Check data/raw/ for PDF files.")
        sys.exit(1)


if __name__ == "__main__":
    main()

