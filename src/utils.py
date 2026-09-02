"""
TQA Pipeline — Shared Utility Functions
========================================
Who:    Imported by every pipeline stage (01–04).
Where:  /content/TQA_Pipeline/src/utils.py
How:    Provides atomic file I/O, logging factories, file-discovery helpers,
        and Google Drive backup/restore utilities for Colab session resilience.
Input:  Various (JSON dicts, JSONL records, directory paths).
Output: Various (loaded data structures, saved files, logger instances).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import LOG_FORMAT, LOG_LEVEL, DriveBackupConfig, logger


# ─────────────────────────────────────────────
# 1 · Logger Factory
# ─────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Create a per-module logger with the global format.

    Who:    Called at module-level by each pipeline stage.
    How:    Re-uses the root-level format / level from config.
    """
    child = logging.getLogger(name)
    child.setLevel(LOG_LEVEL)
    if not child.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        child.addHandler(handler)
    return child


# ─────────────────────────────────────────────
# 2 · File Discovery
# ─────────────────────────────────────────────
def get_files(directory: Path, extension: str = ".pdf") -> list[Path]:
    """
    Recursively discover files matching *extension* in *directory*.

    Who:    Stage 1 (PDFs), Stage 2 (MDs, images).
    Input:  A directory path and a file extension string (e.g. ".pdf").
    Output: Sorted list of Path objects.
    Raises: FileNotFoundError if directory does not exist.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = sorted(directory.rglob(f"*{extension}"))
    logger.info("Found %d '%s' files in %s", len(files), extension, directory)
    return files


# ─────────────────────────────────────────────
# 3 · JSON I/O (Atomic Writes)
# ─────────────────────────────────────────────
def load_json(path: Path) -> Any:
    """
    Read a JSON file and return its parsed content.

    Who:    Stages 2–4 for loading intermediate artifacts.
    Input:  Path to a .json file.
    Output: Parsed Python object (dict / list).
    Raises: FileNotFoundError, json.JSONDecodeError.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    logger.info("Loaded JSON: %s (%d top-level items)", path.name, len(data) if isinstance(data, list) else 1)
    return data


def save_json(data: Any, path: Path) -> Path:
    """
    Atomically write *data* as pretty-printed JSON.

    Who:    Stages 2–4 for persisting intermediate artifacts.
    How:    Writes to a temporary file in the same directory, then performs
            an atomic rename. This prevents JSON corruption if the process
            crashes mid-write (critical for Colab session resilience).
    Input:  Python object (dict / list) + target Path.
    Output: The written Path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in the same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{path.stem}_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

        # Atomic rename (POSIX guarantees atomicity on same filesystem)
        os.replace(tmp_path, str(path))
        logger.info("Saved JSON (atomic): %s", path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return path


def save_jsonl(records: list[dict | BaseModel], path: Path) -> Path:
    """
    Atomically write a list of records as newline-delimited JSON (JSONL).

    Who:    Stage 5 for final dataset output.
    How:    Uses atomic write pattern (temp file + rename).
    Input:  List of dicts or Pydantic models + target Path.
    Output: The written Path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{path.stem}_",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                if isinstance(record, BaseModel):
                    line = record.model_dump_json(ensure_ascii=False)
                else:
                    line = json.dumps(record, ensure_ascii=False)
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, str(path))
        logger.info("Saved JSONL (atomic): %s (%d records)", path, len(records))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return path


# ─────────────────────────────────────────────
# 4 · Text Helpers
# ─────────────────────────────────────────────
def slugify(text: str) -> str:
    """Convert a string to a filesystem-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s-]+", "_", text).strip("_")


def read_text(path: Path) -> str:
    """
    Read a text file (Markdown, etc.) and return its content.

    Raises: FileNotFoundError if file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# 5 · Google Drive Backup Utilities
# ─────────────────────────────────────────────
def verify_drive_mount(cfg: DriveBackupConfig | None = None) -> bool:
    """
    Pre-flight check: verify Google Drive is properly mounted.

    Who:    Called at pipeline startup before any processing.
    How:    Uses DriveBackupConfig.is_drive_mounted() with clear logging.
    Returns: True if Drive is mounted, False otherwise.
    """
    if cfg is None:
        cfg = DriveBackupConfig()

    if cfg.is_drive_mounted():
        logger.info("✅ Google Drive is mounted at %s", cfg.drive_root)
        cfg.ensure_dirs()
        return True
    else:
        logger.warning(
            "⚠️  Google Drive is NOT mounted at %s — "
            "all backups will be DISABLED. Data will only exist on volatile local disk!",
            cfg.drive_root,
        )
        return False


def sync_to_drive(
    local_path: Path,
    drive_path: Path,
    cfg: DriveBackupConfig | None = None,
    label: str = "",
) -> bool:
    """
    Safely copy a local file to Google Drive with error handling.

    Who:    All stages for incremental backup after each unit of work.
    How:    Checks Drive mount → copies file → verifies destination exists.
    Input:  Local file path + Drive destination path.
    Output: True if sync succeeded, False otherwise.
    """
    if cfg is None:
        cfg = DriveBackupConfig()

    if not cfg.is_drive_mounted():
        return False

    if not local_path.exists():
        logger.warning("  ⚠️  Cannot sync (source missing): %s", local_path)
        return False

    try:
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(drive_path))

        if label:
            logger.info("  ☁️  Synced to Drive: %s", label)
        return True

    except Exception as e:
        logger.warning("  ⚠️  Drive sync failed for %s: %s", local_path.name, e)
        return False


def sync_json_to_drive(
    data: Any,
    drive_path: Path,
    cfg: DriveBackupConfig | None = None,
    label: str = "",
) -> bool:
    """
    Write JSON data directly to Google Drive (atomic write).

    Who:    All stages for per-unit checkpoint saving.
    How:    Writes to temp file on Drive → atomic rename.
    Input:  Python data + Drive destination path.
    Output: True if sync succeeded, False otherwise.
    """
    if cfg is None:
        cfg = DriveBackupConfig()

    if not cfg.is_drive_mounted():
        return False

    try:
        drive_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename for atomicity
        tmp_path = drive_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

        os.replace(str(tmp_path), str(drive_path))

        if label:
            logger.info("  ☁️  Checkpoint saved to Drive: %s", label)
        return True

    except Exception as e:
        logger.warning("  ⚠️  Drive checkpoint failed for %s: %s", drive_path.name, e)
        # Clean up temp file
        try:
            tmp_clean = drive_path.with_suffix(".tmp")
            if tmp_clean.exists():
                tmp_clean.unlink()
        except OSError:
            pass
        return False


class AsyncDriveWriter:
    """
    Background thread pool for non-blocking Drive I/O.
    
    Who:    Stages 3-4 for asynchronous checkpointing.
    How:    Uses ThreadPoolExecutor to run sync_json_to_drive in the background,
            preventing the GPU from waiting for slow network/Disk I/O.
    """
    def __init__(self, max_workers: int = 2):
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
    
    def submit(self, data: Any, drive_path: Path, cfg: DriveBackupConfig | None = None, label: str = "") -> None:
        """Submit a JSON sync task to the background."""
        import copy
        # Deep copy data to avoid mutations while it's waiting to be written
        data_copy = copy.deepcopy(data)
        future = self._executor.submit(sync_json_to_drive, data_copy, drive_path, cfg, label)
        self._futures.append(future)
    
    def flush(self) -> None:
        """Wait for all pending writes to complete."""
        from concurrent.futures import as_completed
        for f in as_completed(self._futures):
            try:
                f.result()
            except Exception as e:
                logger.warning("Async write failed: %s", e)
        self._futures.clear()


def restore_from_drive(
    drive_path: Path,
    local_path: Path,
    cfg: DriveBackupConfig | None = None,
    label: str = "",
) -> bool:
    """
    Restore a file from Google Drive to local storage.

    Who:    All stages for recovery after session restart.
    How:    Copies Drive file → local, with directory creation.
    Input:  Drive source path + local destination path.
    Output: True if restore succeeded, False otherwise.
    """
    if cfg is None:
        cfg = DriveBackupConfig()

    if not cfg.is_drive_mounted():
        return False

    if not drive_path.exists():
        return False

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(drive_path), str(local_path))

        if label:
            logger.info("  📥 Restored from Drive: %s", label)
        return True

    except Exception as e:
        logger.warning("  ⚠️  Drive restore failed for %s: %s", drive_path.name, e)
        return False


def load_drive_checkpoint(
    drive_path: Path,
    cfg: DriveBackupConfig | None = None,
) -> Any | None:
    """
    Load a JSON checkpoint from Google Drive.

    Who:    All stages for checking if a unit of work was already completed.
    How:    Reads JSON from Drive path, returns parsed data or None.
    Input:  Drive path to a JSON checkpoint file.
    Output: Parsed Python object, or None if not found/invalid.
    """
    if cfg is None:
        cfg = DriveBackupConfig()

    if not cfg.is_drive_mounted():
        return None

    if not drive_path.exists():
        return None

    try:
        with open(drive_path, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("  ⚠️  Corrupt checkpoint on Drive: %s (%s)", drive_path.name, e)
        return None


# ─────────────────────────────────────────────
# 6 · GPU Optimization Utilities
# ─────────────────────────────────────────────
def batched(iterable: list, batch_size: int):
    """
    Yield successive batches of *batch_size* from *iterable*.

    Who:    Stages 2-4 for batch inference.
    How:    Simple slice-based iterator, no padding.
    Input:  List + batch size.
    Output: Yields list slices.
    """
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def batched_by_length(
    items: list,
    length_fn,
    batch_size: int,
) -> list[tuple[list[int], list]]:
    """
    Group items by similar length for minimal padding waste.

    Who:    Stages 3-4 for dynamic batching (GPU optimization).
    How:    Sorts items by length, batches them, returns (original_indices, items).
            Within each batch, items have similar lengths → less padding tokens.
    Input:  List of items + a length function + batch size.
    Output: List of (original_indices, batch_items) tuples.
    """
    indexed = [(i, length_fn(item), item) for i, item in enumerate(items)]
    indexed.sort(key=lambda x: x[1])

    result = []
    for batch in batched(indexed, batch_size):
        original_indices = [x[0] for x in batch]
        batch_items = [x[2] for x in batch]
        result.append((original_indices, batch_items))
    return result


def try_torch_compile(model, gpu_cfg=None):
    """
    Attempt torch.compile on a model for faster inference.

    Who:    Stages 3-4 (QAGenerator, QAJudge) after model loading.
    How:    Wraps model with torch.compile if available and enabled.
            Falls back silently on failure (e.g., unsupported model).
    Input:  A PyTorch model + optional GPUOptConfig.
    Output: Compiled model (or original model if compile fails/disabled).
    """
    if gpu_cfg is None:
        return model

    if not getattr(gpu_cfg, "use_torch_compile", False):
        return model

    try:
        import torch

        if hasattr(torch, "compile"):
            mode = getattr(gpu_cfg, "compile_mode", "reduce-overhead")
            compiled = torch.compile(model, mode=mode)
            logger.info("  ⚡ torch.compile applied (mode=%s)", mode)
            return compiled
    except Exception as e:
        logger.warning("  ⚠️  torch.compile failed (%s), using eager mode", e)

    return model


def log_gpu_memory(label: str = "") -> None:
    """
    Log current GPU VRAM usage for monitoring.

    Who:    Called periodically during batch inference.
    How:    Uses torch.cuda to query allocated / reserved memory.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return

        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3

        prefix = f"[{label}] " if label else ""
        logger.info(
            "  GPU VRAM: %s%.1f GB alloc / %.1f GB reserved / %.1f GB total",
            prefix,
            alloc,
            reserved,
            total,
        )
    except Exception:
        pass  # non-critical logging


def prefetch_batch_io(
    items: list,
    load_fn,
    max_workers: int = 4,
) -> list:
    """
    Load/preprocess a batch of items in parallel using threads.

    Who:    Stages 2-4 for overlapping I/O with GPU compute.
    How:    ThreadPoolExecutor for I/O-bound work (file reads, image opens).
    Input:  List of items + a function to apply to each.
    Output: List of results (same order as input).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(load_fn, item): idx
            for idx, item in enumerate(items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning("  Prefetch failed for item %d: %s", idx, e)
                results[idx] = None

    return results


def setup_tokenizer_for_batch(tokenizer) -> None:
    """
    Configure a tokenizer for batched causal LM generation.

    Who:    Stages 3-4 (QAGenerator, QAJudge).
    How:    Sets left-padding (required for causal LM batch generation)
            and ensures pad_token is defined.
    Why:    Causal LMs generate from the right, so padding must be on the left
            to keep the actual tokens right-aligned.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
