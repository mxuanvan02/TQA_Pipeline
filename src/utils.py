"""
TQA Pipeline — Shared Utility Functions
========================================
Who:    Imported by every pipeline stage (01–04).
Where:  /content/TQA_Pipeline/src/utils.py
How:    Provides atomic file I/O, logging factories, and file-discovery helpers.
Input:  Various (JSON dicts, JSONL records, directory paths).
Output: Various (loaded data structures, saved files, logger instances).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.config import LOG_FORMAT, LOG_LEVEL, logger


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
# 3 · JSON I/O
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

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded JSON: %s (%d top-level items)", path.name, len(data) if isinstance(data, list) else 1)
    return data


def save_json(data: Any, path: Path) -> Path:
    """
    Atomically write *data* as pretty-printed JSON.

    Who:    Stages 2–4 for persisting intermediate artifacts.
    Input:  Python object (dict / list) + target Path.
    Output: The written Path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("Saved JSON: %s", path)
    return path


def save_jsonl(records: list[dict | BaseModel], path: Path) -> Path:
    """
    Write a list of records as newline-delimited JSON (JSONL).

    Who:    Stage 5 for final dataset output.
    Input:  List of dicts or Pydantic models + target Path.
    Output: The written Path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, BaseModel):
                line = record.model_dump_json(ensure_ascii=False)
            else:
                line = json.dumps(record, ensure_ascii=False)
            f.write(line + "\n")

    logger.info("Saved JSONL: %s (%d records)", path, len(records))
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
