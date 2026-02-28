#!/usr/bin/env bash
# ============================================================
# TQA Pipeline — Full Orchestration Script
# ============================================================
# Who:    Bash shell — runs all Python stages sequentially.
# Where:  /content/TQA_Pipeline/
# How:    Executes Stages 1→2→3→4(+5) with timing and error handling.
# Input:  PDF files in data/raw/
# Output: data/processed/dataset.jsonl
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Optional: limit processing for testing (e.g., --limit 1)
LIMIT_FLAG="${1:+--limit $1}"

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  TQA Pipeline — Multimodal Text-to-QA Dataset    ${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo ""

# ─── Pre-flight: Google Drive Mount Check ───
DRIVE_PATH="/content/drive/MyDrive"
if [ -d "$DRIVE_PATH" ] && [ "$(ls -A $DRIVE_PATH 2>/dev/null)" ]; then
    echo -e "${GREEN}✅ Google Drive is mounted at ${DRIVE_PATH}${NC}"
    BACKUP_DIR="/content/drive/MyDrive/Colab_Workspaces/TQA_Pipeline_Backup"
    mkdir -p "$BACKUP_DIR"
    echo -e "${GREEN}   Backup directory: ${BACKUP_DIR}${NC}"
else
    echo -e "${RED}⚠️  WARNING: Google Drive is NOT mounted!${NC}"
    echo -e "${RED}   Backups will be DISABLED. Data exists only on volatile local disk.${NC}"
    echo -e "${YELLOW}   To mount Drive, run: from google.colab import drive; drive.mount('/content/drive')${NC}"
    echo ""
    read -p "Continue without Drive backup? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted. Mount Drive first, then re-run.${NC}"
        exit 1
    fi
fi
echo ""

TOTAL_START=$(date +%s)

# ─── Stage 1: Digitization ───
echo -e "${YELLOW}▶ Stage 1: Multimodal Document Digitization${NC}"
STAGE_START=$(date +%s)
python -m src.01_digitize ${LIMIT_FLAG}
STAGE_END=$(date +%s)
echo -e "${GREEN}✓ Stage 1 complete ($(( STAGE_END - STAGE_START ))s)${NC}"
echo ""

# ─── Stage 2: Context Structuring ───
echo -e "${YELLOW}▶ Stage 2: Context Structuring & Representation${NC}"
STAGE_START=$(date +%s)
python -m src.02_structuring ${LIMIT_FLAG}
STAGE_END=$(date +%s)
echo -e "${GREEN}✓ Stage 2 complete ($(( STAGE_END - STAGE_START ))s)${NC}"
echo ""

# ─── Stage 3: QA Generation ───
echo -e "${YELLOW}▶ Stage 3: Synthetic QAG Pipeline${NC}"
STAGE_START=$(date +%s)
python -m src.03_qag_generator ${LIMIT_FLAG}
STAGE_END=$(date +%s)
echo -e "${GREEN}✓ Stage 3 complete ($(( STAGE_END - STAGE_START ))s)${NC}"
echo ""

# ─── Stage 4+5: Evaluation & Output ───
echo -e "${YELLOW}▶ Stage 4+5: Evaluation + JSONL Output${NC}"
STAGE_START=$(date +%s)
python -m src.04_evaluate ${LIMIT_FLAG}
STAGE_END=$(date +%s)
echo -e "${GREEN}✓ Stage 4+5 complete ($(( STAGE_END - STAGE_START ))s)${NC}"
echo ""

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))

echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Pipeline finished in ${TOTAL_ELAPSED}s${NC}"
echo -e "${GREEN}  📁 Output: data/processed/dataset.jsonl${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"

# Show output stats
if [ -f "data/processed/dataset.jsonl" ]; then
    LINES=$(wc -l < data/processed/dataset.jsonl)
    echo -e "${GREEN}  📊 Total records: ${LINES}${NC}"
fi
