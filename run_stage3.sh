#!/usr/bin/env bash
# ==============================================================================
# Script: run_stage3.sh
# Stage 3: Question-Answer Generation (QAG)
# ==============================================================================
# Usage: ./run_stage3.sh [LIMIT]
# Examples: 
#   ./run_stage3.sh         -> Chạy toàn bộ dữ liệu
#   ./run_stage3.sh 100     -> Chỉ chạy thử với 100 context đầu tiên
# ==============================================================================

LIMIT=$1

echo "=========================================================="
echo "🚀 Bắt đầu Stage 3: Question-Answer Generation (QAG)"
echo "=========================================================="

# Đảm bảo bạn đang đứng ở thư mục gốc của project
cd "$(dirname "$0")" || exit

if [ -z "$LIMIT" ]; then
    echo "▶ Đang chạy trên toàn bộ dữ liệu... (không có limit)"
    python -m src.03_qag_generator
else
    echo "▶ Đang chạy thử nghiệm với Limit = $LIMIT"
    python -m src.03_qag_generator --limit "$LIMIT"
fi

if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo "✅ Stage 3 hoàn tất thành công!"
    echo "=========================================================="
else
    echo "=========================================================="
    echo "❌ Lỗi xảy ra trong Stage 3. Vui lòng kiểm tra log ở trên."
    echo "=========================================================="
    exit 1
fi
