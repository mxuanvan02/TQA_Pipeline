#!/usr/bin/env bash
# ==============================================================================
# Script: run_stage4.sh
# Stage 4 & 5: Evaluation (LLM-as-a-judge) & Output Formatting
# ==============================================================================
# Usage: ./run_stage4.sh [LIMIT]
# Examples: 
#   ./run_stage4.sh         -> Đánh giá toàn bộ QA pairs
#   ./run_stage4.sh 500     -> Chỉ chạy thử với 500 QA pairs đầu tiên
# ==============================================================================

LIMIT=$1

echo "=========================================================="
echo "🚀 Bắt đầu Stage 4+5: Evaluation & Dataset Formatting"
echo "=========================================================="

# Đảm bảo bạn đang đứng ở thư mục gốc của project
cd "$(dirname "$0")" || exit

if [ -z "$LIMIT" ]; then
    echo "▶ Đang chạy trên toàn bộ dữ liệu... (không có limit)"
    python -m src.04_evaluate
else
    echo "▶ Đang chạy thử nghiệm với Limit = $LIMIT"
    python -m src.04_evaluate --limit "$LIMIT"
fi

if [ $? -eq 0 ]; then
    echo "=========================================================="
    echo "✅ Stage 4+5 hoàn tất thành công! Dataset .jsonl đã sẵn sàng."
    echo "=========================================================="
else
    echo "=========================================================="
    echo "❌ Lỗi xảy ra trong Stage 4. Vui lòng kiểm tra log ở trên."
    echo "=========================================================="
    exit 1
fi
