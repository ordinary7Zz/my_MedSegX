#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ===== 用户需要修改的参数 =====
INPUT_DIR="${INPUT_DIR:-}"                    # 输入图像目录
OUTPUT_DIR="${OUTPUT_DIR:-}"                  # 输出掩码目录
TASK_NAME="${TASK_NAME:-US_ThyroidNodule}"    # 任务名
MODEL_WEIGHT="${MODEL_WEIGHT:-}"              # MedSegX 权重路径
SAM_CKPT_DIR="${SAM_CKPT_DIR:-./playground/SAM}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" || -z "${MODEL_WEIGHT}" ]]; then
    echo "Usage: INPUT_DIR=/path/to/images OUTPUT_DIR=/path/to/output MODEL_WEIGHT=/path/to/weight.pth bash $0"
    exit 1
fi

python "${REPO_ROOT}/infer_simple.py" \
    --input_dir "${INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --task_name "${TASK_NAME}" \
    --checkpoint "${SAM_CKPT_DIR}" \
    --model_type "${MODEL_TYPE}" \
    --model_weight "${MODEL_WEIGHT}" \
    --device "${DEVICE}"
