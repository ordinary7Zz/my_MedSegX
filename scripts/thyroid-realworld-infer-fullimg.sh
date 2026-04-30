#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLAYGROUND_ROOT="${PLAYGROUND_ROOT:-/mnt/wangbd8/workspace/ThyroidAgent/MedSegX-code/playground}"
SAM_CKPT_DIR="${SAM_CKPT_DIR:-${PLAYGROUND_ROOT}/SAM}"
MODEL_WEIGHT="${MODEL_WEIGHT:-${PLAYGROUND_ROOT}/MedSegX/medsegx_vit_b.pth}"
MODEL_TYPE="${MODEL_TYPE:-vit_b}"
SHIFT_TYPE="${SHIFT_TYPE:-cross_site}"
TASK_NAME="${TASK_NAME:-US_ThyroidNodule}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"

DATASETS=(DDTI PKTN ThyroidXL TN3K TN5K)

RESULT_DIR="$(dirname "${MODEL_WEIGHT}")/external_fullimg"
mkdir -p "${RESULT_DIR}"

for dataset in "${DATASETS[@]}"; do
    data_path="${PLAYGROUND_ROOT}/${dataset}/eval/RealWorld"
    inference_dir="${data_path}/${SHIFT_TYPE}/${TASK_NAME}/${dataset}/inference"
    output_dir="${RESULT_DIR}/${dataset}"

    if [[ ! -d "${inference_dir}/npy_imgs" ]]; then
        echo "[skip] ${dataset}: missing ${inference_dir}/npy_imgs"
        continue
    fi

    echo "[run] ${dataset}"
    python "${REPO_ROOT}/infer_external_fullimg.py" \
        --checkpoint "${SAM_CKPT_DIR}" \
        --model_type "${MODEL_TYPE}" \
        --model_weight "${MODEL_WEIGHT}" \
        --input_dir "${inference_dir}" \
        --task_name "${TASK_NAME}" \
        --output_dir "${output_dir}" \
        --device "${DEVICE}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}"

    echo "[done] ${dataset} -> ${output_dir}"
done
