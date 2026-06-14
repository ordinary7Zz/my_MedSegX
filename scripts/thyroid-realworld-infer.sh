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
DEVICE_IDS="${DEVICE_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
METRICS="${METRICS:-dsc hd}"
CI_OUTPUT="${CI_OUTPUT:-${PLAYGROUND_ROOT}/MedSegX/external/thyroid_ci_summary.csv}"

DATASETS=(finall_data DDTI PKTN ThyroidXL TN3K TN5K)

read -r -a DEVICE_ID_ARR <<< "${DEVICE_IDS}"
read -r -a METRIC_ARR <<< "${METRICS}"

RESULT_DIR="$(dirname "${MODEL_WEIGHT}")/external"
mkdir -p "${RESULT_DIR}"

for dataset in "${DATASETS[@]}"; do
    data_path="${PLAYGROUND_ROOT}/${dataset}/eval/RealWorld"
    inference_dir="${data_path}/${SHIFT_TYPE}/${TASK_NAME}/${dataset}/inference"

    if [[ ! -d "${inference_dir}/npy_imgs" || ! -d "${inference_dir}/npy_gts" ]]; then
        echo "[skip] ${dataset}: missing ${inference_dir}/npy_imgs or npy_gts"
        continue
    fi

    echo "[run] ${dataset}"
    python "${REPO_ROOT}/evaluate_external.py" \
        --checkpoint "${SAM_CKPT_DIR}" \
        --model_type "${MODEL_TYPE}" \
        --model_weight "${MODEL_WEIGHT}" \
        --data_path "${data_path}" \
        --shift_type "${SHIFT_TYPE}" \
        --metric "${METRIC_ARR[@]}" \
        --device "${DEVICE}" \
        --device_ids "${DEVICE_ID_ARR[@]}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}"

    src_md="${RESULT_DIR}/RealWorld-site.md"
    src_csv="${RESULT_DIR}/RealWorld-site.csv"
    dst_md="${RESULT_DIR}/${dataset}-RealWorld-site.md"
    dst_csv="${RESULT_DIR}/${dataset}-RealWorld-site.csv"

    [[ -f "${src_md}" ]] && mv -f "${src_md}" "${dst_md}"
    [[ -f "${src_csv}" ]] && mv -f "${src_csv}" "${dst_csv}"

    echo "[done] ${dataset} -> ${dst_md}"
done

echo "[run] calculating 95% confidence intervals"
python "${REPO_ROOT}/scripts/calc_metric_ci.py" \
    --input_dir "${RESULT_DIR}" \
    --pattern "*-RealWorld-site.csv" \
    --output "${CI_OUTPUT}"

echo "[done] confidence intervals -> ${CI_OUTPUT}"
