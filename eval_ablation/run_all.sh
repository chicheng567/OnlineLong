#!/usr/bin/env bash
# End-to-end prune-vs-noprune compressor eval:
#   1. (optional) wait until BOTH training runs have written their final model
#   2. feature-distribution comparison
#   3. caption-ability comparison (non-greedy, repetition_penalty=1.1, dynamic HW)
#   4. stitch a REPORT.md
#
# Trained by shell/pretrain_compressor_prune_ablation.sh:
#   prune  -> work_dirs/compressor_prune_ablation/prune0.30
#   noprune-> work_dirs/compressor_prune_ablation/noprune
#
# Usage:
#   bash eval_ablation/run_all.sh                     # wait for training, then eval on cuda:0
#   WAIT=0 DEVICE=cuda:7 bash eval_ablation/run_all.sh
#   NOPRUNE_DIR=work_dirs/compressor_prune_ablation/noprune/checkpoint-9545 WAIT=0 bash eval_ablation/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

PRUNE_DIR=${PRUNE_DIR:-work_dirs/compressor_prune_ablation/prune0.30}
NOPRUNE_DIR=${NOPRUNE_DIR:-work_dirs/compressor_prune_ablation/noprune}
OUT_DIR=${OUT_DIR:-work_dirs/ablation_eval}
DEVICE=${DEVICE:-cuda:0}
MANIFEST=${MANIFEST:-eval_ablation/manifest.json}
MAX_FRAMES=${MAX_FRAMES:-64}
WINDOW_SIZE=${WINDOW_SIZE:-8}
FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE:-448}
NUM_FEATURE_VIDEOS=${NUM_FEATURE_VIDEOS:-12}
REP_PENALTY=${REP_PENALTY:-1.1}
WAIT=${WAIT:-1}

model_ready () { [ -f "$1/model.safetensors.index.json" ] || [ -f "$1/model.safetensors" ]; }

if [ "${WAIT}" = "1" ]; then
    for d in "${PRUNE_DIR}" "${NOPRUNE_DIR}"; do
        while ! model_ready "${d}"; do
            echo "[wait] $(date +%H:%M:%S)  ${d} has no final model yet; sleeping 120s"
            sleep 120
        done
        echo "[ok] ${d} ready"
    done
fi
for d in "${PRUNE_DIR}" "${NOPRUNE_DIR}"; do
    model_ready "${d}" || { echo "[ERROR] ${d} has no loadable model (use a checkpoint-* dir or WAIT=1)"; exit 1; }
done

mkdir -p "${OUT_DIR}"
if [ ! -f "${MANIFEST}" ]; then
    echo "[manifest] building ${MANIFEST}"
    python eval_ablation/build_manifest.py --num 12
fi

echo "======================================================================"
echo " 1/2  feature-distribution comparison"
echo "======================================================================"
python eval_ablation/feature_distribution.py \
    --models "prune=${PRUNE_DIR}" "noprune=${NOPRUNE_DIR}" \
    --manifest "${MANIFEST}" --num_videos "${NUM_FEATURE_VIDEOS}" \
    --out "${OUT_DIR}/feature_distribution" --device "${DEVICE}" \
    --max_frames "${MAX_FRAMES}" --window_size "${WINDOW_SIZE}" \
    --force_image_size "${FORCE_IMAGE_SIZE}"

echo "======================================================================"
echo " 2/2  caption-ability comparison"
echo "======================================================================"
python eval_ablation/caption_eval.py \
    --models "prune=${PRUNE_DIR}" "noprune=${NOPRUNE_DIR}" \
    --manifest "${MANIFEST}" \
    --out "${OUT_DIR}/caption_eval" --device "${DEVICE}" \
    --max_frames "${MAX_FRAMES}" --window_size "${WINDOW_SIZE}" \
    --force_image_size "${FORCE_IMAGE_SIZE}" --repetition_penalty "${REP_PENALTY}"

{
    echo "# Prune-vs-noprune compressor ablation -- combined report"
    echo
    echo "- prune   : \`${PRUNE_DIR}\`"
    echo "- noprune : \`${NOPRUNE_DIR}\`"
    echo "- manifest: \`${MANIFEST}\`  | max_frames=${MAX_FRAMES} window=${WINDOW_SIZE} force_image_size=${FORCE_IMAGE_SIZE}"
    echo
    echo "---"
    echo
    cat "${OUT_DIR}/feature_distribution/feature_distribution.md"
    echo
    echo "---"
    echo
    cat "${OUT_DIR}/caption_eval/caption_eval.md"
} > "${OUT_DIR}/REPORT.md"

echo
echo "[done] ${OUT_DIR}/REPORT.md"
