#!/usr/bin/env bash
# image-mix-vs-video-only compressor eval.
#   image_mix : work_dirs/compressor_pretrain_image_mix                 (977k InternVid + 2M still images)
#   noprune   : work_dirs/compressor_prune_ablation/noprune            (977k InternVid only)
# Same arch (transformer_decoder_flat, num_queries=64, 8 layers), same compressor_lr=2e-5,
# same frozen LLM/projector/encoder, both 5 epochs, token_prune_ratio=0 for both.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

IMAGEMIX_DIR=${IMAGEMIX_DIR:-work_dirs/compressor_pretrain_image_mix}
NOPRUNE_DIR=${NOPRUNE_DIR:-work_dirs/compressor_prune_ablation/noprune}
OUT_DIR=${OUT_DIR:-work_dirs/image_mix_eval}
MANIFEST=${MANIFEST:-eval_ablation/manifest_imagemix.json}
MAX_FRAMES=${MAX_FRAMES:-64}
WINDOW_SIZE=${WINDOW_SIZE:-8}
FORCE_IMAGE_SIZE=${FORCE_IMAGE_SIZE:-448}
REP_PENALTY=${REP_PENALTY:-1.1}
FD_DEVICE=${FD_DEVICE:-cuda:0}
CAP_DEVICE=${CAP_DEVICE:-cuda:1}
NUM_FEATURE_VIDEOS=${NUM_FEATURE_VIDEOS:-16}

mkdir -p "${OUT_DIR}"

echo "[launch $(date +%H:%M:%S)] feature_distribution -> ${FD_DEVICE}"
python eval_ablation/feature_distribution.py \
    --models "noprune=${NOPRUNE_DIR}" "image_mix=${IMAGEMIX_DIR}" \
    --manifest "${MANIFEST}" --num_videos "${NUM_FEATURE_VIDEOS}" \
    --out "${OUT_DIR}/feature_distribution" --device "${FD_DEVICE}" \
    --max_frames "${MAX_FRAMES}" --window_size "${WINDOW_SIZE}" \
    --force_image_size "${FORCE_IMAGE_SIZE}" \
    > "${OUT_DIR}/feature_distribution.log" 2>&1 &
FD_PID=$!

echo "[launch $(date +%H:%M:%S)] caption_eval -> ${CAP_DEVICE}"
python eval_ablation/caption_eval.py \
    --models "noprune=${NOPRUNE_DIR}" "image_mix=${IMAGEMIX_DIR}" \
    --manifest "${MANIFEST}" --num_videos 0 \
    --out "${OUT_DIR}/caption_eval" --device "${CAP_DEVICE}" \
    --max_frames "${MAX_FRAMES}" --window_size "${WINDOW_SIZE}" \
    --force_image_size "${FORCE_IMAGE_SIZE}" --repetition_penalty "${REP_PENALTY}" \
    > "${OUT_DIR}/caption_eval.log" 2>&1 &
CAP_PID=$!

FAIL=0
wait $FD_PID  || { echo "[ERROR] feature_distribution failed"; FAIL=1; }
wait $CAP_PID || { echo "[ERROR] caption_eval failed"; FAIL=1; }

{
    echo "# image-mix vs video-only compressor pretrain -- combined report"
    echo
    echo "- image_mix : \`${IMAGEMIX_DIR}\`  (977k InternVid cached-feature + 2M still images)"
    echo "- noprune   : \`${NOPRUNE_DIR}\`  (977k InternVid cached-feature only)"
    echo "- manifest  : \`${MANIFEST}\` (28 held-out clips: 16 sharegpt4v + 12 vcg_20k, real gold captions)"
    echo "- eval geometry: max_frames=${MAX_FRAMES} window=${WINDOW_SIZE} force_image_size=${FORCE_IMAGE_SIZE} rep_penalty=${REP_PENALTY}"
    echo
    echo "---"
    echo
    [ -f "${OUT_DIR}/feature_distribution/feature_distribution.md" ] && cat "${OUT_DIR}/feature_distribution/feature_distribution.md"
    echo
    echo "---"
    echo
    [ -f "${OUT_DIR}/caption_eval/caption_eval.md" ] && cat "${OUT_DIR}/caption_eval/caption_eval.md"
} > "${OUT_DIR}/REPORT.md"

echo "[done $(date +%H:%M:%S)] ${OUT_DIR}/REPORT.md  (FAIL=${FAIL})"
exit $FAIL
