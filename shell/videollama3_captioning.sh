#!/usr/bin/env bash
set -xeuo pipefail

export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-}:."
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

META_PATH="anno_data/finetune_online.json"
OUTPUT_DIR="work_dirs/online_caption_batches"

mkdir -p "${OUTPUT_DIR}"

/miniconda/envs/onlinellama3/bin/python videollama3/inference/captionoing.py \
  --model-path pretrained_models/videollama3_7b_local \
  --dataset-meta "${META_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-frames 100 \
  --fps 1 \
  --stride 100 \
  --video-merge-size 2 \
  --max-new-tokens 512 \
  --temperature 0.2 \
  --top-p 0.9 \
  --num-beams 1 \
  --repetition-penalty 1.05 \
  --device-map auto \
  --dtype bfloat16 \
  --attn-impl flash_attention_2 \
  --log-every 1 \
  --seed 42 \
  "$@"
