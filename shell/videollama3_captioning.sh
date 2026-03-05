#!/usr/bin/env bash
set -xeuo pipefail

export OMP_NUM_THREADS=1
export PYTHONPATH="${PYTHONPATH:-}:."
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

META_PATH="anno_data/finetune_online.json"
OUTPUT_DIR="work_dirs/online_caption_batches"
OUTPUT_FILE="${OUTPUT_DIR}/captions.jsonl"

mkdir -p "${OUTPUT_DIR}"

PYTHONPATH=. /miniconda/envs/video/bin/python videollama3/inference/captioning.py \
  --model_name_or_path "pretrained_models/compression/" \
  --meta_data_path "${META_PATH}" \
  --output_file "${OUTPUT_FILE}" \
  --num_inference_repeats 3 \
  --do_sample True \
  --max_new_tokens 512 \
  --temperature 0.2 \
  --top_k 50 \
  --top_p 0.9 \
  --num_beams 1 \
  --repetition_penalty 1.05 \
  --bf16 True \
  --seed 42 \
  --max_frames 70 \
  --fps 1 \
  --video_merge_size 2 \
  --compression_ratio 0.5 \
  --compression_window_size 10 \
  "$@"
