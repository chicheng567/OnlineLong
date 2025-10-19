#!/bin/bash
# Vision Patch Analysis Runner Script

set -e

# Set environment
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Default parameters
VIDEO_PATH="${1:-v__7a80bvsbk8.mp4}"
OUTPUT_DIR="${2:-vision_patch_analysis}"
FPS="${FPS:-1}"
MAX_FRAMES="${MAX_FRAMES:-200}"
NUM_SAMPLE_FRAMES="${NUM_SAMPLE_FRAMES:-10}"
NUM_PATCHES_ANALYZE="${NUM_PATCHES_ANALYZE:-8}"

echo "========================================="
echo "Vision Patch Semantic Analysis"
echo "========================================="
echo "Video: $VIDEO_PATH"
echo "Output: $OUTPUT_DIR"
echo "FPS: $FPS"
echo "Max Frames: $MAX_FRAMES"
echo "Sample Frames: $NUM_SAMPLE_FRAMES"
echo "Patches to Analyze: $NUM_PATCHES_ANALYZE"
echo "========================================="

/miniconda/envs/onlinellama3/bin/python dataset_util/vision_patch_analyze.py \
    --video_path "$VIDEO_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --max_frames "$MAX_FRAMES" \
    --num_sample_frames "$NUM_SAMPLE_FRAMES" \
    --num_patches_analyze "$NUM_PATCHES_ANALYZE" \
    --device cuda:0

echo ""
echo "========================================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
