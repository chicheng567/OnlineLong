#!/bin/bash
# Example script to run patch similarity analysis

set -e

# Configuration
VIDEO_PATH="${1:-/path/to/your/video.mp4}"
MODEL_PATH="${2:-pretrained_models/videollama3_7b_local}"
OUTPUT_DIR="${3:-./patch_analysis_results}"
MAX_FRAMES="${4:-100}"
FPS="${5:-1}"
IMAGE_SIZE="${6:-448}"
VISUALIZE_POSITIONS="${7:-10}"

# Set PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:.

echo "Running patch similarity analysis..."
echo "Video: ${VIDEO_PATH}"
echo "Model: ${MODEL_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Max Frames: ${MAX_FRAMES}"
echo "FPS: ${FPS}"
echo "Image Size: ${IMAGE_SIZE}"
echo ""

python analyze_patch_similarity.py \
    --video_path "${VIDEO_PATH}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_frames ${MAX_FRAMES} \
    --fps ${FPS} \
    --image_size ${IMAGE_SIZE} \
    --patch_size 14 \
    --device cuda \
    --visualize_positions ${VISUALIZE_POSITIONS}

echo ""
echo "Analysis complete! Results saved to ${OUTPUT_DIR}"
echo ""
echo "Output files:"
echo "  - position_similarities.npz: Position-wise similarity matrices [num_patches, T, T]"
echo "  - frame_similarity.npy: Frame-wise similarity matrix [T, T]"
echo "  - frame_similarity.png: Visualization of frame-wise similarity"
echo "  - position_*.png: Visualizations of selected position-wise similarities"
echo "  - statistics.txt: Statistical summary"
