# Patch Similarity Analysis Tool

This tool analyzes the similarity of vision encoder patches across video frames, extracting raw patches **before** the `merge_size` interpolation operation.

## Overview

The script performs three types of similarity analysis:

1. **Position-wise Similarity**: Compares patches at the same spatial location across different frames
   - For each position (0-1023 for 32×32 grid), computes a cosine similarity matrix [T×T]
   - Helps understand temporal changes at specific spatial locations

2. **Frame-wise Similarity**: Compares overall frame representations
   - Averages all patches within each frame to get frame-level embeddings
   - Computes cosine similarity between frames [T×T]
   - Helps understand semantic similarity between frames

3. **Intra-Frame Similarity**: Compares patches within each individual frame
   - For each frame, computes cosine similarity between all patch pairs [1024×1024]
   - Helps understand spatial relationships and patch diversity within frames
   - Useful for analyzing local vs global features

## Requirements

- PyTorch with CUDA support
- transformers
- matplotlib
- seaborn
- numpy
- tqdm
- videollama3 modules (included in this repo)

## Usage

### Basic Usage

```bash
python analyze_patch_similarity.py \
    --video_path /path/to/video.mp4 \
    --model_path pretrained_models/videollama3_7b_local \
    --output_dir ./patch_analysis_results \
    --max_frames 100 \
    --fps 1
```

### Using the Shell Script

```bash
# Default usage
./run_similarity_analysis.sh /path/to/video.mp4

# With custom parameters
./run_similarity_analysis.sh \
    /path/to/video.mp4 \
    pretrained_models/videollama3_7b_local \
    ./results \
    50 \
    1 \
    448 \
    15
```

Parameters for shell script:
1. VIDEO_PATH (required)
2. MODEL_PATH (default: pretrained_models/videollama3_7b_local)
3. OUTPUT_DIR (default: ./patch_analysis_results)
4. MAX_FRAMES (default: 100)
5. FPS (default: 1)
6. IMAGE_SIZE (default: 448)
7. VISUALIZE_POSITIONS (default: 10)

## Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--video_path` | str | **required** | Path to input video (mp4, avi, etc.) |
| `--model_path` | str | `pretrained_models/videollama3_7b_local` | Path to pretrained vision encoder |
| `--output_dir` | str | `./patch_analysis_results` | Output directory for results |
| `--max_frames` | int | 100 | Maximum number of frames to process |
| `--fps` | int | 1 | Frames per second for video sampling |
| `--image_size` | int | 448 | Image size for vision encoder (must match training) |
| `--patch_size` | int | 14 | Patch size (must match training) |
| `--device` | str | `cuda` | Device to run on (cuda/cpu) |
| `--visualize_positions` | int | 10 | Number of position-wise similarities to visualize |

## Output Files

The script generates the following outputs in the specified `output_dir`:

### 1. NumPy Arrays

- **`position_similarities.npz`**: Compressed numpy archive
  - Shape: `[num_patches, T, T]` where num_patches=1024 for 448×448 images
  - Contains cosine similarity matrices for each spatial position
  - Load with: `data = np.load('position_similarities.npz'); sims = data['position_similarities']`

- **`frame_similarity.npy`**: Numpy array
  - Shape: `[T, T]`
  - Frame-level cosine similarity matrix
  - Load with: `sims = np.load('frame_similarity.npy')`

- **`intra_frame_similarities.npz`**: Compressed numpy archive
  - Shape: `[T, num_patches, num_patches]` where num_patches=1024
  - Contains patch-to-patch similarity within each frame
  - Load with: `data = np.load('intra_frame_similarities.npz'); sims = data['intra_frame_similarities']`

### 2. Visualizations

- **`frame_similarity.png`**: Heatmap of frame-wise similarity matrix
  - Shows semantic similarity between all frame pairs
  - Diagonal should be 1.0 (self-similarity)

- **`position_XXXX_rYY_cZZ.png`**: Heatmaps for selected positions
  - XXXX: Position index (0-1023)
  - YY: Row in 32×32 grid (0-31)
  - ZZ: Column in 32×32 grid (0-31)
  - Shows temporal similarity at specific spatial locations

- **`intra_frame_XXXX.png`**: Heatmaps for selected frames
  - XXXX: Frame index
  - Shows patch-to-patch similarity within that frame (1024×1024 matrix)
  - Helps visualize spatial structure and local feature similarity

### 3. Statistics

- **`statistics.txt`**: Text summary with statistics
  - Mean, std, min, max for both similarity types
  - Shape information

## Implementation Details

### Patch Extraction

The script uses a PyTorch hook to extract patches **before** the `merge_size` interpolation:

1. Registers a forward hook on `vision_encoder.post_layernorm`
2. Captures output after transformer layers but before spatial downsampling
3. For 448×448 images with patch_size=14: extracts 32×32=1024 patches per frame
4. Each patch has shape `[hidden_dim]` (e.g., 1024 or 1152 depending on model)

### Similarity Computation

**Position-wise**:
```python
# For each position p:
pos_patches = patches[:, p, :]  # [T, hidden_dim]
pos_patches_norm = F.normalize(pos_patches, p=2, dim=1)
similarity = pos_patches_norm @ pos_patches_norm.T  # [T, T]
```

**Frame-wise**:
```python
# Average all patches per frame:
frame_embs = patches.mean(dim=1)  # [T, hidden_dim]
frame_embs_norm = F.normalize(frame_embs, p=2, dim=1)
similarity = frame_embs_norm @ frame_embs_norm.T  # [T, T]
```

## Example Analysis Workflow

```python
import numpy as np
import matplotlib.pyplot as plt

# Load results
pos_data = np.load('patch_analysis_results/position_similarities.npz')
position_sims = pos_data['position_similarities']  # [1024, T, T]
frame_sims = np.load('patch_analysis_results/frame_similarity.npy')  # [T, T]

# Analyze specific position (e.g., center patch)
center_pos = 32 * 16 + 16  # Row 16, Col 16 (center of 32×32 grid)
center_sim = position_sims[center_pos]

# Find frames with high similarity to frame 0 at center position
sim_to_frame0 = center_sim[0, :]
most_similar_frames = np.argsort(sim_to_frame0)[::-1][:5]
print(f"Frames most similar to frame 0 at center: {most_similar_frames}")

# Compare position-wise vs frame-wise similarity
pos_mean_sim = position_sims.mean(axis=0)  # Average across all positions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(pos_mean_sim, cmap='viridis')
plt.title('Position-wise Similarity (averaged)')
plt.subplot(1, 2, 2)
plt.imshow(frame_sims, cmap='viridis')
plt.title('Frame-wise Similarity')
plt.show()
```

## Notes

- **Memory Usage**: Position-wise similarity requires significant memory for long videos
  - For 100 frames: ~400MB for position_similarities.npz
  - Consider reducing `--max_frames` if memory is limited

- **Processing Time**: Depends on video length and hardware
  - ~1-2 seconds per frame on GPU
  - Use `--max_frames` to limit processing time

- **Image Size**: Must match the vision encoder's training configuration
  - Default: 448 (as per shell/videollama3_test.sh)
  - Do not change unless you know the model was trained differently

- **Patch Grid**: For image_size=448 and patch_size=14:
  - Grid: 32×32 = 1024 patches
  - Position indexing: `pos = row * 32 + col`
  - Convert back: `row = pos // 32`, `col = pos % 32`

## Troubleshooting

**Issue**: `FileNotFoundError: Video not found`
- Ensure video path is correct and file exists
- Supported formats: mp4, avi, and other formats supported by decord

**Issue**: `RuntimeError: CUDA out of memory`
- Reduce `--max_frames`
- Use smaller `--image_size` (not recommended, changes model behavior)
- Use `--device cpu` (much slower)

**Issue**: Model loading fails
- Ensure `--model_path` points to a valid VideoLLaMA3 checkpoint
- Check that the model directory contains the vision encoder weights

**Issue**: Patches shape mismatch
- Verify `--image_size` and `--patch_size` match model training config
- For VideoLLaMA3: image_size=448, patch_size=14 (as per training script)
