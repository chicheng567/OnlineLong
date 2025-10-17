"""
Patch Similarity Analysis Script

This script analyzes the similarity of vision encoder patches across video frames.
It extracts raw patches before the merge_size interpolation and computes:
1. Position-wise similarity: Cosine similarity between patches at the same spatial location across frames
2. Frame-wise similarity: Cosine similarity between frame-level representations (averaged patches)

Usage:
    python analyze_patch_similarity.py \
        --video_path /path/to/video.mp4 \
        --model_path pretrained_models/videollama3_7b_local \
        --output_dir ./patch_analysis_results \
        --max_frames 100 \
        --image_size 448
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import from videollama3
from videollama3.mm_utils import load_video
from videollama3.model.encoder import build_vision_encoder
from videollama3.model.videollama3_encoder.modeling_videollama3_encoder import (
    Videollama3VisionEncoderModel,
)
from transformers import AutoConfig


class PatchExtractor:
    """Hook-based extractor to capture raw patches before merge_size interpolation"""

    def __init__(self):
        self.raw_patches = None
        self.hook_handle = None

    def hook_fn(self, module, input, output):
        """Hook function to capture post_layernorm output"""
        # Output shape: [num_patches, hidden_dim]
        self.raw_patches = output.detach().cpu()

    def register_hook(self, model: Videollama3VisionEncoderModel):
        """Register hook on post_layernorm layer"""
        self.hook_handle = model.post_layernorm.register_forward_hook(self.hook_fn)

    def remove_hook(self):
        """Remove the registered hook"""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def get_patches(self) -> torch.Tensor:
        """Get the captured raw patches"""
        return self.raw_patches


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze patch similarity in video frames")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_models/videollama3_7b_local",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./patch_analysis_results",
        help="Directory to save results",
    )
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames to process")
    parser.add_argument("--fps", type=int, default=1, help="Frames per second for video sampling")
    parser.add_argument("--image_size", type=int, default=448, help="Image size for vision encoder")
    parser.add_argument("--patch_size", type=int, default=14, help="Patch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--visualize_positions", type=int, default=10, help="Number of positions to visualize")

    return parser.parse_args()


def load_model(model_path: str, device: str) -> Tuple[Videollama3VisionEncoderModel, int]:
    """Load vision encoder model"""
    print(f"Loading model from {model_path}...")

    # Create a minimal config for vision encoder
    class MinimalArgs:
        def __init__(self, model_path):
            self.mm_vision_encoder = model_path
            self.mm_vision_select_layer = -1
            self.mm_vision_select_feature = "patch"
            self.mm_attn_implementation = "sdpa"  # Use sdpa for compatibility
            self.torch_dtype = torch.bfloat16

    args = MinimalArgs(model_path)

    # Build vision encoder
    from videollama3.model.encoder import Videollama3VisionEncoder
    vision_encoder = Videollama3VisionEncoder(model_path, args, delay_load=False)
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()

    hidden_size = vision_encoder.hidden_size
    print(f"Model loaded. Hidden size: {hidden_size}")

    return vision_encoder.vision_encoder, hidden_size


def preprocess_frames(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    """
    Preprocess video frames to match vision encoder input format

    Args:
        frames: Tensor of shape [T, C, H, W] (uint8, 0-255)
        image_size: Target image size (448)

    Returns:
        pixel_values: Tensor of shape [T, C, image_size, image_size]
    """
    # Normalize to [0, 1]
    frames = frames.float() / 255.0

    # Resize to target size
    frames = F.interpolate(frames, size=(image_size, image_size), mode="bilinear", align_corners=False)

    # Normalize with ImageNet stats (typical for vision models)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    return frames


def extract_patches_from_frames(
    vision_encoder: Videollama3VisionEncoderModel,
    frames: torch.Tensor,
    image_size: int,
    patch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Extract raw patches from video frames using hook

    Args:
        vision_encoder: The vision encoder model
        frames: Preprocessed frames [T, C, H, W]
        image_size: Image size (448)
        patch_size: Patch size (14)
        device: Device to run on

    Returns:
        all_patches: Tensor of shape [T, num_patches, hidden_dim]
    """
    extractor = PatchExtractor()
    extractor.register_hook(vision_encoder)

    num_frames = frames.shape[0]
    patches_per_side = image_size // patch_size  # 448 // 14 = 32
    num_patches = patches_per_side * patches_per_side  # 1024

    all_patches = []

    print(f"Extracting patches from {num_frames} frames...")
    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Processing frames"):
            frame = frames[i:i+1].to(device)  # [1, C, H, W]

            # Prepare pixel_values in NaViT format
            # NaViT embeddings expects: [num_patches*patch_size*patch_size, C]
            # which will be reshaped internally to [-1, C, patch_size, patch_size]
            C, H, W = frame.shape[1:]

            # Unfold into patches: [C, patches_per_side, patches_per_side, patch_size, patch_size]
            frame_patches = frame.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).squeeze(0)

            # Rearrange to [patches_per_side, patches_per_side, C, patch_size, patch_size]
            frame_patches = frame_patches.permute(1, 2, 0, 3, 4)

            # Flatten to [num_patches, C, patch_size, patch_size]
            frame_patches = frame_patches.reshape(num_patches, C, patch_size, patch_size)

            # Further flatten to [num_patches*patch_size*patch_size, C]
            # This matches what NaViT embeddings.forward() expects
            pixel_values = frame_patches.permute(0, 2, 3, 1).reshape(-1, C).contiguous()

            # Convert to the same dtype as the model (bfloat16)
            pixel_values = pixel_values.to(vision_encoder.dtype)

            # Create grid_sizes and merge_sizes
            grid_sizes = torch.tensor([[1, patches_per_side, patches_per_side]], device=device)
            merge_sizes = torch.tensor([1], device=device)  # Set to 1 to avoid merging

            # Forward pass
            _ = vision_encoder(pixel_values, grid_sizes, merge_sizes)

            # Get captured patches
            patches = extractor.get_patches()  # [num_patches, hidden_dim]
            all_patches.append(patches)

    extractor.remove_hook()

    # Stack all patches: [T, num_patches, hidden_dim]
    all_patches = torch.stack(all_patches, dim=0)
    print(f"Extracted patches shape: {all_patches.shape}")

    return all_patches


def compute_position_wise_similarity(patches: torch.Tensor) -> np.ndarray:
    """
    Compute position-wise cosine similarity across frames

    Args:
        patches: [T, num_patches, hidden_dim]

    Returns:
        similarities: [num_patches, T, T] array of cosine similarities
    """
    T, num_patches, hidden_dim = patches.shape

    print(f"Computing position-wise similarity for {num_patches} positions...")
    similarities = np.zeros((num_patches, T, T), dtype=np.float32)

    for pos in tqdm(range(num_patches), desc="Computing position-wise similarity"):
        # Get patches at this position across all frames: [T, hidden_dim]
        pos_patches = patches[:, pos, :]  # [T, hidden_dim]

        # Normalize
        pos_patches_norm = F.normalize(pos_patches, p=2, dim=1)

        # Compute cosine similarity matrix: [T, T]
        sim_matrix = torch.mm(pos_patches_norm, pos_patches_norm.t())
        similarities[pos] = sim_matrix.float().cpu().numpy()  # Convert to float32 first

    return similarities


def compute_frame_wise_similarity(patches: torch.Tensor) -> np.ndarray:
    """
    Compute frame-wise cosine similarity (averaged patches)

    Args:
        patches: [T, num_patches, hidden_dim]

    Returns:
        similarity: [T, T] array of cosine similarities
    """
    print("Computing frame-wise similarity...")

    # Average patches for each frame: [T, hidden_dim]
    frame_embeddings = patches.mean(dim=1)

    # Normalize
    frame_embeddings_norm = F.normalize(frame_embeddings, p=2, dim=1)

    # Compute cosine similarity matrix: [T, T]
    similarity = torch.mm(frame_embeddings_norm, frame_embeddings_norm.t())

    return similarity.float().cpu().numpy()  # Convert to float32 first


def compute_intra_frame_similarity(patches: torch.Tensor) -> np.ndarray:
    """
    Compute intra-frame patch similarity (within each frame)

    Args:
        patches: [T, num_patches, hidden_dim]

    Returns:
        similarities: [T, num_patches, num_patches] array of cosine similarities
                      For each frame, compute similarity between all patch pairs
    """
    T, num_patches, hidden_dim = patches.shape

    print(f"Computing intra-frame similarity for {T} frames...")
    similarities = np.zeros((T, num_patches, num_patches), dtype=np.float32)

    for frame_idx in tqdm(range(T), desc="Computing intra-frame similarity"):
        # Get all patches in this frame: [num_patches, hidden_dim]
        frame_patches = patches[frame_idx]  # [num_patches, hidden_dim]

        # Normalize
        frame_patches_norm = F.normalize(frame_patches, p=2, dim=1)

        # Compute cosine similarity matrix: [num_patches, num_patches]
        sim_matrix = torch.mm(frame_patches_norm, frame_patches_norm.t())
        similarities[frame_idx] = sim_matrix.float().cpu().numpy()

    return similarities


def visualize_similarity_matrix(
    similarity: np.ndarray,
    title: str,
    save_path: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """Visualize a similarity matrix as heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        similarity,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel("Frame Index")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_position_similarities(
    position_similarities: np.ndarray,
    output_dir: Path,
    num_positions: int = 10,
    patches_per_side: int = 32,
):
    """
    Visualize similarity matrices for selected positions

    Args:
        position_similarities: [num_patches, T, T]
        output_dir: Output directory
        num_positions: Number of positions to visualize
        patches_per_side: Number of patches per side (32)
    """
    num_patches = position_similarities.shape[0]

    # Select positions to visualize (evenly spaced)
    positions = np.linspace(0, num_patches - 1, num_positions, dtype=int)

    print(f"Visualizing {num_positions} position similarity matrices...")

    for pos in tqdm(positions, desc="Creating position visualizations"):
        # Convert position to 2D coordinates
        row = pos // patches_per_side
        col = pos % patches_per_side

        similarity = position_similarities[pos]
        title = f"Position-wise Similarity (Position {pos}: Row {row}, Col {col})"
        save_path = output_dir / f"position_{pos:04d}_r{row}_c{col}.png"

        visualize_similarity_matrix(similarity, title, str(save_path))


def visualize_intra_frame_similarities(
    intra_frame_similarities: np.ndarray,
    output_dir: Path,
    num_frames: int = 5,
):
    """
    Visualize intra-frame similarity matrices for selected frames

    Args:
        intra_frame_similarities: [T, num_patches, num_patches]
        output_dir: Output directory
        num_frames: Number of frames to visualize
    """
    T = intra_frame_similarities.shape[0]

    # Select frames to visualize (evenly spaced)
    frame_indices = np.linspace(0, T - 1, num_frames, dtype=int)

    print(f"Visualizing {num_frames} intra-frame similarity matrices...")

    for frame_idx in tqdm(frame_indices, desc="Creating intra-frame visualizations"):
        similarity = intra_frame_similarities[frame_idx]
        title = f"Intra-Frame Patch Similarity (Frame {frame_idx})"
        save_path = output_dir / f"intra_frame_{frame_idx:04d}.png"

        visualize_similarity_matrix(similarity, title, str(save_path))


def save_results(
    position_similarities: np.ndarray,
    frame_similarity: np.ndarray,
    intra_frame_similarities: np.ndarray,
    output_dir: Path,
):
    """Save similarity arrays to disk"""
    print("Saving results...")

    # Save position-wise similarities (compressed)
    pos_save_path = output_dir / "position_similarities.npz"
    np.savez_compressed(pos_save_path, position_similarities=position_similarities)
    print(f"Saved position similarities to {pos_save_path}")

    # Save frame-wise similarity
    frame_save_path = output_dir / "frame_similarity.npy"
    np.save(frame_save_path, frame_similarity)
    print(f"Saved frame similarity to {frame_save_path}")

    # Save intra-frame similarities (compressed)
    intra_save_path = output_dir / "intra_frame_similarities.npz"
    np.savez_compressed(intra_save_path, intra_frame_similarities=intra_frame_similarities)
    print(f"Saved intra-frame similarities to {intra_save_path}")

    # Save statistics
    stats_path = output_dir / "statistics.txt"
    with open(stats_path, "w") as f:
        f.write("Patch Similarity Analysis Statistics\n")
        f.write("=" * 50 + "\n\n")

        f.write("Position-wise Similarities:\n")
        f.write(f"  Shape: {position_similarities.shape}\n")
        f.write(f"  Mean: {position_similarities.mean():.4f}\n")
        f.write(f"  Std: {position_similarities.std():.4f}\n")
        f.write(f"  Min: {position_similarities.min():.4f}\n")
        f.write(f"  Max: {position_similarities.max():.4f}\n\n")

        f.write("Frame-wise Similarity:\n")
        f.write(f"  Shape: {frame_similarity.shape}\n")
        f.write(f"  Mean: {frame_similarity.mean():.4f}\n")
        f.write(f"  Std: {frame_similarity.std():.4f}\n")
        f.write(f"  Min: {frame_similarity.min():.4f}\n")
        f.write(f"  Max: {frame_similarity.max():.4f}\n\n")

        f.write("Intra-Frame Similarities:\n")
        f.write(f"  Shape: {intra_frame_similarities.shape}\n")
        f.write(f"  Mean: {intra_frame_similarities.mean():.4f}\n")
        f.write(f"  Std: {intra_frame_similarities.std():.4f}\n")
        f.write(f"  Min: {intra_frame_similarities.min():.4f}\n")
        f.write(f"  Max: {intra_frame_similarities.max():.4f}\n")

    print(f"Saved statistics to {stats_path}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check video exists
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")

    # Load video
    print(f"Loading video from {args.video_path}...")
    frames, timestamps = load_video(args.video_path, fps=args.fps, max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames")

    # Convert frames to tensor
    # frames is a list of [C, H, W] arrays (from load_video line 521)
    frames_tensor = torch.stack([torch.from_numpy(f) for f in frames])  # [T, C, H, W]

    # Preprocess frames
    print("Preprocessing frames...")
    preprocessed_frames = preprocess_frames(frames_tensor, args.image_size)

    # Load model
    vision_encoder, hidden_size = load_model(args.model_path, args.device)

    # Extract patches
    patches = extract_patches_from_frames(
        vision_encoder,
        preprocessed_frames,
        args.image_size,
        args.patch_size,
        args.device,
    )

    # Compute similarities
    position_similarities = compute_position_wise_similarity(patches)
    frame_similarity = compute_frame_wise_similarity(patches)
    intra_frame_similarities = compute_intra_frame_similarity(patches)

    # Save results
    save_results(position_similarities, frame_similarity, intra_frame_similarities, output_dir)

    # Visualize frame-wise similarity
    print("Visualizing frame-wise similarity...")
    visualize_similarity_matrix(
        frame_similarity,
        "Frame-wise Cosine Similarity",
        str(output_dir / "frame_similarity.png"),
    )

    # Visualize selected position-wise similarities
    patches_per_side = args.image_size // args.patch_size
    visualize_position_similarities(
        position_similarities,
        output_dir,
        num_positions=args.visualize_positions,
        patches_per_side=patches_per_side,
    )

    # Visualize selected intra-frame similarities
    visualize_intra_frame_similarities(
        intra_frame_similarities,
        output_dir,
        num_frames=5,  # Visualize 5 frames
    )

    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
