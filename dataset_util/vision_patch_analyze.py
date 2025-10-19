import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import argparse
import os

from videollama3.model.videollama3_qwen2 import Videollama3Qwen2ForCausalLM
from videollama3.model.videollama3_encoder.image_processing_videollama3 import Videollama3ImageProcessor
from videollama3.mm_utils import load_video


def compute_cosine_similarity(tensor1, tensor2):
    """
    Compute cosine similarity between two tensors.
    Args:
        tensor1: (N, D) or (D,)
        tensor2: (M, D) or (D,)
    Returns:
        similarity matrix (N, M) or scalar
    """
    # Normalize
    tensor1_norm = tensor1 / (tensor1.norm(dim=-1, keepdim=True) + 1e-8)
    tensor2_norm = tensor2 / (tensor2.norm(dim=-1, keepdim=True) + 1e-8)

    # Compute cosine similarity
    if tensor1.dim() == 1 and tensor2.dim() == 1:
        return (tensor1_norm * tensor2_norm).sum()
    elif tensor1.dim() == 2 and tensor2.dim() == 2:
        return torch.mm(tensor1_norm, tensor2_norm.t())
    else:
        raise ValueError("Unsupported tensor dimensions")


def plot_frame_similarity(patches, timestamps, output_dir):
    """
    Plot 1: Frame-to-frame semantic similarity using average patch embeddings.
    Args:
        patches: (num_frames, num_patches, hidden_size)
        timestamps: list of timestamps
        output_dir: output directory path
    """
    print("Computing frame-to-frame similarity...")

    # Average pooling across patches for each frame
    frame_embeddings = patches.mean(dim=1)  # (num_frames, hidden_size)

    # Compute pairwise cosine similarity
    similarity_matrix = compute_cosine_similarity(frame_embeddings, frame_embeddings)
    similarity_matrix = similarity_matrix.cpu().float().numpy()

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        cmap='coolwarm',
        center=0.5,
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=[f"{t:.1f}s" for t in timestamps],
        yticklabels=[f"{t:.1f}s" for t in timestamps],
        cbar_kws={'label': 'Cosine Similarity'}
    )
    plt.title('Frame-to-Frame Semantic Similarity', fontsize=16, fontweight='bold')
    plt.xlabel('Frame Timestamp', fontsize=12)
    plt.ylabel('Frame Timestamp', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'frame_similarity_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved frame similarity heatmap to: {output_path}")
    plt.close()

    return similarity_matrix


def plot_token_similarity_and_pca(patches, frame_idx, timestamp, output_dir, original_frame=None):
    """
    Plot 2: For a specific frame, show token-to-token similarity and PCA visualization.
    Args:
        patches: (num_patches, hidden_size) for a single frame
        frame_idx: frame index
        timestamp: timestamp of the frame
        output_dir: output directory path
        original_frame: original frame image for reference (optional)
    """
    print(f"Analyzing frame {frame_idx} at {timestamp:.1f}s...")

    # Compute token-to-token cosine similarity
    token_similarity = compute_cosine_similarity(patches, patches)
    token_similarity = token_similarity.cpu().float().numpy()

    # PCA to 3D
    pca = PCA(n_components=3)
    patches_np = patches.cpu().float().numpy()
    patches_3d = pca.fit_transform(patches_np)

    # Normalize PCA coordinates to [0, 1] for RGB visualization
    patches_3d_normalized = (patches_3d - patches_3d.min(axis=0)) / (patches_3d.max(axis=0) - patches_3d.min(axis=0) + 1e-8)

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))

    # Subplot 1: Token similarity heatmap
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(token_similarity, cmap='coolwarm', vmin=0, vmax=1)
    ax1.set_title(f'Token-to-Token Similarity\nFrame {frame_idx} @ {timestamp:.1f}s',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Token Index', fontsize=11)
    ax1.set_ylabel('Token Index', fontsize=11)
    plt.colorbar(im1, ax=ax1, label='Cosine Similarity')

    # Subplot 2: PCA 3D scatter plot
    ax2 = plt.subplot(1, 3, 2, projection='3d')
    ax2.scatter(
        patches_3d[:, 0],
        patches_3d[:, 1],
        patches_3d[:, 2],
        c=patches_3d_normalized,  # RGB colors from normalized PCA
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidths=0.5
    )
    ax2.set_title(f'PCA 3D Visualization\nFrame {frame_idx} @ {timestamp:.1f}s',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
    ax2.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)', fontsize=10)

    # Subplot 3: Original frame (if provided) or PCA 2D projection
    ax3 = plt.subplot(1, 3, 3)
    if original_frame is not None:
        # Handle different image formats (PIL Image, torch tensor, numpy array)
        if isinstance(original_frame, torch.Tensor):
            # Convert tensor to numpy: (C, H, W) -> (H, W, C)
            img_np = original_frame.permute(1, 2, 0).cpu().numpy()
        elif hasattr(original_frame, 'mode'):
            # PIL Image
            img_np = np.array(original_frame)
        else:
            # Numpy array
            img_np = np.array(original_frame)
            # If shape is (C, H, W), transpose to (H, W, C)
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:
                img_np = np.transpose(img_np, (1, 2, 0))

        # Normalize to [0, 1] if needed
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

        ax3.imshow(img_np)
        ax3.set_title(f'Original Frame {frame_idx}', fontsize=14, fontweight='bold')
        ax3.axis('off')
    else:
        ax3.scatter(
            patches_3d[:, 0],
            patches_3d[:, 1],
            c=patches_3d_normalized,
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5
        )
        ax3.set_title(f'PCA 2D Projection (PC1 vs PC2)\nFrame {frame_idx} @ {timestamp:.1f}s',
                      fontsize=14, fontweight='bold')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'frame_{frame_idx:04d}_token_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved token analysis for frame {frame_idx} to: {output_path}")
    plt.close()

    return token_similarity, patches_3d, pca


def main():
    parser = argparse.ArgumentParser(description='Vision Patch Semantic Analysis for VideoLLaMA3')
    parser.add_argument('--video_path', type=str, default='v__7a80bvsbk8.mp4',
                        help='Path to input video file')
    parser.add_argument('--model_path', type=str, default='pretrained_models/videollama3_7b_local',
                        help='Path to pretrained VideoLLaMA3 model')
    parser.add_argument('--output_dir', type=str, default='vision_patch_analysis',
                        help='Output directory for analysis results')
    parser.add_argument('--fps', type=int, default=1,
                        help='Frames per second for video sampling')
    parser.add_argument('--max_frames', type=int, default=200,
                        help='Maximum number of frames to process')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run model on')
    parser.add_argument('--sample_frames', type=int, nargs='+', default=None,
                        help='Specific frame indices to analyze in detail (e.g., --sample_frames 0 10 20)')
    parser.add_argument('--num_sample_frames', type=int, default=5,
                        help='Number of evenly-spaced frames to analyze if --sample_frames not specified')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    vl3 = Videollama3Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    vl3 = vl3.eval()

    # Initialize image processor
    image_processor = Videollama3ImageProcessor(
        do_convert_rgb=True,
        do_normalize=True,
        do_rescale=True,
        do_resize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        max_tokens=16384,
        min_tokens=16,
        patch_size=14,
        resample=3,
        rescale_factor=0.00392156862745098,
        merge_size=1,
        force_size=[448, 448],
    )

    # Load video
    print(f"Loading video from {args.video_path}...")
    frames, timestamps = load_video(args.video_path, fps=args.fps, max_frames=args.max_frames)
    print(f"Loaded {len(frames)} frames")

    # Forward through vision encoder
    print("Processing frames through vision encoder...")
    vision_encoder = vl3.get_vision_encoder()

    with torch.no_grad():
        inputs = image_processor(frames, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=args.device, dtype=torch.bfloat16)
        merge_sizes = inputs["merge_sizes"].to(args.device)
        grid_sizes = inputs["grid_sizes"].to(args.device)

        patches = vision_encoder(pixel_values=pixel_values, merge_sizes=merge_sizes, grid_sizes=grid_sizes)
        patches = patches.view(len(frames), -1, patches.size(-1))  # (num_frames, num_patches, hidden_size)

    print(f"Patches shape: {patches.shape}")  # (num_frames, num_patches, hidden_size)

    # Analysis 1: Frame-to-frame similarity
    plot_frame_similarity(patches, timestamps, args.output_dir)

    # Analysis 2: Token-level analysis for selected frames
    if args.sample_frames is not None:
        sample_indices = args.sample_frames
    else:
        # Evenly sample frames
        sample_indices = np.linspace(0, len(frames) - 1, args.num_sample_frames, dtype=int).tolist()

    print(f"\nAnalyzing {len(sample_indices)} sample frames: {sample_indices}")

    for frame_idx in sample_indices:
        if frame_idx >= len(frames):
            print(f"Warning: Frame index {frame_idx} out of range, skipping...")
            continue

        frame_patches = patches[frame_idx]  # (num_patches, hidden_size)
        frame_timestamp = timestamps[frame_idx]
        original_frame = frames[frame_idx]  # PIL Image or numpy array

        plot_token_similarity_and_pca(
            frame_patches,
            frame_idx,
            frame_timestamp,
            args.output_dir,
            original_frame=original_frame
        )

    print(f"\nAnalysis complete! All results saved to: {args.output_dir}")
    print(f"Generated files:")
    print(f"  - frame_similarity_heatmap.png")
    for idx in sample_indices:
        if idx < len(frames):
            print(f"  - frame_{idx:04d}_token_analysis.png")


if __name__ == "__main__":
    main()