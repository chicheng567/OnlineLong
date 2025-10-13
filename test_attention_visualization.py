import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from videollama3.model.videollama3_qwen2 import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.model.videollama3_encoder.image_processing_videollama3 import Videollama3ImageProcessor
from transformers import AutoTokenizer
from videollama3.constants import DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN
from videollama3.mm_utils import load_video

# ============ Configuration ============
model_local = "pretrained_models/videollama3_7b_local"
video_path = "/workspace/datasetfortest/ActivityNets/v__Af_9cK5x4E.mp4"
device = "cuda:0"
output_dir = "./attention_visualizations"
Path(output_dir).mkdir(parents=True, exist_ok=True)

# ============ Load Model ============
print("Loading model...")
model = Videollama3Qwen2ForCausalLM.from_pretrained(
    model_local,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",  # Changed from flash_attention_2 to capture attention weights
)
model.eval()

# ============ Load Processor ============
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
    rescale_factor=0.00392156862745098
)
tokenizer = AutoTokenizer.from_pretrained("pretrained_models/videollama3_7b_local")
tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN, STREAM_START_TOKEN, STREAM_END_TOKEN], special_tokens=True)
processor = Videollama3Processor(image_processor, tokenizer)

# ============ Prepare Input ============
print("Loading video...")
frames, timestamps = load_video(video_path, fps=1, max_frames=8)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
            {"type": "text", "text": "<video>\nDescribe this video."},
        ]
    }
]
inputs = processor(
    images=[frames],
    text=conversation,
    merge_size=2,  # Use merge_size=2 to reduce visual tokens
    return_tensors="pt",
)
inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
inputs["modals"] = ["video"]

# Print input sequence length
print(f"Input sequence length: {inputs['input_ids'].shape[1]}")
print(f"Number of frames: {len(frames)}")
print(f"Estimated attention matrix size per layer: {inputs['input_ids'].shape[1]}x{inputs['input_ids'].shape[1]}")

# ============ Attention Capture Setup ============
attention_maps = {}

num_layers = len(model.model.layers)
print(f"Total layers: {num_layers}")

# Only capture attention from a few representative layers
layers_to_capture = [0, num_layers//4, num_layers//2, num_layers*3//4, num_layers-1]
print(f"Capturing attention from layers: {layers_to_capture}")

# ============ Forward Pass ============
print("Running forward pass with attention capture...")
with torch.no_grad():
    # Single forward pass with output_attentions=True
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        pixel_values=inputs.get("pixel_values"),
        grid_sizes=inputs.get("grid_sizes"),
        merge_sizes=inputs.get("merge_sizes"),
        modals=inputs["modals"],
        output_attentions=True,  # Enable attention output
    )

    # Extract attention weights from outputs
    if hasattr(outputs, 'attentions') and outputs.attentions is not None:
        print(f"Successfully captured attentions from {len(outputs.attentions)} layers")
        # Only keep the layers we're interested in
        for layer_idx in layers_to_capture:
            if layer_idx < len(outputs.attentions):
                attn = outputs.attentions[layer_idx]
                if attn is not None:
                    attention_maps[f"layer_{layer_idx}"] = attn.detach().cpu()
                    print(f"  layer_{layer_idx} attention shape: {attn.shape}")
    else:
        print("Warning: No attention weights in outputs!")

# Clean up memory before generation
print("Cleaning up memory...")
import gc
torch.cuda.empty_cache()
gc.collect()

# Generate response (skip if still OOM)
try:
    print("Generating response...")
    output_ids = model.generate(**inputs, max_new_tokens=512, output_attentions=False)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\n" + "="*50)
    print("GENERATED RESPONSE:")
    print("="*50)
    print(response)
    print("="*50 + "\n")
except torch.cuda.OutOfMemoryError:
    print("Skipping generation due to OOM (attention visualization complete)")
    response = "[Generation skipped - attention visualization completed]"

# ============ Visualize Attention Maps ============
print(f"\nCaptured attention maps for {len(attention_maps)} layers")

if len(attention_maps) == 0:
    print("Error: No attention maps captured!")
    print("Make sure the model is using eager attention implementation.")

# Visualize attention maps
print("Creating visualizations...")

def plot_attention_head(attention, layer_name, head_idx, save_path):
    """Plot a single attention head"""
    # attention shape: [batch, heads, seq_len, seq_len]
    # Convert to float32 first to avoid bfloat16 issues with matplotlib
    attn_head = attention[0, head_idx].float().numpy()  # [seq_len, seq_len]

    plt.figure(figsize=(14, 12))

    # Use log scale to make small values visible
    import numpy as np
    attn_log = np.log10(attn_head + 1e-10)  # Add small epsilon to avoid log(0)

    # Get statistics for title
    vmax = attn_head.max()
    vmin = attn_head[attn_head > 0].min() if (attn_head > 0).any() else 0

    sns.heatmap(attn_log, cmap='hot', cbar=True, square=True,
                cbar_kws={'label': 'log10(Attention Weight)'})
    plt.title(f'{layer_name} - Head {head_idx}\n(Max: {vmax:.6f}, Min>0: {vmin:.8f})', fontsize=12)
    plt.xlabel('Key Position', fontsize=10)
    plt.ylabel('Query Position', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_average_attention(attention, layer_name, save_path):
    """Plot average attention across all heads"""
    # Average across heads: [batch, heads, seq_len, seq_len] -> [seq_len, seq_len]
    # Convert to float32 first to avoid bfloat16 issues with matplotlib
    avg_attn = attention[0].float().mean(dim=0).numpy()

    plt.figure(figsize=(14, 12))

    # Use log scale
    import numpy as np
    attn_log = np.log10(avg_attn + 1e-10)

    # Get statistics
    vmax = avg_attn.max()
    vmin = avg_attn[avg_attn > 0].min() if (avg_attn > 0).any() else 0

    sns.heatmap(attn_log, cmap='hot', cbar=True, square=True,
                cbar_kws={'label': 'log10(Attention Weight)'})
    plt.title(f'{layer_name} - Average Attention (Log Scale)\n(Max: {vmax:.6f}, Min>0: {vmin:.8f})', fontsize=12)
    plt.xlabel('Key Position', fontsize=10)
    plt.ylabel('Query Position', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_attention_summary(attention_maps, save_path):
    """Plot attention pattern summary for all layers"""
    import numpy as np

    # Adjust subplot layout based on number of layers
    ncols = min(5, len(attention_maps))
    _, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
    if ncols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Sort by layer number, not string
    layer_items = [(int(name.split('_')[1]), name, attn) for name, attn in attention_maps.items()]
    layer_items.sort()

    for idx, (layer_num, layer_name, attention) in enumerate(layer_items):
        if idx >= len(axes):
            break
        # Convert to float32 first to avoid bfloat16 issues
        avg_attn = attention[0].float().mean(dim=0).numpy()

        # Subsample if too large
        max_size = 200
        if avg_attn.shape[0] > max_size:
            step = avg_attn.shape[0] // max_size
            avg_attn = avg_attn[::step, ::step]

        # Use log scale for better visibility
        attn_log = np.log10(avg_attn + 1e-10)
        vmax = avg_attn.max()

        sns.heatmap(attn_log, ax=axes[idx], cmap='hot', cbar=True, square=True,
                   cbar_kws={'label': 'log10'})
        axes[idx].set_title(f'{layer_name}\n(max={vmax:.6f})', fontsize=10)
        axes[idx].set_xlabel('Key', fontsize=8)
        axes[idx].set_ylabel('Query', fontsize=8)

    # Hide unused subplots
    for idx in range(len(attention_maps), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Attention Patterns Across Selected Layers (Log Scale)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# Generate visualizations
if len(attention_maps) > 0:
    # Summary plot
    print("Creating summary plot...")
    plot_attention_summary(attention_maps, f"{output_dir}/attention_summary.png")

    # Plot all captured layers in detail (sorted by layer number, not string)
    # Extract layer numbers and sort numerically
    layer_items = [(int(name.split('_')[1]), name) for name in attention_maps.keys()]
    layer_items.sort()  # Sort by layer number
    sorted_layer_names = [name for _, name in layer_items]

    print(f"Plotting detailed visualizations for layers: {sorted_layer_names}")
    for layer_name in sorted_layer_names:
        attention = attention_maps[layer_name]
        num_heads = attention.shape[1]

        # Average attention
        plot_average_attention(attention, layer_name, f"{output_dir}/{layer_name}_avg.png")

        # First 4 heads
        for head_idx in range(min(4, num_heads)):
            plot_attention_head(attention, layer_name, head_idx,
                              f"{output_dir}/{layer_name}_head_{head_idx}.png")

    print(f"\nVisualization complete! Files saved to: {output_dir}/")
    print(f"- attention_summary.png: Overview of all layers")
    print(f"- layer_*_avg.png: Average attention for specific layers")
    print(f"- layer_*_head_*.png: Individual attention heads")
else:
    print("\nNo attention maps were captured.")
    print("Note: flash_attention_2 does not return attention weights.")
    print("The script uses 'eager' attention implementation to capture weights.")

# ============ Save Metadata ============
metadata_path = f"{output_dir}/metadata.txt"

# 分析序列組成以提供詳細的 metadata
input_ids = inputs['input_ids'][0]
image_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

# 計算實際序列長度（來自 attention matrix）
actual_seq_len = list(attention_maps.values())[0].shape[2] if len(attention_maps) > 0 else 0

# 估算 vision 和 text tokens 的分界點
# 基於 image placeholders 的最後位置
if len(image_positions) > 0:
    last_image_pos = image_positions[-1].item()
    # 在實際 forward pass 中，placeholders 被替換為 visual features
    # 估算 vision tokens 結束位置（實際序列長度 - 剩餘 text tokens）
    num_text_tokens = len(input_ids) - last_image_pos - 1
    vision_tokens_end = actual_seq_len - num_text_tokens

    # 解碼文本部分
    text_tokens = input_ids[last_image_pos + 1:]
    text_content = processor.tokenizer.decode(text_tokens, skip_special_tokens=False)
else:
    vision_tokens_end = 0
    num_text_tokens = 0
    text_content = ""

with open(metadata_path, 'w') as f:
    f.write("="*70 + "\n")
    f.write("ATTENTION VISUALIZATION METADATA\n")
    f.write("="*70 + "\n\n")

    f.write("## INPUT INFORMATION\n")
    f.write(f"Video: {video_path}\n")
    f.write(f"Frames: {len(frames)}\n")
    f.write(f"FPS: 1\n")
    f.write(f"Merge size: {inputs.get('merge_sizes', 'N/A')}\n")
    f.write(f"Grid sizes: {inputs.get('grid_sizes', 'N/A')}\n\n")

    f.write("## SEQUENCE COMPOSITION\n")
    f.write(f"Processor output length: {inputs['input_ids'].shape[1]} tokens\n")
    f.write(f"  - <image> placeholders: {len(image_positions)}\n")
    f.write(f"  - Other tokens: {inputs['input_ids'].shape[1] - len(image_positions)}\n\n")

    f.write(f"Actual sequence length (in forward pass): {actual_seq_len} tokens\n")
    f.write(f"  - Vision tokens: positions 0-{vision_tokens_end-1} ({vision_tokens_end} tokens)\n")
    f.write(f"  - Text tokens: positions {vision_tokens_end}-{actual_seq_len-1} ({num_text_tokens} tokens)\n\n")

    f.write("## TOKEN BREAKDOWN\n")
    f.write(f"Vision region:  [0:{vision_tokens_end}]  <- Video visual features\n")
    f.write(f"Text region:    [{vision_tokens_end}:{actual_seq_len}]  <- Prompt text\n\n")

    f.write("Text content:\n")
    f.write(f"{text_content}\n\n")

    f.write("## PROMPT\n")
    f.write(f"User prompt: {conversation[0]['content'][1]['text']}\n\n")

    f.write("## MODEL RESPONSE\n")
    f.write(f"{response}\n\n")

    f.write("## ATTENTION DETAILS\n")
    f.write(f"Number of layers captured: {len(attention_maps)}\n")
    f.write(f"Layers: {sorted([int(k.split('_')[1]) for k in attention_maps.keys()])}\n")
    if len(attention_maps) > 0:
        first_layer = list(attention_maps.values())[0]
        f.write(f"Attention shape: {first_layer.shape}\n")
        f.write(f"  - [batch, num_heads, seq_len, seq_len]\n")
        f.write(f"  - [{first_layer.shape[0]}, {first_layer.shape[1]}, {first_layer.shape[2]}, {first_layer.shape[3]}]\n\n")

    f.write("## ATTENTION MATRIX INTERPRETATION\n")
    f.write(f"Matrix dimensions: {actual_seq_len} × {actual_seq_len}\n\n")
    f.write("Regions:\n")
    f.write(f"  [0:{vision_tokens_end}, 0:{vision_tokens_end}]           -> Vision-to-Vision attention\n")
    f.write(f"  [{vision_tokens_end}:{actual_seq_len}, 0:{vision_tokens_end}]     -> Text-to-Vision attention\n")
    f.write(f"  [{vision_tokens_end}:{actual_seq_len}, {vision_tokens_end}:{actual_seq_len}] -> Text-to-Text attention\n\n")

    f.write("="*70 + "\n")

print(f"Metadata saved to: {metadata_path}")
