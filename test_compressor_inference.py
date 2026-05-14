"""
Ad-hoc inference script for the compressor model.

Supports three ways to specify which frame windows to compress:

  --compression_ratio 0.5 --compression_window_size 10
      Auto-selects non-overlapping windows via select_compression_parts().

  --compress_windows "0-9,30-39"
      Manually specify frame windows (inclusive, 0-indexed).
      Multiple windows separated by commas, each as "start-end".

  --compress_all [--compression_window_size 10]
      Compress the entire video. Frames are divided into consecutive
      non-overlapping windows of --compression_window_size. Leftover
      frames form an extra window only if remainder > 3, otherwise discarded.

If none of the above is given, runs without compression (plain video QA).

Usage examples:
  python test_compressor_inference.py \
      --model_path pretrained_models/compressor_videollama3 \
      --video_path /path/to/video.mp4 \
      --prompt "Summarize the video." \
      --compression_ratio 0.5 --compression_window_size 10

  python test_compressor_inference.py \
      --model_path pretrained_models/compressor_videollama3 \
      --video_path /path/to/video.mp4 \
      --prompt "What happens between frame 0 and 9?" \
      --compress_windows "0-9"

  python test_compressor_inference.py \
      --model_path pretrained_models/compressor_videollama3 \
      --video_path /path/to/video.mp4 \
      --prompt "Summarize the video." \
      --compress_all --compression_window_size 10
"""

import argparse
import random

import torch

from videollama3.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from videollama3.mm_utils import load_video
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.train.videollama3_chat_finetune_compressor import select_compression_parts, select_full_compression_parts


def parse_args():
    p = argparse.ArgumentParser(description="Compressor model single-video inference")
    p.add_argument("--model_path", default="pretrained_models/compressor_videollama3")
    p.add_argument("--video_path", required=True)
    p.add_argument("--prompt", default="<video>\nSummarize all the events in this video. For each event. Format your response as: <start time> - <end time>, <description>")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--merge_size", type=int, default=2)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)

    # Compression — pick one mode
    comp = p.add_mutually_exclusive_group()
    comp.add_argument(
        "--compression_ratio",
        type=float,
        default=None,
        help="Fraction of frames to compress (auto window selection). "
             "Also requires --compression_window_size.",
    )
    comp.add_argument(
        "--compress_windows",
        type=str,
        default=None,
        help='Manual frame windows to compress, e.g. "0-9,30-39" (inclusive, 0-indexed).',
    )
    comp.add_argument(
        "--compress_all",
        action="store_true",
        default=False,
        help="Compress the entire video using consecutive non-overlapping windows of "
             "--compression_window_size. Leftover frames (> 3) form one extra window.",
    )
    p.add_argument(
        "--compression_window_size",
        type=int,
        default=10,
        help="Window size (frames) used with --compression_ratio or --compress_all.",
    )
    return p.parse_args()


def build_compression_parts_auto(
    total_frames: int,
    total_vision_tokens: int,
    ratio: float,
    window_size: int,
    seed: int,
) -> list:
    rng = random.Random(seed)
    parts = select_compression_parts(
        total_frames=total_frames,
        total_vision_tokens=total_vision_tokens,
        ratio=ratio,
        window_size=window_size,
        rng=rng,
    )
    return parts


def build_compression_parts_full(
    total_frames: int,
    total_vision_tokens: int,
    window_size: int,
) -> list:
    return select_full_compression_parts(
        total_frames=total_frames,
        total_vision_tokens=total_vision_tokens,
        window_size=window_size,
    )


def build_compression_parts_manual(
    window_str: str,
    total_frames: int,
    total_vision_tokens: int,
) -> list:
    tokens_per_frame = total_vision_tokens // total_frames
    parts = []
    for seg in window_str.split(","):
        seg = seg.strip()
        start_f, end_f = (int(x) for x in seg.split("-"))
        assert 0 <= start_f <= end_f < total_frames, (
            f"Window {seg} out of range for {total_frames} frames."
        )
        parts.append([start_f * tokens_per_frame, (end_f + 1) * tokens_per_frame])
    # Sort by start token index
    parts.sort(key=lambda x: x[0])
    return parts


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading model from {args.model_path} ...")
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.to(args.device)
    model.eval()

    processor = Videollama3Processor.from_pretrained(
        args.model_path,
        trust_remote_code=False,
    )
    if processor.tokenizer.pad_token is None and processor.tokenizer.unk_token is not None:
        processor.tokenizer.pad_token = processor.tokenizer.unk_token

    print(f"Loading video from {args.video_path} (fps={args.fps}, max_frames={args.max_frames}) ...")
    frames, timestamps = load_video(args.video_path, fps=args.fps, max_frames=args.max_frames)
    num_frames = len(frames)
    print(f"Loaded {num_frames} frames.")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": num_frames},
                {"type": "text", "text": f"<video>\n{args.prompt}"},
            ],
        }
    ]

    inputs = processor(
        images=[frames],
        text=conversation,
        merge_size=args.merge_size,
        return_tensors="pt",
    )
    inputs = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    # Compute total vision tokens from input_ids
    image_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    total_vision_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
    print(f"Total vision tokens in sequence: {total_vision_tokens} ({total_vision_tokens // num_frames} per frame)")

    # Build compression_parts
    compression_parts = None
    if args.compression_ratio is not None:
        compression_parts = build_compression_parts_auto(
            total_frames=num_frames,
            total_vision_tokens=total_vision_tokens,
            ratio=args.compression_ratio,
            window_size=args.compression_window_size,
            seed=args.seed,
        )
        print(f"Auto compression_parts ({len(compression_parts)} windows): {compression_parts}")
    elif args.compress_windows is not None:
        compression_parts = build_compression_parts_manual(
            window_str=args.compress_windows,
            total_frames=num_frames,
            total_vision_tokens=total_vision_tokens,
        )
        print(f"Manual compression_parts ({len(compression_parts)} windows): {compression_parts}")
    elif args.compress_all:
        compression_parts = build_compression_parts_full(
            total_frames=num_frames,
            total_vision_tokens=total_vision_tokens,
            window_size=args.compression_window_size,
        )
        print(f"Full compression_parts ({len(compression_parts)} windows): {compression_parts}")
    else:
        print("No compression specified — running plain video inference.")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            compression_parts=compression_parts,
            max_new_tokens=args.max_new_tokens,
            modals=["video"],
        )

    response = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\n=== Model Response ===")
    print(response)


if __name__ == "__main__":
    main()
