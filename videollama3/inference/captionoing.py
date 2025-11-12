#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from decord import VideoReader, cpu
from tqdm import tqdm

from videollama3.mm_utils import (
    get_model_name_from_path,
    preprocess_videollama3,
    read_frames_decord,
)
from videollama3.model import load_pretrained_model
from videollama3.model.processor import Videollama3Processor


logger = logging.getLogger(__name__)

TRAINING_PREFIX = (
    "There is a streaming video provided. Below are some captions describing the events in the video "
    "at different timestamps in ascending order.\n"
)
TRAINING_SUFFIX = "The following clip contains only the last few seconds of the ongoing stream.\n"
BASE_INSTRUCTION = "Describe the events that took place within this video clip as a detailed chronicle."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sliding window captioning with VideoLLaMA3.")
    parser.add_argument("--model-path", required=True, help="Path or HF repo id of the pretrained checkpoint.")
    parser.add_argument(
        "--model-base",
        default=None,
        help="Optional base model path when loading LoRA/QLoRA checkpoints.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="JSON/JSONL file that lists samples. Each item must contain a 'video' field.",
    )
    parser.add_argument(
        "--video-root",
        default=None,
        help="Directory that stores the raw videos. Relative manifest paths resolve against this directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination JSON file that will store generated captions.",
    )
    parser.add_argument(
        "--dataset-meta",
        default=None,
        help="Meta JSON that lists multiple datasets (see training data_path meta format).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory used to store per-dataset outputs when --dataset-meta is provided.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=128,
        help="Number of frames to feed the model per window.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Temporal sampling rate used by decord when extracting windows.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frame stride between windows. Defaults to num_frames (no overlap).",
    )
    parser.add_argument(
        "--video-merge-size",
        type=int,
        default=1,
        help="Spatial merge size used by the VideoLLaMA3 image processor.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per window.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--device-map", default="auto", help="Device map passed to load_pretrained_model.")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Computation dtype override.",
    )
    parser.add_argument(
        "--attn-impl",
        default="flash_attention_2",
        choices=["flash_attention_2", "eager"],
        help="Attention backend to request when loading the model.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optionally limit how many videos are processed.")
    parser.add_argument("--log-every", type=int, default=1, help="Log progress every N videos.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for frame sampling.")
    return parser.parse_args()


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_manifest(manifest_path: str) -> List[Dict]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)

    if isinstance(records, dict):
        # Accept common wrappers while keeping the script simple.
        for key in ("data", "videos", "samples"):
            if key in records and isinstance(records[key], list):
                records = records[key]
                break
        else:
            raise ValueError(
                "Manifest JSON must be a list or contain a top-level 'data'/'videos'/'samples' list."
            )

    if not isinstance(records, list):
        raise ValueError("Manifest file must encode a list of samples.")

    return records


def load_dataset_meta(meta_path: str) -> Dict[str, Dict]:
    path = Path(meta_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset meta not found: {meta_path}")
    with path.open("r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError("Dataset meta JSON must be a dictionary keyed by dataset name.")
    return content


def build_dataset_specs_from_meta(meta_path: str, output_dir: str) -> List[Dict]:
    meta = load_dataset_meta(meta_path)
    specs: List[Dict] = []
    for name, cfg in meta.items():
        annotation = cfg.get("annotation")
        data_root = cfg.get("data_root")
        if annotation is None:
            raise ValueError(f"Dataset '{name}' is missing 'annotation' field.")
        if data_root is None:
            raise ValueError(f"Dataset '{name}' is missing 'data_root' field.")
        annotation_path = annotation if os.path.isabs(annotation) else str(Path(annotation).resolve())
        data_root_path = data_root if os.path.isabs(data_root) else str(Path(data_root).resolve())
        output_path = os.path.join(output_dir, f"{name}_captions.json")
        specs.append(
            {
                "name": name,
                "manifest": annotation_path,
                "video_root": data_root_path,
                "output": output_path,
            }
        )
    return specs


def build_single_dataset_spec(manifest: str, video_root: Optional[str], output: str) -> List[Dict]:
    dataset_name = Path(output).stem
    return [
        {
            "name": dataset_name,
            "manifest": manifest,
            "video_root": video_root,
            "output": output,
        }
    ]


def resolve_video_path(
    raw_path: str,
    video_root: Optional[str],
    manifest_dir: Path,
) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    candidate_roots = [video_root, manifest_dir]
    for root in candidate_roots:
        if root is None:
            continue
        candidate = os.path.join(root, raw_path)
        if os.path.exists(candidate):
            return candidate
    return raw_path


def probe_duration(video_path: str) -> Tuple[float, float]:
    reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        fps = 1.0
    frame_count = len(reader)
    last_ts = reader.get_frame_timestamp(frame_count - 1)[-1] if frame_count > 0 else 0.0
    duration = max(last_ts, frame_count / fps)
    del reader
    return fps, duration


def build_windows(duration: float, window_sec: float, stride_sec: float) -> List[Tuple[float, float]]:
    if duration <= 0:
        return []
    eps = 1e-6
    if duration <= window_sec + eps:
        return [(0.0, duration)]

    windows: List[Tuple[float, float]] = []
    start = 0.0
    while True:
        end = start + window_sec
        if end >= duration - eps:
            windows.append((max(0.0, duration - window_sec), duration))
            break
        windows.append((start, end))
        start += stride_sec
        if start >= duration:
            break
    # Remove duplicates that can appear when duration aligns with window boundaries.
    deduped = []
    for win in windows:
        if not deduped or abs(deduped[-1][0] - win[0]) > eps:
            deduped.append((round(win[0], 3), round(win[1], 3)))
    return deduped


def prepare_messages(
    message_value: str,
    timestamps: Sequence[float],
) -> List[Dict]:
    rounded = np.array([round(t, 1) for t in timestamps], dtype=np.float32)
    query_time = (rounded[-1] if len(rounded) else 0.0) + 0.1
    conversations = [
        {
            "from": "human",
            "timestamps": float(query_time),
            "value": message_value,
        }
    ]
    return preprocess_videollama3(conversations, rounded)


def build_prompt_with_history(events: Sequence[str], base_instruction: str) -> str:
    if not events:
        return f"<video>\n{base_instruction}"
    history = "".join(f"{evt}\n" for evt in events if evt)
    return f"{TRAINING_PREFIX}{history}{TRAINING_SUFFIX}<video>\n{base_instruction}"


def caption_window(
    frames: List,
    timestamps: Sequence[float],
    message_value: str,
    processor: Videollama3Processor,
    model,
    tokenizer,
    merge_size: int,
    generation_kwargs: Dict,
) -> str:
    if len(frames) == 0 or len(timestamps) == 0:
        raise ValueError("Window has no decodable frames.")

    messages = prepare_messages(message_value, timestamps)
    batch = processor(
        images=frames,
        text=messages,
        merge_size=merge_size,
        return_tensors="pt",
    )
    # Align processor outputs with the model device/dtype for flash-attention compatibility.
    try:
        model_device = getattr(model, "device", next(model.parameters()).device)
    except StopIteration:
        model_device = torch.device("cpu")
    try:
        model_dtype = getattr(model, "dtype", next(model.parameters()).dtype)
    except StopIteration:
        model_dtype = torch.float16
    batch = batch.to(model_device)
    for key, value in batch.items():
        if torch.is_tensor(value) and torch.is_floating_point(value) and value.dtype != model_dtype:
            batch[key] = value.to(dtype=model_dtype)
    batch["modals"] = ["video"] * len(frames)

    input_len = batch["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            **generation_kwargs,
        )
    response_ids = output_ids[:, input_len:]
    text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
    return text.strip()


def process_dataset(
    dataset_name: str,
    manifest_path: str,
    video_root: Optional[str],
    output_path: str,
    processor: Videollama3Processor,
    model,
    tokenizer,
    base_instruction: str,
    num_frames: int,
    fps: float,
    stride_sec: float,
    window_sec: float,
    generation_kwargs: Dict,
    video_merge_size: int,
    limit: Optional[int],
    log_every: int,
) -> None:
    records = load_manifest(manifest_path)
    manifest_dir = Path(manifest_path).parent
    iterator = records if limit is None else records[:limit]

    results = []
    dataset_desc = f"Captioning {dataset_name}"

    for idx, sample in enumerate(tqdm(iterator, desc=dataset_desc)):
        video_field = sample.get("video") or sample.get("video_path")
        if video_field is None:
            logger.warning("Skipping sample without 'video' field: %s", sample)
            continue
        video_path = resolve_video_path(video_field, video_root, manifest_dir)
        if not os.path.exists(video_path):
            logger.warning("Video not found, skip: %s", video_path)
            continue

        try:
            _, duration = probe_duration(video_path)
        except Exception as exc:
            logger.exception("Failed to probe %s: %s", video_path, exc)
            continue

        windows = build_windows(duration, window_sec, stride_sec)
        if not windows:
            logger.warning("No valid window computed for %s", video_path)
            continue

        segments = []
        events_history: List[str] = []
        for window_idx, (start_sec, end_sec) in enumerate(windows):
            prompt_value = build_prompt_with_history(events_history, base_instruction)
            try:
                frames, timestamps = read_frames_decord(
                    video_path,
                    num_frames=num_frames,
                    sample=f"fps{fps}",
                    clip=(start_sec, end_sec),
                    min_num_frames=1,
                    return_timestamps=True,
                    force_context_length=True,
                )
            except Exception as exc:
                logger.exception(
                    "Failed to decode window %s (%.2f-%.2f)s: %s",
                    video_path,
                    start_sec,
                    end_sec,
                    exc,
                )
                continue

            rounded_timestamps = [float(round(float(t), 1)) for t in timestamps]
            try:
                caption = caption_window(
                    frames,
                    rounded_timestamps,
                    prompt_value,
                    processor,
                    model,
                    tokenizer,
                    video_merge_size,
                    generation_kwargs,
                )
            except Exception as exc:
                logger.exception(
                    "Generation failed for %s (%.2f-%.2f)s: %s",
                    video_path,
                    start_sec,
                    end_sec,
                    exc,
                )
                continue

            events_history.append(caption)

            segments.append(
                {
                    "window_index": window_idx,
                    "start": float(round(float(start_sec), 2)),
                    "end": float(round(float(end_sec), 2)),
                    "num_frames": len(frames),
                    "timestamps": rounded_timestamps,
                    "prompt": prompt_value,
                    "caption": caption,
                }
            )

        results.append(
            {
                "video": video_field,
                "video_path": video_path,
                "segments": segments,
            }
        )

        if (idx + 1) % log_every == 0:
            logger.info(
                "[%s] Processed %d/%d videos",
                dataset_name,
                idx + 1,
                len(iterator),
            )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("[%s] Saved %d video captions to %s", dataset_name, len(results), output_path)


def main():
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    stride = args.stride if args.stride is not None else args.num_frames
    if stride <= 0:
        raise ValueError("--stride / --num-frames must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(args.dtype, None)

    cuda_available = torch.cuda.is_available()
    requested_attn = args.attn_impl
    if (
        requested_attn == "flash_attention_2"
        and torch_dtype is not None
        and torch_dtype not in (torch.float16, torch.bfloat16)
    ):
        logger.warning(
            "flash_attention_2 requires float16/bfloat16 tensors but dtype %s was requested; "
            "falling back to eager attention.",
            args.dtype,
        )
        requested_attn = "eager"

    if not cuda_available and requested_attn == "flash_attention_2":
        logger.warning("CUDA is unavailable; falling back from flash_attention_2 to eager attention.")
        requested_attn = "eager"

    device_map = args.device_map
    if device_map == "auto" and not cuda_available:
        device_map = "cpu"

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        device_map=device_map,
        attn_implementation=requested_attn,
        torch_dtype=torch_dtype,
    )
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    model.eval()

    processor = Videollama3Processor(image_processor, tokenizer)

    stride_sec = stride / args.fps
    window_sec = args.num_frames / args.fps

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.num_beams == 1 and args.temperature > 0,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if args.dataset_meta:
        if args.output_dir is None:
            raise ValueError("--output-dir must be provided when using --dataset-meta.")
        dataset_specs = build_dataset_specs_from_meta(args.dataset_meta, args.output_dir)
    else:
        if args.manifest is None or args.output is None:
            raise ValueError("Provide --manifest/--output for single dataset or --dataset-meta/--output-dir for multiple datasets.")
        dataset_specs = build_single_dataset_spec(args.manifest, args.video_root, args.output)

    for spec in dataset_specs:
        logger.info(
            "Processing dataset '%s' (manifest=%s, video_root=%s) -> %s",
            spec["name"],
            spec["manifest"],
            spec["video_root"],
            spec["output"],
        )
        process_dataset(
            dataset_name=spec["name"],
            manifest_path=spec["manifest"],
            video_root=spec["video_root"],
            output_path=spec["output"],
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            base_instruction=BASE_INSTRUCTION,
            num_frames=args.num_frames,
            fps=args.fps,
            stride_sec=stride_sec,
            window_sec=window_sec,
            generation_kwargs=generation_kwargs,
            video_merge_size=args.video_merge_size,
            limit=args.limit,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
