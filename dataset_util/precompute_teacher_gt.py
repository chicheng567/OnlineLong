#!/usr/bin/env python3
"""
Precompute teacher soft labels for knowledge distillation.

For every sample (whether it has an "assistant" GT in the annotation or not),
this script runs the teacher model ONCE via
`model.generate(..., output_logits=True, return_dict_in_generate=True)` and
saves both:

  1. **Soft labels** (distribution GT)
     Per-step teacher logits over the tokens the teacher itself generated,
     stored as a float16 tensor of shape [N_generated, vocab_size].
     → {output_dir}/{dataset_name}/{video_stem}.pt

  2. **Text GT**
     The decoded teacher response, stored inline in the updated annotation
     JSON as "teacher_text_gt" (list with one string per sample).

The original "assistant" turns in the annotation are IGNORED — the teacher
sees only the user prompt (from the annotation's user turn, or
--default_prompt if none). There is NO second teacher-forcing pass, so every
saved logit row corresponds one-to-one with a generated token.

After the run, an updated meta JSON is written to
{output_dir}/meta_with_teacher.json. Point --meta_path at this file when
launching compressor training, and set DataArguments.teacher_cache_dir to
`output_dir`.

NOTE: the student training side MUST construct the assistant turn from
`teacher_text_gt` (NOT the original annotation GT) so that label positions
line up with the cached logits.

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────
Single GPU:
    python dataset_util/precompute_teacher_gt.py \\
        --teacher_model_path pretrained_models/videollama3 \\
        --meta_path anno_data/finetune_online.json \\
        --output_dir teacher_cache \\
        --fps 1 --max_frames 10

Multi-GPU with torchrun:
    torchrun --nproc_per_node=8 dataset_util/precompute_teacher_gt.py ...

Resume is supported: already-computed .pt files are skipped automatically.
"""

import argparse
import json
import logging
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

sys.path.append("./")

from transformers import AutoTokenizer
from videollama3.model.videollama3_qwen2 import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.mm_utils import load_video, load_images


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

# Separate gloo process group used exclusively for barriers. Pure CPU, so it
# is not subject to NCCL's watchdog: ranks that finish early can safely wait
# many hours for stragglers (e.g. when corrupted videos make shards uneven).
_GLOO_PG = None


def _init_dist() -> Tuple[int, int, int]:
    """Initialize process group if under torchrun; otherwise act as rank 0."""
    global _GLOO_PG
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=8))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _GLOO_PG = dist.new_group(backend="gloo", timeout=timedelta(days=1))
    else:
        rank = local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        world_size = 1
    return rank, local_rank, world_size


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=_GLOO_PG)


# ---------------------------------------------------------------------------
# Conversation builder
# ---------------------------------------------------------------------------

def _build_conversation(
    sample: Dict,
    dataset_root: str,
    fps: int,
    max_frames: int,
    default_prompt: str,
) -> Tuple[List[Dict], List, Optional[str]]:
    """
    Load media and build a USER-ONLY conversation for teacher generation.

    Assistant / gpt / system turns in `sample["conversations"]` are ignored.
    If no usable user text is found, `default_prompt` is used as the single
    user turn. Media is attached to that single user turn.

    Returns:
        messages   : single-element list with one user message
        media      : [frames] for video, [image_list] for image, [] otherwise
        modal_type : "video", "image", or None
    """
    has_video = bool(sample.get("video"))
    has_image = bool(sample.get("image"))

    media: List = []
    modal_type: Optional[str] = None
    timestamps = None
    num_media_frames: int = 0
    image_files: List[str] = []

    if has_video:
        video_file = sample["video"]
        if isinstance(video_file, (list, tuple)):
            video_file = video_file[0]
        video_path = os.path.join(dataset_root, video_file)
        frames, timestamps = load_video(video_path, fps=fps, max_frames=max_frames)
        media = [frames]
        num_media_frames = len(frames)
        modal_type = "video"
    elif has_image:
        image_file = sample["image"]
        image_files = image_file if isinstance(image_file, list) else [image_file]
        images = load_images([os.path.join(dataset_root, f) for f in image_files])
        media = [images]
        modal_type = "image"

    # Collect only user-role text; drop assistant/gpt/system entirely.
    user_texts: List[str] = []
    for c in sample.get("conversations", []):
        role = c.get("role") or c.get("from", "")
        if role not in ("human", "user"):
            continue
        text = c.get("value", c.get("content", ""))
        text = str(text).replace("<video>", "").replace("<image>", "").strip()
        if text:
            user_texts.append(text)
    if not user_texts:
        user_texts.append(default_prompt)

    content: List[Dict] = []
    if has_video:
        video_block: Dict = {"type": "video", "num_frames": num_media_frames}
        if timestamps is not None:
            video_block["timestamps"] = timestamps
        content.append(video_block)
    elif has_image:
        for _ in image_files:
            content.append({"type": "image"})
    for t in user_texts:
        content.append({"type": "text", "text": t})

    messages = [{"role": "user", "content": content}]
    return messages, media, modal_type


# ---------------------------------------------------------------------------
# Teacher inference
# ---------------------------------------------------------------------------

def _inputs_to_device(processed, device: torch.device, dtype: torch.dtype) -> Dict:
    """Move all tensors to device; cast pixel_values to dtype; add batch dim where needed."""
    inputs: Dict = {}
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if k == "pixel_values":
                v = v.to(dtype)
            if v.dim() == 1 and k == "input_ids":
                v = v.unsqueeze(0)
            inputs[k] = v
        else:
            inputs[k] = v
    return inputs


@torch.no_grad()
def _run_teacher_inference(
    model,
    processor,
    conversation: List[Dict],
    media: List,
    modal_type: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> Tuple[str, torch.Tensor]:
    """
    A single teacher `generate()` call. Returns:
      - decoded response string (for `teacher_text_gt`)
      - per-step logits tensor, shape [N_generated, vocab_size], float16, CPU

    Each logit row is the teacher's prediction distribution for the matching
    token in `output.sequences[0]`. No second forward pass is performed.
    """
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    modals = [modal_type] if modal_type else []

    processed = processor(
        images=media if media else None,
        text=conversation,
        merge_size=2,
        return_tensors="pt",
    )
    inputs = _inputs_to_device(processed, device, dtype)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_token_id,
        modals=modals,
        output_logits=True,
        return_dict_in_generate=True,
    )

    # Videollama3Qwen2.generate delegates to super().generate(inputs_embeds=...),
    # so output.sequences contains ONLY the newly generated tokens (no prompt prefix).
    output_ids = output.sequences[0]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    # output.logits: tuple of N tensors, each [1, vocab_size] — stack to [N, V]
    if not output.logits:
        raise RuntimeError("model.generate() returned empty logits; check output_logits=True support")
    logits = torch.stack(list(output.logits), dim=0).squeeze(1).to(torch.float16).cpu()
    return text, logits


# ---------------------------------------------------------------------------
# Unannotated video discovery
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".flv"}


def _find_unannotated_videos(
    dataset_root: str,
    annotation: List[Dict],
    default_prompt: str,
) -> List[Dict]:
    """
    Walk *dataset_root* recursively and return synthetic samples for every
    video file that is NOT already referenced in *annotation*. Each synthetic
    sample has a single user turn using *default_prompt*.
    """
    if not dataset_root or not os.path.isdir(dataset_root):
        return []

    annotated: set = set()
    for sample in annotation:
        v = sample.get("video")
        if not v:
            continue
        if isinstance(v, (list, tuple)):
            v = v[0]
        annotated.add(os.path.normpath(v))

    new_samples: List[Dict] = []
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        # Deterministic walk so samples[rank::world_size] shards consistently.
        dirnames.sort()
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in VIDEO_EXTENSIONS:
                continue
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, dataset_root)
            if os.path.normpath(rel_path) in annotated:
                continue
            new_samples.append({
                "video": rel_path,
                "conversations": [
                    {"from": "human", "value": f"<video>\n{default_prompt}"},
                ],
                "_unannotated": True,
            })

    return new_samples


def _build_logits_filename(sample: Dict) -> str:
    """Name the .pt file after the source video stem."""
    v = sample["video"]
    if isinstance(v, (list, tuple)):
        v = v[0]
    return f"{Path(str(v)).stem}.pt"


# ---------------------------------------------------------------------------
# Per-dataset processing loop
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_name: str,
    dataset_cfg: Dict,
    model,
    processor,
    fps: int,
    max_frames: int,
    output_dir: Path,
    max_new_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    overwrite: bool,
    rank: int,
    world_size: int,
    annotate_unannotated: bool,
    default_prompt: str,
) -> Optional[Dict]:
    """
    Process this rank's shard of the dataset. Each rank writes a per-rank
    partial annotation file; rank 0 merges them at the end.
    """
    annotation_path = dataset_cfg["annotation"]
    dataset_root = dataset_cfg.get("data_root", "")

    logger.info(f"[rank {rank}][{dataset_name}] Loading annotation: {annotation_path}")
    with open(annotation_path) as f:
        annotation: List[Dict] = json.load(f)

    if annotate_unannotated and dataset_root:
        extra = _find_unannotated_videos(dataset_root, annotation, default_prompt)
        if extra:
            logger.info(
                f"[rank {rank}][{dataset_name}] Found {len(extra)} unannotated video(s) "
                f"in {dataset_root}; appending as synthetic samples."
            )
            annotation = annotation + extra

    n_total = len(annotation)
    logger.info(f"[rank {rank}][{dataset_name}] {n_total} samples total")

    logits_dir = output_dir / dataset_name
    logits_dir.mkdir(parents=True, exist_ok=True)

    shard_indices = list(range(n_total))[rank::world_size]
    partial_anno_path = output_dir / f"{dataset_name}_rank{rank}_partial.json"

    existing: Dict[int, Dict] = {}
    if partial_anno_path.exists() and not overwrite:
        try:
            with open(partial_anno_path) as f:
                saved: List[Dict] = json.load(f)
            for item in saved:
                idx = item.get("_precompute_idx")
                if idx is not None:
                    existing[int(idx)] = item
            logger.info(f"[rank {rank}][{dataset_name}] Resume: {len(existing)} already done")
        except Exception:
            logger.warning(f"[rank {rank}][{dataset_name}] Could not read partial file, starting fresh")

    partial_results: Dict[int, Dict] = dict(existing)
    n_ok = n_skip = n_fail = 0

    pbar = tqdm(
        shard_indices,
        desc=f"[rank {rank}] {dataset_name}",
        position=rank,
        leave=True,
        dynamic_ncols=True,
        file=sys.stderr,
    )
    for idx in pbar:
        sample = annotation[idx]
        logits_path = logits_dir / _build_logits_filename(sample)

        if not overwrite and logits_path.exists() and idx in existing:
            n_skip += 1
            pbar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
            continue
        try:
            conversation, media, modal_type = _build_conversation(
                sample, dataset_root, fps, max_frames, default_prompt
            )
            text, logits = _run_teacher_inference(
                model, processor, conversation, media, modal_type,
                device, dtype, max_new_tokens,
            )

            torch.save(logits, logits_path)

            updated_sample = dict(sample)
            updated_sample["_precompute_idx"] = idx
            updated_sample["teacher_logits_path"] = str(logits_path.relative_to(output_dir))
            updated_sample["teacher_text_gt"] = [text]
            partial_results[idx] = updated_sample
            n_ok += 1

        except Exception:
            logger.error(f"[rank {rank}][{dataset_name}] idx={idx} failed:\n{traceback.format_exc()}")
            fallback = dict(sample)
            fallback["_precompute_idx"] = idx
            partial_results[idx] = fallback
            n_fail += 1

        pbar.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
        if (n_ok + n_fail) % 100 == 1:
            logger.info(f"[rank {rank}][{dataset_name}] ok={n_ok} skip={n_skip} fail={n_fail}")
            _save_partial(partial_results, partial_anno_path)

    pbar.close()
    logger.info(f"[rank {rank}][{dataset_name}] Finished: ok={n_ok} skip={n_skip} fail={n_fail}")
    _save_partial(partial_results, partial_anno_path)

    _barrier()
    if rank != 0:
        return None

    return _merge_dataset(dataset_name, annotation, output_dir, logits_dir, world_size, dataset_cfg)


def _save_partial(results: Dict[int, Dict], path: Path):
    """Save this rank's partial results as a list sorted by index."""
    with open(path, "w") as f:
        json.dump(
            sorted(results.values(), key=lambda x: x.get("_precompute_idx", 0)),
            f, indent=2, ensure_ascii=False,
        )


def _merge_dataset(
    dataset_name: str,
    original_annotation: List[Dict],
    output_dir: Path,
    logits_dir: Path,
    world_size: int,
    original_cfg: Dict,
) -> Dict:
    """Merge per-rank partials into one final annotation file (rank 0)."""
    logger.info(f"[rank 0][{dataset_name}] Merging {world_size} partial annotation(s) ...")

    merged: List[Dict] = list(original_annotation)
    for r in range(world_size):
        partial_path = output_dir / f"{dataset_name}_rank{r}_partial.json"
        if not partial_path.exists():
            logger.warning(f"[rank 0][{dataset_name}] Missing partial file for rank {r}: {partial_path}")
            continue
        with open(partial_path) as f:
            items: List[Dict] = json.load(f)
        for item in items:
            idx = item.get("_precompute_idx")
            if idx is not None:
                merged[int(idx)] = item

    final_anno_path = output_dir / f"{dataset_name}_teacher_anno.json"
    with open(final_anno_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info(f"[rank 0][{dataset_name}] Final annotation → {final_anno_path}")

    for r in range(world_size):
        partial_path = output_dir / f"{dataset_name}_rank{r}_partial.json"
        if partial_path.exists():
            partial_path.unlink()

    updated_cfg = dict(original_cfg)
    updated_cfg["annotation"] = str(final_anno_path)
    updated_cfg["teacher_logits_dir"] = str(logits_dir)
    return updated_cfg


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_teacher(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load teacher model + processor using Videollama3Qwen2ForCausalLM."""
    logger.info(f"Loading teacher model from {model_path} on {device} ...")

    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
        attn_implementation="flash_attention_2",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=30000, padding_side="right", use_fast=True,
    )
    vision_encoder = model.get_vision_encoder()
    processor = Videollama3Processor(vision_encoder.image_processor, tokenizer)

    logger.info(f"Teacher model ready on {device}.")
    return model, processor


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_model_path", required=True)
    p.add_argument("--meta_path", required=True,
                   help="Meta JSON (same format as anno_data/finetune_online.json)")
    p.add_argument("--output_dir", required=True,
                   help="Root dir for all cached outputs")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=10)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--annotate_unannotated", action="store_true",
                   help="Also process video files in data_root that are not referenced "
                        "in the annotation (they get --default_prompt as the user turn)")
    p.add_argument("--default_prompt", default="Describe this video.",
                   help="Prompt used as the user turn when the annotation has no usable "
                        "user text, or for unannotated videos discovered under data_root")
    return p.parse_args()


def _setup_logging(output_dir: Path, rank: int):
    fmt = logging.Formatter("%(asctime)s [rank %(rankno)s][%(levelname)s] %(message)s")

    class _RankFilter(logging.Filter):
        def __init__(self, r):
            super().__init__()
            self._rank = r
        def filter(self, record):
            record.rankno = self._rank
            return True

    handler_list = [logging.StreamHandler(sys.stdout)]
    handler_list.append(logging.FileHandler(output_dir / f"precompute_rank{rank}.log", mode="a"))
    for h in handler_list:
        h.setFormatter(fmt)
        h.addFilter(_RankFilter(rank))

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in handler_list:
        root.addHandler(h)


def main():
    args = _parse_args()
    rank, local_rank, world_size = _init_dist()

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    _setup_logging(output_dir, rank)

    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    compute_dtype = dtype_map[args.dtype]

    model, processor = load_teacher(args.teacher_model_path, device, compute_dtype)

    with open(args.meta_path) as f:
        meta: Dict = json.load(f)

    updated_meta: Dict = {}
    for dataset_name, dataset_cfg in meta.items():
        if rank == 0:
            logger.info(f"===== Dataset: {dataset_name} =====")
        _barrier()

        updated_cfg = process_dataset(
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg,
            model=model,
            processor=processor,
            fps=args.fps,
            max_frames=args.max_frames,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            device=device,
            dtype=compute_dtype,
            overwrite=args.overwrite,
            rank=rank,
            world_size=world_size,
            annotate_unannotated=args.annotate_unannotated,
            default_prompt=args.default_prompt,
        )
        if rank == 0 and updated_cfg is not None:
            updated_meta[dataset_name] = updated_cfg

    if rank == 0:
        meta_out = output_dir / "meta_with_teacher.json"
        with open(meta_out, "w") as f:
            json.dump(updated_meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Updated meta JSON → {meta_out}")
        logger.info("All done.")

    _barrier()


if __name__ == "__main__":
    main()
