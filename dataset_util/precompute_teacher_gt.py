#!/usr/bin/env python3
"""
Precompute teacher model ground truth for knowledge distillation.

For every sample in each dataset listed in a meta JSON, this script runs the
teacher model and saves two types of per-sample ground truth:

  For every sample, the teacher model first generates its own response, then a
  second forward pass computes soft labels on that teacher-generated response.
  Original GT annotations are used only as conversational context (prior turns),
  not as supervision targets.

  1. **Soft labels** (distribution GT)
     Teacher logits at teacher-generated token positions, stored as a
     float16 tensor of shape [N_tokens, vocab_size].
     → {output_dir}/{dataset_name}/{sample_idx}.pt

  2. **Text GT**
     Teacher-generated response, stored inline in the updated annotation JSON
     as "teacher_text_gt".

After the run, an updated meta JSON is written to {output_dir}/meta_with_teacher.json.
Point --meta_path at this file when launching compressor training to skip
the live teacher forward pass entirely (set DataArguments.teacher_cache_dir to output_dir).

─────────────────────────────────────────────────────────────────────────────
Usage
─────────────────────────────────────────────────────────────────────────────
Single GPU:
    python dataset_util/precompute_teacher_gt.py \\
        --teacher_model_path pretrained_models/videollama3 \\
        --meta_path anno_data/finetune_online.json \\
        --output_dir teacher_cache \\
        --fps 1 --max_frames 10

Multi-GPU with torchrun (recommended):
    torchrun --nproc_per_node=8 dataset_util/precompute_teacher_gt.py \\
        --teacher_model_path pretrained_models/videollama3 \\
        --meta_path anno_data/finetune_online.json \\
        --output_dir teacher_cache \\
        --fps 1 --max_frames 10

Each rank processes samples[rank::world_size] independently (no gradient sync
needed — pure inference). Rank 0 merges all per-rank annotation files at the
end and writes the final meta JSON.

Resume is supported: already-computed .pt files are skipped automatically.
─────────────────────────────────────────────────────────────────────────────
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

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

# Separate gloo process group used exclusively for barriers. Pure CPU, so it
# is not subject to NCCL's watchdog: ranks that finish early can safely wait
# many hours for stragglers (e.g. when corrupted videos make shards uneven).
_GLOO_PG = None


def _init_dist() -> Tuple[int, int, int]:
    """
    Initialize the process group if running under torchrun, otherwise treat
    the current process as the only rank.

    Returns (rank, local_rank, world_size).
    """
    global _GLOO_PG
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Bind this process to its own GPU before NCCL init so collectives
        # (barrier, all_reduce, etc.) target the correct device.
        torch.cuda.set_device(local_rank)
        # Large NCCL timeout as a safety net; barriers go through gloo below.
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
) -> Tuple[List[Dict], List, Optional[str]]:
    """
    Load media and convert an annotation sample into the conversation format
    expected by Videollama3Processor(images=..., text=...).

    Returns:
        messages    : list of message dicts (role / content)
        media       : list to pass as `images=` to the processor;
                      [frames] for video, [image_list] for image, [] otherwise
        modal_type  : "video", "image", or None
    Supports:
      - image samples  (sample["image"] present)
      - video samples  (sample["video"] present)
      - single-turn and multi-turn conversations
      - both {"from":"human"/"gpt"} and {"role":"user"/"assistant"} formats
    """
    has_video = bool(sample.get("video"))
    has_image = bool(sample.get("image"))

    # ------------------------------------------------------------------
    # Load media up-front and record what the conversation block should say
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Normalise to list of {"role": ..., "value": ...}
    # ------------------------------------------------------------------
    raw_convs = sample.get("conversations", [])
    norm: List[Dict] = []
    for c in raw_convs:
        role = c.get("role") or c.get("from", "")
        if role in ("human",):
            role = "user"
        elif role in ("gpt",):
            role = "assistant"
        norm.append({"role": role, "value": c.get("value", c.get("content", ""))})

    # Skip leading system turns
    start = 0
    while start < len(norm) and norm[start]["role"] == "system":
        start += 1
    norm = norm[start:]

    messages: List[Dict] = []
    first_user_seen = False

    for conv in norm:
        role = conv["role"]
        value: str = conv["value"]

        if role == "user":
            # Strip placeholder tokens
            text = value.replace("<video>", "").replace("<image>", "").strip()
            content: List[Dict] = []

            # Attach media reference to the first user turn only
            if not first_user_seen:
                first_user_seen = True
                if has_video:
                    video_block: Dict = {"type": "video", "num_frames": num_media_frames}
                    if timestamps is not None:
                        video_block["timestamps"] = timestamps
                    content.append(video_block)
                elif has_image:
                    for _ in image_files:
                        content.append({"type": "image"})

            if text:
                content.append({"type": "text", "text": text})

            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            messages.append({"role": "assistant", "content": value})

    return messages, media, modal_type


# ---------------------------------------------------------------------------
# Teacher inference helpers
# ---------------------------------------------------------------------------

def _inputs_to_device(processed, device: torch.device, dtype: torch.dtype) -> Dict:
    """Move all tensors to device; cast pixel_values to dtype; add batch dim where needed."""
    inputs: Dict = {}
    for k, v in processed.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if k == "pixel_values":
                v = v.to(dtype)
            # Videollama3Processor returns 1-D input_ids when return_labels=True
            if v.dim() == 1 and k == "input_ids":
                v = v.unsqueeze(0)
            inputs[k] = v
        else:
            inputs[k] = v
    return inputs


@torch.no_grad()
def _run_soft_labels(
    model,
    processor,
    conversation: List[Dict],
    media: List,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """
    Full teacher forward → float16 logits at valid label positions.
    Returns tensor of shape [N_valid_tokens, vocab_size] or None.
    """
    processed = processor(
        images=media if media else None,
        text=conversation,
        merge_size=2,
        return_labels=True,
        return_tensors="pt",
    )

    # Pop labels before moving to model (they stay on CPU for masking)
    raw_labels: torch.Tensor = processed.pop("labels")
    if raw_labels.dim() == 1:
        labels = raw_labels  # keep 1-D for mask indexing after shift
    else:
        labels = raw_labels[0]  # take first batch element

    inputs = _inputs_to_device(processed, device, dtype)
    outputs = model(**inputs)

    logits: torch.Tensor = outputs.logits   # [1, T, V]
    shift_logits = logits[0, :-1, :]        # [T-1, V]
    shift_labels = labels[:-1]              # [T-1]
    mask = shift_labels != IGNORE_INDEX

    if not mask.any():
        return None

    return shift_logits[mask.to(device)].half().cpu()   # [N_valid, V]


@torch.no_grad()
def _run_text_generation(
    model,
    processor,
    conversation: List[Dict],
    media: List,
    modal_type: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> List[str]:
    """
    Generate one teacher response per assistant turn, using GT prior context.
    Returns a list of strings, one per assistant turn.
    """
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    modals = [modal_type] if modal_type else []

    generated_texts: List[str] = []

    def _generate(context: List[Dict]) -> str:
        processed = processor(
            images=media if media else None,
            text=context,
            merge_size=2,
            return_tensors="pt",
        )
        inputs = _inputs_to_device(processed, device, dtype)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            modals=modals,
        )
        # model.generate() calls super().generate(inputs_embeds=...) internally,
        # so output_ids contains ONLY the newly generated tokens (no prompt prefix).
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Walk through conversation; whenever we hit an assistant turn,
    # generate using the context up to (but not including) that turn.
    context_so_far: List[Dict] = []
    for msg in conversation:
        if msg["role"] == "assistant":
            if not context_so_far:
                generated_texts.append("")
                continue
            generated_texts.append(_generate(context_so_far))
            # Add GT assistant turn to context for subsequent turns
            context_so_far.append(msg)
        else:
            context_so_far.append(msg)

    # Handle a trailing user turn (e.g. unannotated videos: the synthetic
    # sample carries only one user turn and no GT assistant turn). Without
    # this, nothing would be generated and soft labels would have no valid
    # tokens either.
    if context_so_far and context_so_far[-1]["role"] == "user":
        generated_texts.append(_generate(context_so_far))

    return generated_texts


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
    video file that is NOT already referenced in *annotation*.

    Each returned sample has the same structure as a real annotation entry
    (a single user turn, no assistant turn) so it can be fed through the
    normal processing pipeline.  The soft-labels step will produce no .pt
    file (no GT labels to supervise against), but text-GT generation
    teacher generation always runs and the result is attached as "teacher_text_gt".
    """
    if not dataset_root or not os.path.isdir(dataset_root):
        return []

    # Build a normalised set of video paths already covered by the annotation.
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
        # Sort dirnames in-place so os.walk descends in a deterministic order
        # across ranks. Without this, filesystem-dependent ordering can make
        # samples[rank::world_size] shard a different list on each rank.
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
    """Name the .pt file after the source video so the cache is self-describing.
    Dataset separation is handled by the parent directory (logits_dir = output_dir / dataset_name)."""
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
    annotate_unannotated: bool = False,
    default_prompt: str = "Describe this video.",
) -> Optional[Dict]:
    """
    Process this rank's shard of the dataset.

    Each rank writes a per-rank partial annotation file.
    Returns the updated dataset config dict (rank 0 only, after merge).

    When *annotate_unannotated* is True, video files found in data_root that
    are not referenced in the annotation are appended as synthetic samples
    (single user turn, no GT assistant turn).  Their soft-labels step will
    be skipped (no GT), but text-GT generation still runs if --generate_text_gt
    is set.
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

    # This rank owns samples[rank::world_size]
    shard_indices = list(range(n_total))[rank::world_size]

    # Per-rank partial annotation file (avoids write conflicts between ranks)
    partial_anno_path = output_dir / f"{dataset_name}_rank{rank}_partial.json"

    # Load existing partial results for resume
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
            conversation, media, modal_type = _build_conversation(sample, dataset_root, fps, max_frames)

            # Always generate teacher responses first — soft labels are computed
            # on the teacher's own output, not the original GT annotations.
            text_gt = _run_text_generation(
                model, processor, conversation, media, modal_type, device, dtype, max_new_tokens
            )

            # Build a teacher-complete conversation by appending generated turns.
            teacher_conversation = list(conversation)
            if text_gt:
                for generated_text in text_gt:
                    teacher_conversation.append({"role": "assistant", "content": generated_text})

            # Soft labels on teacher-generated conversation.
            soft_labels = _run_soft_labels(model, processor, teacher_conversation, media, device, dtype)
            if soft_labels is None:
                logger.warning(f"[rank {rank}][{dataset_name}] idx={idx}: no valid labels, skipping .pt")
            else:
                torch.save(soft_labels, logits_path)

            updated_sample = dict(sample)
            updated_sample["_precompute_idx"] = idx
            if soft_labels is not None:
                updated_sample["teacher_logits_path"] = str(logits_path.relative_to(output_dir))
            if text_gt is not None:
                updated_sample["teacher_text_gt"] = text_gt
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

    # ---- Merge (rank 0 only, after all ranks finish) ----
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
    """
    Merge per-rank partial annotations into one final annotation file (rank 0).
    Partial files are removed after a successful merge.
    """
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

def load_teacher(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Load teacher model and processor using Videollama3Qwen2ForCausalLM / Videollama3Processor."""
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
        model_path,
        model_max_length=30000,
        padding_side="right",
        use_fast=True,
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
                        "in the annotation")
    p.add_argument("--default_prompt", default="Describe this video.",
                   help="Prompt used for unannotated videos (only relevant with --annotate_unannotated)")
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
    _barrier()  # ensure dir exists before other ranks try to log

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
