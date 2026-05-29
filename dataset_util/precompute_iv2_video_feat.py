#!/usr/bin/env python3
"""
Precompute InternVideo2-L global video features for the compressor pretrain
semantic loss.

For every sample in --data_path, decode ``num_frames`` frames, run them
through the frozen InternVideo2-L Stage-2 backbone, and save the resulting
``(768,)`` CLIP-aligned feature vector as a fp16 ``.pt`` file under
``--output_dir/{video_stem}.pt``.

The script is resume-safe (cached samples are skipped unless ``--overwrite``).
Multi-GPU sharding via ``torchrun`` is supported and required for large
datasets — each rank processes a strided shard ``samples[rank::world_size]``.

Usage
-----
Single GPU:
    /miniconda/envs/video/bin/python dataset_util/precompute_iv2_video_feat.py \\
        --iv2_ckpt pretrained_models/iv2_L/pytorch_model.bin \\
        --data_path anno_online/testt.json \\
        --data_root /workspace/datasetfortest/ActivityNets \\
        --output_dir iv2_cache

Multi-GPU:
    torchrun --nproc_per_node=4 dataset_util/precompute_iv2_video_feat.py ...

Output layout
-------------
    iv2_cache/
      {video_stem}.pt              # (768,) fp16
      meta_with_iv2.json           # input meta + 'iv2_feat_path' per sample
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

sys.path.append("./")

from videollama3.mm_utils import read_frames_decord
from videollama3.model.internvideo2_vendor import (
    IV2_IMAGE_MEAN,
    IV2_IMAGE_STD,
    load_internvideo2_l,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed
# ---------------------------------------------------------------------------

_GLOO_PG = None


def _init_dist() -> Tuple[int, int, int]:
    global _GLOO_PG
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=4))
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
# Data loading: accept both list-of-samples and dict-of-datasets formats.
# ---------------------------------------------------------------------------

def _load_samples(data_path: str, data_root: Optional[str]) -> List[Dict]:
    with open(data_path) as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [dict(s, _data_root=data_root or "") for s in raw]

    items: List[Dict] = []
    for _, ds_cfg in raw.items():
        ann_path = ds_cfg["annotation"]
        root = ds_cfg.get("data_root", data_root or "")
        with open(ann_path) as fa:
            ann = json.load(fa)
        for entry in ann:
            entry = dict(entry)
            entry["_data_root"] = root
            items.append(entry)
    return items


# ---------------------------------------------------------------------------
# Frame → IV2 input tensor
# ---------------------------------------------------------------------------

class FrameToIV2:
    """Resize PIL frames to 224 + CLIP-normalize → (3, T, 224, 224)."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self._mean = torch.tensor(IV2_IMAGE_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IV2_IMAGE_STD).view(3, 1, 1)

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        tensors = []
        for fr in frames:
            fr = fr.convert("RGB").resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
            arr = np.asarray(fr, dtype=np.float32) / 255.0  # (H, W, 3)
            t = torch.from_numpy(arr).permute(2, 0, 1)       # (3, H, W)
            t = (t - self._mean) / self._std
            tensors.append(t)
        # (T, 3, H, W) → (3, T, H, W)
        return torch.stack(tensors, dim=0).permute(1, 0, 2, 3).contiguous()


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def _process_sample(
    sample: Dict,
    output_dir: Path,
    model,
    transform: FrameToIV2,
    num_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    overwrite: bool,
) -> Optional[str]:
    """
    Returns the relative .pt path if cached/computed; None on failure.
    """
    root = sample.get("_data_root", "")
    video_field = sample["video"]
    if isinstance(video_field, (list, tuple)):
        video_field = video_field[0]
    video_path = os.path.join(root, video_field) if root else video_field

    stem = Path(str(video_field)).stem
    out_path = output_dir / f"{stem}.pt"

    if out_path.exists() and not overwrite:
        return str(out_path)

    try:
        # return_timestamps=False (default) → List[PIL.Image].
        frames = read_frames_decord(video_path, num_frames=num_frames, sample="middle")
        assert isinstance(frames, list)
    except Exception as exc:
        logger.warning("decord failed on %s: %s", video_path, exc)
        return None

    # Pad / trim to exactly num_frames so pos_embed lookup is consistent.
    if len(frames) == 0:
        logger.warning("empty frames for %s", video_path)
        return None
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    elif len(frames) > num_frames:
        frames = frames[:num_frames]

    tensor = transform(frames).unsqueeze(0).to(device=device, dtype=dtype)  # (1, 3, T, 224, 224)
    with torch.no_grad():
        feat = model(tensor).squeeze(0).to(torch.float16).cpu()              # (768,)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feat, out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iv2_ckpt", required=True)
    parser.add_argument("--data_path", required=True,
                        help="Path to JSON: list-of-samples or dict-of-datasets.")
    parser.add_argument("--data_root", default=None,
                        help="Optional prefix for video paths (overridden by per-dataset data_root).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Frames sampled per video; must equal IV2 pretrain config (8).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rank, local_rank, world_size = _init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        logger.info("Loading IV2-L from %s", args.iv2_ckpt)
    model = load_internvideo2_l(args.iv2_ckpt, device=device, dtype=dtype)
    transform = FrameToIV2(image_size=args.image_size)

    samples = _load_samples(args.data_path, args.data_root)
    shard = samples[rank::world_size]
    if rank == 0:
        logger.info("Total samples: %d  |  this rank: %d  |  world_size: %d",
                    len(samples), len(shard), world_size)

    processed: List[Dict] = []
    for sample in tqdm(shard, desc=f"[rank {rank}]", disable=(rank != 0)):
        feat_path = _process_sample(
            sample, output_dir, model, transform,
            args.num_frames, device, dtype, args.overwrite,
        )
        if feat_path is None:
            continue
        out_sample = {k: v for k, v in sample.items() if k != "_data_root"}
        out_sample["iv2_feat_path"] = feat_path
        processed.append(out_sample)

    # Write per-rank partial meta; rank 0 merges.
    partial_path = output_dir / f".meta_partial_rank{rank}.json"
    with open(partial_path, "w") as f:
        json.dump(processed, f)
    _barrier()

    if rank == 0:
        merged: List[Dict] = []
        for r in range(world_size):
            p = output_dir / f".meta_partial_rank{r}.json"
            if not p.exists():
                logger.warning("Partial meta missing for rank %d (%s)", r, p)
                continue
            with open(p) as f:
                merged.extend(json.load(f))
            p.unlink()
        meta_path = output_dir / "meta_with_iv2.json"
        with open(meta_path, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info("Wrote %s with %d samples", meta_path, len(merged))

    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
