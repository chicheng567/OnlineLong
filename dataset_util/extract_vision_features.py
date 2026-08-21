#!/usr/bin/env python3
"""
Extract per-video VideoLLaMA3 vision-encoder features for a dataset.

For every sample in --data_path, decode frames at a fixed 1 FPS, run them
through the frozen VideoLLaMA3 vision encoder (SigLIP-NaViT, extracted from the
model checkpoint at --model_path), and save the resulting token features as a
.pt file under ``--output_dir/{video_stem}.pt``.

Frame sampling is hard-wired to ``sample="fps1"``: one frame per second of
video, so T tracks the real video duration. Only when a video is long enough
that 1 FPS would yield more than ``--max_frames`` frames are those 1-FPS indices
uniformly subsampled down to exactly ``--max_frames``. Videos shorter than
``--max_frames`` seconds therefore produce fewer than ``--max_frames`` frames
(no padding); ``--min_frames`` still filters out the very short ones.

This mirrors dataset_util/precompute_iv2_video_feat.py but targets the video
model's own vision encoder instead of InternVideo2-L, using the same
frame-decode + patchify pipeline as AE pretraining
(videollama3/train/videollama3_pretrain_compressor.py): read_frames_decord ->
Videollama3ImageProcessor -> Videollama3VisionEncoderModel.

Each saved feature tensor has shape (T, HW, hidden_size) in frame-major order
(or (T*HW, hidden_size) with --flatten), where T is the number of decoded
frames (``min(ceil(duration_sec), max_frames)``, i.e. variable across videos)
and HW = (force_image_size // (patch_size * video_merge_size)) ** 2.

Frame timestamps
----------------
The .pt tensors carry no time axis, so the exact wall-clock second of every
saved frame is recorded in the output meta instead (``frame_timestamps``, one
float per row of the T dimension, plus ``video_duration`` / ``video_fps`` /
``num_frames``). Downstream consumers must read these rather than re-deriving a
grid: 1-FPS sampling puts frame i at ``i + 0.5`` seconds -- NOT at ``i`` -- and
for videos longer than max_frames seconds those 1-FPS indices are uniformly
subsampled, so the spacing is no longer 1 second either. Both regimes are
already baked into what is written here.

For samples whose feature file is reused from an earlier run (resume without
--overwrite) no frames are decoded, so the timestamps are recomputed from
decord *metadata only* (container fps + frame count), replaying the very same
get_frame_indices() call. If the video file is gone the entry gets
``frame_timestamps: null`` and is counted in the run summary.

The script is resume-safe (cached samples are skipped unless --overwrite).
Multi-GPU sharding via torchrun is supported — each rank processes a strided
shard samples[rank::world_size].

Usage
-----
Single GPU:
    /miniconda/envs/video/bin/python dataset_util/extract_vision_features.py \\
        --model_path pretrained_models/videollama3_7b_local \\
        --data_path anno_online/testt.json \\
        --data_root /workspace/datasetfortest/ActivityNets \\
        --output_dir vision_feat_cache

Multi-GPU:
    torchrun --nproc_per_node=4 dataset_util/extract_vision_features.py ...

Output layout
-------------
    vision_feat_cache/
      {video_stem}.pt              # (T, HW, hidden_size) or (T*HW, hidden_size), fp16
      meta_with_vision_feat.json   # input meta + per sample:
                                   #   vision_feat_path : str
                                   #   frame_timestamps : [float] | null  (seconds, len == T)
                                   #   num_frames       : int   | null
                                   #   video_duration   : float | null    (seconds)
                                   #   video_fps        : float | null    (container avg fps)
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("./")

from videollama3.mm_utils import get_frame_indices, read_frames_decord
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.videollama3_encoder import Videollama3ImageProcessor


logger = logging.getLogger(__name__)


# Meta keys this script adds on top of whatever the input annotation carried.
TIME_META_KEYS = ("frame_timestamps", "num_frames", "video_duration", "video_fps")


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

_VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def _scan_video_root(data_root: str) -> List[Dict]:
    """Build a sample list by recursively globbing video files under data_root
    (same extensions/recursive convention as iterative_build_infer_dataset.py).
    Each sample's "video" is stored relative to data_root so it round-trips
    through the same os.path.join(root, video) logic as the JSON-driven path."""
    root = Path(data_root)
    items: List[Dict] = []
    for ext in _VIDEO_EXTS:
        for p in sorted(root.rglob(f"*{ext}")):
            items.append({"video": str(p.relative_to(root)), "_data_root": data_root})
    return items


def _load_samples(data_path: Optional[str], data_root: Optional[str]) -> List[Dict]:
    if data_path is None:
        assert data_root, "Must pass --data_root when --data_path is omitted."
        return _scan_video_root(data_root)

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
# Frame timestamps
# ---------------------------------------------------------------------------

def _probe_frame_times(video_path: str, max_frames: int) -> Dict:
    """Recover the sampled frame times without decoding a single frame.

    Mirrors read_frames_decord(sample="fps1", ...) exactly: same get_frame_indices
    call, same `round(idx / fps, 1)` timestamps. Only the container header is
    read, so this is cheap enough to run for every cached sample on a resume.
    """
    from decord import VideoReader

    vr = VideoReader(video_path, num_threads=1)
    vlen, fps = len(vr), float(vr.get_avg_fps())
    if vlen == 0 or fps <= 0:
        raise ValueError(f"unusable video metadata (vlen={vlen}, fps={fps})")
    frame_indices = get_frame_indices(
        max_frames, vlen, sample="fps1", input_fps=fps, max_num_frames=max_frames
    )
    return {
        "frame_timestamps": [round(idx / fps, 1) for idx in frame_indices],
        "num_frames": len(frame_indices),
        "video_duration": round(vlen / fps, 2),
        "video_fps": round(fps, 3),
    }


# ---------------------------------------------------------------------------
# Vision encoder loading (same pattern as
# videollama3_pretrain_compressor.py::_load_vision_encoder): load the full
# checkpoint just to extract the vision tower, then drop the LLM weights.
# ---------------------------------------------------------------------------

def _load_vision_encoder(model_path: str, dtype: torch.dtype):
    logger.info("Loading vision encoder from %s ...", model_path)
    full_model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    ve = full_model.get_vision_encoder()
    ve_hidden_size = ve.hidden_size
    ve = ve.to("cpu")
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Vision encoder loaded. hidden_size=%d", ve_hidden_size)
    return ve, ve_hidden_size


# ---------------------------------------------------------------------------
# Dataset: parallel frame decode + patchify (CPU only; the encoder forward
# happens batched on GPU in the main loop).
# ---------------------------------------------------------------------------

class _VideoFrameDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict],
        output_dir: Path,
        image_processor: Videollama3ImageProcessor,
        max_frames: int,
        min_frames: int,
        merge_size: int,
        overwrite: bool,
    ):
        self.samples = samples
        self.output_dir = output_dir
        self.image_processor = image_processor
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.merge_size = merge_size
        self.overwrite = overwrite

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict:
        sample = self.samples[i]
        root = sample.get("_data_root", "")
        video_field = sample["video"]
        if isinstance(video_field, (list, tuple)):
            video_field = video_field[0]
        video_path = os.path.join(root, video_field) if root else video_field

        stem = Path(str(video_field)).stem
        out_path = self.output_dir / f"{stem}.pt"

        out_sample = {k: v for k, v in sample.items() if k != "_data_root"}
        result: Dict = {"sample": out_sample, "out_path": str(out_path)}

        if out_path.exists() and not self.overwrite:
            # The features are reused as-is, but the meta still has to describe
            # them, so recover the frame times from the container header alone.
            result["status"] = "cached"
            try:
                result["time_meta"] = _probe_frame_times(video_path, self.max_frames)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No frame timestamps for cached %s: %s", video_path, exc)
                result["time_meta"] = None
            return result

        try:
            # Fixed 1-FPS sampling; get_frame_indices only falls back to a
            # uniform subsample of those 1-FPS indices when the video is long
            # enough to exceed max_frames. return_timestamps gives us the exact
            # second of each kept frame for free -- same decord read, no extra IO.
            decoded = read_frames_decord(
                video_path,
                num_frames=self.max_frames,
                sample="fps1",
                max_num_frames=self.max_frames,
                return_timestamps=True,
            )
            assert isinstance(decoded, tuple) and len(decoded) == 2
            frames, timestamps = decoded
            assert isinstance(frames, list) and len(frames) > 0
            assert len(timestamps) == len(frames)
        except Exception as exc:
            logger.warning("decord failed on %s: %s", video_path, exc)
            result["status"] = "fail"
            return result

        if len(frames) < self.min_frames:
            logger.warning(
                "Skipping %s: decoded %d frames < min_frames=%d",
                video_path, len(frames), self.min_frames,
            )
            result["status"] = "fail"
            return result

        try:
            data_dict = self.image_processor(
                images=[frames], merge_size=self.merge_size, return_tensors="pt"
            )
        except Exception as exc:
            logger.warning("Image processor failed on %s: %s", video_path, exc)
            result["status"] = "fail"
            return result

        # `timestamps` came out of the real decode, so it is authoritative. The
        # header-only probe is run alongside it purely to keep duration/fps in
        # the meta and to assert that the probe -- which is all the cached path
        # on a resume has to go on -- reproduces the decode exactly.
        time_meta = {
            "frame_timestamps": [float(x) for x in timestamps],
            "num_frames": len(frames),
            "video_duration": None,
            "video_fps": None,
        }
        try:
            probed = _probe_frame_times(video_path, self.max_frames)
            time_meta["video_duration"] = probed["video_duration"]
            time_meta["video_fps"] = probed["video_fps"]
            if probed["frame_timestamps"] != time_meta["frame_timestamps"]:
                logger.warning(
                    "Timestamp probe disagrees with the decode on %s (probe=%s, decode=%s); "
                    "a resumed run would record the probe's values for this video.",
                    video_path, probed["frame_timestamps"], time_meta["frame_timestamps"],
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Duration/fps probe failed on %s: %s", video_path, exc)

        result["pixel_values"] = data_dict["pixel_values"]    # (patches, patch_dim)
        result["grid_sizes"] = data_dict["grid_sizes"]         # (1, 3) = [t, h, w] pre-merge
        result["merge_sizes"] = data_dict["merge_sizes"]       # (1,)
        result["n_frames"] = len(frames)
        result["time_meta"] = time_meta
        result["status"] = "ok"
        return result


def _identity_collate(batch: List[Dict]) -> List[Dict]:
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="pretrained_models/videollama3_7b_local",
                        help="VideoLLaMA3 checkpoint dir; only the vision encoder is extracted.")
    parser.add_argument("--data_path", default=None,
                        help="Path to JSON: list-of-samples or dict-of-datasets. If omitted, "
                             "--data_root is scanned recursively for video files instead.")
    parser.add_argument("--data_root", default=None,
                        help="Prefix for video paths named in --data_path (overridden by "
                             "per-dataset data_root); required on its own when --data_path is omitted.")
    parser.add_argument("--output_dir", required=True)

    # Frame sampling hyperparameters. Sampling itself is fixed at 1 FPS.
    parser.add_argument("--max_frames", type=int, default=10,
                        help="Upper bound on frames decoded per video (== T in the saved "
                             "(T, HW, hidden) tensor). Videos are sampled at 1 FPS; only when "
                             "that yields more than this many frames are the 1-FPS indices "
                             "uniformly subsampled down to exactly max_frames.")
    parser.add_argument("--min_frames", type=int, default=4,
                        help="Skip videos that decode to fewer than this many frames.")

    # Patchify hyperparameters (must match how the encoder/compressor expects tokens).
    parser.add_argument("--video_merge_size", type=int, default=2,
                        help="Spatial merge size; default 2 -> 256 tokens/frame at force_image_size=448.")
    parser.add_argument("--force_image_size", type=int, default=448,
                        help="Force every frame to this square size before patching. Set <=0 to "
                             "disable forcing and let the processor dynamically resize instead.")

    # Compute / IO hyperparameters.
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Videos per encoder forward pass.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers for parallel video decoding + patchify.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Compute dtype for the vision encoder forward.")
    parser.add_argument("--save_dtype", choices=["bf16", "fp16", "fp32"], default="fp16",
                        help="Dtype the output feature tensors are saved as.")
    parser.add_argument("--flatten", action="store_true",
                        help="Save features as (T*HW, hidden_size) instead of (T, HW, hidden_size).")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.data_path is None and not args.data_root:
        parser.error("Must pass --data_path, --data_root, or both.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rank, local_rank, world_size = _init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    save_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.save_dtype]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vision_encoder, hidden_size = _load_vision_encoder(args.model_path, dtype=compute_dtype)
    vision_encoder = vision_encoder.to(device).eval()

    image_processor = Videollama3ImageProcessor.from_pretrained(args.model_path)
    if args.force_image_size > 0:
        image_processor.force_size = [args.force_image_size] * 2
        patch_size = image_processor.patch_size
        side = args.force_image_size // (patch_size * args.video_merge_size)
        if rank == 0:
            logger.info(
                "force_image_size=%d, patch_size=%d, video_merge_size=%d -> %d tokens/frame",
                args.force_image_size, patch_size, args.video_merge_size, side * side,
            )

    samples = _load_samples(args.data_path, args.data_root)
    shard = samples[rank::world_size]
    if rank == 0:
        logger.info("Total samples: %d  |  this rank: %d  |  world_size: %d",
                    len(samples), len(shard), world_size)

    dataset = _VideoFrameDataset(
        shard, output_dir, image_processor,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        merge_size=args.video_merge_size,
        overwrite=args.overwrite,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=_identity_collate,
    )

    processed: List[Dict] = []
    n_fail = 0
    n_no_ts = 0
    with tqdm(total=len(shard), desc=f"[rank {rank}]", disable=(rank != 0)) as pbar:
        for batch in loader:
            to_run = [r for r in batch if r["status"] == "ok"]
            if to_run:
                pixel_values = torch.cat([r["pixel_values"] for r in to_run], dim=0).to(
                    device=device, dtype=compute_dtype
                )
                grid_sizes = torch.cat([r["grid_sizes"] for r in to_run], dim=0).to(device)
                merge_sizes = torch.cat([r["merge_sizes"] for r in to_run], dim=0).to(device)

                with torch.no_grad():
                    visual_tokens = vision_encoder(
                        pixel_values=pixel_values,
                        grid_sizes=grid_sizes,
                        merge_sizes=merge_sizes,
                    )  # (sum_i T_i*HW_i, hidden_size), frame-major per video

                # Split the flat encoder output back into per-video (T, HW, hidden)
                # chunks. Post-merge token count per video = t * (h//m) * (w//m).
                counts = [
                    int(gs[0]) * (int(gs[1]) // int(ms)) * (int(gs[2]) // int(ms))
                    for gs, ms in zip(grid_sizes.tolist(), merge_sizes.tolist())
                ]
                per_video = visual_tokens.split(counts, dim=0)

                for r, feat, gs in zip(to_run, per_video, grid_sizes.tolist()):
                    t = gs[0]
                    feat = feat.view(t, -1, hidden_size) if not args.flatten else feat.view(-1, hidden_size)
                    feat = feat.to(save_dtype).cpu().clone()
                    Path(r["out_path"]).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(feat, r["out_path"])

            for r in batch:
                if r["status"] == "fail":
                    n_fail += 1
                    continue
                s = dict(r["sample"])
                s["vision_feat_path"] = r["out_path"]
                # Overwrites any same-named field from the source annotation on
                # purpose: these describe the frames actually saved in the .pt,
                # which is what downstream consumers need to align against.
                time_meta = r.get("time_meta")
                if time_meta is None:
                    n_no_ts += 1
                    time_meta = {k: None for k in TIME_META_KEYS}
                s.update(time_meta)
                processed.append(s)
            pbar.update(len(batch))
    if n_fail:
        logger.warning("rank %d: %d samples failed to decode/process and were skipped", rank, n_fail)
    if n_no_ts:
        logger.warning(
            "rank %d: %d samples have no frame_timestamps (video unreadable); downstream "
            "timestamp_mode=meta will fall back for those.", rank, n_no_ts,
        )

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
        meta_path = output_dir / "meta_with_vision_feat.json"
        with open(meta_path, "w") as f:
            json.dump(merged, f, indent=2)
        logger.info("Wrote %s with %d samples", meta_path, len(merged))

    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
