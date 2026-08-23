#!/usr/bin/env python3
"""
Re-annotate a video dataset with detailed captions, starting from *pre-extracted*
vision features instead of raw video.

Input  : the .pt feature files written by dataset_util/extract_vision_features.py
         -- shape (T, HW, hidden_size) or (T*HW, hidden_size), i.e. exactly the
         frozen vision-encoder output *before* the mm_projector.
Output : one detailed caption per video, as JSONL (streamed, resume-safe) plus a
         merged conversation-format JSON that can be dropped straight into
         anno_online/ and referenced from an anno_data/*.json registry.

Why this works without touching pixels
--------------------------------------
Videollama3MetaForCausalLM.encode_images() is just

    vision_encoder(pixels) -> [optional compressor] -> mm_projector

and prepare_inputs_labels_for_multimodal() then scatters those projected
features into the text embedding sequence at every <image> token. Since the
cached .pt files already hold the vision_encoder output, this script replays
only the second half of that pipeline (mm_projector + scatter) and calls the
plain Qwen2 generate() on the resulting inputs_embeds. The vision encoder is
never run -- and by default is dropped from memory right after load.

NOTE: the model's own `generate()` refuses `inputs_embeds`, so we deliberately
bypass it via `super(Videollama3Qwen2ForCausalLM, model).generate(...)`, which
lands on the vanilla HF generation loop.

Timestamps
----------
Feature files carry no time information, but the chat template wants a
"Time X.0s:" prefix per frame. --timestamp_mode controls where those come from:
  auto     (default) meta 'frame_timestamps' -> duration field -> video
           metadata -> fabricated --fake_fps grid
  meta     ONLY the 'frame_timestamps' extract_vision_features.py recorded
           (fails loudly-ish rather than falling back to a guessed grid)
  video    read fps/frame-count via decord from --video_root/<video> (metadata
           only, no decoding) and replay the 'uniform' sampling grid
  duration use meta's --duration_key and spread T frames evenly over it
  index    frame index as seconds (equivalent to 1 fps)
  fake     ignore every real source and fabricate a --fake_fps grid
  none     no timestamps at all

Prefer 'meta'/'auto': the extractor samples at 1 FPS, which puts frame i at
i + 0.5 seconds, and subsamples that grid for videos longer than its
--max_frames, so *every* re-derived grid here is an approximation. 'index' is
only close for short videos and is badly wrong (off by the subsample factor)
for long ones. Whatever the source, the resulting times are the only wall-clock
signal the LLM gets -- the Qwen2 side sees plain 1-D RoPE over the flattened
token sequence, which encodes frame *order* but not frame *rate*.

Fabricated grids (--fake_fps, used by 'auto' as a last resort and by 'fake'
unconditionally) are a deliberate LIE about wall-clock time, so every record
they produce is tagged "timestamps_synthetic": true in the JSONL and
"synthetic_timestamps": true in --annotation_out, and the run warns with a
count at the end. Use them for plain captioning, where only frame order
matters; never for second-referencing annotation such as temporal grounding or
dense captioning. Pass --fake_fps 0 to restore the old behaviour of emitting no
timestamps at all when no real source is available.

Usage
-----
Single GPU:
    /miniconda/envs/video/bin/python dataset_util/recaption_from_features.py \
        --feat_dir vision_feat_cache \
        --output_file recaption/detail_captions.jsonl \
        --annotation_out anno_online/detail_caption_recap.json

Multi-GPU (samples are sharded rank::world_size, same as the extractor):
    torchrun --nproc_per_node=2 dataset_util/recaption_from_features.py ...
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from tqdm import tqdm

sys.path.append("./")

from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import DEFAULT_CHAT_TEMPLATE, Videollama3Processor


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Describe this video in detail. Cover the main subjects and their appearance, "
    "the actions and events in the order they happen, the scene and background, "
    "any notable camera movement, and any visible text. "
    "Write one coherent, factual paragraph and do not speculate about what is not shown."
)

_INTERNAL_KEYS = ("_data_root", "vision_feat_path")


# ---------------------------------------------------------------------------
# Distributed (mirrors dataset_util/extract_vision_features.py)
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
# Sample collection
# ---------------------------------------------------------------------------

def _feat_path_for(stem: str, feat_dir: Optional[Path]) -> Optional[Path]:
    if feat_dir is None:
        return None
    p = feat_dir / f"{stem}.pt"
    return p if p.exists() else None


def _load_samples(
    meta_path: Optional[str],
    feat_dir: Optional[Path],
    data_root: Optional[str],
) -> List[Dict]:
    """Build the work list. Every item is
    {"video": <relative path or stem>, "feat_path": str, "video_id": stem, "raw": {...}}."""
    samples: List[Dict] = []

    if meta_path is None:
        assert feat_dir is not None, "Need --meta_path or --feat_dir."
        for p in sorted(feat_dir.glob("*.pt")):
            samples.append(
                {"video": p.stem, "feat_path": str(p), "video_id": p.stem, "raw": {"video": p.stem}}
            )
        return samples

    with open(meta_path) as f:
        raw = json.load(f)

    # dict-of-datasets registry (anno_data/*.json) -> flatten the annotations
    if isinstance(raw, dict):
        entries: List[Dict] = []
        for _, ds_cfg in raw.items():
            with open(ds_cfg["annotation"]) as fa:
                ann = json.load(fa)
            for entry in ann:
                entry = dict(entry)
                entry.setdefault("_data_root", ds_cfg.get("data_root", data_root or ""))
                entries.append(entry)
        raw = entries

    seen = set()
    for entry in raw:
        video_field = entry.get("video")
        if isinstance(video_field, (list, tuple)):
            video_field = video_field[0]
        if video_field is None:
            continue
        stem = Path(str(video_field)).stem
        if stem in seen:  # one caption per video, even if the meta has many turns
            continue
        seen.add(stem)

        feat = entry.get("vision_feat_path") or _feat_path_for(stem, feat_dir)
        if feat is None or not Path(feat).exists():
            logger.warning("No feature file for %s -- skipped.", video_field)
            continue
        samples.append(
            {"video": str(video_field), "feat_path": str(feat), "video_id": stem, "raw": entry}
        )
    return samples


def _load_done_ids(output_file: Path) -> set:
    """Collect already-captioned video_ids from every rank's shard file."""
    done = set()
    for p in output_file.parent.glob(f"{output_file.stem}.rank*{output_file.suffix}"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    continue
    return done


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

def _uniform_frame_indices(num_frames: int, vlen: int) -> List[int]:
    """Same grid as mm_utils.get_frame_indices(sample='uniform')."""
    if vlen <= 1:
        return [0] * num_frames
    import numpy as np

    return np.linspace(0, vlen - 1, num_frames).round().astype(int).tolist()


def _video_timestamps(video_path: str, num_frames: int) -> Optional[List[float]]:
    try:
        from decord import VideoReader

        vr = VideoReader(video_path, num_threads=1)  # metadata only, no decode
        vlen, fps = len(vr), float(vr.get_avg_fps())
        if vlen == 0 or fps <= 0:
            return None
        return [idx / fps for idx in _uniform_frame_indices(num_frames, vlen)]
    except Exception as exc:  # noqa: BLE001
        logger.debug("decord metadata read failed on %s: %s", video_path, exc)
        return None


def _meta_timestamps(
    sample: Dict, num_frames: int, kept_indices: Optional[List[int]]
) -> Optional[List[float]]:
    """The frame times extract_vision_features.py recorded for this .pt file.

    These are ground truth -- they come from the decode that produced the cached
    features -- so no sampling grid has to be guessed. `kept_indices` mirrors any
    --max_frames subsample applied on top of the cached tensor.
    """
    ts = sample["raw"].get("frame_timestamps")
    if not isinstance(ts, (list, tuple)) or not ts:
        return None
    ts = [float(x) for x in ts]
    if kept_indices is not None:
        if max(kept_indices) >= len(ts):
            logger.warning(
                "%s: frame_timestamps has %d entries but the cached tensor has more frames; "
                "ignoring the cached timestamps.", sample["video_id"], len(ts),
            )
            return None
        ts = [ts[i] for i in kept_indices]
    if len(ts) != num_frames:
        logger.warning(
            "%s: frame_timestamps length %d != %d frames in the feature file -- the meta and "
            "the .pt are out of sync; ignoring the cached timestamps.",
            sample["video_id"], len(ts), num_frames,
        )
        return None
    return ts


def fake_fps_meta(num_frames: int, fps: float = 1.0) -> Dict:
    """Fabricate the time fields extract_vision_features.py would have recorded,
    pretending the features were sampled at a constant `fps`.

    Mirrors mm_utils.get_frame_indices(sample="fps<N>"): frame i stands for the
    clip [i/fps, (i+1)/fps), so its timestamp is that clip's *midpoint* --
    (i + 0.5)/fps, not i/fps. Keys match the extractor's TIME_META_KEYS so the
    result drops straight into a meta entry, plus a 'synthetic_timestamps'
    marker so nothing downstream mistakes it for a real decode.

    This is a deliberate LIE about wall-clock time. The extractor caps its
    output at --max_frames, so any video longer than num_frames/fps seconds has
    its frames spread across the entire runtime rather than the num_frames/fps
    window claimed here; the error grows with video length and is unbounded.
    Fine when only frame order matters (plain captioning) -- never for
    second-referencing annotation (temporal grounding, dense captioning).
    """
    delta = 1.0 / fps
    return {
        "frame_timestamps": [round((i + 0.5) * delta, 1) for i in range(num_frames)],
        "num_frames": num_frames,
        "video_duration": round(num_frames * delta, 2),
        "video_fps": float(fps),
        "synthetic_timestamps": True,
    }


def _build_timestamps(
    sample: Dict,
    num_frames: int,
    mode: str,
    video_root: Optional[str],
    duration_key: str,
    kept_indices: Optional[List[int]] = None,
    fake_fps: float = 0.0,
) -> Tuple[Optional[List[float]], bool]:
    """Returns (timestamps, is_synthetic)."""
    if mode == "none":
        return None, False

    # 'fake' skips every real source on purpose: it exists to force a chosen
    # frame rate onto features whose true timing is known but unwanted.
    if mode == "fake":
        return fake_fps_meta(num_frames, fake_fps or 1.0)["frame_timestamps"], True

    # Highest priority: the exact times the extractor decoded. Every other mode
    # re-derives a sampling grid and can only ever approximate them.
    if mode in ("auto", "meta"):
        ts = _meta_timestamps(sample, num_frames, kept_indices)
        if ts is not None:
            return ts, False
        if mode == "meta":
            return None, False

    if mode == "index":
        return [float(i) for i in range(num_frames)], False

    duration = sample["raw"].get(duration_key)
    if mode == "auto" and not isinstance(duration, (int, float)):
        # Written by extract_vision_features.py even when the source annotation
        # has no duration field of its own.
        duration = sample["raw"].get("video_duration")
    if mode in ("auto", "duration") and isinstance(duration, (int, float)) and duration > 0:
        if num_frames == 1:
            return [0.0], False
        step = float(duration) / (num_frames - 1)
        return [i * step for i in range(num_frames)], False
    if mode == "duration":
        return None, False

    if video_root or os.path.isabs(str(sample["video"])):
        root = sample["raw"].get("_data_root") or video_root or ""
        video_path = os.path.join(root, str(sample["video"])) if root else str(sample["video"])
        if os.path.exists(video_path):
            ts = _video_timestamps(video_path, num_frames)
            if ts is not None:
                return ts, False

    # No real source panned out. Fabricate a constant-rate grid rather than drop
    # timestamps entirely -- without them the chat template degrades to bare
    # "<image>" separators and the caption loses every wall-clock cue. The caller
    # counts these and warns at the end; --fake_fps 0 restores the old None.
    if mode == "auto" and fake_fps > 0:
        return fake_fps_meta(num_frames, fake_fps)["frame_timestamps"], True
    return None, False


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _load_feature(
    path: str, tokens_per_frame: int, max_frames: int
) -> Tuple[torch.Tensor, int, int, Optional[List[int]]]:
    """Return (feat[T, HW, C], T, HW, kept_indices).

    kept_indices is None when every cached frame is used, else the indices into
    the *original* T that survived the --max_frames subsample -- the caller needs
    them to slice the cached frame_timestamps the same way.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("feat", "feature", "features", "vision_feat"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path}: expected a tensor, got {type(obj)}")

    if obj.dim() == 3:
        t, hw, c = obj.shape
    elif obj.dim() == 2:
        n, c = obj.shape
        hw = tokens_per_frame
        if n % hw != 0:
            raise ValueError(
                f"{path}: flat feature length {n} is not divisible by "
                f"--tokens_per_frame {hw}."
            )
        t = n // hw
        obj = obj.view(t, hw, c)
    else:
        raise ValueError(f"{path}: unexpected feature shape {tuple(obj.shape)}")

    kept: Optional[List[int]] = None
    if max_frames > 0 and t > max_frames:
        idx = torch.linspace(0, t - 1, max_frames).round().long()
        obj = obj[idx]
        kept = idx.tolist()
        t = max_frames
    return obj, t, hw, kept


# ---------------------------------------------------------------------------
# Prompt / embedding assembly
# ---------------------------------------------------------------------------

def _build_prompt_ids(
    processor: Videollama3Processor,
    num_frames: int,
    tokens_per_frame: int,
    merge_size: int,
    timestamps: Optional[List[float]],
    prompt: str,
) -> torch.Tensor:
    side = int(round(math.sqrt(tokens_per_frame)))
    if side * side != tokens_per_frame:
        raise ValueError(f"tokens_per_frame={tokens_per_frame} is not a perfect square.")

    content: List[Dict] = [{"type": "video", "num_frames": num_frames}]
    if timestamps is not None:
        content[0]["timestamps"] = [float(t) for t in timestamps]
    content.append({"type": "text", "text": prompt})
    conversation = [{"role": "user", "content": content}]

    text = processor.tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, add_system_prompt=True
    )
    # process_text expands each <image> into grid_size.prod() image tokens.
    grid_sizes = torch.tensor([[num_frames, side * merge_size, side * merge_size]])
    merge_sizes = torch.tensor([merge_size])
    text_inputs = processor.process_text(
        [text],
        {"grid_sizes": grid_sizes, "merge_sizes": merge_sizes},
        return_tensors="pt",
    )
    return text_inputs["input_ids"][0]


@torch.no_grad()
def _embed_sample(
    model: Videollama3Qwen2ForCausalLM,
    input_ids: torch.Tensor,
    feat: torch.Tensor,
    image_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Text embeddings with the projected vision features scattered into the
    <image> slots -- the inputs_embeds half of prepare_inputs_labels_for_multimodal."""
    input_ids = input_ids.to(device)
    inputs_embeds = model.get_model().embed_tokens(input_ids).clone()

    flat = feat.reshape(-1, feat.shape[-1]).to(device=device, dtype=dtype)
    mm_features = model.get_model().mm_projector(flat)

    image_selected = input_ids == image_token_id
    n_slots = int(image_selected.sum().item())
    if n_slots != mm_features.shape[0]:
        raise RuntimeError(
            f"image-token slots ({n_slots}) != vision tokens ({mm_features.shape[0]})."
        )
    inputs_embeds[image_selected] = mm_features.to(inputs_embeds.dtype)
    return inputs_embeds


def _left_pad(
    embeds_list: List[torch.Tensor], pad_embed: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_len = max(e.shape[0] for e in embeds_list)
    hidden = embeds_list[0].shape[-1]
    device, dtype = embeds_list[0].device, embeds_list[0].dtype

    batch = torch.empty(len(embeds_list), max_len, hidden, device=device, dtype=dtype)
    attn = torch.zeros(len(embeds_list), max_len, device=device, dtype=torch.long)
    for i, e in enumerate(embeds_list):
        pad = max_len - e.shape[0]
        if pad:
            batch[i, :pad] = pad_embed
        batch[i, pad:] = e
        attn[i, pad:] = 1
    return batch, attn


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str, dtype: torch.dtype, attn_impl: str, drop_vision_encoder: bool):
    logger.info("Loading %s ...", model_path)
    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if drop_vision_encoder and getattr(model.get_model(), "vision_encoder", None) is not None:
        # Never invoked here: the features are already encoded.
        model.get_model().vision_encoder = None
        logger.info("Dropped the vision encoder (features are pre-extracted).")
    if model.config.use_cache is None:
        model.config.use_cache = True
    return model.eval()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", default="pretrained_models/videollama3_7b_local",
                        help="Pretrained VideoLLaMA3 checkpoint (only LLM + mm_projector are used).")
    parser.add_argument("--feat_dir", default=None,
                        help="Directory of {video_stem}.pt features from extract_vision_features.py. "
                             "Scanned directly when --meta_path is omitted.")
    parser.add_argument("--meta_path", default=None,
                        help="meta_with_vision_feat.json, a plain sample list, or an "
                             "anno_data-style dict-of-datasets registry. Defaults to "
                             "--feat_dir/meta_with_vision_feat.json when that file exists.")
    parser.add_argument("--data_root", default=None, help="Fallback root for relative video paths.")
    parser.add_argument("--video_root", default=None,
                        help="Video root used ONLY to recover exact timestamps via decord metadata "
                             "(no frames are decoded). Defaults to --data_root.")
    parser.add_argument("--output_file", default="recaption/detail_captions.jsonl",
                        help="JSONL results; each rank streams to {stem}.rank{r}{suffix}, "
                             "rank 0 merges into this path at the end.")
    parser.add_argument("--annotation_out", default=None,
                        help="Also write a conversation-format JSON (anno_online style) here.")

    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt_file", default=None, help="Read the prompt from this file instead.")

    parser.add_argument("--tokens_per_frame", type=int, default=256,
                        help="HW per frame; must match extraction (448/(14*2) -> 16*16=256).")
    parser.add_argument("--merge_size", type=int, default=2,
                        help="Spatial merge size used at extraction time.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Uniformly subsample cached features down to this many frames "
                             "(0 = use all cached frames).")

    parser.add_argument("--timestamp_mode", default="auto",
                        choices=["auto", "meta", "video", "duration", "index", "fake", "none"])
    parser.add_argument("--fake_fps", type=float, default=1.0,
                        help="Frame rate to fabricate when no real timestamp source is available "
                             "('auto') or when forcing one ('fake'): frame i is placed at "
                             "(i+0.5)/fps seconds, the same midpoint convention as the "
                             "extractor's fps1 sampling. Fabricated grids are tagged synthetic in "
                             "every output. 0 disables fabrication, so 'auto' emits no timestamps "
                             "at all when nothing real is found.")
    parser.add_argument("--duration_key", default="duration",
                        help="Meta field holding video duration in seconds.")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Videos per generate() call. >1 is faster, but bf16 batched matmuls "
                             "shift logits slightly, so greedy captions are no longer reproducible "
                             "token-for-token against a batch_size=1 run.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Greedy decoding on a video the model is unsure about falls into a "
                             "self-sustaining loop that runs to --max_new_tokens, and the looping "
                             "captions were also measurably more hallucinated. 1.1 only flips "
                             "near-ties, so necessary repeats ('the watch') survive; 1.0 disables "
                             "the penalty entirely.")

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    parser.add_argument("--keep_vision_encoder", action="store_true",
                        help="Keep the (unused) vision encoder in memory.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing results instead of resuming.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)

    rank, local_rank, world_size = _init_dist()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    feat_dir = Path(args.feat_dir) if args.feat_dir else None
    meta_path = args.meta_path
    if meta_path is not None and not Path(meta_path).exists():
        # A typo'd path should not silently become a fabricated-timestamp run, so
        # this only degrades to scanning when there is a --feat_dir to scan.
        if feat_dir is None:
            parser.error(f"--meta_path {meta_path} does not exist and there is no --feat_dir "
                         f"to fall back on.")
        if rank == 0:
            logger.warning("--meta_path %s does not exist; scanning %s instead.", meta_path, feat_dir)
        meta_path = None
    if meta_path is None and feat_dir is not None:
        default_meta = feat_dir / "meta_with_vision_feat.json"
        if default_meta.exists():
            meta_path = str(default_meta)
            if rank == 0:
                logger.info("Using %s", meta_path)
        elif rank == 0 and args.timestamp_mode == "auto" and args.fake_fps > 0:
            logger.warning(
                "No meta next to the features (%s), so frame timestamps will be FABRICATED at "
                "%.4g FPS. Captions stay usable for content, but their times are made up -- "
                "pass --fake_fps 0 to emit none instead.", default_meta, args.fake_fps)
    if meta_path is None and feat_dir is None:
        parser.error("Pass --feat_dir and/or --meta_path.")

    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rank_file = output_file.parent / f"{output_file.stem}.rank{rank}{output_file.suffix}"
    if args.overwrite and rank_file.exists():
        rank_file.unlink()

    samples = _load_samples(meta_path, feat_dir, args.data_root)
    if args.limit > 0:
        samples = samples[: args.limit]
    done = set() if args.overwrite else _load_done_ids(output_file)
    todo = [s for s in samples if s["video_id"] not in done]
    shard = todo[rank::world_size]
    if rank == 0:
        logger.info(
            "Samples: %d total | %d already captioned | %d to do | %d on this rank",
            len(samples), len(samples) - len(todo), len(todo), len(shard),
        )

    model = _load_model(args.model_path, dtype, args.attn_implementation, not args.keep_vision_encoder)
    model.to(device)

    processor = Videollama3Processor.from_pretrained(args.model_path)
    # from_pretrained loads the checkpoint's chat_template.jinja, which references an
    # undefined `image_token` variable; use the repo's template (identical otherwise).
    processor.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    tokenizer = processor.tokenizer
    image_token_id = processor.image_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        pad_embed = model.get_model().embed_tokens(
            torch.tensor([pad_token_id], device=device)
        )[0]

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        use_cache=True,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if args.do_sample:
        generation_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
    # The model's forward defaults to num_logits_to_keep=0, i.e. it runs lm_head over
    # the *whole* prefill sequence (vocab is ~152k, so that is GBs once a video
    # contributes tens of thousands of image tokens). Only the last position matters
    # for generation. HF 4.57 only auto-fills the newer `logits_to_keep` name, which
    # this forward does not declare, so pass it explicitly.
    if "num_logits_to_keep" in inspect.signature(model.forward).parameters:
        generation_kwargs["num_logits_to_keep"] = 1

    expected_dim = getattr(model.config, "mm_hidden_size", None)
    video_root = args.video_root or args.data_root

    n_fail = 0
    n_no_ts = 0
    n_fake_ts = 0
    fout = open(rank_file, "a", encoding="utf-8")
    pbar = tqdm(total=len(shard), desc=f"[rank {rank}] captioning", disable=(rank != 0))
    for start in range(0, len(shard), args.batch_size):
        batch = shard[start : start + args.batch_size]
        embeds_list, metas = [], []

        for sample in batch:
            try:
                feat, t, hw, kept = _load_feature(
                    sample["feat_path"], args.tokens_per_frame, args.max_frames
                )
                if expected_dim is not None and feat.shape[-1] != expected_dim:
                    raise ValueError(
                        f"feature dim {feat.shape[-1]} != model mm_hidden_size {expected_dim}; "
                        f"features were extracted with a different vision encoder."
                    )
                if hw != args.tokens_per_frame:
                    raise ValueError(
                        f"cached tokens/frame {hw} != --tokens_per_frame {args.tokens_per_frame}"
                    )
                timestamps, ts_synthetic = _build_timestamps(
                    sample, t, args.timestamp_mode, video_root, args.duration_key, kept,
                    args.fake_fps,
                )
                if timestamps is None:
                    n_no_ts += 1
                elif ts_synthetic:
                    n_fake_ts += 1
                input_ids = _build_prompt_ids(
                    processor, t, hw, args.merge_size, timestamps, prompt
                )
                embeds_list.append(
                    _embed_sample(model, input_ids, feat, image_token_id, device, dtype)
                )
                metas.append((sample, t, timestamps, ts_synthetic))
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                logger.warning("Skipping %s: %s", sample["video_id"], exc)

        if not embeds_list:
            pbar.update(len(batch))
            continue

        inputs_embeds, attention_mask = _left_pad(embeds_list, pad_embed)
        with torch.no_grad():
            # The model's own generate() rejects inputs_embeds; go straight to the
            # vanilla Qwen2/HF generation loop.
            output_ids = super(Videollama3Qwen2ForCausalLM, model).generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for (sample, t, timestamps, ts_synthetic), caption in zip(metas, captions):
            record = {
                "video_id": sample["video_id"],
                "video": sample["video"],
                "feat_path": sample["feat_path"],
                "num_frames": t,
                "timestamps": [round(float(x), 2) for x in timestamps] if timestamps else None,
                # True => the times above were made up, not decoded. Filter on this
                # before using these captions for anything that cites seconds.
                "timestamps_synthetic": ts_synthetic,
                "prompt": prompt,
                "caption": caption.strip(),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()
        pbar.update(len(batch))

    pbar.close()
    fout.close()
    if n_fail:
        logger.warning("rank %d: %d samples skipped.", rank, n_fail)
    if n_fake_ts:
        logger.warning(
            "rank %d: %d captions used FABRICATED timestamps at %.4g FPS -- no real source was "
            "available. They are tagged \"timestamps_synthetic\": true in the output; the frame "
            "order is real but the seconds are not, so do not use them for temporal grounding or "
            "dense captioning.", rank, n_fake_ts, args.fake_fps,
        )
    if n_no_ts:
        logger.warning(
            "rank %d: %d captions were generated with NO timestamps (the prompt carried frame "
            "order but no wall-clock time). Re-run extract_vision_features.py so the meta has "
            "frame_timestamps, or pass --video_root / --timestamp_mode explicitly.",
            rank, n_no_ts,
        )

    _barrier()

    if rank == 0:
        # Glob rather than range(world_size): a run resumed under a different GPU
        # count still has to pick up the shards written by the earlier run.
        by_video: Dict[str, Dict] = {}
        for p in sorted(output_file.parent.glob(f"{output_file.stem}.rank*{output_file.suffix}")):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    by_video[rec["video_id"]] = rec  # last write wins
        records = list(by_video.values())
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %s with %d captions", output_file, len(records))

        if args.annotation_out:
            by_id = {s["video_id"]: s for s in samples}
            anno = []
            for rec in records:
                base = dict(by_id.get(rec["video_id"], {}).get("raw", {}))
                for key in _INTERNAL_KEYS:
                    base.pop(key, None)
                base["video"] = rec["video"]
                # Taken from the merged record, not from `raw`: these reflect any
                # --max_frames subsample, and in a multi-GPU run only the rank that
                # owned this sample ever saw the timestamps it actually used.
                if rec.get("timestamps") is not None:
                    base["frame_timestamps"] = rec["timestamps"]
                    base["num_frames"] = rec["num_frames"]
                    if rec.get("timestamps_synthetic"):
                        base["synthetic_timestamps"] = True
                base["conversations"] = [
                    {"from": "human", "value": f"<video>\n{rec['prompt']}"},
                    {"from": "gpt", "value": rec["caption"]},
                ]
                anno.append(base)
            anno_path = Path(args.annotation_out)
            anno_path.parent.mkdir(parents=True, exist_ok=True)
            with open(anno_path, "w", encoding="utf-8") as f:
                json.dump(anno, f, ensure_ascii=False, indent=2)
            logger.info("Wrote %s with %d samples", anno_path, len(anno))

    _barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
