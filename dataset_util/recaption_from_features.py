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
never run: --llm_impl stock never constructs one, and --llm_impl vendored drops
it from memory right after load.

NOTE: the multimodal `generate()` refuses `inputs_embeds`. By default
(--llm_impl stock) the LLM is loaded as a plain `transformers.Qwen2ForCausalLM`
alongside a standalone mm_projector, so `model.generate()` is the vanilla HF
loop already; --llm_impl vendored restores the repo's `qwen2/` copy and reaches
the same loop via `super(Videollama3Qwen2ForCausalLM, model).generate(...)`.
The two backends were verified bitwise identical (teacher-forced, cache off, on
real features: max|dlogit| == 0 across every caption position); stock is kept as
the default only because it decodes ~1.35x faster at batch 1 and ~2.25x faster
at batch 4.

Throughput
----------
Generation is ~90% decode, which is memory-bandwidth bound, so batching is
close to free while prefill is already compute-saturated and gains nothing from
it. Samples are ordered longest-first and packed under --max_batch_tokens, a
*padded*-token budget (batch_size x longest member) that predicts peak VRAM far
better than --batch_size does. Note that batching -- and any change in batch
composition, including a resume -- perturbs bf16 logits enough to flip near-ties,
so greedy captions are not reproducible token-for-token across runs.

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

Multi-GPU (the work list is sorted, then sharded rank::world_size, which hands
every rank a near-identical length distribution):
    torchrun --nproc_per_node=4 dataset_util/recaption_from_features.py ...
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
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.append("./")

from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import DEFAULT_CHAT_TEMPLATE, Videollama3Processor
from videollama3.model.projector import build_vision_projector


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
    embed_tokens: torch.nn.Module,
    projector: torch.nn.Module,
    input_ids: torch.Tensor,
    feat: torch.Tensor,
    image_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Text embeddings with the projected vision features scattered into the
    <image> slots -- the inputs_embeds half of prepare_inputs_labels_for_multimodal."""
    input_ids = input_ids.to(device)
    inputs_embeds = embed_tokens(input_ids).clone()

    flat = feat.reshape(-1, feat.shape[-1]).to(device=device, dtype=dtype)
    mm_features = projector(flat)

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
# Length estimation, ordering and token-budget batching
# ---------------------------------------------------------------------------

# The prompt is overwhelmingly image tokens (T * tokens_per_frame); the rest is
# the chat template plus one "Time X.0s:" run per frame. Measured overhead was
# 12.0-12.8 tokens/frame, so these deliberately over-estimate -- a batch planned
# on them never exceeds --max_batch_tokens once the real ids are built.
_TEXT_TOKENS_PER_FRAME = 20
_TEXT_TOKENS_FIXED = 256


def _probe_num_frames(sample: Dict, tokens_per_frame: int, max_frames: int) -> int:
    """Frames this sample will contribute, without reading the feature payload.

    Prefers the extractor's `num_frames`; otherwise reads only the .pt header
    (~1 ms) via a meta-device load.
    """
    t = sample["raw"].get("num_frames")
    if not isinstance(t, int) or t <= 0:
        try:
            obj = torch.load(sample["feat_path"], map_location="meta", weights_only=True, mmap=True)
            shape = tuple(obj.shape)
            t = shape[0] if len(shape) == 3 else max(1, shape[0] // tokens_per_frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("frame probe failed on %s (%s); assuming 1 frame.", sample["video_id"], exc)
            t = 1
    if max_frames > 0:
        t = min(t, max_frames)
    return int(t)


def _estimate_seq_len(num_frames: int, tokens_per_frame: int) -> int:
    return num_frames * tokens_per_frame + _TEXT_TOKENS_PER_FRAME * num_frames + _TEXT_TOKENS_FIXED


def _pack_batches(
    items: List[Dict], max_batch_tokens: int, max_batch_size: int
) -> List[List[Dict]]:
    """Group a length-ordered list into batches under a *padded*-token budget.

    Peak VRAM tracks padded tokens (batch_size x longest member), not batch size:
    measured 4x8514 and 2x16994 both peaked at the same 22.1 GB. So the budget,
    not --batch_size, is the real memory knob; --batch_size only caps the
    per-sample Python overhead on very short videos.
    """
    batches: List[List[Dict]] = []
    cur: List[Dict] = []
    cur_max = 0
    for item in items:
        nxt_max = max(cur_max, item["est_len"])
        if cur and (len(cur) + 1) * nxt_max > max_batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            nxt_max = item["est_len"]
        cur.append(item)
        cur_max = nxt_max
        if len(cur) >= max_batch_size:
            batches.append(cur)
            cur, cur_max = [], 0
    if cur:
        batches.append(cur)

    over = [b for b in batches if len(b) == 1 and b[0]["est_len"] > max_batch_tokens]
    if over:
        logger.warning(
            "%d sample(s) exceed --max_batch_tokens on their own (longest %d tokens) and run "
            "unbatched; raise the budget or lower --max_frames if they OOM.",
            len(over), max(b[0]["est_len"] for b in over),
        )
    return batches


def _identity_collate(batch):
    """The dataset already yields a whole batch; DataLoader must not re-collate."""
    return batch


class _FeatureBatchDataset(Dataset):
    """Does the CPU-side prep (torch.load + timestamps + prompt ids) off the main
    process so it overlaps with generate() instead of stalling the GPU."""

    def __init__(self, batches, processor, args, video_root, prompt):
        self.batches = batches
        self.processor = processor
        self.args = args
        self.video_root = video_root
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int) -> List[Dict]:
        a = self.args
        out = []
        for item in self.batches[idx]:
            sample = item["sample"]
            try:
                feat, t, hw, kept = _load_feature(sample["feat_path"], a.tokens_per_frame, a.max_frames)
                if hw != a.tokens_per_frame:
                    raise ValueError(
                        f"cached tokens/frame {hw} != --tokens_per_frame {a.tokens_per_frame}"
                    )
                timestamps, ts_synthetic = _build_timestamps(
                    sample, t, a.timestamp_mode, self.video_root, a.duration_key, kept, a.fake_fps,
                )
                input_ids = _build_prompt_ids(
                    self.processor, t, hw, a.merge_size, timestamps, self.prompt
                )
                out.append({
                    "sample": sample, "feat": feat, "t": t, "input_ids": input_ids,
                    "timestamps": timestamps, "ts_synthetic": ts_synthetic, "error": None,
                })
            except Exception as exc:  # noqa: BLE001
                out.append({"sample": sample, "error": str(exc)})
        return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_mm_projector(model_path: str, dtype: torch.dtype) -> torch.nn.Module:
    """Rebuild just `model.mm_projector` from the checkpoint, without the LLM.

    Only four tensors (mlp2x_gelu: 1152->3584->3584, ~30 MB), so this is read
    straight out of the shard rather than through from_pretrained.
    """
    with open(os.path.join(model_path, "config.json")) as f:
        raw_cfg = json.load(f)
    mm_hidden = raw_cfg["vision_encoder_config"]["hidden_size"]

    state = {}
    prefix = "model.mm_projector."
    shards = sorted(Path(model_path).glob("*.safetensors"))
    if shards:
        from safetensors.torch import load_file

        for shard in shards:
            for k, v in load_file(str(shard)).items():
                if k.startswith(prefix):
                    state[k[len(prefix):]] = v
    else:
        # mmap keeps this from paging in the whole (30 GB, fp32) single-file checkpoint.
        full = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu", weights_only=True, mmap=True,
        )
        state = {k[len(prefix):]: v for k, v in full.items() if k.startswith(prefix)}
    if not state:
        raise RuntimeError(f"{model_path}: no '{prefix}*' weights found.")

    projector = build_vision_projector(
        SimpleNamespace(
            hidden_size=raw_cfg["hidden_size"],
            mm_projector_type=raw_cfg.get("mm_projector_type", "mlp2x_gelu"),
        ),
        mm_hidden,
    )
    projector.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)
    return projector.to(dtype).eval()


def _load_model(model_path: str, dtype: torch.dtype, attn_impl: str, llm_impl: str):
    """Returns (model, projector, embed_tokens, generate_fn).

    llm_impl="stock" loads the LLM as a plain `transformers.Qwen2ForCausalLM`
    instead of the repo's vendored `qwen2/` copy. The two were verified to be
    *bitwise* identical (teacher-forced, cache disabled, real features:
    max|dlogit| == 0 over every caption position), but the vendored copy -- a
    transformers-4.46.3-era snapshot kept only because the trainable-compressor
    training path subclasses it -- decodes ~1.35x slower at batch 1 and ~2.25x
    slower at batch 4. Nothing here needs the multimodal subclass: the features
    are pre-encoded, so only the mm_projector and the plain LLM are ever run.

    Greedy captions still will not match a vendored run token-for-token, but that
    is bf16 KV-cache rounding, not an implementation difference -- the vendored
    model does not reproduce its own cache-free output either, and every observed
    divergence sat on a top-2 logit margin at or below bf16 resolution.
    """
    logger.info("Loading %s (llm_impl=%s) ...", model_path, llm_impl)

    if llm_impl == "vendored":
        model = Videollama3Qwen2ForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation=attn_impl, low_cpu_mem_usage=True,
        )
        if getattr(model.get_model(), "vision_encoder", None) is not None:
            # Never invoked here: the features are already encoded.
            model.get_model().vision_encoder = None
            logger.info("Dropped the vision encoder (features are pre-extracted).")
        if model.config.use_cache is None:
            model.config.use_cache = True
        model = model.eval()
        projector = model.get_model().mm_projector
        embed_tokens = model.get_model().embed_tokens
        # The multimodal generate() rejects inputs_embeds; go straight to the
        # vanilla HF generation loop on the parent class.
        def generate_fn(**kwargs):
            return super(Videollama3Qwen2ForCausalLM, model).generate(**kwargs)
        return model, projector, embed_tokens, generate_fn

    config = Qwen2Config.from_pretrained(model_path)
    model = Qwen2ForCausalLM.from_pretrained(
        model_path, config=config, dtype=dtype,
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
    ).eval()
    if model.config.use_cache is None:
        model.config.use_cache = True
    projector = _load_mm_projector(model_path, dtype)
    return model, projector, model.get_input_embeddings(), model.generate


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

    parser.add_argument("--batch_size", type=int, default=16,
                        help="Hard cap on videos per generate() call. The real memory knob is "
                             "--max_batch_tokens; this only bounds the per-sample Python overhead "
                             "on very short videos. NOTE: bf16 batched matmuls shift logits "
                             "slightly, so greedy captions are not reproducible token-for-token "
                             "against a batch_size=1 run (neither is a rerun with a different "
                             "batch composition).")
    parser.add_argument("--max_batch_tokens", type=int, default=131072,
                        help="Padded-token budget per generate() call -- batch_size x the longest "
                             "member. This, not --batch_size, predicts peak VRAM: measured ~199 KB "
                             "per padded token on top of the weights, consistently across shapes. "
                             "131072 suits an 80 GB card; use ~32768 on 24 GB.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers prefetching features / building prompt ids.")
    parser.add_argument("--no_sort_by_length", action="store_true",
                        help="Process in meta order instead of longest-first. Sorting keeps a "
                             "batch from being padded out to its longest member (cached T ranges "
                             "4..100 frames), and longest-first makes an over-budget run OOM in "
                             "the first minute rather than hours in.")
    parser.add_argument("--llm_impl", choices=["stock", "vendored"], default="stock",
                        help="Which Qwen2 implementation runs the LLM. 'stock' uses "
                             "transformers.Qwen2ForCausalLM plus a standalone mm_projector; it is "
                             "bitwise identical to 'vendored' (verified teacher-forced with the "
                             "cache off) but decodes ~1.35x faster at batch 1 and ~2.25x faster at "
                             "batch 4. 'vendored' restores the repo's qwen2/ copy.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
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

    items = [
        {"sample": s, "est_len": _estimate_seq_len(
            _probe_num_frames(s, args.tokens_per_frame, args.max_frames), args.tokens_per_frame)}
        for s in todo
    ]
    if not args.no_sort_by_length:
        # Sort the whole list *before* sharding: round-robin over a sorted list
        # hands every rank a near-identical length distribution for free, which
        # sharding first and sorting second only achieves by luck.
        items.sort(key=lambda it: -it["est_len"])
    shard = items[rank::world_size]
    batches = _pack_batches(shard, args.max_batch_tokens, args.batch_size)
    if rank == 0:
        lens = [it["est_len"] for it in items]
        sizes = [len(b) for b in batches]
        logger.info(
            "Samples: %d total | %d already captioned | %d to do | %d on this rank",
            len(samples), len(samples) - len(todo), len(todo), len(shard),
        )
        if shard:
            logger.info(
                "Est. seq len %d..%d tokens | %d batches on this rank, size %d..%d (mean %.1f) "
                "under a %d-token budget",
                min(lens), max(lens), len(batches), min(sizes), max(sizes),
                sum(sizes) / len(sizes), args.max_batch_tokens,
            )

    model, projector, embed_tokens, generate_fn = _load_model(
        args.model_path, dtype, args.attn_implementation, args.llm_impl
    )
    model.to(device)
    projector.to(device)

    processor = Videollama3Processor.from_pretrained(args.model_path)
    # from_pretrained loads the checkpoint's chat_template.jinja, which references an
    # undefined `image_token` variable; use the repo's template (identical otherwise).
    processor.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    tokenizer = processor.tokenizer
    image_token_id = processor.image_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        pad_embed = embed_tokens(torch.tensor([pad_token_id], device=device))[0]

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
    # Without this the forward runs lm_head over the *whole* prefill sequence (vocab
    # is ~152k, so that is GBs once a video contributes tens of thousands of image
    # tokens); only the last position matters for generation. Stock Qwen2 declares
    # the newer `logits_to_keep` name and HF 4.57 auto-fills it to 1, but the
    # vendored copy still spells it `num_logits_to_keep`, which nothing auto-fills.
    if "num_logits_to_keep" in inspect.signature(model.forward).parameters:
        generation_kwargs["num_logits_to_keep"] = 1

    # The projector's input width is the vision encoder's hidden size, and it is
    # present under both backends (stock's Qwen2Config has no mm_hidden_size).
    first_linear = next(m for m in projector.modules() if isinstance(m, torch.nn.Linear))
    expected_dim = first_linear.in_features
    video_root = args.video_root or args.data_root

    n_fail = 0
    n_no_ts = 0
    n_fake_ts = 0
    fout = open(rank_file, "a", encoding="utf-8")
    loader = DataLoader(
        _FeatureBatchDataset(batches, processor, args, video_root, prompt),
        batch_size=None,                 # the dataset already yields whole batches
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=_identity_collate,
    )
    pbar = tqdm(total=len(shard), desc=f"[rank {rank}] captioning", disable=(rank != 0))
    for prepared in loader:
        embeds_list, metas = [], []

        for item in prepared:
            sample = item["sample"]
            if item["error"] is not None:
                n_fail += 1
                logger.warning("Skipping %s: %s", sample["video_id"], item["error"])
                continue
            try:
                feat = item["feat"]
                if feat.shape[-1] != expected_dim:
                    raise ValueError(
                        f"feature dim {feat.shape[-1]} != projector input width {expected_dim}; "
                        f"features were extracted with a different vision encoder."
                    )
                if item["timestamps"] is None:
                    n_no_ts += 1
                elif item["ts_synthetic"]:
                    n_fake_ts += 1
                embeds_list.append(
                    _embed_sample(embed_tokens, projector, item["input_ids"], feat,
                                  image_token_id, device, dtype)
                )
                metas.append((sample, item["t"], item["timestamps"], item["ts_synthetic"]))
            except Exception as exc:  # noqa: BLE001
                n_fail += 1
                logger.warning("Skipping %s: %s", sample["video_id"], exc)

        if not embeds_list:
            pbar.update(len(prepared))
            continue

        inputs_embeds, attention_mask = _left_pad(embeds_list, pad_embed)
        with torch.no_grad():
            output_ids = generate_fn(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        del inputs_embeds, attention_mask, embeds_list

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
        pbar.update(len(prepared))

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
