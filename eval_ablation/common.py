"""
Shared plumbing for the compressor prune-vs-noprune ablation eval.

Everything both eval scripts need: load a compressor checkpoint, decode a video the
exact way training did (frozen SigLIP-NaViT -> token compressor -> mm_projector ->
LLM), and turn it into `(input_ids, pixel_values, grid_sizes, merge_sizes,
compression_parts, compression_ts_info)` where the video is cut into fixed
`window_size`-frame groups (training used fixed_frames=8, one group per clip; here
a longer clip becomes several consecutive groups).

Dynamic HW: compression is always driven through the model's own
`_grid_hw_for_compression_parts` + `compressor.output_hw_for(h, w)` path -- the
per-window (h, w) actually seen by the encoder is what sizes the cross-attention
RoPE and the compressed placeholder block. Nothing here hardcodes 16x16 / 256.
`describe_grid()` returns those per-window grids so callers can log that the
dynamic path ran.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from videollama3.constants import DEFAULT_IMAGE_TOKEN  # noqa: E402
from videollama3.model import Videollama3Qwen2ForCausalLM  # noqa: E402
from videollama3.model.processor import Videollama3Processor  # noqa: E402
from videollama3.model.videollama3_arch import _grid_hw_for_compression_parts  # noqa: E402
from videollama3.mm_utils import load_video  # noqa: E402
from videollama3.train.videollama3_chat_finetune_compressor import (  # noqa: E402
    select_full_compression_parts,
)

# The instruction the two compressors were pretrained under (anno_online/
# detail_caption_recap.json). Using it verbatim keeps the caption eval on
# distribution.
TRAIN_CAPTION_PROMPT = (
    "Describe this video in detail. Cover the main subjects and their appearance, "
    "the actions and events in the order they happen, the scene and background, any "
    "notable camera movement, and any visible text. Write one coherent, factual "
    "paragraph and do not speculate about what is not shown."
)

DEFAULT_WINDOW_SIZE = 8      # frames per compressor group  (== training fixed_frames)
DEFAULT_MAX_FRAMES = 64      # relaxed frame budget for the caption eval
DEFAULT_MERGE_SIZE = 2       # == training video_merge_size
DEFAULT_FPS = 1
DEFAULT_FORCE_IMAGE_SIZE = 448   # == training force_image_size (16x16 post-merge grid)


# --------------------------------------------------------------------------------------
# model / processor
# --------------------------------------------------------------------------------------
def load_model(model_path: str, device: str = "cuda:0",
               dtype: torch.dtype = torch.bfloat16) -> Videollama3Qwen2ForCausalLM:
    try:
        model = Videollama3Qwen2ForCausalLM.from_pretrained(
            model_path, dtype=dtype, attn_implementation="flash_attention_2")
    except TypeError:  # older transformers
        model = Videollama3Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation="flash_attention_2")
    if getattr(model.get_model(), "token_compressor", None) is None:
        raise RuntimeError(
            f"{model_path} has no token_compressor -- not a compressor checkpoint."
        )
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model


def free_model(model) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_processor(model_path: str,
                   force_image_size: int = DEFAULT_FORCE_IMAGE_SIZE,
                   native_max_tokens: int = 1600) -> Videollama3Processor:
    """`force_image_size` > 0 -> every frame resized to that square (training
    geometry, all windows land on a 16x16 post-merge grid). `force_image_size` in
    {0, None} -> keep the encoder's native aspect-ratio resolution, but clamp
    max_tokens so 64 frames stay tractable; grids then vary video-to-video and the
    dynamic-HW path carries the real (h, w). Off-distribution -- use only to stress
    dynamic HW."""
    proc = Videollama3Processor.from_pretrained(
        model_path, trust_remote_code=False, fix_mistral_regex=True)
    if proc.tokenizer.pad_token is None and proc.tokenizer.unk_token is not None:
        proc.tokenizer.pad_token = proc.tokenizer.unk_token
    if not force_image_size:
        proc.image_processor.force_size = None
        proc.image_processor.max_tokens = int(native_max_tokens)
    else:
        proc.image_processor.force_size = [int(force_image_size)] * 2
    return proc


# --------------------------------------------------------------------------------------
# video -> model inputs
# --------------------------------------------------------------------------------------
def _build_ts_info(timestamps: List[float], parts: List[List[int]],
                   total_frames: int, total_vision_tokens: int, tokenizer):
    """Per-window (old "Time X.0s:" token length, new "Time:Xs-Ye:" token ids).

    Byte-for-byte the same construction as
    videollama3_chat_finetune_compressor._build_ts_info, so
    prepare_inputs_labels_for_multimodal trims/splices exactly as in training."""
    if not parts or not timestamps or total_vision_tokens <= 0:
        return [(0, []) for _ in parts]
    tpf = total_vision_tokens // total_frames
    info = []
    for s, e in parts:
        fs, fe = s // tpf, e // tpf
        ts_start = float(timestamps[fs])
        ts_end = float(timestamps[min(fe, len(timestamps)) - 1])
        old_ids = tokenizer.encode(f"Time {ts_start:.1f}s:", add_special_tokens=False)
        new_ids = tokenizer.encode(f"Time:{ts_start:.1f}s-{ts_end:.1f}s:", add_special_tokens=False)
        info.append((len(old_ids), new_ids))
    return info


def prepare_video_sample(
    proc: Videollama3Processor,
    video_path: str,
    prompt: str = TRAIN_CAPTION_PROMPT,
    fps: int = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    window_size: int = DEFAULT_WINDOW_SIZE,
    merge_size: int = DEFAULT_MERGE_SIZE,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    out_hw_fn=None,
) -> Dict:
    """Decode `video_path` and package it the training way.

    `out_hw_fn(h, w) -> (oh, ow)` maps a window's input grid to its compressed
    output grid; pass `model.get_token_compressor().output_hw_for` so the logged
    token counts are exact (transformer_decoder_flat emits num_queries, not h*w).
    Defaults to identity for a model-free call.

    Returns a dict with the model.generate kwargs plus `meta` (num_frames,
    timestamps, per-window frame spans, per-window input (h, w) and output grids,
    token counts).
    """
    if out_hw_fn is None:
        out_hw_fn = lambda h, w: (h, w)  # noqa: E731
    frames, timestamps = load_video(video_path, fps=fps, max_frames=max_frames)
    num_frames = len(frames)
    timestamps = [float(t) for t in timestamps]

    prompt_text = prompt.replace("<video>", "").strip()
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "timestamps": timestamps, "num_frames": num_frames},
            {"type": "text", "text": prompt_text},
        ],
    }]

    inputs = proc(images=[frames], text=conversation,
                  merge_size=merge_size, return_tensors="pt")

    image_token_id = proc.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    total_vision_tokens = int((inputs["input_ids"] == image_token_id).sum().item())
    if total_vision_tokens % num_frames != 0:
        raise RuntimeError(
            f"{video_path}: {total_vision_tokens} vision tokens not divisible by "
            f"{num_frames} frames."
        )
    tokens_per_frame = total_vision_tokens // num_frames

    parts = select_full_compression_parts(
        total_frames=num_frames,
        total_vision_tokens=total_vision_tokens,
        window_size=window_size,
    )
    if not parts:
        raise RuntimeError(
            f"{video_path}: only {num_frames} frames (< window_size {window_size}); "
            f"no compression window."
        )
    ts_info = _build_ts_info(timestamps, parts, num_frames, total_vision_tokens,
                             proc.tokenizer)

    grid_hws = _grid_hw_for_compression_parts(parts, inputs["grid_sizes"],
                                              inputs["merge_sizes"])

    window_frames = [[s // tokens_per_frame, e // tokens_per_frame] for s, e in parts]
    window_out_hw = [[int(a), int(b)] for (a, b) in
                     (out_hw_fn(int(h), int(w)) for (h, w) in grid_hws)]
    compressed_tokens_total = int(sum(oh * ow for oh, ow in window_out_hw))
    out = {
        "input_ids": inputs["input_ids"].to(device),
        "pixel_values": inputs["pixel_values"].to(device=device, dtype=dtype),
        "grid_sizes": inputs["grid_sizes"].to(device),
        "merge_sizes": inputs["merge_sizes"].to(device),
        "modals": ["video"],
        "compression_parts": parts,
        "compression_ts_info": ts_info,
        "meta": {
            "video_path": video_path,
            "num_frames": num_frames,
            "fps": fps,
            "timestamps": timestamps,
            "tokens_per_frame": tokens_per_frame,
            "total_vision_tokens": total_vision_tokens,
            "n_windows": len(parts),
            "window_frame_spans": window_frames,
            "window_grid_hw": [list(map(int, hw)) for hw in grid_hws],
            "window_out_hw": window_out_hw,
            "compressed_tokens_total": compressed_tokens_total,
            "prompt": prompt_text,
        },
    }
    return out


def describe_grid(meta: Dict) -> str:
    spans = meta["window_frame_spans"]
    uniq_in = sorted({tuple(g) for g in meta["window_grid_hw"]})
    uniq_out = sorted({tuple(g) for g in meta["window_out_hw"]})
    return (f"{meta['num_frames']}f -> {meta['n_windows']} windows "
            f"(spans {spans[0]}..{spans[-1]}), input (h,w)={uniq_in} "
            f"-> output {uniq_out} = {meta['compressed_tokens_total']} tokens")


# --------------------------------------------------------------------------------------
# raw feature extraction (encoder -> compressor -> projector), no LLM
# --------------------------------------------------------------------------------------
@torch.no_grad()
def extract_features(model, sample: Dict) -> Dict[str, np.ndarray]:
    """Run the frozen encoder and the trained compressor on one packaged sample.

    Returns float32 numpy arrays:
        comp        (n_out, Cvis)   compressor output, pre-projector
        comp_proj   (n_out, Cllm)   what the LLM actually receives for those tokens
        raw         (n_raw, Cvis)   a random sample of the uncompressed encoder tokens
        raw_proj    (n_raw, Cllm)   their projection (reference manifold)
        n_per_window (n_windows,)   output tokens contributed by each window
    """
    m = model.get_model()
    pv = sample["pixel_values"]
    gs = sample["grid_sizes"]
    ms = sample["merge_sizes"]
    parts = sample["compression_parts"]

    mm = m.get_vision_encoder()(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)  # (N, Cvis)
    grid_hws = _grid_hw_for_compression_parts(parts, gs, ms)

    comp = model.compress_visual_tokens_with_compressor(mm.clone(), parts, grid_hws)  # (n_out, Cvis)
    comp_proj = m.mm_projector(comp)                                                  # (n_out, Cllm)

    n_raw = min(mm.shape[0], 2000)
    idx = torch.randperm(mm.shape[0], device=mm.device)[:n_raw]
    raw = mm.index_select(0, idx)
    raw_proj = m.mm_projector(raw)

    # tokens per window: compressor.output_hw_for(h, w) product, in part order
    compressor = model.get_token_compressor()
    n_per_window = np.array(
        [int(np.prod(compressor.output_hw_for(int(h), int(w)))) for (h, w) in grid_hws],
        dtype=np.int64,
    )

    f32 = lambda t: t.detach().float().cpu().numpy()
    return {
        "comp": f32(comp),
        "comp_proj": f32(comp_proj),
        "raw": f32(raw),
        "raw_proj": f32(raw_proj),
        "n_per_window": n_per_window,
    }
