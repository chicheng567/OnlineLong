"""Raw segments of raw/qbase hybrid windows (``qbase_llm_full_ft.py``, training only).

The flat qbase (``compressor.py``) cuts each window into segments, marks some of them
raw (``_last_seg_lens`` / ``_last_raw_segments``), compresses the rest and returns
their K rows only. This module turns that into the window's LLM rows, in time order:
K qbase rows per qbase segment, every frame's raw encoder rows per raw segment. The
arch only wires it in (``compress_visual_tokens_with_compressor``) and lays the rows
out as text (``prepare_inputs_labels_for_multimodal``).

A raw frame holds at most ``hybrid_raw_max_tokens_per_frame`` rows. A larger one is
re-encoded, not pooled: its pixels are resized the way the image processor resizes
and go through the frozen encoder again -- how VideoLLaMA3 itself gives a frame fewer
tokens (pixel resize, then its fixed merge 2). Pooling the full-resolution features
instead (the encoder's bilinear merge at a larger factor) lands far from the base
model's raw tokens: per-token cosine to the processor's own encoding at the same grid
0.75 (0.81 antialiased) vs 0.98 re-encoded, 8 training videos (2026-10-21). The qbase
segments always read the full-resolution features; only a copy of the raw frames'
pixels is resized.
"""
import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .videollama3_encoder.image_processing_videollama3 import patchify, unpatchify

# One window's LLM rows, in time order: (first frame, end frame, raw grid). A qbase
# segment has grid None (K rows); a raw one the merged grid (h', w') of each frame.
Segment = Tuple[int, int, Optional[Tuple[int, int]]]


def raw_frame_grid(h: int, w: int, max_tokens: int) -> Tuple[int, int]:
    """Merged grid ``(h', w')`` of a raw frame capped at ``max_tokens`` rows: the image
    processor's own downscale rule (``simple_batched_resize``: keep the aspect ratio,
    floor to whole merged tokens) on the frame's ``h x w`` grid. Unchanged when it
    already fits or ``max_tokens <= 0``."""
    h, w, max_tokens = int(h), int(w), int(max_tokens)
    if max_tokens <= 0 or h * w <= max_tokens:
        return h, w
    beta = math.sqrt(h * w / max_tokens)
    return max(1, math.floor(h / beta)), max(1, math.floor(w / beta))


def window_layout(seg_lens: Sequence[int], raw_segments: Sequence[int],
                  grid_hw: Tuple[int, int], max_tokens: int) -> List[Segment]:
    """The window's segments in time order, raw ones with their capped frame grid."""
    raw, grid = set(raw_segments), raw_frame_grid(grid_hw[0], grid_hw[1], max_tokens)
    layout, s = [], 0
    for j, n in enumerate(seg_lens):
        layout.append((s, s + int(n), grid if j in raw else None))
        s += int(n)
    return layout


def window_rows(layout: Sequence[Segment], k: int) -> int:
    """LLM rows of a window: K per qbase segment, h' * w' per raw frame."""
    return sum(k if g is None else (b - a) * g[0] * g[1] for a, b, g in layout)


def assemble(qbase_rows: torch.Tensor, layouts: Sequence[Sequence[Segment]], k: int,
             raw_rows: Callable[[int, int, int, Tuple[int, int]], torch.Tensor]) -> torch.Tensor:
    """Every window's rows in time order: the next K of ``qbase_rows`` (the
    compressor's output, qbase segments in order) per qbase segment,
    ``raw_rows(window, a, b, grid)`` per raw segment ``[a, b)``."""
    out, q = [], 0
    for wi, layout in enumerate(layouts):
        for a, b, grid in layout:
            if grid is None:
                out.append(qbase_rows[q: q + k])
                q += k
            else:
                rows = raw_rows(wi, a, b, grid)
                assert rows.shape[0] == (b - a) * grid[0] * grid[1], (
                    f"window {wi} raw frames [{a}, {b}): {rows.shape[0]} rows for grid {grid}")
                out.append(rows.to(qbase_rows.dtype))
    assert q == qbase_rows.shape[0], f"used {q} of {qbase_rows.shape[0]} qbase rows"
    return torch.cat(out, dim=0)


def reencode_frames(encode: Callable, pixel_values: torch.Tensor, grid_sizes: torch.Tensor,
                    merge_sizes: torch.Tensor, tok_start: int, frames: range,
                    new_hw: Tuple[int, int]) -> torch.Tensor:
    """Encoder rows of ``frames`` of the video whose vision tokens start at
    ``tok_start``, re-encoded at the merged grid ``new_hw``: the frames' pixel rows are
    put back into images, resized like the image processor (bicubic, antialiased --
    resizing is linear, so the normalised values resize like the uint8 frame up to
    rounding), re-cut into patches and passed to ``encode(pixel_values, grid_sizes,
    merge_sizes)`` (the arch's frozen-encoder forward)."""
    tok_off = patch_off = 0
    for (t, gh, gw), m in zip(grid_sizes.tolist(), merge_sizes.tolist()):
        n_tok = t * gh * gw // (m * m)
        if tok_start < tok_off + n_tok:
            break
        tok_off += n_tok
        patch_off += t * gh * gw
    else:
        raise ValueError(f"vision token {tok_start} lies past the last grid row")
    hw = (gh // m) * (gw // m)
    assert (tok_start - tok_off) % hw == 0, "a compression part must start on a frame"
    f0 = (tok_start - tok_off) // hw
    assert f0 + frames.stop <= t, f"frames up to {f0 + frames.stop} of a {t}-frame row"
    p = int(round(math.sqrt(pixel_values.shape[-1] / 3)))
    assert 3 * p * p == pixel_values.shape[-1], f"{pixel_values.shape[-1]} values per row != 3 * p * p"
    per_frame = gh * gw
    rows = pixel_values[patch_off + (f0 + frames.start) * per_frame: patch_off + (f0 + frames.stop) * per_frame]
    img = unpatchify(rows, len(frames), gh, gw, m, p)
    rh, rw = int(new_hw[0]) * m, int(new_hw[1]) * m             # patch grid
    img = F.interpolate(img.float(), size=(rh * p, rw * p), mode="bicubic", antialias=True, align_corners=False)
    grid = torch.tensor([[len(frames), rh, rw]], dtype=grid_sizes.dtype, device=grid_sizes.device)
    merge = torch.tensor([m], dtype=merge_sizes.dtype, device=merge_sizes.device)
    return encode(patchify(img, m, p).to(pixel_values.dtype), grid, merge)
