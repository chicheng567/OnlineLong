# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import contextlib
import copy
import os
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import warnings
import einops
import torch
import torch.distributed as dist
import torch.nn as nn

from ..constants import IGNORE_INDEX, MODAL_INDEX_MAP, NUM_FRAMES
from .encoder import build_vision_encoder
from .projector import build_vision_projector, load_mm_projector
from .compressor import build_token_compressor


def _vl3_memlog(tag, **tensors):
    """VL3_LOG_MEM=1 -> one CUDA-memory line per instrumented point, rank 0 only.
    ``live`` is still-referenced tensors, so a jump between two tags is memory that
    stage RETAINED (for backward), not a transient spike."""
    import os as _os
    if _os.environ.get("VL3_LOG_MEM") != "1" or not torch.cuda.is_available():
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    from tqdm import tqdm as _tqdm
    g = 2 ** 30
    extra = " ".join(
        f"{k}={tuple(v.shape)}/{str(v.dtype).replace('torch.', '')}"
        f"={v.numel() * v.element_size() / g:.2f}G"
        for k, v in tensors.items() if torch.is_tensor(v)
    )
    _tqdm.write(f"[VL3_MEM] {tag}: live={torch.cuda.memory_allocated() / g:.2f}G "
                f"peak={torch.cuda.max_memory_allocated() / g:.2f}G {extra}")
    # VL3_MEM_DUMP=1 -> once, name every live CUDA tensor >0.5G by shape/dtype.
    # memory_allocated() only gives a total; this says WHAT is holding it.
    if _os.environ.get("VL3_MEM_DUMP") == "1" and not getattr(_vl3_memlog, "_dumped", False):
        _vl3_memlog._dumped = True
        import gc as _gc
        seen, rows = set(), []
        for o in _gc.get_objects():
            try:
                t = o if torch.is_tensor(o) else getattr(o, "data", None)
                if not torch.is_tensor(t) or not t.is_cuda:
                    continue
                key = (t.data_ptr(), t.numel())
                if key in seen or t.data_ptr() == 0:
                    continue
                seen.add(key)
                nb = t.numel() * t.element_size()
                if nb > 0.5 * g:
                    rows.append((nb / g, tuple(t.shape), str(t.dtype).replace("torch.", ""),
                                 type(o).__name__, bool(t.requires_grad)))
            except Exception:
                continue
        rows.sort(reverse=True)
        _tqdm.write(f"[VL3_MEM] --- live CUDA tensors >0.5G (total tracked "
                    f"{sum(r[0] for r in rows):.2f}G over {len(rows)}) ---")
        for nb, shp, dt, cls, rg in rows[:25]:
            _tqdm.write(f"[VL3_MEM]   {nb:6.2f}G {dt:9s} requires_grad={rg!s:5s} {cls:12s} {shp}")


def _compressed_len(compressor, n_frames, h, w):
    """Number of compressed tokens a window of ``n_frames`` frames (post-merge grid
    ``h x w``) produces. Prefers the compressor's ``output_len_for`` (frame-count
    aware — the fixed-count adaptive segmenter's ``N * num_queries``); falls back to
    ``prod(output_hw_for(h, w))`` for the fixed-length compressors."""
    if hasattr(compressor, "output_len_for"):
        return int(compressor.output_len_for(int(n_frames), int(h), int(w)))
    oh, ow = compressor.output_hw_for(int(h), int(w))
    return int(oh) * int(ow)


def _grid_hw_for_compression_parts(compression_parts, grid_sizes, merge_sizes):
    """
    Map each compression part's [start, end) vision-token range to the post-merge
    (h, w) patch grid of the single grid_sizes entry it falls inside.

    grid_sizes: (num_grids, 3) rows of (t, h, w) in pre-merge patch units, one entry
    per video/image in the sample. merge_sizes: (num_grids,) merge_size per entry.
    A compression part is expected to come from exactly one video/image (compression
    windows are built per-video upstream), so it must fall entirely within one grid
    entry's token range.

    Returns: List[Tuple[int, int]], one (h, w) per compression part, aligned to
    compression_parts' input order (NOT sorted).
    """
    tokens_per_grid = []
    grid_hw = []
    for (t, h, w), m in zip(grid_sizes.tolist(), merge_sizes.tolist()):
        oh, ow = h // m, w // m
        tokens_per_grid.append(t * oh * ow)
        grid_hw.append((oh, ow))
    offsets = [0]
    for n in tokens_per_grid:
        offsets.append(offsets[-1] + n)

    result = []
    for start, end in compression_parts:
        owner = None
        for i in range(len(tokens_per_grid)):
            if start >= offsets[i] and end <= offsets[i + 1]:
                owner = i
                break
        assert owner is not None, (
            f"Compression part [{start}, {end}) does not fall within a single "
            f"grid_sizes entry (offsets={offsets}); every compression window must "
            f"come from one video/image so the compressor can assign it one (h, w)."
        )
        result.append(grid_hw[owner])
    return result


def spatial_downsampling(features, grid_thws, stride=2):
    n, c = features.shape

    flatten_grid_thws = torch.cat([grid_thw for batch_grid_thws in grid_thws for grid_thw in batch_grid_thws])
    split_sizes = [grid_thw.prod() for grid_thw in flatten_grid_thws]
    features = torch.split(features, split_sizes)

    new_features = []
    for feature, grid_thw in zip(features, flatten_grid_thws):
        # NOTE: adapted for reshape in image processor 
        feature = feature.view(grid_thw[0], grid_thw[1] // stride, grid_thw[2] // stride, stride, stride,  c).permute(0, 1, 3, 2, 4, 5)
        feature = feature.reshape(grid_thw[0], grid_thw[1], grid_thw[2], c).permute(0, 3, 1, 2)
        # NOTE: previous version model is align_corners=True
        new_feature = torch.nn.functional.interpolate(feature, (math.ceil(grid_thw[1] / stride), math.ceil(grid_thw[2] / stride)), mode='bilinear')
        # new_feature = nn.functional.avg_pool2d(feature, stride)
        # new_feature = nn.functional.max_pool2d(feature, stride)
        new_features.append(new_feature.permute(0, 2, 3, 1).view(-1, c))
    new_features = torch.cat(new_features)

    return new_features


class Videollama3MetaModel:

    def __init__(self, config):
        super(Videollama3MetaModel, self).__init__(config)

        if hasattr(config, "vision_encoder") or hasattr(config, "mm_vision_encoder"):
            self.vision_encoder = build_vision_encoder(config, delay_load=False)
            self.mm_projector = build_vision_projector(config, self.vision_encoder.hidden_size)
        if hasattr(config, "trainable_mm_compressor") and config.trainable_mm_compressor:
            self.token_compressor = build_token_compressor(config)
            self._maybe_build_split_projectors()

    def _maybe_build_split_projectors(self):
        """TwoStageCompressor (`+mamba`) output is two distributions -- the
        qbase-only replay stream and the fold readout -- that used to share one
        `mm_projector` (it had to "track two moving compressor manifolds",
        CLAUDE.md). Give each its own copy instead. Only declared for two-stage
        checkpoints; single-stage (Phase 1) / no-compressor models are untouched
        and `get_mm_projector_{qbase,fold}()` fall back to the shared projector.
        Random init here (a fresh deepcopy of whatever `mm_projector` currently
        is, itself not yet loaded at __init__ time) only needs to have the right
        shape for `from_pretrained`'s state-dict load to fill in on a genuine
        split-projector checkpoint; a fresh training run overwrites both with a
        real copy of the (by-then loaded) `mm_projector` explicitly -- see
        compressor_pretrain_with_videollama3.py.
        """
        compressor = getattr(self, "token_compressor", None)
        if compressor is not None and hasattr(compressor, "compress_windows"):
            self.mm_projector_qbase = copy.deepcopy(self.mm_projector)
            self.mm_projector_fold = copy.deepcopy(self.mm_projector)

    def get_vision_encoder(self):
        vision_encoder = getattr(self, 'vision_encoder', None)
        if type(vision_encoder) is list:
            vision_encoder = vision_encoder[0]
        return vision_encoder

    def get_mm_projector(self):
        return self.mm_projector

    def get_mm_projector_qbase(self):
        return getattr(self, "mm_projector_qbase", None) or self.mm_projector

    def get_mm_projector_fold(self):
        return getattr(self, "mm_projector_fold", None) or self.mm_projector

    def get_token_compressor(self):
        compressor = getattr(self, 'token_compressor', None)
        return compressor
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_encoder = model_args.vision_encoder
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature

        self.config.mm_vision_encoder = vision_encoder

        if self.get_vision_encoder() is None:
            vision_encoder = build_vision_encoder(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_encoder = [vision_encoder]
            else:
                self.vision_encoder = vision_encoder
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_encoder = self.vision_encoder[0]
            else:
                vision_encoder = self.vision_encoder
            # NOTE: only compatible with delay_load encoder
            # vision_encoder.load_model(vision_encoder.cfg_only)

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_encoder.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

class Videollama3MetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_encoder(self):
        return self.get_model().get_vision_encoder()

    def get_mm_projector(self):
        return self.get_model().get_mm_projector()

    def get_mm_projector_qbase(self):
        return self.get_model().get_mm_projector_qbase()

    def get_mm_projector_fold(self):
        return self.get_model().get_mm_projector_fold()

    def get_token_compressor(self):
        return self.get_model().get_token_compressor()

    def compress_visual_tokens_with_compressor(
        self,
        vision_tokens: torch.FloatTensor,
        compression_parts: List[List[int]],
        grid_hws: List[Tuple[int, int]],
        unit_counts: Optional[List[Optional[int]]] = None,
        seed: Optional[int] = None,
        qbase_only: Optional[List[bool]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[List[dict]]]:
        # compression_parts: [[start, end], [start, end], ...] — index into the
        # vision (image) tokens only.  Single-stage compressors: one part == one
        # compressed block of _compressed_len tokens.  TwoStageCompressor: one part
        # == one WHOLE-VIDEO window that is split model-side into U content-adaptive
        # units (§ docs/two_stage_compression_design.md §4 Phase 2); the compressor
        # returns unit_meta and this method scatters sum_u M rows per part.
        # unit_counts / seed: forwarded to compress_windows (per-window U; Gumbel
        # seed).  grid_hws: one (h, w) per part.
        device = vision_tokens.device
        vision_tokens = vision_tokens.squeeze(0)  # [num_tokens, dim]
        compressor = self.get_token_compressor()
        two_stage = hasattr(compressor, "compress_windows")

        compression_cu_seqlens = [0]
        need_compress_parts = torch.zeros(vision_tokens.shape[0], device=device, dtype=torch.bool)
        replace_mask = torch.zeros(vision_tokens.shape[0], device=device, dtype=torch.bool)
        part_starts: List[int] = []
        for part, (h, w) in zip(compression_parts, grid_hws):
            part_len = part[1] - part[0]
            need_compress_parts[part[0]: part[1]] = True
            part_starts.append(part[0])
            compression_cu_seqlens.append(compression_cu_seqlens[-1] + part_len)
            if not two_stage:
                n_frames = part_len // (h * w)
                n_out = _compressed_len(compressor, n_frames, h, w)
                replace_mask[part[0]: part[0] + n_out] = True
        compression_cu_seqlens = torch.tensor(compression_cu_seqlens, device=device, dtype=torch.long)

        original_tokens_to_reconstruct = vision_tokens[need_compress_parts]
        unit_meta = None
        self._last_compression_kind = None
        if two_stage:
            compressed, unit_meta = compressor.compress_windows(
                original_tokens_to_reconstruct,
                compression_cu_seqlens,
                grid_hws,
                unit_counts=unit_counts,
                seed=seed,
                qbase_only=qbase_only,
            )
            # Per part (== window), reserve sum of its units' n_out rows, and mark
            # which projector each of those rows wants (design doc; qbase-only vs
            # fold-readout are two distributions -- see get_mm_projector_{qbase,fold}).
            # 0 = raw/uncompressed (never touched below -- the shared mm_projector,
            # same as pre-split), 1 = qbase-only replay, 2 = fold readout.
            n_out_per_part = [0] * len(compression_parts)
            kind = torch.zeros(vision_tokens.shape[0], dtype=torch.uint8, device=device)
            for m in unit_meta:
                pi = m["window"]
                start = part_starts[pi] + n_out_per_part[pi]
                n_out = int(m["n_out"])
                kind[start: start + n_out] = 2 if m.get("kind") == "fold" else 1
                n_out_per_part[pi] += n_out
            for pi, ps in enumerate(part_starts):
                # The rows are reserved INSIDE the part they replace, so a window
                # that emits more rows than its part holds silently spills into the
                # next part's region -- the union under-counts and the scatter below
                # dies on an opaque broadcast error. Name the offending window here.
                part_len = compression_parts[pi][1] - compression_parts[pi][0]
                assert n_out_per_part[pi] <= part_len, (
                    f"compression window {pi} emits {n_out_per_part[pi]} rows but its part holds "
                    f"only {part_len} vision tokens (grid_hw={grid_hws[pi] if grid_hws else None}). "
                    f"A window's output must fit in the part it replaces: qbase-only emits N*K, a "
                    f"fold U*M, so the per-frame grid needs >= K tokens. Raise --vision_min_tokens."
                )
                replace_mask[ps: ps + n_out_per_part[pi]] = True
        else:
            compressed = compressor(
                original_tokens_to_reconstruct,
                compression_cu_seqlens,
                grid_hws,
            )
        keeping_masks = ~need_compress_parts | replace_mask
        vision_tokens[replace_mask] = compressed
        vision_tokens = vision_tokens[keeping_masks]
        if two_stage:
            self._last_compression_kind = kind[keeping_masks]
        return vision_tokens, unit_meta

    def _run_vision_encoder(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Frozen-encoder forward, optionally in ``vision_encoder_chunk_frames``-frame
        groups (design doc §5 item 4).

        The encoder has **no cross-frame attention** — its ``cu_seqlens`` is built
        per frame (``repeat_interleave(h*w, t)``) and the RoPE table is 2-D spatial,
        repeated identically for every frame of a row — so splitting a ``(t, h, w)``
        row into ``(t_chunk, h, w)`` rows is mathematically a no-op. It only bounds
        the peak activation, which at Phase-3 lengths (1200 frames x 1024 patches)
        is otherwise the single largest allocation in the step. ``0`` (default)
        keeps the single-shot forward Phase 1/2 ran.
        """
        encoder = self.get_model().get_vision_encoder()
        chunk = int(getattr(self.config, "vision_encoder_chunk_frames", 0) or 0)
        if chunk <= 0:
            return encoder(pixel_values=pixel_values, grid_sizes=grid_sizes, merge_sizes=merge_sizes)
        # Frozen encoder -> no graph to keep. Only skip it if something in there is
        # actually being trained (an unfrozen vision tower), where the chunked
        # forward must stay differentiable.
        trainable = any(p.requires_grad for p in encoder.parameters())
        ctx = contextlib.nullcontext() if trainable else torch.no_grad()
        outs = []
        tok_off = 0
        with ctx:
            for row, ms in zip(grid_sizes, merge_sizes):
                t, h, w = int(row[0]), int(row[1]), int(row[2])
                per_frame = h * w
                for s in range(0, t, chunk):
                    n = min(chunk, t - s)
                    pv = pixel_values[tok_off + s * per_frame: tok_off + (s + n) * per_frame]
                    gs = torch.tensor([[n, h, w]], device=row.device, dtype=grid_sizes.dtype)
                    outs.append(encoder(pixel_values=pv, grid_sizes=gs, merge_sizes=ms.reshape(1)))
                tok_off += t * per_frame
        assert tok_off == pixel_values.shape[0], (
            f"chunked vision forward consumed {tok_off} of {pixel_values.shape[0]} patch rows"
        )
        return torch.cat(outs, dim=0)

    def encode_images(
        self,
        pixel_values: torch.FloatTensor,
        grid_sizes: torch.LongTensor,
        merge_sizes: torch.LongTensor,
        compression_parts: Optional[List[List[int]]] = None,
        grid_hws: Optional[List[Tuple[int, int]]] = None,
        unit_counts: Optional[List[Optional[int]]] = None,
        seed: Optional[int] = None,
        qbase_only: Optional[List[bool]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[List[dict]]]:
        unit_meta = None
        mm_features = self._run_vision_encoder(pixel_values, grid_sizes, merge_sizes)
        _vl3_memlog("  after vision_encoder", enc_out=mm_features)
        kind = None
        if getattr(self.config, "trainable_mm_compressor", False) and compression_parts is not None and len(compression_parts) > 0:
            mm_features, unit_meta = self.compress_visual_tokens_with_compressor(
                mm_features,
                compression_parts,
                grid_hws,
                unit_counts=unit_counts,
                seed=seed,
                qbase_only=qbase_only,
            )
            kind = self._last_compression_kind
            _vl3_memlog("  after compressor", comp_out=mm_features)
        mm_features = self._apply_mm_projector(mm_features, kind)
        _vl3_memlog("  after mm_projector", proj_out=mm_features)
        return mm_features, unit_meta

    def _apply_mm_projector(self, mm_features: torch.FloatTensor, kind: Optional[torch.Tensor]):
        """Route each row to the projector for its compressor stream. `kind` (from
        `compress_visual_tokens_with_compressor`) is None for single-stage
        compressors / uncompressed input -- the plain shared `mm_projector`, same
        as before this split existed. For a TwoStageCompressor: 0 = raw/uncompressed
        rows (e.g. a partial-window split) -> the shared `mm_projector`, same as
        always; 1 = qbase-only replay rows; 2 = fold-readout rows -- each of the
        latter two gets its own projector (both initialized as a copy of
        `mm_projector`, see _maybe_build_split_projectors).

        Runs ALL THREE projectors on the FULL `mm_features` unconditionally and
        selects per-row with `torch.where`, rather than boolean-indexing only the
        rows each kind actually has. This is single-video-per-forward (batch size
        1), so which kinds are even present is decided entirely by that one
        video's data -- with a skip-if-absent (`if qbase_rows.any(): ...`) some
        ranks would call e.g. get_mm_projector_qbase() this step and others
        wouldn't (qbase-only replay is ~6.6% of Phase-2 data, and
        group_by_compression_depth deliberately makes each grad-accum window
        depth-homogeneous, so an all-fold or all-qbase-only window per rank is
        common, not an edge case). Under ZeRO stage 1 (overlap_comm=false, one
        flat-buffer allreduce per accumulation window) that makes the flattened
        gradient buffer a DIFFERENT SIZE on ranks that touched the param that step
        vs ranks that didn't -- the same numbered collective call then disagrees
        on element count across ranks, which is a NCCL hang / "invalid peer GPU
        memory access" waiting to happen (reproduced deterministically at a fixed
        step). mm_projector is a small linear/MLP, so always running all three is
        cheap next to the LLM forward, and keeps every rank's graph identical
        regardless of data composition.
        """
        if kind is None:
            return self.get_model().mm_projector(mm_features)
        raw_out = self.get_model().mm_projector(mm_features)
        qbase_out = self.get_mm_projector_qbase()(mm_features)
        fold_out = self.get_mm_projector_fold()(mm_features)
        kind = kind.unsqueeze(-1)
        out = torch.where(kind == 1, qbase_out, raw_out)
        out = torch.where(kind == 2, fold_out, out)
        return out
    
    def _time_range_token_ids(self, a_sec: int, b_sec: int) -> List[int]:
        """`Time:{a}s-{b}s:` as token ids, assembled from the digit / fragment
        tables baked into config at setup (bake_time_tokens). Returns [] when the
        tables are absent (e.g. an old checkpoint) so the range string is simply
        omitted, matching the single-stage empty-`new_ts_ids` path."""
        cfg = self.config
        digits = getattr(cfg, "time_tok_digits", None)
        if not digits:
            return []
        hi = len(digits) - 1
        a = max(0, min(int(a_sec), hi))
        b = max(0, min(int(b_sec), hi))
        return (list(cfg.time_tok_open) + list(digits[a]) + list(cfg.time_tok_mid)
                + list(digits[b]) + list(cfg.time_tok_close))

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        grid_sizes: Optional[torch.LongTensor] = None,
        merge_sizes: Optional[torch.LongTensor] = None,
        modals: Optional[torch.LongTensor] = None, # This parameter is currently not used in the model, but can be used to indicate the modality of each token for more flexible multimodal modeling.
        compression_parts: Optional[List[List[int]]] = None,
        compression_ts_info: Optional[List[Tuple[int, List[int]]]] = None,
        compression_units: Optional[List[Optional[int]]] = None,
        compression_seed: Optional[List[int]] = None,
        compression_frame_sec: Optional[List[List[int]]] = None,
        compression_qbase_only: Optional[List[bool]] = None,
    ):
        B, N = input_ids.shape
        device = input_ids.device
        if getattr(self.config, "trainable_mm_compressor", False) and pixel_values is not None:
            assert position_ids is not None, "Currently model only supports position_ids and flatten input."
            # Compression parts should like: [[1, 3], [4, 10], [16, 20]],  where each part indicates the start and end position of vision tokens to be compressed.
            assert B == 1, "Currently model only supports batch size 1 for trainable token compression."
        vision_encoder = self.get_vision_encoder()
        # NOTE: text-only situation
        if vision_encoder is None or pixel_values is None or input_ids.shape[1] == 1:
            return input_ids, attention_mask, position_ids, past_key_values, None, labels
        # 1. flatten text inputs
        input_ids = input_ids.view(B * N)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B * N)
        if position_ids is not None:
            position_ids = position_ids.view(B * N)
        if labels is not None:
            labels = labels.view(B * N)

        # 2. embed visual tokens and compress if needed
        image_selected = (input_ids == self.config.image_token_index)
        image_positions = torch.nonzero(image_selected, as_tuple=False).squeeze(-1) # vision token's positions among all tokens
        grid_hws = None
        if compression_parts is not None and len(compression_parts) > 0:
            grid_hws = _grid_hw_for_compression_parts(compression_parts, grid_sizes, merge_sizes)
        _seed = None
        if compression_seed is not None and len(compression_seed) > 0:
            _seed = list(compression_seed)
        _qbase_only = None
        if compression_qbase_only is not None and len(compression_qbase_only) > 0:
            _qbase_only = list(compression_qbase_only)
        _vl3_memlog("before encode_images", pixel_values=pixel_values)
        mm_features, unit_meta = self.encode_images(
            pixel_values, grid_sizes, merge_sizes, compression_parts, grid_hws,
            unit_counts=compression_units, seed=_seed, qbase_only=_qbase_only,
        )
        _vl3_memlog("after encode_images (encoder+compressor+projector)", mm_features=mm_features)

        if compression_parts is not None and len(compression_parts) > 0:
            compressor = self.get_token_compressor()
            two_stage = unit_meta is not None
            # unit_meta grouped by window index (== part index; parts are collator-sorted).
            units_by_win: dict = {}
            if two_stage:
                for m in unit_meta:
                    units_by_win.setdefault(m["window"], []).append(m)

            # List-based construction: build the new token sequence piece-by-piece.
            # Single-stage: replace the per-frame "Time X.0s:" text with one range
            # "Time:{a}s-{b}s:" before the compressed block. Two-stage: the model
            # split each whole-video part into U units, so emit U {range-ts, <cs>,
            # M placeholders, <ce>} blocks; placeholder blocks carry a
            # `pos_block` = (slot offsets, unit_span) for the strided position_ids.
            ids_segs, lbl_segs, attn_segs, is_start_segs, pos_plan = [], [], [], [], []

            def _append_seg(tok_ids, lbl_fill, attn_fill, is_sample_start_mask=None, pos_block=None):
                if tok_ids is None or len(tok_ids) == 0:
                    return
                ids_segs.append(tok_ids)
                if labels is not None:
                    lbl_segs.append(lbl_fill)
                if attention_mask is not None:
                    attn_segs.append(attn_fill)
                if position_ids is not None:
                    if is_sample_start_mask is not None:
                        is_start_segs.append(is_sample_start_mask)
                    else:
                        is_start_segs.append(torch.zeros(len(tok_ids), device=device, dtype=torch.bool))
                    pos_plan.append(pos_block)

            def _append_ids(ids_list):
                if not ids_list:
                    return
                t = torch.tensor(ids_list, device=device, dtype=input_ids.dtype)
                _append_seg(
                    t,
                    torch.full([len(ids_list)], IGNORE_INDEX, device=device, dtype=labels.dtype) if labels is not None else None,
                    torch.ones(len(ids_list), device=device, dtype=attention_mask.dtype) if attention_mask is not None else None,
                )

            def _append_placeholders(n_out, pos_block):
                img_toks = torch.full([n_out], self.config.image_token_index, device=device, dtype=input_ids.dtype)
                _append_seg(
                    img_toks,
                    torch.full([n_out], IGNORE_INDEX, device=device, dtype=labels.dtype) if labels is not None else None,
                    torch.ones(n_out, device=device, dtype=attention_mask.dtype) if attention_mask is not None else None,
                    pos_block=pos_block,
                )

            cs_id, ce_id = self.config.compression_start_token_id, self.config.compression_end_token_id
            prev = 0
            parts_with_hw = sorted(zip(compression_parts, grid_hws), key=lambda pair: pair[0][0])
            for part_idx, (part, (part_h, part_w)) in enumerate(parts_with_hw):
                part_start = image_positions[part[0]].item()
                part_end = image_positions[part[1] - 1].item()

                old_ts_len = 0
                ts_extra = None
                if compression_ts_info is not None and part_idx < len(compression_ts_info):
                    old_ts_len, ts_extra = compression_ts_info[part_idx]

                # 1. Keep everything up to (but not including) the old "Time X.0s:" text.
                keep_end = max(prev, part_start - old_ts_len)
                if keep_end > prev:
                    _append_seg(
                        input_ids[prev:keep_end],
                        labels[prev:keep_end] if labels is not None else None,
                        attention_mask[prev:keep_end] if attention_mask is not None else None,
                        (position_ids[prev:keep_end] == 0) if position_ids is not None else None,
                    )

                if two_stage:
                    frame_sec = None
                    if compression_frame_sec is not None and part_idx < len(compression_frame_sec):
                        frame_sec = compression_frame_sec[part_idx]
                    for m in units_by_win.get(part_idx, []):
                        if frame_sec:
                            nfs = len(frame_sec)
                            a_sec = frame_sec[min(m["a_frame"], nfs - 1)]
                            b_sec = frame_sec[min(max(m["b_frame"] - 1, 0), nfs - 1)]
                        else:  # fps = 1 -> frame index is seconds
                            a_sec, b_sec = m["a_frame"], max(m["b_frame"] - 1, m["a_frame"])
                        _append_ids(self._time_range_token_ids(a_sec, b_sec))
                        _append_ids([cs_id])
                        _append_placeholders(int(m["n_out"]), (m["pos_offsets"], int(m["unit_span"])))
                        _append_ids([ce_id])
                else:
                    n_frames = (part[1] - part[0]) // (part_h * part_w)
                    compact = _compressed_len(compressor, n_frames, part_h, part_w)
                    _append_ids(list(ts_extra) if ts_extra else [])
                    _append_ids([cs_id])
                    _append_placeholders(compact, None)
                    _append_ids([ce_id])

                prev = part_end + 1

            # last text run
            if prev < input_ids.shape[0]:
                _append_seg(
                    input_ids[prev:],
                    labels[prev:] if labels is not None else None,
                    attention_mask[prev:] if attention_mask is not None else None,
                    (position_ids[prev:] == 0) if position_ids is not None else None,
                )

            input_ids = torch.cat(ids_segs)
            if labels is not None:
                labels = torch.cat(lbl_segs)
            if attention_mask is not None:
                attention_mask = torch.cat(attn_segs)
            if position_ids is not None:
                # Text runs: stride 1, resetting to 0 at each sample start. Placeholder
                # blocks: `base + slot offsets`, then advance the counter by the unit's
                # full slot span (N_u*K) so the compressed region keeps its Phase-1
                # RoPE footprint (design doc §1 / §5 item 11).
                pieces = []
                cur = 0
                for seg_ids, is_start, pblock in zip(ids_segs, is_start_segs, pos_plan):
                    n = len(seg_ids)
                    if pblock is None:
                        starts = torch.nonzero(is_start, as_tuple=False).squeeze(-1).tolist()
                        if not starts:
                            pieces.append(torch.arange(cur, cur + n, device=device, dtype=torch.long))
                            cur += n
                        else:
                            p = torch.arange(n, device=device, dtype=torch.long)
                            off = torch.full((n,), cur, device=device, dtype=torch.long)
                            for s in starts:
                                off[s:] = -s
                            pieces.append(p + off)
                            cur = n - starts[-1]
                    else:
                        offsets, span = pblock
                        pieces.append(cur + offsets.to(device=device, dtype=torch.long))
                        cur = cur + int(span)
                position_ids = torch.cat(pieces).to(torch.long)

        # 3. embed text tokens
        inputs_embeds = self.get_model().embed_tokens(input_ids).clone()

        # 4. replace multimodal tokens with features
        image_selected = (input_ids == self.config.image_token_index)
        inputs_embeds[image_selected] = inputs_embeds[image_selected] * 0.0 + mm_features   

        # 5. reshape back to batched format
        C = inputs_embeds.shape[-1]
        inputs_embeds = inputs_embeds.reshape(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.view(B, -1)
        if labels is not None:
            labels = labels.view(B, -1)
        if position_ids is not None:
            position_ids = position_ids.view(B, -1)

        return None, attention_mask, position_ids, past_key_values, inputs_embeds, labels
