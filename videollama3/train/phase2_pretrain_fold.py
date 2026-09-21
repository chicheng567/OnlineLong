#!/usr/bin/env python3
"""Plan X Phase 2 — train the Mamba-2 segment fold on top of the Phase-1 qbase.

Thin wrapper over ``compressor_pretrain_with_videollama3.py`` — same
frozen-LLM CE recipe, single forward (``use_dual_forward=False``), same
collator/trainer — with these changes:

* ``compressor_type`` is forced to ``"<base>+mamba"`` → ``TwoStageCompressor``
  (the flat qbase + a ``SegmentAggregator`` fold);
* each video is **one whole-video ``compression_part``**; the compressor does the
  content-adaptive segment cut *and* the ``U``-unit split model-side (seeded
  per-``(epoch, index)`` Gumbel). The dataset only draws ``U`` and the seed;
* the qbase is warm-started from ``--stage1_pretrained`` (a bare ``.pt`` **or** an
  HF checkpoint dir). A single knob controls whether it trains: ``--qbase_lr > 0``
  trains it, in its own optimizer group at that LR, alongside the fold;
  ``--qbase_lr 0`` (the dataclass default -- the shell wrapper defaults it to
  ``$MAMBA_LR`` for joint training instead) freezes it (``requires_grad=False``)
  for a staggered cold-fold start, later resumed with ``--qbase_lr <small>``
  (~10x below ``--mamba_lr``) so it moves slower than the fold. ``stage1`` is
  the load-bearing name of the qbase module;
* a meta-JSON entry with ``"qbase_only": true`` becomes the **pure-qbase replay**
  stream (no unit split, no fold — ``N*K`` qbase tokens straight to the projector),
  so the unfrozen projector / qbase stay anchored to the raw-qbase manifold.

There is no retained-K mechanism: a unit's fold readout (``M`` tokens) is the
*only* representation of its segments the LLM sees. (An earlier design reserved a
per-unit raw-qbase escape hatch for Phase 3; dropped -- with ``M == K`` it gave the
LLM a cheap incentive to read the verbatim tokens instead of the fold summary,
undermining the reason the fold exists. See docs/two_stage_compression_design.md.)

Everything else (token add / embed resize / DeepSpeed / trainable-LR wiring / save)
is reused from the base script via ``base.train()``'s keyword-only injection hooks
(``model_args_cls`` / ``data_args_cls`` / ``dataset_cls`` /
``build_token_compressor_config`` / ``configure_image_processor`` /
``on_compressor_built``). See ``docs/two_stage_compression_design.md`` §4 Phase 2.

Run from repo root with ``PYTHONPATH=.`` (the shell wrapper exports it)::

    PYTHONPATH=. torchrun ... videollama3/train/phase2_pretrain_fold.py \
        --compressor_type transformer_decoder_flat --num_queries 64 \
        --compressor_num_layers 8 --compressor_num_attention_heads 8 --match_encoder_scale True \
        --stage1_pretrained work_dirs/phase1_qbase_internvid \
        --stage2_n_summary_tokens 64 --stage2_max_units 5 --mm_projector_lr 2e-5 \
        --group_by_compression_depth True --durations_json anno_data/internVid_durations.json \
        --multi_dataset True --data_path anno_data/phase2_training.json ...
"""
from __future__ import annotations

import json
import logging
import multiprocessing as _mp
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

sys.path.append("./")

import torch

import videollama3.train.compressor_pretrain_with_videollama3 as base
from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.model.compressor import TwoStageCompressor, adaptive_segment_count
from videollama3.train.compressor_pretrain_with_videollama3 import (
    _build_token_compressor_config as _orig_build_cfg,
)
from videollama3.train.data.common import cast_pixel_values_, rank0_print
from videollama3.train.data.global_compressor import (
    GlobalCompressorLazySupervisedDataset,
    _rewrite_image_block_as_single_frame_video,
    get_video_content,
    resample_video_frames,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset — same as GlobalCompressorLazySupervisedDataset, but the tail emits
# U contiguous units instead of one whole-video part.
# ---------------------------------------------------------------------------

class Phase2FoldDataset(GlobalCompressorLazySupervisedDataset):
    """Plan-X Phase 2 dataset. Emits ONE whole-video ``compression_part`` per video
    plus the seeded per-``(epoch, index)`` draw the model consumes: ``U`` (via a
    depth-class ``N̄_u``) and a reproducible ``compression_seed`` for the model-side
    Gumbel (segment + unit boundaries). ``N = ⌊T/segment_target_frames⌋+1`` is a
    pure function of the frame count, so no feature decode is needed here.
    Content-adaptive segmentation and unit-boundary placement happen model-side in
    ``TwoStageCompressor`` (unit grouping on the qbase segment tokens, not the raw
    encoder feature).

    A meta-JSON entry with ``"qbase_only": true`` turns every sample of that
    sub-dataset into the pure-qbase replay: one whole-video part, no unit split, no
    fold -- the model emits ``N*K`` qbase tokens straight to the projector so the
    unfrozen ``mm_projector`` / qbase stay anchored to the raw-qbase manifold.
    """

    def __init__(self, *args, **kwargs):
        # ``_epoch`` / ``_progress`` are written by ``_DatasetProgressCallback`` in
        # the MAIN process but read inside dataloader workers. A plain attribute does
        # not survive ``dataloader_persistent_workers`` and does not propagate with
        # ``num_workers > 0`` -- which silently freezes the per-``(epoch, index)``
        # re-partition seed and the variance curriculum
        # (docs/two_stage_compression_design.md §4 Phase 2). Back them with
        # fork-shared memory so every worker sees each later write. (Assumes the
        # ``fork`` start method -- the Linux / torch default.)
        self._epoch_val = _mp.Value("l", 0, lock=False)
        self._progress_val = _mp.Value("d", 0.0, lock=False)
        super().__init__(*args, **kwargs)
        self._durations = None
        dj = getattr(self.data_args, "durations_json", None)
        if dj and os.path.exists(dj):
            try:
                raw = json.load(open(dj))
                # Accepts two shapes: a ``{clip: est_frames_1fps | {est_frames_1fps: ...}}``
                # map, OR the list-of-records the ffprobe scan emits
                # (``anno_data/internVid_durations.json``:
                #  ``[{"video"/"video_id", "est_frames_1fps"|"duration_sec", "ok"}, ...]``).
                items = raw.items() if isinstance(raw, dict) else (
                    (r.get("video_id") or r.get("video"), r) for r in raw
                )
                d = {}
                for k, v in items:
                    if k is None:
                        continue
                    if isinstance(v, dict):
                        if v.get("ok") is False:
                            continue
                        ef = v.get("est_frames_1fps") or v.get("duration_sec")
                    else:
                        ef = v
                    if ef:
                        stem = os.path.splitext(os.path.basename(str(k)))[0]
                        d[stem] = int(round(float(ef)))
                self._durations = d
                rank0_print(f"[phase2] durations_json: {len(d)} clips for depth-class grouping")
            except Exception as e:  # pragma: no cover
                rank0_print(f"[phase2] failed to load durations_json {dj}: {e}")

    # -- epoch / progress: fork-shared so dataloader workers see main-process writes
    @property
    def _epoch(self) -> int:
        return int(self._epoch_val.value)

    @_epoch.setter
    def _epoch(self, value) -> None:
        self._epoch_val.value = int(value)

    @property
    def _progress(self) -> float:
        return float(self._progress_val.value)

    @_progress.setter
    def _progress(self, value) -> None:
        self._progress_val.value = float(value)

    def _fold_knobs(self):
        ma, da = self.model_args, self.data_args
        m = int(getattr(ma, "stage2_n_summary_tokens", 64)) if ma is not None else 64
        k = int(getattr(ma, "num_queries", 64)) if ma is not None else 64
        tf = int(getattr(ma, "segment_target_frames", 4)) if ma is not None else 4
        return (
            tf,                                              # segment_target_frames
            int(getattr(da, "stage2_max_units", 5)),         # U cap
            m,                                               # M readout tokens / unit
            k,                                               # K qbase tokens / segment
        )

    def _segment_count(self, n_frames: int) -> int:
        """``N`` for a clip — Phase 2's fixed-frames rule. A pure function of the
        frame count, identical to the model's (``compressor.segment_count_for``), so
        no decode is needed here. Phase 3 overrides it with the target-``N`` rule."""
        target_f, _, _, _ = self._fold_knobs()
        return adaptive_segment_count(int(n_frames), target_f)

    @staticmethod
    def _depth_class_range(n: int) -> "Tuple[int, int]":
        """Narrow N̄_u draw range by total segment count (design doc §4 Phase 2)."""
        if n < 23:
            return 4, 8          # shallow
        if n < 46:
            return 8, 12         # mid
        return 12, 16            # deep

    def _draw_units(self, i: int, N: int, tpf: int, total_vision_tokens: int):
        """Return (U, seed). ``U`` seeded by (epoch, index): a depth-class ``N̄_u``
        draw -> ``round(N / N̄_u)``; the cold-fold window narrows it to
        ``min(3, ⌊N/4⌋)``."""
        _, max_units, M, _K = self._fold_knobs()
        da = self.data_args
        cold_frac = float(getattr(da, "variance_cold_frac", 0.15))
        seed = (int(self._epoch) * 1_000_003 + int(i)) & 0x7FFFFFFF
        rng = random.Random(seed)

        lo, hi = self._depth_class_range(N)
        u_cap = max(1, min(max_units, N // 4 if N >= 4 else 1))
        prog = float(self._progress)
        if prog < cold_frac:                       # cold fold: fewer, simpler units
            U = min(3, u_cap)
        else:
            n_bar = max(1, min(rng.randint(lo, hi), N))
            U = max(1, min(round(N / n_bar), u_cap))

        # The compressed region must fit inside the video's vision-token slots
        # (arch reserves replace_mask[ps : ps + sum n_out]).
        while U > 1 and U * M > total_vision_tokens:
            U -= 1
        return U, seed

    @property
    def compression_depths(self):
        if not self._durations:
            return None
        # Grouping N must match the N the model actually cuts: --max_frames caps the
        # decoded frame count, so a 400 s clip run at --max_frames 320 has N from 320,
        # not 400. Clamp here so the depth-class megabatch stays homogeneous.
        cap = int(getattr(self.data_args, "max_frames", 0) or 0)
        out = []
        for s in self.list_data_dict:
            vid = s.get("video")
            vid = vid[0] if isinstance(vid, list) and vid else vid
            stem = os.path.splitext(os.path.basename(str(vid)))[0] if vid else None
            T = self._durations.get(stem)
            if not T:
                out.append(32)
                continue
            T = min(int(T), cap) if cap > 0 else int(T)
            out.append(self._segment_count(T))
        return out

    def _convert_normal(self, data_dict):
        """Some caption annotations tag the clip with ``<image>``; the base
        ``_convert_normal`` splits *video* turns on ``<video>``, so the ``<image>``
        tag is not consumed -- it leaks into the text as a spurious image
        placeholder (``grid_sizes`` IndexError) and a ``<video>`` is auto-prepended.
        Normalise the human turns to ``<video>`` first."""
        if data_dict.get("video") is not None and data_dict.get("image") is None:
            convs = data_dict.get("conversations") or []
            if any(
                c.get("from") in ("human", "system")
                and isinstance(c.get("value"), str)
                and "<image>" in c["value"]
                and "<video>" not in c["value"]
                for c in convs
            ):
                data_dict = dict(data_dict)
                data_dict["conversations"] = [
                    {**c, "value": c["value"].replace("<image>", "<video>")}
                    if c.get("from") in ("human", "system") and isinstance(c.get("value"), str)
                    else c
                    for c in convs
                ]
        return super()._convert_normal(data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.list_data_dict[i]
            if self.online_mode:
                modal, images, messages, merge_size = self._convert_online_video(sample)
            else:
                modal, images, messages, merge_size = self._convert_normal(sample)
            is_still_image = modal == "image"
            if is_still_image:
                _rewrite_image_block_as_single_frame_video(messages)
                modal, merge_size = "video", self.data_args.video_merge_size
            assert modal == "video", "Compressor training currently only supports video data."
            content = get_video_content(messages)
            if self.fixed_frames > 0 and not is_still_image:
                images = resample_video_frames(images, content, self.fixed_frames)
            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=self.return_label,
                return_tensors="pt",
            )
            # This class overrides GlobalCompressorLazySupervisedDataset.__getitem__
            # wholesale, so the cast has to be repeated here -- fp32 patches are what
            # crosses worker -> main through /dev/shm. See cast_pixel_values_.
            cast_pixel_values_(data_dict, getattr(self.data_args, "pixel_values_dtype", None))
            data_dict["modals"] = [modal] * len(images)

            total_frames = int(content["num_frames"])
            assert total_frames > 0, f"Sample {i} has no frames."

            max_len = self.vlprocessor.tokenizer.model_max_length
            seq_len = int(data_dict["input_ids"].shape[-1])
            if seq_len > max_len:
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                logger.warning(
                    "Sample %s: pre-compression length %d exceeds model_max_length %d (%d frames). "
                    "Lower --max_frames or --fixed_frames. Retrying with sample %s.",
                    i, seq_len, max_len, total_frames, backup_idx,
                )
                return self.__getitem__(backup_idx)

            image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            total_vision_tokens = int((data_dict["input_ids"] == image_token_id).sum().item())
            assert total_vision_tokens % total_frames == 0, (
                f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
            )

            _target_f, _max_units, m_tok, _k = self._fold_knobs()
            if total_vision_tokens < m_tok:
                raise ValueError(
                    f"sample {i}: {total_vision_tokens} image tokens (< M={m_tok}); too small to fold"
                )
            tpf = total_vision_tokens // total_frames
            N = self._segment_count(total_frames)
            seed = (int(self._epoch) * 1_000_003 + int(i)) & 0x7FFFFFFF

            # One whole-video part.
            data_dict["compression_parts"] = [[0, total_vision_tokens]]
            data_dict["compression_seed"] = [seed]
            qbase_only_sample = bool(getattr(self, "qbase_only", False))
            if qbase_only_sample and N * _k > total_vision_tokens:
                # The arch reserves replace_mask[ps : ps + n_out] INSIDE the part it
                # replaces, so a window can never emit more rows than it has vision
                # tokens. The fold path is clamped in _draw_units (U*M); the
                # qbase-only path emits N*K rows, which fits only while the per-frame
                # grid holds at least K tokens (N <= T => N*K <= T*K <= T*hw). Phase
                # 2's N ~ T/4 made that automatic against the min_tokens=16 floor;
                # Phase 3's target-N rule pushes N to ~min(target_n, T), so a
                # low-resolution clip (hw < K) now overflows into the next part and
                # the scatter in compress_visual_tokens_with_compressor dies on a
                # shape mismatch. Demote the sample to the (clamped) fold path rather
                # than pay another decode to resample it.
                if not getattr(type(self), "_qbase_fit_warned", False):
                    type(self)._qbase_fit_warned = True
                    logger.warning(
                        "Sample %s: qbase-only replay wants N*K = %d*%d = %d rows but the clip "
                        "holds only %d vision tokens (%d frames x %d tok/frame). Demoting clips "
                        "this low-resolution to the fold path; pass --vision_min_tokens >= K "
                        "(%d) to keep them on the replay stream.",
                        i, N, _k, N * _k, total_vision_tokens, total_frames, tpf, _k,
                    )
                qbase_only_sample = False
            if qbase_only_sample:
                # Pure-qbase replay: no unit split, no fold -- the model routes this
                # window straight through stage-1 (N*K qbase tokens, Phase-1 layout).
                # unit_counts[wi] is never read on this path (compress_windows
                # returns early for qbase_only windows) -- None is a placeholder.
                data_dict["compression_units"] = [None]
                data_dict["compression_qbase_only"] = [True]
            else:
                U, _seed = self._draw_units(i, N, tpf, total_vision_tokens)
                data_dict["compression_units"] = [U]
                data_dict["compression_qbase_only"] = [False]

            # timestamps may be a list OR a numpy array (the decoder synthesises
            # per-frame seconds when the annotation carries none); ``x or y`` on a
            # multi-element ndarray raises, so test explicitly (cf. build_range_ts_info).
            ts = content.get("timestamps")
            if ts is None or len(ts) == 0:
                ts = list(range(total_frames))
            frame_sec = [int(round(float(t))) for t in ts][:total_frames]
            if len(frame_sec) < total_frames:
                frame_sec += [frame_sec[-1] if frame_sec else 0] * (total_frames - len(frame_sec))
            data_dict["compression_frame_sec"] = [frame_sec]
            tok = self.vlprocessor.tokenizer
            first_ts = float(ts[0]) if len(ts) else 0.0
            old_ts_len = len(tok.encode(f"Time {round(first_ts, 1)}s:", add_special_tokens=False))
            data_dict["compression_ts_info"] = [(old_ts_len, [])]

        except Exception:
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            logger.exception("Failed to process sample %s. Fallback index: %s.", i, backup_idx)
            return self.__getitem__(backup_idx)
        return data_dict


# ---------------------------------------------------------------------------
# Argument dataclasses — extend the base ones with the fold knobs.
# ---------------------------------------------------------------------------

@dataclass
class Phase2ModelArguments(base.ModelArguments):
    compressor_type: str = field(default="transformer_decoder_flat")
    # Phase 2 needs a NEW segment partition each epoch (fold must be robust to any
    # partition, not one) -> default the Gumbel-top-k segment jitter ON. The base
    # class defaults this to 0.0 for the Phase-1 qbase pretrain.
    segment_sample_tau: float = field(
        default=0.5,
        metadata={"help": "Gumbel-top-k temperature for the per-epoch segment-boundary jitter "
                          "(seeded by compression_seed). 0 = deterministic top-diff (ablation only)."},
    )
    stage1_pretrained: str = field(
        default="",
        metadata={"help": "Warm-start for the qbase (token_compressor.stage1.*): a bare .pt/.bin OR an "
                          "HF checkpoint dir. Leave empty when the whole compressor is warm-started via "
                          "--pretrained_compressor_path."},
    )
    stage2_n_summary_tokens: int = field(default=64, metadata={"help": "M readout tokens per unit."})
    stage2_d_model: int = field(
        default=1024,
        metadata={"help": "Fold working width (bottleneck below the compressor hidden 1152). "
                          "SegmentAggregator.output_proj decodes the M readout tokens from this width "
                          "back to 1152 -> the SHARED frozen mm_projector (Option A/B align them). "
                          "The SSM state stays its own (nheads, headdim, d_state) object."},
    )
    stage2_n_layers: int = field(default=4)
    stage2_d_state: int = field(default=128)
    stage2_headdim: int = field(default=64)
    stage2_dropout: float = field(
        default=0.1,
        metadata={"help": "Fold dropout (Mamba2Block residual branches + MLP). docs §4 Phase 2 "
                          "readout-regularization recommends 0.1; set 0.0 to disable."},
    )
    stage2_time_embed: str = field(default="rel_gap_mlp", metadata={"help": "rel_gap_mlp (per-segment (gap,duration) seconds, "
                                                                     "built model-side from the adaptive cut) | seconds_mlp | "
                                                                     "index_sincos | none. First runs used none; temporal_blindness "
                                                                     "0.955->0.998 (qbase->fold) on eval_ablation/grounding_probe.py "
                                                                     "motivated switching the default to rel_gap_mlp."})
    stage2_final_norm: str = field(default="rmsnorm", metadata={"help": "readout norm: rmsnorm|layernorm|scale|none (ablatable)."})
    stage2_rope_slot_scale: str = field(default="ratio", metadata={"help": "'ratio' (readout stride N_u*K/M) or float S slots/sec."})


@dataclass
class Phase2DataArguments(base.DataArguments):
    stage2_max_units: int = field(default=5, metadata={"help": "Hard cap on U (so U*M <= budget)."})
    durations_json: Optional[str] = field(
        default=None,
        metadata={"help": "Optional {clip: {est_frames_1fps}} map; enables the depth-class grouped "
                          "sampler (--group_by_compression_depth). Missing -> sampler falls back."},
    )
    variance_cold_frac: float = field(
        default=0.15, metadata={"help": "Cold-fold window: fraction of training with U narrowed to min(3, N//4)."},
    )
    # Dynamic-HW knobs. Leave --force_image_size UNSET so the image processor keeps
    # each video's native aspect ratio / resolution (smart_resize), scaled to a
    # per-VIDEO token budget shared across its frames. The two-stage compressor
    # still emits a fixed U*M <= 320 tokens to the LLM, so a large budget only
    # costs the frozen encoder forward + the qbase cross-attention KV.
    vision_max_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Override image_processor.max_tokens: total vision-token budget PER VIDEO "
                          "(shared across its frames). ~= tokens/frame * n_frames. None keeps the "
                          "checkpoint value (16384). Ignored when --force_image_size is set."},
    )
    vision_min_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Override image_processor.min_tokens (per-frame floor). None keeps the "
                          "checkpoint value (16 -> 4x4 grid)."},
    )


# ---------------------------------------------------------------------------
# Injection hooks passed to base.train() (no monkeypatching).
# ---------------------------------------------------------------------------

def _build_phase2_token_compressor_config(model_config, model_args, data_args) -> Dict:
    d = _orig_build_cfg(model_config, model_args, data_args)
    if not str(d["compressor_type"]).endswith("+mamba"):
        d["compressor_type"] = f"{d['compressor_type']}+mamba"
    d.update(
        # The fold consumes per-segment qbase tokens -> the qbase must self-segment.
        adaptive_segmentation=True,
        stage2_n_summary_tokens=model_args.stage2_n_summary_tokens,
        stage2_d_model=model_args.stage2_d_model,
        stage2_n_layers=model_args.stage2_n_layers,
        stage2_d_state=model_args.stage2_d_state,
        stage2_headdim=model_args.stage2_headdim,
        stage2_dropout=model_args.stage2_dropout,
        stage2_time_embed=model_args.stage2_time_embed,
        stage2_final_norm=model_args.stage2_final_norm,
        stage2_rope_slot_scale=model_args.stage2_rope_slot_scale,
        compressor_gradient_checkpointing=bool(
            getattr(model_args, "compressor_gradient_checkpointing", False)
        ),
    )
    return d


def _configure_phase2_image_processor(image_processor, model_args, data_args) -> None:
    """Push the image processor into DYNAMIC-HW mode before it is wrapped.

    ``force_size`` is left as-is (so passing --force_image_size still forces a
    square); we only widen the token budget so native-resolution frames are not
    shrunk. ``max_tokens`` is a per-video budget shared across frames.
    """
    mt = data_args.vision_max_tokens
    mn = data_args.vision_min_tokens
    if mt:
        image_processor.max_tokens = int(mt)
    if mn:
        image_processor.min_tokens = int(mn)
    rank0_print(
        f"[phase2] image processor: force_size={image_processor.force_size}, "
        f"min_tokens={image_processor.min_tokens}, max_tokens={image_processor.max_tokens} "
        f"(dynamic HW unless force_size is set; max_tokens is a per-video budget)"
    )


def _warmstart_and_freeze_stage1(compressor, model_args, data_args, training_args) -> None:
    if not isinstance(compressor, TwoStageCompressor):
        return
    if model_args.stage1_pretrained:
        compressor.load_stage1_pretrained(model_args.stage1_pretrained)
    # Single knob: qbase_lr > 0 trains the qbase (its own optimizer group at that
    # LR, set up in create_optimizer); qbase_lr <= 0 freezes it. No separate
    # freeze/unfreeze flag -- see docs/two_stage_compression_design.md §4 Phase 2.
    qbase_lr = getattr(training_args, "qbase_lr", 0.0) or 0.0
    if qbase_lr > 0:
        compressor.stage1_frozen = False
        for p in compressor.stage1.parameters():
            p.requires_grad_(True)
        compressor.stage1.train()
        s1_state = f"TRAINABLE @ qbase_lr={qbase_lr}"
    else:
        compressor.freeze_stage1()
        s1_state = "frozen (qbase_lr<=0)"
    n1 = sum(p.numel() for p in compressor.stage1.parameters() if p.requires_grad)
    n2 = sum(p.numel() for p in compressor.stage2.parameters() if p.requires_grad)
    rank0_print(
        f"[phase2] TwoStageCompressor: qbase {s1_state} ({n1 / 1e6:.2f}M trainable), "
        f"fold trainable ({n2 / 1e6:.2f}M params); "
        f"segment_target_frames={compressor.segment_target_frames}, "
        f"K={compressor.tokens_per_segment}, M={compressor.n_summary_tokens}, "
        f"final_norm={compressor.stage2.cfg.final_norm}, dropout={compressor.stage2.cfg.dropout}, "
        f"rope_slot_scale={compressor.rope_slot_scale}"
    )


if __name__ == "__main__":
    # base.train() defaults attn_implementation to None, which trips
    # `assert model.config._attn_implementation == "flash_attention_2"`.
    base.train(
        attn_implementation="flash_attention_2",
        model_args_cls=Phase2ModelArguments,
        data_args_cls=Phase2DataArguments,
        dataset_cls=Phase2FoldDataset,
        build_token_compressor_config=_build_phase2_token_compressor_config,
        configure_image_processor=_configure_phase2_image_processor,
        on_compressor_built=_warmstart_and_freeze_stage1,
    )
