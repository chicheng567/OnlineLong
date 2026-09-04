#!/usr/bin/env python3
"""Stage-2a: train the Mamba-2 segment fold on top of a frozen Stage-1 qbase.

Thin wrapper over ``compressor_pretrain_with_videollama3.py`` — same
frozen-everything-but-the-compressor CE recipe, single forward
(``use_dual_forward=False``), same collator/trainer — with three changes:

* the video is cut into ``U <= --stage2_max_units`` contiguous **readout units**
  (not one whole-video window); each unit is one ``compression_part`` and is
  subdivided *inside the compressor* into ``<= --frames_per_segment``-frame Stage-1
  segments;
* ``compressor_type`` is forced to ``"<base>+mamba"`` → ``TwoStageCompressor``
  (Stage-1 flat qbase + Stage-2 ``SegmentAggregator`` fold);
* Stage-1 is warm-started from ``--stage1_pretrained`` (a bare qbase ``.pt`` **or**
  an HF checkpoint dir) and **frozen**; only the Stage-2 fold trains.

Everything else (token add / embed resize / DeepSpeed / trainable-LR wiring / save)
is reused verbatim from the base script via monkeypatch, so this file stays small.

Run from repo root with ``PYTHONPATH=.`` (the shell wrapper exports it)::

    PYTHONPATH=. torchrun ... videollama3/train/stage2a_pretrain_compressor_fold.py \
        --compressor_type transformer_decoder_flat --num_queries 64 \
        --compressor_num_layers 8 --compressor_num_attention_heads 8 --match_encoder_scale True \
        --stage1_pretrained work_dirs/compressor_pretrain_video_norm \
        --stage2_n_summary_tokens 64 --frames_per_segment 4 --segs_per_unit 6 --stage2_max_units 5 \
        --multi_dataset True --data_path anno_data/stage_2_training.json ...
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.append("./")

import torch

import videollama3.train.compressor_pretrain_with_videollama3 as base
from videollama3.model.compressor import TwoStageCompressor

# ---------------------------------------------------------------------------
# Unit partition — U <= max_units contiguous frame chunks, one compression_part
# each. Fold depth (segments/unit) is frames_per_unit / frames_per_segment.
# ---------------------------------------------------------------------------

def select_stage2_units(
    total_frames: int,
    total_vision_tokens: int,
    frames_per_segment: int,
    segs_per_unit: int,
    max_units: int,
    min_part_tokens: int = 0,
) -> List[List[int]]:
    """Contiguous ``[start_tok, end_tok)`` image-token ranges, one per readout unit.

    ``frames_per_unit = frames_per_segment * segs_per_unit``; if that gives more than
    ``max_units`` units the unit is widened (rounded up to a whole number of
    segments) so it fits. Any trailing remainder is absorbed into the last unit.
    Indices count image tokens only, not sequence positions.

    ``min_part_tokens`` (pass ``M``): every unit must carry at least this many image
    tokens. The fold always emits ``M`` readout tokens per unit and
    ``compress_visual_tokens_with_compressor`` reserves exactly ``M`` placeholder
    slots per part, so a unit shorter than ``M`` makes the compressed output overrun
    its slot (RuntimeError at the scatter). Undersized units — in practice only the
    trailing one, whose start can land within ``<M`` tokens of the end when
    ``total_frames`` is just over a multiple of ``frames_per_unit`` — are merged into
    their predecessor.
    """
    if total_frames <= 0 or frames_per_segment <= 0 or segs_per_unit <= 0 or max_units <= 0:
        return []
    assert total_vision_tokens % total_frames == 0, (
        f"vision tokens {total_vision_tokens} not divisible by frames {total_frames}"
    )
    tpf = total_vision_tokens // total_frames
    frames_per_unit = frames_per_segment * segs_per_unit
    n_units = math.ceil(total_frames / frames_per_unit)
    if n_units > max_units:
        n_units = max_units
        frames_per_unit = math.ceil(total_frames / max_units)
        frames_per_unit = math.ceil(frames_per_unit / frames_per_segment) * frames_per_segment

    parts: List[List[int]] = []
    for u in range(n_units):
        fs = u * frames_per_unit
        if fs >= total_frames:
            break
        fe = min(total_frames, fs + frames_per_unit)
        parts.append([fs * tpf, fe * tpf])
    if parts:
        parts[-1][1] = total_vision_tokens  # absorb the remainder

    # Fold any unit with < min_part_tokens image tokens into its predecessor so the
    # fold's M-token output always has a placeholder slot to land in.
    if min_part_tokens > 0 and len(parts) > 1:
        merged: List[List[int]] = [parts[0]]
        for s, e in parts[1:]:
            if (e - s) < min_part_tokens or (merged[-1][1] - merged[-1][0]) < min_part_tokens:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        parts = merged
    return parts


def build_stage2_ts_info(
    parts: List[List[int]], content: Dict, tokenizer, total_frames: int, total_vision_tokens: int
) -> List[Tuple[int, List[int]]]:
    """Per-part ``(old "Time X.0s:" token length, new "Time:{a}s-{b}s:" token ids)``.

    Degrades to ``(0, [])`` per part when the sample carries no per-frame timestamps
    (then ``prepare_inputs_labels_for_multimodal`` leaves the timestamp text alone),
    exactly like ``build_range_ts_info``.
    """
    ts = content.get("timestamps", None)
    if not parts or ts is None or total_vision_tokens <= 0 or total_frames <= 0:
        return [(0, []) for _ in parts]
    tpf = total_vision_tokens // total_frames
    out: List[Tuple[int, List[int]]] = []
    for s, e in parts:
        fs = min(s // tpf, len(ts) - 1)
        fe = min(e // tpf, len(ts)) - 1
        a, b = float(ts[fs]), float(ts[max(fe, fs)])
        old_ids = tokenizer.encode(f"Time {round(a, 1)}s:", add_special_tokens=False)
        new_ids = tokenizer.encode(f"Time:{a:.1f}s-{b:.1f}s:", add_special_tokens=False)
        out.append((len(old_ids), new_ids))
    return out


# ---------------------------------------------------------------------------
# Dataset — same as GlobalCompressorLazySupervisedDataset, but the tail emits
# U contiguous units instead of one whole-video part.
# ---------------------------------------------------------------------------

class Stage2UnitDataset(base.GlobalCompressorLazySupervisedDataset):
    def _stage2_knobs(self):
        da = self.data_args
        return (
            int(getattr(da, "frames_per_segment", 4)),
            int(getattr(da, "segs_per_unit", 6)),
            int(getattr(da, "stage2_max_units", 5)),
            int(getattr(base, "_STAGE2_M", 64)),   # M readout tokens / unit (from model_args)
        )

    def _convert_normal(self, data_dict):
        """videoxl caption annotations tag the clip with ``<image>``; the base
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
            feat_path = self._feature_path(sample)
            if feat_path is not None:
                modal, feat, messages, merge_size = self._convert_feature(sample, feat_path)
                content = base.get_video_content(messages)
                data_dict = self._process_feature(feat, messages, merge_size)
            else:
                if self.online_mode:
                    modal, images, messages, merge_size = self._convert_online_video(sample)
                else:
                    modal, images, messages, merge_size = self._convert_normal(sample)
                is_still_image = modal == "image"
                if is_still_image:
                    base._rewrite_image_block_as_single_frame_video(messages)
                    modal, merge_size = "video", self.data_args.video_merge_size
                assert modal == "video", "Compressor training currently only supports video data."
                content = base.get_video_content(messages)
                if self.fixed_frames > 0 and not is_still_image:
                    images = base.resample_video_frames(images, content, self.fixed_frames)
                data_dict = self.vlprocessor(
                    images=images,
                    text=messages,
                    merge_size=merge_size,
                    return_labels=self.return_label,
                    return_tensors="pt",
                )
                data_dict["modals"] = [modal] * len(images)

            total_frames = int(content["num_frames"])
            assert total_frames > 0, f"Sample {i} has no frames."

            max_len = self.vlprocessor.tokenizer.model_max_length
            seq_len = int(data_dict["input_ids"].shape[-1])
            if seq_len > max_len:
                backup_idx = base.random.randint(0, len(self.list_data_dict) - 1)
                base.logger.warning(
                    "Sample %s: pre-compression length %d exceeds model_max_length %d (%d frames). "
                    "Lower --max_frames or --fixed_frames. Retrying with sample %s.",
                    i, seq_len, max_len, total_frames, backup_idx,
                )
                return self.__getitem__(backup_idx)

            image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(base.DEFAULT_IMAGE_TOKEN)
            total_vision_tokens = int((data_dict["input_ids"] == image_token_id).sum().item())
            assert total_vision_tokens % total_frames == 0, (
                f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
            )

            fps_seg, segs_per_unit, max_units, m_tok = self._stage2_knobs()
            parts = select_stage2_units(
                total_frames, total_vision_tokens, fps_seg, segs_per_unit, max_units,
                min_part_tokens=m_tok,
            )
            if not parts:  # too short to fold — fall back to one whole-video unit
                parts = [[0, total_vision_tokens]]
            if len(parts) == 1 and (parts[0][1] - parts[0][0]) < m_tok:
                # whole video carries fewer image tokens than one unit's readout
                # width; the fold still emits M and would overrun the slot. Rare
                # (tiny still image / a few low-res frames) — skip and resample.
                raise ValueError(
                    f"sample {i}: {parts[0][1] - parts[0][0]} image tokens (< M={m_tok}); too small to fold"
                )
            data_dict["compression_parts"] = parts
            data_dict["compression_ts_info"] = build_stage2_ts_info(
                parts, content, self.vlprocessor.tokenizer, total_frames, total_vision_tokens
            )

        except Exception:
            backup_idx = base.random.randint(0, len(self.list_data_dict) - 1)
            base.logger.exception("Failed to process sample %s. Fallback index: %s.", i, backup_idx)
            return self.__getitem__(backup_idx)
        return data_dict


# ---------------------------------------------------------------------------
# Argument dataclasses — extend the base ones with the Stage-2 knobs.
# ---------------------------------------------------------------------------

@dataclass
class Stage2ModelArguments(base.ModelArguments):
    compressor_type: str = field(default="transformer_decoder_flat")
    stage1_pretrained: str = field(
        default="",
        metadata={"help": "Warm-start for Stage-1: a bare qbase .pt/.bin OR an HF checkpoint dir "
                          "(model.token_compressor.* pulled from the shards). Stage-1 is then frozen."},
    )
    stage2_n_summary_tokens: int = field(default=64, metadata={"help": "M readout tokens per unit."})
    stage2_d_model: int = field(default=1024)
    stage2_n_layers: int = field(default=4)
    stage2_d_state: int = field(default=128)
    stage2_headdim: int = field(default=64)
    stage2_time_embed: str = field(default="index_sincos")

    def __post_init__(self):
        # stash for the monkeypatched _set_module_trainable (fires mid-train())
        base._STAGE1_PRETRAINED = self.stage1_pretrained or None
        # stash M for Stage2UnitDataset._stage2_knobs (min unit size = M tokens)
        base._STAGE2_M = int(self.stage2_n_summary_tokens)


@dataclass
class Stage2DataArguments(base.DataArguments):
    frames_per_segment: int = field(default=4, metadata={"help": "Frames per Stage-1 segment, clamp [1, 8]."})
    segs_per_unit: int = field(default=6, metadata={"help": "Target segments per readout unit (fold depth)."})
    stage2_max_units: int = field(default=5, metadata={"help": "Hard cap on U (so U*M <= budget)."})
    # Dynamic-HW knobs. Leave --force_image_size UNSET so the image processor keeps
    # each video's native aspect ratio / resolution (smart_resize), scaled to a
    # per-VIDEO token budget shared across its frames. The two-stage compressor
    # still emits a fixed U*M <= 320 tokens to the LLM, so a large budget only
    # costs the frozen encoder forward + Stage-1 cross-attention KV.
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

    def __post_init__(self):
        # stash for the monkeypatched Videollama3Processor factory (fires mid-train())
        base._VISION_MAX_TOKENS = self.vision_max_tokens
        base._VISION_MIN_TOKENS = self.vision_min_tokens


# ---------------------------------------------------------------------------
# Config builder + trainable hook (monkeypatched into the base module).
# ---------------------------------------------------------------------------

_orig_build_cfg = base._build_token_compressor_config
_orig_set_trainable = base._set_module_trainable
_orig_processor_cls = base.Videollama3Processor
base._STAGE1_PRETRAINED = None
base._VISION_MAX_TOKENS = None
base._VISION_MIN_TOKENS = None
base._STAGE2_M = 64


def _stage2_processor(image_processor=None, tokenizer=None, *args, **kwargs):
    """Push the image processor into DYNAMIC-HW mode before it is wrapped.

    ``force_size`` is left as-is (so passing --force_image_size still forces a
    square); we only widen the token budget so native-resolution frames are not
    shrunk. ``max_tokens`` is a per-video budget shared across frames.
    """
    if image_processor is not None:
        mt = getattr(base, "_VISION_MAX_TOKENS", None)
        mn = getattr(base, "_VISION_MIN_TOKENS", None)
        if mt:
            image_processor.max_tokens = int(mt)
        if mn:
            image_processor.min_tokens = int(mn)
        base.rank0_print(
            f"[stage2a] image processor: force_size={image_processor.force_size}, "
            f"min_tokens={image_processor.min_tokens}, max_tokens={image_processor.max_tokens} "
            f"(dynamic HW unless force_size is set; max_tokens is a per-video budget)"
        )
    return _orig_processor_cls(image_processor, tokenizer, *args, **kwargs)


def _build_stage2_token_compressor_config(model_config, model_args, data_args) -> Dict:
    d = _orig_build_cfg(model_config, model_args, data_args)
    if not str(d["compressor_type"]).endswith("+mamba"):
        d["compressor_type"] = f"{d['compressor_type']}+mamba"
    d.update(
        stage2_n_summary_tokens=model_args.stage2_n_summary_tokens,
        stage2_frames_per_segment=data_args.frames_per_segment,
        stage2_d_model=model_args.stage2_d_model,
        stage2_n_layers=model_args.stage2_n_layers,
        stage2_d_state=model_args.stage2_d_state,
        stage2_headdim=model_args.stage2_headdim,
        stage2_time_embed=model_args.stage2_time_embed,
    )
    return d


def _set_module_trainable_with_stage2(module, trainable):
    _orig_set_trainable(module, trainable)
    if isinstance(module, TwoStageCompressor):
        if getattr(base, "_STAGE1_PRETRAINED", None):
            module.load_stage1_pretrained(base._STAGE1_PRETRAINED)
        module.freeze_stage1()
        n2 = sum(p.numel() for p in module.stage2.parameters() if p.requires_grad)
        base.rank0_print(
            f"[stage2a] TwoStageCompressor: stage-1 frozen, stage-2 trainable "
            f"({n2/1e6:.2f}M params); frames_per_segment={module.frames_per_segment}, "
            f"K={module.tokens_per_segment}, M={module.n_summary_tokens}"
        )


base.ModelArguments = Stage2ModelArguments
base.DataArguments = Stage2DataArguments
base.GlobalCompressorLazySupervisedDataset = Stage2UnitDataset
base._build_token_compressor_config = _build_stage2_token_compressor_config
base._set_module_trainable = _set_module_trainable_with_stage2
base.Videollama3Processor = _stage2_processor


if __name__ == "__main__":
    # base.train()'s own __main__ passes this; base.train() defaults it to None,
    # which trips `assert model.config._attn_implementation == "flash_attention_2"`.
    base.train(attn_implementation="flash_attention_2")
