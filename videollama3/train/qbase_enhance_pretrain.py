#!/usr/bin/env python3
"""qbase enhance — single-stage qbase CE pretrain on the Phase-3 blend, mixed 1–4 frames/segment.

Why: at an ample budget (one frame per segment, every unit a raw bypass — segmentation
and the fold drop out) the qbase loses to plain 4x4 feature pooling and to resizing by
4.5–6.8 pp on Video-MME at equal tokens (``eval_ablation/qbase_vs_pool.py``,
2026-09-29). The fold is parked until the qbase alone clears that bar. The Phase-3
blend (``/root/datasets`` dense captions + QA/MCQ + 20 % still images) replaces
Phase 1's InternVid-caption stream.

Thin wrapper over ``compressor_pretrain_with_videollama3.py`` (same hooks as Phase 2/3,
no monkeypatching). Three changes:

* **Mixed granularity.** Each sample draws its frames/segment from
  ``--segment_target_frames_choices`` (default ``1,2,3,4``) via a per-sample
  ``compression_seed``; ``N = T // tf + 1``. The model (``pick_target_frames`` in
  ``compressor.py``) derives the same ``tf`` from the same seed, so the collator's
  placeholder count and the model's cut agree. One frame/segment is the eval
  operating point of ``qbase_vs_pool.py``; 2–4 keep the multi-frame segments the
  fold will need later. Eval / inference (no seed) uses ``--segment_target_frames``.
* **Phase-3 annotations.** Some caption rows tag a clip with ``<image>``; they are
  normalised to ``<video>`` exactly as ``Phase2FoldDataset`` does.
* **Rank load balance** (``--group_by_encode_cost`` + ``--media_geometry_json``). The
  dataset exposes ``encode_costs`` -- each sample's estimated frozen-encoder + LLM
  time from its native geometry (``dataset_util/probe_media_geometry.py``) -- and the
  trainer deals every optimizer step's samples into micro-batches of equal cost, so
  no rank idles at the step-boundary allreduce. The step's sample set is still a
  uniform random draw.

Warm start: ``--pretrained_compressor_path`` (qbase + its projector + the two
compression-token rows; a two-stage source contributes ``stage1.*`` and
``mm_projector_qbase``).

    PYTHONPATH=. torchrun --nproc_per_node 8 videollama3/train/qbase_enhance_pretrain.py \\
        --compressor_type transformer_decoder_flat --adaptive_segmentation True \\
        --segment_target_frames_choices 1,2,3,4 --max_frames 64 \\
        --pretrained_compressor_path work_dirs/phase1_qbase_internvid \\
        --multi_dataset True --data_path anno_data/phase3_blend_img.json ...
(canonical invocation: ``shell/pretrain_qbase_enhance.sh``)
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional

sys.path.append("./")

import videollama3.train.compressor_pretrain_with_videollama3 as base
from videollama3.model.compressor import adaptive_segment_count, pick_target_frames
from videollama3.train.data.common import rank0_print
from videollama3.train.data.global_compressor import GlobalCompressorLazySupervisedDataset


def _parse_choices(s) -> list:
    return [int(x) for x in str(s or "").split(",") if x.strip()]


# Per-sample step cost, in seconds of one H100, for --group_by_encode_cost. Fit on
# profiled 8-GPU runs of shell/pretrain_qbase_enhance.sh (2026-09-30, fused-rotary
# encoder): frozen-encoder time = a * T*P + b * T*P^2 (P = pre-merge patches per
# frame; the second term is the per-frame attention; R^2 0.99 on unbalanced
# micro-batches); LLM forward + checkpointed backward (+ the qbase) is linear in the
# rewritten sequence length. Only the RATIOS matter to the sampler.
_ENC_S_PER_PATCH = 2.3e-6
_ENC_S_PER_ATTN = 5.0e-10
_LLM_S_PER_TOKEN = 1.1e-4
_TOKENS_PER_WORD = 1.3


@lru_cache(maxsize=2)
def _load_geometry(path: str, mtime: float) -> Dict[str, list]:
    """``{media path: [w, h, duration]}`` from dataset_util/probe_media_geometry.py,
    parsed once per process (every sub-dataset of the blend shares it)."""
    geo = json.load(open(path))
    rank0_print(f"[qbase-enhance] media geometry: {len(geo)} media ({os.path.basename(path)})")
    return geo


def _patches_per_frame(height: int, width: int, n_frames: int, proc, merge_size: int) -> int:
    """Pre-merge patches of one frame after the image processor's resize -- the
    ``simple_batched_resize`` arithmetic (image_processing_videollama3.py) with its
    per-video budget ``max_tokens`` shared over ``n_frames`` frames and the optional
    ``max_tokens_per_frame`` cap."""
    factor = proc.patch_size * merge_size
    if proc.force_size is not None:
        h_bar, w_bar = proc.force_size
    else:
        max_pixels = proc.max_tokens * factor * factor
        min_pixels = proc.min_tokens * factor * factor
        frame_pixels = max_pixels // n_frames
        if getattr(proc, "max_tokens_per_frame", None):
            frame_pixels = min(frame_pixels, proc.max_tokens_per_frame * factor * factor)
        h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
        if h_bar * w_bar > frame_pixels:
            beta = math.sqrt((height * width) / frame_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        if h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
    return (h_bar // proc.patch_size) * (w_bar // proc.patch_size)


class QbaseEnhanceDataset(GlobalCompressorLazySupervisedDataset):
    """Phase-1 dataset + a per-sample ``compression_seed`` that picks the sample's
    frames/segment. The seed is fresh every call, so a clip's granularity changes
    across epochs."""

    _cur_seed = None

    def _convert_normal(self, data_dict):
        if data_dict.get("video") is not None and data_dict.get("image") is None:
            convs = data_dict.get("conversations") or []
            if any(c.get("from") in ("human", "system") and isinstance(c.get("value"), str)
                   and "<image>" in c["value"] and "<video>" not in c["value"] for c in convs):
                data_dict = dict(data_dict)
                data_dict["conversations"] = [
                    {**c, "value": c["value"].replace("<image>", "<video>")}
                    if c.get("from") in ("human", "system") and isinstance(c.get("value"), str)
                    else c
                    for c in convs
                ]
        return super()._convert_normal(data_dict)

    def _segment_count(self, n_frames: int) -> int:
        ma = self.model_args
        tf = pick_target_frames(_parse_choices(getattr(ma, "segment_target_frames_choices", "")),
                                self._cur_seed, int(getattr(ma, "segment_target_frames", 4)))
        return adaptive_segment_count(int(n_frames), tf, int(getattr(ma, "segment_force_every", 8)))

    def __getitem__(self, i, _retries: int = 0) -> Dict:
        self._cur_seed = random.getrandbits(31)
        seed = self._cur_seed
        data_dict = super().__getitem__(i, _retries)
        # A rejected sample recurses through _backup_sample -> self.__getitem__, which
        # stamps the replacement's own seed first; keep that one.
        data_dict.setdefault("compression_seed", [seed])
        return data_dict

    @property
    def encode_costs(self) -> Optional[list]:
        """Estimated step cost per sample, for ``--group_by_encode_cost``: frozen-encoder
        time from the frame grid the processor WILL produce (native size from
        ``--media_geometry_json``, frame count from the duration, fps and
        ``--max_frames``), plus LLM time for the expected ``E_tf[N] * K`` compressed
        tokens and the text. None without a geometry file; a medium missing from it
        gets the sub-dataset's median."""
        if hasattr(self, "_encode_costs"):
            return self._encode_costs
        self._encode_costs = None
        path = getattr(self.data_args, "media_geometry_json", None)
        if not path or not os.path.exists(path):
            return None
        geo = _load_geometry(path, os.path.getmtime(path))
        ma, da = self.model_args, self.data_args
        proc = self.vlprocessor.image_processor
        k = int(getattr(ma, "num_queries", 64))
        fe = int(getattr(ma, "segment_force_every", 8))
        tfs = (_parse_choices(getattr(ma, "segment_target_frames_choices", ""))
               or [int(getattr(ma, "segment_target_frames", 4))])
        fps = float(getattr(da, "fps", 0) or 0)
        max_frames = int(getattr(da, "max_frames", 0) or 0)
        merge = int(da.video_merge_size)   # a still image is compressed as a 1-frame video

        rows = self.list_data_dict
        cols = getattr(rows, "column_names", None)   # HF Dataset: whole-column reads

        def column(name):
            if cols is None:
                return [r.get(name) for r in rows]
            return rows[name] if name in cols else [None] * len(rows)

        root = self.dataset_root or ""
        costs = []
        for vid, img, conv in zip(column("video"), column("image"), column("conversations")):
            is_image = bool(img) and not vid
            m = img if is_image else vid
            m = m[0] if isinstance(m, list) and len(m) == 1 else m
            g = geo.get(os.path.join(root, m)) if isinstance(m, str) else None
            if g is None or (not is_image and not fps):
                costs.append(None)
                continue
            w, h, dur = g
            t = 1 if is_image else max(1, round(float(dur) * fps))
            if max_frames:
                t = min(t, max_frames)
            p = _patches_per_frame(h, w, t, proc, merge)
            n_llm = (statistics.mean(adaptive_segment_count(t, tf, fe) for tf in tfs) * k
                     + _TOKENS_PER_WORD * sum(len(str(c.get("value", "")).split()) for c in conv or []))
            costs.append(_ENC_S_PER_PATCH * t * p + _ENC_S_PER_ATTN * t * p * p
                         + _LLM_S_PER_TOKEN * n_llm)
        known = [c for c in costs if c is not None]
        if not known:
            return None
        fill = statistics.median(known)
        if len(known) < len(costs):
            rank0_print(f"[qbase-enhance] {self.dataset_name}: {len(costs) - len(known)}/{len(costs)} "
                        f"media missing from the geometry file -> median cost")
        self._encode_costs = [fill if c is None else c for c in costs]
        return self._encode_costs


@dataclass
class QbaseEnhanceDataArguments(base.DataArguments):
    media_geometry_json: Optional[str] = field(
        default=None,
        metadata={"help": "{media path: [w, h, duration]} from dataset_util/probe_media_geometry.py; "
                          "gives the dataset its encode_costs for --group_by_encode_cost."},
    )


@dataclass
class QbaseEnhanceModelArguments(base.ModelArguments):
    segment_target_frames_choices: str = field(
        default="1,2,3,4",
        metadata={"help": "Comma-separated frames/segment choices, one drawn per sample. "
                          "Empty = fixed --segment_target_frames."},
    )
    compressor_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Recompute qbase layer activations in backward (the qbase re-projects "
                          "the whole kv window every layer; at --vision_max_tokens 65536 it "
                          "dominates the step)."},
    )


def _build_config(model_config, model_args, data_args) -> Dict:
    d = base._build_token_compressor_config(model_config, model_args, data_args)
    choices = _parse_choices(model_args.segment_target_frames_choices)
    fe = int(model_args.segment_force_every)
    if any(c < 1 or c >= fe for c in choices):
        raise ValueError(f"--segment_target_frames_choices {choices} must lie in [1, "
                         f"force_every={fe}) or the count/cut contract breaks")
    d["segment_target_frames_choices"] = choices or None
    d["compressor_gradient_checkpointing"] = bool(model_args.compressor_gradient_checkpointing)
    return d


def _on_built(compressor, model_args, data_args, training_args) -> None:
    rank0_print(f"[qbase-enhance] frames/segment choices="
                f"{compressor.segment_target_frames_choices} (per-sample seed), eval default "
                f"tf={compressor.segment_target_frames}, force_every={compressor.segment_force_every}, "
                f"tau={compressor.segment_sample_tau}")


if __name__ == "__main__":
    base.train(
        attn_implementation="flash_attention_2",
        model_args_cls=QbaseEnhanceModelArguments,
        data_args_cls=QbaseEnhanceDataArguments,
        dataset_cls=QbaseEnhanceDataset,
        build_token_compressor_config=_build_config,
        configure_image_processor=base._configure_image_processor,
        on_compressor_built=_on_built,
    )
