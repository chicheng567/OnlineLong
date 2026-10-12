#!/usr/bin/env python3
"""qbase + LLM full finetune, with raw/qbase hybrid samples (2026-10-07).

Why: the qbase trails raw frames on Video-MME, and the still-image probe
(``eval_ablation/image_detail.py``) says much of that is the LLM not reading the
qbase tokens: the Phase-2 qbase keeps as much probe-decodable word information as a
64-token resize, yet the frozen LLM answers TextVQA 11.8 pp worse from it. So the LLM
trains too -- every parameter of it (embed_tokens, layers, norm, lm_head) at
``--llm_lr`` -- jointly with the qbase (``--compressor_lr``) and ``mm_projector``
(``--mm_projector_lr``); the vision encoder stays frozen.

Thin wrapper over ``qbase_enhance_pretrain.py`` (its dataset, segmenter and rank load
balance; same hooks into ``compressor_pretrain_with_videollama3.train``). One addition:

* **Raw/qbase hybrid samples.** With probability ``--hybrid_raw_sample_prob`` a video
  sample becomes a hybrid: of its ``N`` adaptive segments, ``round(--hybrid_raw_seg_frac
  * N)`` are drawn uniformly from the ``--hybrid_raw_pool_frac`` lowest-change ones
  (change = mean ``1 - cos`` of consecutive per-frame mean encoder features, the
  segmenter's ``diff``) and reach the LLM as the segment's middle frame's raw encoder
  tokens (merge 2, the sample's own resolution, ``h*w`` rows, shared ``mm_projector``)
  behind that frame's own ``Time X.Xs:`` marker, instead of the segment's K qbase
  rows. Everything stays inside the sample's one ``<cs>...<ce>`` block behind its
  ``Time:{a}s-{b}s:`` range. A segment is either raw or qbase, never both (the
  CLAUDE.md training rule). The draw is keyed on the sample's ``compression_seed``
  (``compressor.py::hybrid_rng``), independent of its tf, and only happens in
  training; still images (one segment) never get a raw segment.
  Model side: ``TransformerDecoderFlatCompressor.hybrid_n_raw`` / ``forward``;
  token rewrite: ``videollama3_arch.py``. This dataset supplies the per-frame
  ``Time X.Xs:`` ids (``compression_frame_ts_ids``), tokenised exactly as the chat
  template renders them.

* **Budgeted raw/qbase windows** (``--rawdp_budget B``, 2026-10-10; exclusive with
  the hybrid draw). Every frame reaches the LLM raw (merge 2, the processor's
  resolution, own ``Time X.Xs:``) unless the window holds more than ``B`` rows; then
  ``compressor.py::rawdp_layout`` keeps the most raw frames ``r`` that fit next to
  ``Q = ⌊(T - r)/tf⌋ + 1`` qbase blocks (tf from ``--segment_target_frames_choices``
  per sample) and ``rawdp_place`` places them by exact DP on within-block WCSS of the
  per-frame mean encoder features. Every window keeps >= 1 qbase block.
  Launcher: ``shell/sft_qbase_rawdp_long.sh`` (videos > 60 s, native resolution).

Canonical invocation: ``shell/sft_qbase_full_hybrid.sh``. Checkpoints are full HF
saves (LLM included); evaluate them directly, no merge step.
"""
from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from typing import Dict

sys.path.append("./")

import videollama3.train.qbase_enhance_pretrain as qe
from videollama3.model.compressor import adaptive_segment_count, hybrid_raw_count, rawdp_layout
from videollama3.train.data.common import rank0_print
from videollama3.train.data.global_compressor import get_video_content

base = qe.base


class QbaseHybridDataset(qe.QbaseEnhanceDataset):
    """QbaseEnhanceDataset + each video sample's per-frame ``Time X.Xs:`` token ids."""

    _cur_content = None

    def _convert_normal(self, data_dict):
        out = super()._convert_normal(data_dict)
        # The content dict the vlprocessor renders from (timestamps already final).
        self._cur_content = get_video_content(out[2]) if out[0] == "video" else None
        return out

    def _frame_ts_ids(self, t: float) -> list:
        key = f"Time {round(float(t), 1)}s:"      # the chat template's per-frame marker
        cache = self.__dict__.setdefault("_ts_id_cache", {})
        if key not in cache:
            cache[key] = self.vlprocessor.tokenizer.encode(key, add_special_tokens=False)
        return cache[key]

    def __getitem__(self, i, _retries: int = 0) -> Dict:
        self._cur_content = None
        data_dict = super().__getitem__(i, _retries)
        # A rejected sample recurses through _backup_sample -> self.__getitem__, which
        # attaches the replacement's own ids first; keep those.
        if "compression_frame_ts_ids" in data_dict:
            return data_dict
        is_image = bool((data_dict.get("compression_is_image") or [False])[0])
        content = None if is_image else self._cur_content
        ts = None if content is None else content.get("timestamps", None)
        if ts is None or len(ts) == 0:
            data_dict["compression_frame_ts_ids"] = [[]]
            return data_dict
        assert len(ts) == int(content["num_frames"]), (
            f"{len(ts)} timestamps for {content['num_frames']} frames")
        data_dict["compression_frame_ts_ids"] = [[self._frame_ts_ids(t) for t in ts]]
        return data_dict

    def _expected_vision_llm_tokens(self, t, hw, is_image, tfs, fe, k) -> float:
        ma = self.model_args
        budget = int(getattr(ma, "rawdp_budget", 0) or 0)
        if budget > 0 and not is_image:
            return statistics.mean(r * hw + q * k for r, q in
                                   (rawdp_layout(t, hw, tf, budget, k, fe) for tf in tfs))
        n = super()._expected_vision_llm_tokens(t, hw, is_image, tfs, fe, k)
        p = float(getattr(ma, "hybrid_raw_sample_prob", 0.0) or 0.0)
        frac = float(getattr(ma, "hybrid_raw_seg_frac", 0.0) or 0.0)
        if is_image or p <= 0.0 or frac <= 0.0:
            return n
        n_raw = statistics.mean(hybrid_raw_count(adaptive_segment_count(t, tf, fe), frac) for tf in tfs)
        return n + p * n_raw * (hw - k)


@dataclass
class QbaseHybridModelArguments(qe.QbaseEnhanceModelArguments):
    hybrid_raw_sample_prob: float = field(
        default=0.25,
        metadata={"help": "Share of video samples that become raw/qbase hybrids (training only)."},
    )
    hybrid_raw_seg_frac: float = field(
        default=0.3,
        metadata={"help": "Share of a hybrid sample's segments sent raw (middle frame, merge 2)."},
    )
    hybrid_raw_pool_frac: float = field(
        default=0.5,
        metadata={"help": "Raw segments are drawn uniformly from this lowest-change share of segments."},
    )
    rawdp_budget: int = field(
        default=0,
        metadata={"help": "> 0: every frame raw unless the window exceeds this many LLM rows, then "
                          "DP-placed raw frames + qbase blocks under it (training only). 0 = off."},
    )


def _build_config(model_config, model_args, data_args) -> Dict:
    d = qe._build_config(model_config, model_args, data_args)
    if not 0.0 <= model_args.hybrid_raw_seg_frac < 1.0:
        raise ValueError(f"--hybrid_raw_seg_frac {model_args.hybrid_raw_seg_frac} must lie in [0, 1)")
    if model_args.hybrid_raw_seg_frac > model_args.hybrid_raw_pool_frac:
        raise ValueError("--hybrid_raw_seg_frac cannot exceed --hybrid_raw_pool_frac (the draw pool)")
    d["hybrid_raw_sample_prob"] = float(model_args.hybrid_raw_sample_prob)
    d["hybrid_raw_seg_frac"] = float(model_args.hybrid_raw_seg_frac)
    d["hybrid_raw_pool_frac"] = float(model_args.hybrid_raw_pool_frac)
    if model_args.rawdp_budget > 0:
        if model_args.hybrid_raw_sample_prob > 0.0:
            raise ValueError("--rawdp_budget needs --hybrid_raw_sample_prob 0 (the two are exclusive)")
        tfs = [int(x) for x in str(model_args.segment_target_frames_choices or "").split(",") if x.strip()]
        if any(tf < 2 for tf in tfs):
            raise ValueError("--rawdp_budget: tf choices must be >= 2 (a 1-frame block is a raw frame)")
        if model_args.rawdp_budget < int(model_args.num_queries):
            raise ValueError("--rawdp_budget must hold at least one qbase block")
    d["rawdp_budget"] = int(model_args.rawdp_budget)
    return d


def _on_built(compressor, model_args, data_args, training_args) -> None:
    qe._on_built(compressor, model_args, data_args, training_args)
    if compressor.rawdp_budget > 0:
        rank0_print(f"[qbase-full-ft] rawdp: budget={compressor.rawdp_budget} rows/window, "
                    f"tf choices={compressor.segment_target_frames_choices}, "
                    f"block <= {compressor.segment_force_every} frames")
    rank0_print(f"[qbase-full-ft] hybrid: p_sample={compressor.hybrid_raw_sample_prob}  "
                f"raw share of segments={compressor.hybrid_raw_seg_frac}  "
                f"drawn from lowest-change {compressor.hybrid_raw_pool_frac}")


if __name__ == "__main__":
    base.train(
        attn_implementation="flash_attention_2",
        model_args_cls=QbaseHybridModelArguments,
        data_args_cls=qe.QbaseEnhanceDataArguments,
        dataset_cls=QbaseHybridDataset,
        build_token_compressor_config=_build_config,
        configure_image_processor=base._configure_image_processor,
        on_compressor_built=_on_built,
    )
