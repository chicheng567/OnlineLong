#!/usr/bin/env python3
"""Plan X Phase 3 — budget-driven length-adaptive fold on long video (420–1200 s).

Thin wrapper over ``phase2_pretrain_fold.py`` (itself a wrapper over
``compressor_pretrain_with_videollama3.py``): same frozen-LLM CE recipe, same
collator/trainer, same ``TwoStageCompressor``, warm-started from a Phase-2
checkpoint. Three mechanisms change, all of them documented in
``docs/two_stage_compression_design.md`` §4 Phase 3 / §5 items 13, 16, 17:

* **§5 item 17 — target-``N`` qbase segmentation.** The first-level cut is
  denominated in TOKENS like the budget is, not in frames:
  ``N = clip(c*⌊B/M⌋, N_min(T), T)`` with ``c = 4``, ``min_adapt = 0.5``.
  ``min_adapt = 0.5`` reproduces Phase 1/2's ``target_frames = 4`` exactly on long
  clips (``target_frames = force_every*(1-min_adapt)``), so the operating point the
  warm-started checkpoint was trained at is undisturbed; ``c`` only moves short
  clips, taking the "every unit a raw bypass, zero fold gradient" rate on the
  ``unified``/``vcd`` blend from 69.0 % to 20.4 % (the rest are genuinely too short
  to fill the budget).
* **§5 item 13 — budget-driven ``U``.** ``U = min(N, ⌊B/M⌋)`` (``B = 1024`` ⇒
  ``U ≤ 16``) replaces Phase 2's depth-class ``N̄_u`` draw. A unit costs ``M``
  tokens whether it folds or passes through raw, so the mechanism reduces to
  choosing how many final units there are; a unit left holding one segment emits
  that segment's ``K`` qbase tokens verbatim (``--stage2_bypass_singletons``).
  That is **not** the banned retained-K: a segment is either raw or absorbed into
  some other unit's fold, never both.
* **§5 item 16 — DP unit placement.** ``--stage2_unit_placement dp``: exact
  ``O(N²U)`` DP over ``WCSS + λ·depth_cost``, ``N_u_soft = ratio·N/U``. Phase 2's
  greedy ``_place_unit_boundaries`` is left untouched (its ``min_gap`` is a floor on
  every unit's size, so it can never emit a bypass unit, and it is worse than the DP
  on depth: ``N_u_max`` 75 vs 34 measured).

Per-epoch partition variance still comes from the **segment-level** Gumbel jitter
(``--segment_sample_tau``, default 0.5, inherited); the DP itself is deterministic
(a WCSS-space jitter analogue is an open item in the design doc). Phase 2's
``variance_cold_frac`` no longer narrows ``U`` — with a budget-derived ``U`` there
is no draw to narrow; it now scopes the optional warm-start budget ramp
(``--token_budget_cold``, off by default), which addresses the design doc's
"warm-start distribution shift" (problem #8) by growing ``B`` rather than by
ramping frames. ``N`` never depends on the ramp, so the collator/model length
contract holds throughout.

Run from repo root with ``PYTHONPATH=.`` (see ``shell/`` for the canonical
invocation)::

    PYTHONPATH=. torchrun ... videollama3/train/phase3_pretrain_budget.py \\
        --compressor_type transformer_decoder_flat --num_queries 64 \\
        --compressor_num_layers 8 --compressor_num_attention_heads 8 \\
        --match_encoder_scale True --compressor_distr_loss_weight 0.05 \\
        --pretrained_compressor_path work_dirs/phase2_fold_internvid \\
        --token_budget 1024 --stage2_n_summary_tokens 64 \\
        --stage2_unit_placement dp --stage2_bypass_singletons True \\
        --mm_projector_lr 2e-5 --qbase_lr 5e-6 --mamba_lr 5e-5 \\
        --group_by_compression_depth True --durations_json anno_data/internVid_durations.json \\
        --multi_dataset True --data_path anno_data/phase3_training.json ...
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, Tuple

sys.path.append("./")

import videollama3.train.compressor_pretrain_with_videollama3 as base
from videollama3.model.compressor import (
    TwoStageCompressor,
    budget_unit_count,
    target_segment_count,
)
from videollama3.train.data.common import rank0_print
from videollama3.train.phase2_pretrain_fold import (
    Phase2DataArguments,
    Phase2FoldDataset,
    Phase2ModelArguments,
    _build_phase2_token_compressor_config,
    _configure_phase2_image_processor,
    _warmstart_and_freeze_stage1,
)


# ---------------------------------------------------------------------------
# Dataset — Phase 2's, with the budget rule replacing the depth-class draw.
# ---------------------------------------------------------------------------

class Phase3BudgetDataset(Phase2FoldDataset):
    """Phase-3 dataset. One whole-video ``compression_part`` per clip (unchanged);
    what changes is the two numbers it hands the model:

      * ``N`` (implicitly, via the segment-count rule the model applies to the same
        frame count) — ``target_segment_count``, the token-denominated rule;
      * ``U = min(N, ⌊B/M⌋)`` — the budget, not a depth-class draw.

    Both stay pure functions of the frame count, so the collator's placeholder
    reservation and the model's cut still agree without decoding features.
    """

    def _budget(self) -> int:
        """Effective ``B`` for this step. Equal to ``--token_budget`` unless a
        warm-start ramp is configured (``--token_budget_cold`` > 0), which holds a
        smaller budget for the first ``--variance_cold_frac`` of training so a
        Phase-2-warm-started fold is not handed its deepest units on step 1
        (design doc §4 Phase 3, "Potential problems" #8)."""
        ma, da = self.model_args, self.data_args
        full = int(getattr(ma, "token_budget", 1024)) if ma is not None else 1024
        cold = int(getattr(da, "token_budget_cold", 0) or 0)
        if cold <= 0:
            return full
        if float(self._progress) < float(getattr(da, "variance_cold_frac", 0.15)):
            return max(1, min(cold, full))
        return full

    def _fold_knobs(self):
        tf, _max_units, m, k = super()._fold_knobs()
        ma = self.model_args
        budget = int(getattr(ma, "token_budget", 1024)) if ma is not None else 1024
        # U cap is the budget itself, not Phase 2's stage2_max_units = 5.
        return tf, max(1, budget // max(1, m)), m, k

    def _segment_count(self, n_frames: int) -> int:
        """§5 item 17: ``N = clip(c*⌊B/M⌋, N_min(T), T)``. Must match the model's
        ``compressor.segment_count_for`` exactly — both call the same helper, and the
        Phase-3 compressor config is built from the same ``c`` / ``min_adapt`` /
        ``force_every`` below."""
        ma = self.model_args
        _tf, u_cap, _m, _k = self._fold_knobs()
        c = int(getattr(ma, "segment_target_c", 4)) if ma is not None else 4
        fe = int(getattr(ma, "segment_force_every", 8)) if ma is not None else 8
        min_adapt = float(getattr(ma, "segment_min_adapt", 0.5)) if ma is not None else 0.5
        return target_segment_count(int(n_frames), c * u_cap, fe, min_adapt)

    def _draw_units(self, i: int, N: int, tpf: int, total_vision_tokens: int) -> "Tuple[int, int]":
        """§5 item 13: ``U = min(N, ⌊B/M⌋)``. No depth-class draw, no ``N//4``
        clamp (which capped short clips below the budget). The seed is still drawn
        per ``(epoch, index)`` — the model's segment-level Gumbel uses it."""
        _tf, _u_cap, M, _K = self._fold_knobs()
        seed = (int(self._epoch) * 1_000_003 + int(i)) & 0x7FFFFFFF
        U = budget_unit_count(N, self._budget(), M)
        # The compressed region must fit inside the video's vision-token slots
        # (the arch reserves replace_mask[ps : ps + sum n_out]).
        while U > 1 and U * M > total_vision_tokens:
            U -= 1
        return U, seed


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Phase3ModelArguments(Phase2ModelArguments):
    # Phase-3 clips are 420-1200 frames; the single-shot encoder forward is the
    # largest allocation in the step at that length (§5 item 4, §8 step 7).
    vision_encoder_chunk_frames: int = field(default=8)
    compressor_gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Recompute the qbase / fold layer activations in backward. The qbase "
                          "re-projects the whole kv window (up to --vision_max_tokens) in EVERY "
                          "layer, so its stored activations dominate the step at Phase-3 lengths; "
                          "--gradient_checkpointing only covers the LLM, never the compressor."},
    )
    token_budget: int = field(
        default=1024,
        metadata={"help": "B -- per-video vision-token budget the LLM sees. U = min(N, B/M); "
                          "keep it a multiple of M so nothing is left on the table."},
    )
    segment_target_c: int = field(
        default=4,
        metadata={"help": "Target segment count as a multiple of the unit cap: N aims at "
                          "c*floor(B/M), clamped by the force_every / min_adapt floors. c=4 beats "
                          "c=2 (same bypass rate and budget fill, ~40%% more fold gradient from "
                          "short clips); c=1 sits exactly on the all-bypass boundary."},
    )
    segment_min_adapt: float = field(
        default=0.5,
        metadata={"help": "Minimum fraction of segment cuts that stay discretionary. 0.5 with "
                          "force_every=8 reproduces Phase 1/2's target_frames=4 on long clips "
                          "(target_frames = force_every*(1-min_adapt)); 0 halves the N*K RoPE "
                          "footprint but mostly disables the adaptive segmenter."},
    )
    stage2_unit_placement: str = field(
        default="dp",
        metadata={"help": "'dp' = exact DP on WCSS + lambda*depth_cost (Phase 3); 'greedy' = "
                          "Phase 2's top-diff + min_gap ranking (cannot emit a bypass unit)."},
    )
    stage2_dp_lambda: float = field(
        default=0.05,
        metadata={"help": "depth_cost weight. Insensitive above ~0.01 (lambda=0.01 vs 0.1 differ "
                          "by 1-2 segments); lambda=0 lets real video reach N_u=196."},
    )
    stage2_dp_soft_ratio: float = field(
        default=1.75,
        metadata={"help": "N_u_soft = ratio * N/U. MUST scale with N/U (1.5-2), never a fixed 16: "
                          "below N/U every unit pays the quadratic and the DP degenerates to a "
                          "uniform split."},
    )
    stage2_bypass_singletons: bool = field(
        default=True,
        metadata={"help": "A unit holding one segment emits that segment's K qbase tokens raw "
                          "(kind 'qbase') instead of folding a singleton."},
    )


@dataclass
class Phase3DataArguments(Phase2DataArguments):
    token_budget_cold: int = field(
        default=0,
        metadata={"help": "Optional warm-start ramp: hold this smaller B for the first "
                          "--variance_cold_frac of training, then switch to --token_budget. "
                          "0 disables. Affects U only -- N never depends on it, so the "
                          "collator/model length contract is untouched."},
    )


# ---------------------------------------------------------------------------
# Injection hooks
# ---------------------------------------------------------------------------

def _build_phase3_token_compressor_config(model_config, model_args, data_args) -> Dict:
    d = _build_phase2_token_compressor_config(model_config, model_args, data_args)
    u_cap = max(1, int(model_args.token_budget) // max(1, int(model_args.stage2_n_summary_tokens)))
    d.update(
        # §5 item 17 -- the model derives N from the frame count with these, exactly
        # as Phase3BudgetDataset._segment_count does.
        segment_count_rule="target_n",
        segment_target_n=int(model_args.segment_target_c) * u_cap,
        segment_min_adapt=float(model_args.segment_min_adapt),
        # §5 items 13 / 16.
        stage2_unit_placement=model_args.stage2_unit_placement,
        stage2_dp_lambda=model_args.stage2_dp_lambda,
        stage2_dp_soft_ratio=model_args.stage2_dp_soft_ratio,
        stage2_bypass_singletons=model_args.stage2_bypass_singletons,
    )
    return d


def _phase3_on_compressor_built(compressor, model_args, data_args, training_args) -> None:
    _warmstart_and_freeze_stage1(compressor, model_args, data_args, training_args)
    if not isinstance(compressor, TwoStageCompressor):
        return
    M = compressor.n_summary_tokens
    u_cap = max(1, int(model_args.token_budget) // max(1, M))
    rank0_print(
        f"[phase3] budget B={model_args.token_budget} -> U <= {u_cap} units x M={M} tokens; "
        f"segmentation rule={compressor.stage1.segment_count_rule} "
        f"(target_n={compressor.stage1.segment_target_n} = c{model_args.segment_target_c}*{u_cap}, "
        f"min_adapt={compressor.stage1.segment_min_adapt}, "
        f"force_every={compressor.stage1.segment_force_every}); "
        f"unit placement={compressor.unit_placement} "
        f"(lambda={compressor.dp_lambda}, N_u_soft={compressor.dp_soft_ratio}*N/U), "
        f"bypass_singletons={compressor.bypass_singletons}"
    )
    for T in (60, 180, 420, 900, 1200):
        N = compressor.segment_count_for(T)
        U = budget_unit_count(N, model_args.token_budget, M)
        rank0_print(
            f"[phase3]   T={T:5d}f -> N={N:4d} segments, U={U:3d} units, "
            f"mean N_u={N / U:5.1f}, N*K={N * compressor.tokens_per_segment:6d} RoPE slots"
        )


if __name__ == "__main__":
    base.train(
        attn_implementation="flash_attention_2",
        model_args_cls=Phase3ModelArguments,
        data_args_cls=Phase3DataArguments,
        dataset_cls=Phase3BudgetDataset,
        build_token_compressor_config=_build_phase3_token_compressor_config,
        configure_image_processor=_configure_phase2_image_processor,
        on_compressor_built=_phase3_on_compressor_built,
    )
