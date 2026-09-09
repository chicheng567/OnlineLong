# Dynamic video compression — joint InternVid pretrain (Plan X)

**Status:** design agreed. This document **supersedes** the earlier plan (kept in
git history): the qbase pretrained on whole-video windows against a frozen LLM, then
an SSD fold bolted on and trained separately, then the qbase unfrozen. That plan
*ran to completion* (`work_dirs/stage2a_fold_videoxl`,
`work_dirs/stage2_unfreeze_full`) and its fold readout **collapsed cross-video** —
worse caption quality than the qbase alone despite 2× the tokens. §2 has the
measured post-mortem; the new plan is built to not repeat it.

Training-phase vocabulary is now **Phase 1 / 2 / 3** (§4). The `stage1` / `stage2`
names that remain below are load-bearing *code* symbols (`TwoStageCompressor.stage1`,
`stage2_*` config fields, `Stage2UnitDataset`, `stage1_lr`) — the trainer's save
filter and per-group LR logic key on them — not training phases.

**What changed vs. the superseded plan**

| | superseded | Plan X |
|---|---|---|
| qbase / fold trained | separately (qbase frozen while fold trains) | **jointly**, from Phase 2 on |
| pretrain data | videoxl caption sets (~73k) | **InternVid** (`anno_data/internVid.json`, 977,423; Qwen3-VL re-captions) |
| video length | ≤160 f, caps most clips | **length-bucketed curriculum**, fps=1, real depth |
| qbase segments | whole-video window, or uniform 4-frame (superseded runs) | **fixed count `⌊T/4⌋+1`, adaptively placed** 1–8 frame segments (forced-8 + top-`diff`) |
| LLM | frozen | frozen (unchanged) |
| `mm_projector` | frozen | **unfrozen** (tracks the compressor manifold; low risk) |
| fold-readout regularization | none | **Option A + Option B + cross-sample decorrelation** |
| collapse gate | `feature_distribution.py` (missed it) | **grounding probe: centered cross-video cosine + `struct_ρ` vs encoder** |

**Unchanged:** the architecture. qbase = `transformer_decoder_flat`
(`compressor.py`, K learned queries), fold = `SegmentAggregator`
(`segment_aggregator.py`, pure-PyTorch scalar-decay SSD). `TwoStageCompressor`
wires them. Only the *training method*, the *curriculum*, the *segmenter*, and the
*fold-output regularization* change.

---

## 1. Goal and the LLM-facing representation

Compression must be **dynamic and non-uniform**: dense where content changes, cheap
where it does not; the same mechanism for streaming and offline; a spare token
budget spent on keeping resolution, not on a fixed ratio.

Plan X gives the LLM **two levels of detail** per video:

- **qbase (hi-res).** The video is cut into `N = ⌊T/4⌋+1` **1–8 frame segments** —
  a *fixed count* (so the compressed length is predictable, §4), *adaptively
  placed*: boundaries land on the largest frame-to-frame encoder-feature changes,
  with a forced cut every 8 frames. Each segment → **K = 64** LLM-readable tokens.
  Dense action ⇒ 1-frame segments there; a calm span ⇒ 8-frame segments — same
  total, redistributed.
- **fold (coarse / gist).** An SSD associative fold folds a *unit*'s segment
  summaries into **M = 64** tokens with an **O(1) recurrent state** — depth grows
  with video length, token count does not.

Per video, in time order, the LLM sees for each unit `u` (U ≤ 5 contiguous units):

```
[ Time:{a}s-{b}s: ] <|compression_start|>
    interleave(  retained segments' K tokens ,  unit u's M readout tokens  )   # time-ordered
<|compression_end|>
```

- **Retained set:** a small **no-gradient** budget of segments per unit (highest
  inter-frame motion / lowest neighbour-cosine, capped at ≤ 3/unit) contribute
  their raw qbase K tokens directly, interleaved with the M readout tokens. This is
  the ∞-former / compressive-transformer escape hatch for detail a finite SSD state
  cannot hold (§3).
- **Vision-token cost to the LLM** = `U·M + (Σ retained)·K` ≤ `5·64 + 15·64 = 1280`
  — bounded and predictable for any video length.

**Phase 1 is the degenerate case:** no fold, no units, retained = *every* segment ⇒
the LLM sees `N·K` tokens (qbase per semantic segment, straight to the projector).

---

## 2. Why the superseded plan collapsed — measured

### 2.1 Cross-video attribution (`eval_ablation`, 12 videoxl + 12 InternVid clips)

Per video: one mean-pooled vector per representation. `xcos_raw` = mean off-diagonal
cosine of L2-normed pooled vectors (this is `grounding_probe.content_collapse`).
`xcos_cen` = same after subtracting the across-video mean vector. `cc_frac` =
`‖mean_v p_v‖ / mean_v‖p_v‖` (shared-component energy). `struct_ρ` = Spearman of
this rep's video-to-video cosine ranking against the encoder's.

| representation | xcos_raw | xcos_cen | cc_frac | struct_ρ vs encoder |
|---|---:|---:|---:|---:|
| raw encoder (videoxl) | 0.869 | −0.081 | **0.937** | 1.000 (ref) |
| qbase (videoxl) | 0.883 | −0.081 | 0.944 | **0.999** |
| **fold readout (videoxl)** | **0.999** | −0.079 | **1.000** | **0.386** |
| raw encoder (InternVid) | 0.907 | −0.088 | 0.956 | 1.000 (ref) |
| qbase (InternVid) | 0.918 | −0.088 | 0.962 | 0.999 |
| **fold readout (InternVid)** | **1.000** | −0.075 | **1.000** | **0.717** |

Reading:

1. **`content_collapse ≈ 0.87–0.91` is an inherent SigLIP-NaViT property, not the
   compressor.** ~94–96 % of a mean-pooled encoder vector's energy is a single
   shared direction (`cc_frac`). A raw `content_collapse` near 0.9 is the *baseline*.
2. **qbase passes the encoder geometry through faithfully** — `struct_ρ = 0.999`,
   `cc_frac` unchanged. `match_encoder_scale` (Option A) matches encoder stats by
   design. The "qbase does hi-res" half of Plan X is on solid ground.
3. **The fold does two harmful things.** It drives `cc_frac` to exactly 1.000 (the
   readout's `final_norm` RMSNorm pins every vector to the same norm ⇒ near-collinear
   ⇒ `xcos_raw` → 1.000), and it **scrambles the discriminative residual**
   (`struct_ρ` 0.999 → 0.39 on videoxl). It is not that information is destroyed to
   zero — `xcos_cen` magnitude survives — it is that the surviving signal no longer
   tracks which video is which.

### 2.2 Contributing causes

- **No encoder-scale match on the fold output.** qbase output is affine-matched to
  the encoder scale; `SegmentAggregator.input_norm` (RMSNorm) then renormalizes to
  RMS≈1 and the readout comes out at **~13× the encoder-token norm** (REPORT
  `norm ratio compressed/raw = 12.94`). The frozen `mm_projector` + LLM were aligned
  on ~1.5-norm tokens. The superseded doc predicted this — the qbase output must be
  on a sane scale before the fold's `input_norm`, *not only* before `mm_projector` —
  and nothing acted on it for the fold's *output*.
- **CE gives almost no gradient.** The superseded fold run's train CE went
  0.289 → 0.247 and was flat after ~step 30 of 288. The frozen 7B autocompletes
  videoxl captions from its text
  prior; with the readout mis-scaled it leans entirely on that prior; the fold's
  gradient ≈ noise; the CE optimum is "emit a constant".
- **qbase run out of distribution.** Pretrained on whole-video windows, then called
  per 4-frame segment in `compress_windows`.
- Fixed fold depth, absolute-index `index_sincos` time embedding, `weight_decay=0`,
  `dropout=0`, and the Option-B aux loss silently a **no-op** for the fold
  (`_compressor_distr_loss` reads `distr_loss_weight` off the `TwoStageCompressor`
  wrapper, which does not have it — it lives on `.stage1`).

### 2.3 Consequence for gating

**Stop gating on raw `content_collapse`.** Gate on **`xcos_cen`** (centered
cross-video cosine — should sit near the encoder's, ≈ −0.08 for a dozen clips) and
**`struct_ρ` vs the encoder** (target ≈ qbase's 0.999; a fold value below ~0.9 is a
red flag). Keep `feature_distribution.py` as a secondary check.

---

## 3. The fold — mechanism (unchanged, condensed)

Full derivation is in the superseded doc's §2–4 (git history). The load-bearing
points:

- **Monoid fold over the segment stream.** `lift`: raw span → qbase K tokens →
  fold into a fresh state. `append`: `state ⊕ lift(span)` via the SSD step. `merge`:
  `state ⊕ state` via the SSD associative-scan operator
  `(A₁,b₁)∘(A₂,b₂) = (A₁A₂, A₁b₂+b₁)`. `readout`: M learned queries through the
  recurrence, run from a *copy* of the state (non-destructive).
- **Chunk-invariance / associativity** is what keeps this clean: the fold result
  must not depend on how the stream was cut. The SSD recurrence has it for the SSM
  state; a heuristic token-merge does not. So `merge(fold(A), fold(B)) ≡ fold(A++B)`
  exactly on `ssm_state`.
- **The canonical object is the SSM state**, not a token set. `state` and
  `readout(state)` live in different spaces; readout is mandatory before the LLM.
- **`append` only** for the fold (whole unit in hand). `merge` is reserved for the
  future streaming budget controller (two already-materialised unit states forced
  together to hold U at its cap); with `d_conv` at default it carries a bounded
  `d_conv−1` (=3) token conv seam, acceptable for that coarse/old-memory use.
- **A finite state floods** on a uniformly high-action long passage. `merge` does
  **not** fix this (it carries no information a deep `append` would not). The
  **retained set** (§1) is the answer.

### Vocabulary

| sym | meaning | set by | typ. |
|---|---|---|---|
| N | segments the video is cut into | `⌊T/4⌋+1` (fixed count, adaptive placement); each seg 1–8 f | ~16–80 |
| **K** | tokens qbase emits **per segment** — fixed | `num_queries` (pretrained) | 64 |
| **M** | tokens **one** fold readout emits **per unit** | `stage2_n_summary_tokens` | 64 |
| **U** | readout units (independent folds), **capped** | `stage2_max_units` | ≤ 5 |
| — | frames per segment | segmenter, clamp **[1, 8]** | ~1–8 |
| — | segments per unit (fold depth) | **randomized in training**, grows with length at inference | U[4,16] train |
| — | retained segments per unit | no-gradient heuristic, capped | ≤ 3 |
| — | vision tokens to the LLM | `U·M + (Σ retained)·K` | ≤ 1280 |

---

## 4. Training curriculum — 3 phases, **LLM frozen throughout**

**Common to every phase**

- Data: InternVid, `anno_data/internVid.json` (977,423 clips), Qwen3-VL-8B
  re-captions (`recaption/qwen3vl/`). fps = 1 always. Dynamic HW (native aspect
  ratio, per-video token budget; `--force_image_size` **not** set). On-the-fly
  decode + frozen-encoder forward, **no feature cache**.
- Single CE forward, `B == 1` flattened collator, `use_dual_forward=False`.
  `zero1` for Phase 1, `zero2` from Phase 2 (qbase + fold + projector trainable).
- **`mm_projector` unfrozen** every phase (it must be free to track two moving
  compressor manifolds; small, low risk). **LLM frozen** every phase.
- **Option A + Option B on every compressor output** — qbase output (already wired)
  *and*, from Phase 2, the fold readout (new code, §5).
- Length buckets from `anno_data/internVid_durations.json` (built by the full
  ffprobe scan): **< 180 s ≈ 264k · 180–420 s ≈ 262k · 420–1200 s ≈ 451k**
  (> 1200 s: 98 clips, dropped).
- **No LLM warm-up / adaptation stage.** (The earlier "Phase 0: finetune the LLM on
  Qwen3-VL-style captions" is dropped — it lowers the CE floor and *weakens* the
  compressor gradient, and it muddies the frozen-LLM control.) The residual
  style-mismatch in CE is not compressor signal; that is why the gate is the
  representation-level probe (§2.3), and why caption eval always also runs
  **compressor + the original frozen LLM**.

### Phase 1 — qbase only, content-adaptive segments, **NO fold**

- **Data:** every clip **< 180 s** (~264k) — **including the very short ones**.
  Short clips are useless for the fold but perfectly good qbase signal; do not drop
  them.
- **Segmenter** (new, `compressor.py` or a small module): causal, non-learned,
  model-side on the frozen-encoder **per-frame mean feature**. **Fixed segment
  count, adaptive placement** — a deliberate trade (this *is* a fixed compression
  ratio, against the §1 "not a fixed ratio" wish) taken because it keeps the
  compressed length a pure function of the frame count: the collator and the model
  compute `N` identically **without decoding features**, so Phase 1 needs no
  `prepare_inputs_labels_for_multimodal` refactor (§5.6) and no dynamic-length
  plumbing. Requires one new method `compressor.output_len_for(n_frames, h, w)`
  (arch uses it for the placeholder count in place of `output_hw_for`).
  - `diff[i] = 1 − cos(f_i, f_{i−1})` on the per-frame mean-pooled encoder feature.
  - **Forced boundary every 8 frames**, then spend the remaining budget
    `k = ⌊T/4⌋ + 1 − 1 − n_forced` on the largest-`diff` non-forced positions.
    ⇒ **`N = ⌊T/4⌋ + 1` exactly**, every segment in `[1, 8]`.
  - **Measured** (40 real InternVid clips, frozen SigLIP-NaViT, 1 fps, 448 px,
    merge 2): forced-8 budget ⇒ 0 % of segments over 8 f, length CV 0.66, longest
    10 % of segments hold 20 % of frames (uniform ≈ 10 %), and cuts land on real
    change (mean `diff` at chosen boundaries **14×** the interior mean; uniform-4 is
    1.05×). Segment-length histogram is bimodal — ~25 % 1-frame (adaptive cuts at
    scene turns), ~20 % 8-frame (calm spans at the ceiling) — which is the intended
    behaviour, bounded. **Bare top-k `k = ⌊T/4⌋` (no forced-8) is rejected:** it
    clusters — 10.8 % of segments > 8 f, worst 58 f, CV 1.06, longest 10 % hold
    35 % of frames.
  - **Per-epoch augmentation:** draw the `k` non-forced boundaries stochastically —
    Gumbel-top-k over `softmax(diff / τ)`, `τ ≈ 0.5` — so the adaptive cuts move
    each epoch (~35 % change between draws, measured) while `N` stays exact. This
    replaces the earlier per-sample τ-jitter idea (which would have made `N`
    unpredictable).
  - **Non-learned** — no new parameters, so qbase is the only thing training.
  - Alternatives to ablate (all keep the forced-8 budget + `[1,8]` clamp): `diff`
    on common-component-removed features (`f_i − mean_i f_i`; measured ≈ no change
    on InternVid, keep as a fallback for low-motion domains); RGB shot boundary
    (PySceneDetect / TransNetV2) as the non-forced score.
- **Path:** chunked encoder forward (≤ 8-frame groups, `no_grad`) → segmenter →
  qbase per segment → K = 64 tokens/segment → **all** segments' tokens → frozen
  `mm_projector` → frozen LLM → CE.
- **Trainable:** qbase + `mm_projector`. Warm-start qbase from
  `pretrained_models/compressor_pretrain_video_norm` (147 tensors, clean load;
  `match_encoder_scale=True`, `distr_loss_weight=0.05`). Keep A + B on the qbase
  output.
- **Token math:** `N = ⌊T/4⌋ + 1`, so a 60–180 s clip ⇒ ~16–46 segments ⇒
  ~1.0k–2.9k tokens to the LLM — well inside `model_max_length`, and already a
  large cut vs the uncompressed `T×HW` (~16×).
- **Gate to Phase 2:** on a held-out probe — qbase `struct_ρ` vs encoder ≥ ~0.95;
  `xcos_cen` ≈ encoder's; caption eval (**compressor + original frozen LLM**) not
  worse than the whole-video qbase baseline.

### Phase 2 — add the fold; hybrid representation; **+ 180–420 s**

- **Data:** 180–420 s (~262k) **+ 30 % replay** of the Phase-1 pool.
  **Length-homogeneous minibatches** — bucket by frame count *within* each
  grad-accum window; recurrence depth must not swing step to step.
- **New machinery:** unit grouping (U ≤ 5 contiguous), SSD fold per unit (fresh
  `init_state`), readout M = 64, **retained-subset selection** (no-gradient
  heuristic, ≤ 3 segments/unit), **interleave** retained K tokens + M readout tokens
  in time order per unit.
- **fold-depth randomization ON:** per-unit segment count `N_u ~ U[4, 16]`,
  `Σ N_u = N`, `U ≤ 5` (`Stage2UnitDataset`).
- **time_embed:** **not** absolute `index_sincos`. Use per-segment
  `(gap_seconds, duration_seconds)` → `time_mlp` (`rel_gap_mlp`), or `none` for the
  first runs.
- **Readout regularization (the anti-collapse core):**
  - **Option A** (`_match_encoder_scale`) with `ref` = that forward's encoder
    tokens (available in `compress_windows` as `kv`).
  - **Option B** (`_distribution_match_loss`), wired into the trainer for the fold
    (§5 fixes the wrapper bug).
  - **Cross-sample decorrelation** — VICReg variance + covariance terms on the
    readout over the grad-accum window. This is what defends `struct_ρ`; A + B do
    **not** (§2.1). Logged as `loss_decorr`. *(Distribution moments, not per-token
    targets — not the banned reconstruction objective.)*
  - `weight_decay 0.03–0.05`, fold `dropout 0.1`.
- **Trainable:** qbase + fold + `mm_projector`, LLM frozen. **Stagger the cold
  fold:** freeze qbase (or 10× lower LR) for the first ~15 % of Phase 2; fold LR
  3–10× the qbase LR. `create_optimizer` already has the `stage1_lr` split.
- **Gate to Phase 3:** fold `struct_ρ` within ~0.1 of qbase's; `xcos_cen` ≈
  encoder's; **`cc_frac` back near ~0.94, not 1.0**; caption metrics plateaued; a
  budget ablation (U capped vs. uncapped) shows the cap is nearly free.

### Phase 3 — deep folds; **420–1200 s**

- **Data:** 420–1200 s (~451k) + layered replay (≈ 60 % long / 25 % mid / 15 %
  short). Drop the 98 clips > 1200 s.
- **Trainable:** same as Phase 2. Deeper `segs_per_unit`. **TBPTT** over a window
  of recent segments once the offline `[N·K ; M]` sequence stops fitting (older
  segments detached, forward streamed).
- **Length curriculum inside the phase** — grow the max frame count gradually; do
  not start at 1200 s.
- **Gate:** needle / frame-order probes on long video; frozen-LLM caption eval
  holds; `struct_ρ` holds at depth.

---

## 5. Implementation requirements (code changes)

| # | change | where | GPU? |
|---|---|---|---|
| 1 | **Option A/B on the fold readout** — after `SegmentAggregator.output_proj`, apply `_match_encoder_scale` (ref = the forward's encoder tokens, passed through `compress_windows`) and stash `_last_distr_loss` from `_distribution_match_loss`. | `segment_aggregator.py`, `compressor.py::TwoStageCompressor.compress_windows` | no |
| 2 | **Trainer wiring fix** — `_compressor_distr_loss` does `getattr(comp, "distr_loss_weight", 0.0)` on the `TwoStageCompressor` wrapper → 0. Sum `_last_distr_loss` from `.stage1` **and** `.stage2`. | `videollama3_trainer.py` | no |
| 3 | **Cross-sample decorrelation loss** — VICReg variance + covariance on the readout over the grad-accum window; add to CE, log `loss_decorr`. | trainer + `compressor.py` | no |
| 4 | **Fixed-count adaptive causal segmenter** — model-side, non-learned. `N = ⌊T/4⌋+1` from the frame count (collator + model compute it identically ⇒ no dynamic-length plumbing, no §5.6 refactor for Phase 1). Forced boundary every 8 f + remaining budget to top-`diff` positions on the per-frame encoder mean feature; Gumbel-top-k draw for per-epoch augmentation; `[1,8]` clamp. `frames_per_segment` is now just the clamp max. Validated: `eval_ablation/segmenter_validate.py` (+ `_hist.py`). | new segmenter + `_segment_cu_seqlens` in `compressor.py`; **new** `compressor.output_len_for(n_frames,h,w)`; `videollama3_arch.py` uses it for the placeholder count | no |
| 5 | **Chunked encoder forward** — ≤ 8-frame groups under `no_grad` (encoder frozen, no cross-frame attention). Required at InternVid lengths. | `videollama3_arch.py::encode_images` | no |
| 6 | **Retained-subset selection** — no-gradient heuristic, **fixed count/unit** (§7: shortest segments, e.g. 3 or `round(N_u/8)`), interleaved with the readout. Per-unit output length `M + retained·K` stays a pure function of `N_u` ⇒ the `prepare_inputs_labels_for_multimodal` refactor is about *layout only* (model returns unit structure; resolve the tokenised-`Time:` question — tokenizer in, or a pre-tokenised collator lookup), not length. **Phase 1 does not need this** (§4). | `videollama3_arch.py`, `stage2a_pretrain_compressor_fold.py` dataset | no |
| 7 | **fold-depth randomization** — per-unit `N_u ~ U[4,16]`, `Σ=N`, `U≤5`. | `Stage2UnitDataset` | no |
| 8 | **time_embed** — `rel_gap_mlp` (per-segment gap + duration seconds) or `none`; drop `index_sincos` for the fold. | `segment_aggregator.py` config | no |
| 9 | **K/M knob** — default K = M = 64. Decoupling (K > M, e.g. 128/64) only needs `TwoStageCompressor.__init__` to stop default-tying `n_summary_tokens` to `tokens_per_segment` (`SegmentAggregatorConfig` already has them separate). Open decision §6. | `compressor.py` | no |
| 10 | **`struct_ρ` / `xcos_cen` probe** — add the centered cross-video cosine and the encoder-similarity Spearman to `grounding_probe.py` (or a new probe) so they can gate phases. | `eval_ablation/` | yes (eval) |
| 11 | **LR groups** — `create_optimizer` `stage1`/`stage2` split (present) + `mm_projector` group (present). No change, just enable. | — | — |

**Keep, do not touch:** no feature cache; CE only; **no reconstruction / MSE /
masked-feature** (collapses to order-invariant — Option A/B and VICReg var/cov are
distribution moments, not per-token targets, and are allowed);
`match_encoder_scale` + `distr_loss_weight=0.05` on qbase.

---

## 6. Data

| bucket | duration | ~count | phase |
|---|---|---:|---|
| short | < 180 s | 264k | 1 (qbase), replay 2/3 |
| mid | 180–420 s | 262k | 2, replay 3 |
| long | 420–1200 s | 451k | 3 |
| — | > 1200 s | 98 | dropped |

- `anno_data/internVid.json` — 977,423 entries `{"video": "<id>.mp4", ...}`, 1 file
  missing on disk. Videos in `/share/dataset/internVid/`, stored at native 2 fps
  (decode stride-2 for fps = 1).
- **Captions:** original InternVid captions are ASR-derived and unusable. Use the
  Qwen3-VL-8B re-captions (`recaption/qwen3vl/captions.rank*.jsonl` →
  `dataset_util/captions_jsonl_to_stage1_anno.py` → `anno_online/` + `anno_data/`
  meta). ~37 % complete at 2026-09-08; Phases 1–2 can start on the finished
  short/mid subset.
- **Durations:** `anno_data/internVid_durations.json` (per-clip
  `duration_sec` / `fps` / `est_frames_1fps`, full ffprobe scan). Median ≈ 366 s,
  p90 ≈ 900 s. Bucket the manifest from this file.
- **Domain caveat:** InternVid (YouTube vlog / gameplay / how-to) ≠ the eval sets
  (ShareGPT4Video, cinepile, VCG). Keep **all evaluation on the target
  distribution**, and always also report **compressor + the original frozen LLM** so
  a gain cannot be the LLM memorising InternVid style.

---

## 7. Open decisions

- **K = M vs K > M.** Default K = M = 64 (matches the pretrained qbase, keeps the
  interleave trivially aligned). K > M (e.g. 128 / 64) makes qbase genuinely
  higher-resolution and the fold a real 2:1 reduction — but "hi-res" already comes
  from *short content-adaptive segments*, so this is optional.
- **Segmenter criterion — RESOLVED (§4 Phase 1, measured):** fixed count
  `N = ⌊T/4⌋+1`, forced boundary every 8 f + remaining budget to top-`diff`
  positions (Gumbel-top-k for augmentation). Bare threshold-τ / bare top-k
  rejected. Still ablatable: the `diff` score (raw vs common-component-removed vs
  RGB shot boundary) as the *non-forced* ranking.
- **Retained-subset budget & heuristic** — motion vs. neighbour-cosine; cap. **Must
  also be a fixed count per unit** (e.g. exactly 3, or `round(N_u/8)`), not a
  threshold, so Phase 2's per-unit output length `M + retained·K` stays predictable
  and §5.6's arch refactor is only about *layout*, not *length*. Note the tension:
  the segmenter cuts at high `diff`, so segments are internally low-motion by
  construction — the retained signal should be *shortest segments* (where the
  segmenter was forced to cut often = dense span), not per-segment motion.
- **Learned segmenter** — Phase 1 keeps the fixed-count `diff` rule; a learned
  causal boundary head is a later option (would have to keep the fixed output
  count).
- **`mamba_ssm` varlen packed scan** vs. the current per-unit Python loop
  (throughput only; same math).
- **Streaming budget controller** (`merge` of adjacent oldest unit states to hold
  U ≤ 5 on an unbounded stream) — deferred; needed for real streaming inference,
  not for the pretrain.

---

## 8. Build order

1. **Code (no GPU):** items 1–3, 7–9, 11 of §5 — Option A/B on the readout, trainer
   distr-loss fix, decorrelation loss, fold-depth randomization, `time_embed` swap,
   K/M knob, LR groups.
2. **Code:** item 4 (fixed-count adaptive segmenter + `output_len_for`) + item 5
   (chunked encoder forward). Item 4's fixed `N = ⌊T/4⌋+1` means Phase 1 does
   **not** wait on the §5.6 arch refactor — collator and model agree on the length.
3. **Phase 1 run** on short InternVid → gate on the §2.3 probe (`struct_ρ`,
   `xcos_cen`) + frozen-LLM caption eval.
4. **Code:** item 6 (retained subset + `prepare_inputs_labels_for_multimodal`
   refactor to consume model-returned unit structure + tokenised `Time:`).
5. **Phase 2 run** → gate (`struct_ρ`, `cc_frac` back off 1.0, caption plateau,
   budget ablation).
6. **Phase 3 run** + long-video needle / frame-order probes + length curriculum.

---

## Load-bearing facts (unchanged)

- **LLM cross-entropy only.** No reconstruction / MSE / masked-feature objective —
  every such variant collapses the bottleneck to be perfectly order-invariant
  (`cos(real, shuffled) = 1.0000`). Option A (affine mean/std match), Option B
  (CORAL centroid + covariance + norm), and the VICReg variance/covariance
  decorrelation term operate on **distribution moments**, not per-token targets, and
  are allowed.
- **No pre-extracted vision-feature cache** anywhere. Frames decoded + encoded on
  the fly every step; encoder runs in ≤ 8-frame chunks (no cross-frame attention).
- **Gate on the grounding probe** — centered cross-video cosine (`xcos_cen`) and
  `struct_ρ` vs the encoder — **not** raw `content_collapse`, which is dominated by
  an inherent SigLIP-NaViT common component (`cc_frac ≈ 0.94`, `content_collapse ≈
  0.87` on the raw encoder itself).
- **LLM frozen every phase; `mm_projector` unfrozen.** Every checkpoint is also
  evaluated as *compressor + the original frozen LLM* against a no-compression
  baseline, so a gain is attributable to compression and not to anything downstream.
