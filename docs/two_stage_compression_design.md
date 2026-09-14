# Dynamic video compression — joint InternVid pretrain (Plan X)

**Status:** design agreed. This document **supersedes** the earlier plan (kept in
git history): the qbase pretrained on whole-video windows against a frozen LLM, then
an SSD fold bolted on and trained separately on videoxl caption sets, then the qbase
unfrozen. That plan *ran to completion* and its fold readout **collapsed
cross-video** — worse caption quality than the qbase alone despite 2× the tokens.
§2 has the measured post-mortem; the new plan is built to not repeat it.

Training-phase vocabulary is **Phase 1 / 2 / 3** (§4). The `stage1` / `stage2` names
that remain below are load-bearing *code* symbols (`TwoStageCompressor.stage1` /
`.stage2`, the `stage2_*` config fields) — the trainer's save filter and per-group
LR logic key on them — not training phases. The per-group LR CLI flags are named
for the modules they address: `--qbase_lr` (was `--stage1_lr`) and `--mamba_lr`
(overrides `--compressor_lr` for the fold group when set).

**What changed vs. the superseded plan**

| | superseded | Plan X |
|---|---|---|
| qbase / fold trained | separately (qbase frozen while fold trains) | **jointly**, from Phase 2 on |
| pretrain data | videoxl caption sets (~73k) | **InternVid** (`anno_data/internVid.json`, 977,423; Qwen3-VL re-captions) |
| video length | ≤160 f, caps most clips | **length-bucketed curriculum**, fps=1, real depth |
| qbase segments | whole-video window, or uniform 4-frame (superseded runs) | **fixed count `⌊T/4⌋+1`, adaptively placed** 1–8 frame segments (forced-8 + top-`diff`) |
| LLM | frozen | frozen (unchanged) |
| `mm_projector` | frozen | **unfrozen**; Phase 2 on gives the qbase-only replay stream and the fold readout their own copy each (`mm_projector_qbase`/`mm_projector_fold`, both deepcopied from the shared one at build time) instead of forcing one projector to track both moving manifolds |
| fold-readout regularization | none | **Option A + Option B** |
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
Phase 2:  [ Time:{a}s-{b}s: ] <|compression_start|>  unit u's M readout tokens  <|compression_end|>
Phase 3:  [ Time:{a}s-{b}s: ] <|compression_start|>
              interleave(  retained segments' K tokens ,  unit u's M readout tokens  )  # by RoPE slot
          <|compression_end|>
```

- **Retained set — Phase 3 only.** A small **no-gradient** budget of segments per
  unit (`r_u`, `≤ 3/unit`) whose raw qbase `K` tokens are emitted **in addition to**
  the `M` readout tokens (the retained segments are still folded — this is an extra
  verbatim copy, not a hold-out). It is the ∞-former / compressive-transformer
  escape hatch for detail a finite SSD state floods on (§3), and it is introduced in
  **Phase 3** where that flooding is real. **Phase 2 runs `r_u ≡ 0`**: a moderate
  `E[r_u]` with `M = K` lets the raw-qbase tokens outnumber the fold output and the
  LLM learns to lean on them instead of the fold. The `compress_windows` selection +
  interleave stay in the code guarded behind `r_u > 0`. Phase 2 keeps the raw-qbase
  path alive a different way — a **pure-qbase replay** stream (§4).
- **Vision-token cost to the LLM** = `U·M` in Phase 2 (`≤ 5·64 = 320`), `U·M + (Σ
  retained)·K ≤ 1280` in Phase 3 — bounded and predictable for any video length.
- **RoPE-index footprint — sparsified readout.** The compressed tokens are **not**
  given consecutive `position_ids`. Readout token `m` of unit `u` is placed at
  stride `N_u·K / M` (the unit's compression ratio; `= N_u` at `K = M = 64`), so
  unit `u` spans `N_u·K` position slots and the whole compressed region spans `N·K`
  — its uncompressed qbase footprint — regardless of how many tokens it actually
  holds. Phase 1 and Phase 2 then place the video on an identical positional scale.
  (Approximation: the region also carries the per-unit `Time:` / `<cs>` / `<ce>`
  text at stride 1, so its true footprint is `Σ N_u·K` + those few text tokens.)
  In Phase 3 a retained segment's `K` tokens sit at stride 1 on the slots of their
  real qbase offset, and `compress_windows` **emits the unit's rows in ascending
  slot order** (retained-K physically interleaved among the readout rows), with
  `pos_offsets` carrying that same order. The frozen LLM reads relative distance
  from RoPE; `Time:{a}s-{b}s:` is only a discrete hint. Same formula at train and
  inference so streaming readout positions track real elapsed time. Scale for very
  long clips: §7.

**Phase 1 is the degenerate case:** no fold, no units, retained = *every* segment ⇒
the LLM sees `N·K` tokens (qbase per semantic segment, straight to the projector).
Phase 2's **pure-qbase replay** stream (§4) reuses exactly this path.

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
  **retained set** (§1) is the answer — introduced in **Phase 3**, where clips are
  long enough for the flooding to bite; Phase 2 runs `r_u ≡ 0`.

### Vocabulary

| sym | meaning | set by | typ. |
|---|---|---|---|
| **T** | decoded frames (fps = 1 ⇒ ≈ seconds) | `load_video`, `--max_frames` | ~180–320 (Ph2) |
| **N** | **segments** the video is cut into (qbase does not change this — it changes *tokens per segment*) | `⌊T/4⌋+1` (fixed count, adaptive placement); each seg 1–8 f | ~46–81 (Ph2) |
| **K** | tokens qbase emits **per segment** — fixed | `num_queries` (pretrained) | 64 |
| **M** | tokens **one** fold readout emits **per unit** | `stage2_n_summary_tokens` | 64 |
| **U** | readout units (independent folds), **capped** | `round(N / N̄_u)`, clamp `[1, min(stage2_max_units, ⌊N/4⌋)]` | ≤ 5 |
| **N_u** | segments in unit `u` = that fold's depth; `Σ_u N_u = N` | boundary placement (§4 Phase 2), grows with length at inference | soft-bounded by `min_gap ≈ N/(2U)`; no hard `[4,16]` clamp |
| **r_u** | retained raw-K segments in unit `u` — **Phase 3 only** (`≡ 0` in Phase 2) | no-gradient heuristic, per-unit draw | 0–3 |
| — | frames per segment | segmenter, clamp **[1, 8]** | ~1–8 |
| — | vision tokens to the LLM | `U·M` (Phase 2) / `Σ_u (M + r_u·K)` (Phase 3) | ≤ 320 / ≤ 1280 |
| — | `position_ids` slots the compressed region spans (RoPE footprint) | readout stride `N_u·K/M` (+ retained-K stride 1 in Phase 3) | `≈ N·K` (= Phase-1 footprint; + the per-unit `Time:`/`<cs>`/`<ce>` text tokens) |

Chain (fps = 1, **Phase 2**): `T frames → N = ⌊T/4⌋+1 segments → N·K qbase tokens →
U units (Σ N_u = N) → per unit: fold N_u segments → M readout tokens`. Worked
example, T = 240: N = 61; draw `N̄_u = 13` ⇒ U = 5; partition e.g.
`N_u = [13,12,13,11,12]`; the LLM sees `5·64 = 320` compressed tokens across
`≈ N·K = 3904` `position_ids` slots. **Phase 3** adds `r_u` retained segments' raw
`K` tokens per unit (e.g. `r_u = [1,0,2,1,0]` ⇒ `320 + 4·64 = 576` tokens).

---

## 4. Training curriculum — 3 phases, **LLM frozen throughout**

**Common to every phase**

- Data: InternVid, `anno_data/internVid.json` (977,423 clips), Qwen3-VL-8B
  re-captions (`recaption/qwen3vl/`). fps = 1 always. Dynamic HW (native aspect
  ratio, per-video token budget; `--force_image_size` **not** set). On-the-fly
  decode + frozen-encoder forward, **no feature cache**.
- Single CE forward, `B == 1` flattened collator, `use_dual_forward=False`.
  `zero1` for Phase 1, `zero2` from Phase 2 (qbase + fold + projector trainable).
- **`mm_projector` unfrozen** every phase. Phase 1 has one compressed-token
  manifold (qbase), so it keeps the single shared `mm_projector`. Phase 2 on has
  two independently-moving manifolds (qbase-only replay, fold readout) sharing
  the vision-encoder scale but not the same distribution, so `TwoStageCompressor`
  models get `mm_projector_qbase` / `mm_projector_fold` instead — both start as a
  deepcopy of the shared projector (`Videollama3MetaModel._maybe_build_split_projectors`,
  `compress_visual_tokens_with_compressor`'s per-row routing in `videollama3_arch.py`)
  and are free to diverge; a raw/uncompressed row (partial-window callers) still
  uses the plain shared `mm_projector`. **LLM frozen** every phase.
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
  `prepare_inputs_labels_for_multimodal` refactor (§5 item 5) and no dynamic-length
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
    Gumbel-top-k over the **z-normalised** `diff` (`z = (diff − mean) / std`,
    scores `z / τ + gumbel`, `τ ≈ 0.5`), active only in `.train()` (deterministic
    top-`diff` at eval) — so the adaptive cuts move each epoch while `N` stays
    exact. (Z-normalise because raw `diff ∈ ~[0.01, 0.2]`; `diff / τ` alone is
    swamped by the Gumbel noise.) This replaces the earlier per-sample τ-jitter
    idea (which would have made `N` unpredictable).
  - **Non-learned** — no new parameters, so qbase is the only thing training.
  - Alternatives to ablate (all keep the forced-8 budget + `[1,8]` clamp): `diff`
    on common-component-removed features (`f_i − mean_i f_i`; measured ≈ no change
    on InternVid, keep as a fallback for low-motion domains); RGB shot boundary
    (PySceneDetect / TransNetV2) as the non-forced score.
- **Path:** frozen encoder forward (single-shot for now; §5 item 4 chunking still
  pending, fine at Phase-1 lengths) → segmenter → qbase per segment → K = 64
  tokens/segment → **all** segments' tokens → frozen `mm_projector` → frozen LLM → CE.
- **Trainable:** qbase + `mm_projector`. Warm-start qbase from
  `pretrained_models/compressor_pretrain_video_norm` (147 tensors, clean load;
  `match_encoder_scale=True`, `distr_loss_weight=0.05`). Keep A + B on the qbase
  output.
- **Token math:** `N = ⌊T/4⌋ + 1`, so a 60–180 s clip ⇒ ~16–46 segments ⇒
  ~1.0k–2.9k tokens to the LLM — well inside `model_max_length`, and already a
  large cut vs the uncompressed `T×HW` (~16×).
- **Gate to Phase 2:** on a held-out probe (`eval_ablation/struct_probe.py`) —
  qbase `struct_ρ` vs encoder ≥ ~0.95; `xcos_cen` ≈ encoder's; `cc_frac` ≈
  encoder's (not pinned to 1.0); caption eval (**compressor + original frozen
  LLM**) not worse than the whole-video qbase baseline.

### Phase 2 — add the fold; qbase + fold streams; **+ 180–420 s**

- **Data:** 180–420 s (~273k, Qwen3-VL re-captions complete) **+ 30 % replay** of
  the Phase-1 `< 180 s` pool. The replay is not only short-video retention — it is
  where the **shallow-fold** training signal comes from (a 60 s replay clip has
  `N ≈ 16` ⇒ `U = 1–2`, `N_u ≈ 8–16`; a 30 s clip `N ≈ 8`).
- **Pure-qbase replay** (new, separate meta-JSON entry with `"qbase_only": true`).
  A small slice (~5–10 % of steps) of short clips routed **straight through
  stage-1** — no unit split, no fold: the model emits the `N·K` qbase tokens to the
  projector in the Phase-1 layout (stride-1 slots). Retained-K is off in Phase 2
  (see §1), so this is what keeps the raw-qbase → projector → LLM path exercised:
  the unfrozen `mm_projector` (and, once unfrozen, the qbase) must not drift onto
  the fold-readout manifold and forget how to read a plain qbase sequence. The
  30 % fold-replay above still supplies the shallow-fold signal — the two replay
  streams are separate.
- **Working frame cap ≈ 320.** With `U ≤ 5` and `N_u` typically `≲ 16` a training
  sample folds `⌊T/4⌋+1 ≲ 80` segments ⇒ `T ≲ 320`. So the 180–420 s bucket is
  either frame-capped (`--max_frames ~320`) or a contiguous sub-span is taken per
  epoch. (Open decision §7 — the variance design below assumes ≲ 320.)
- **New machinery:** unit grouping (U ≤ 5 contiguous), SSD fold per unit (fresh
  `init_state`), readout M = 64, the `qbase_only` passthrough branch in
  `compress_windows`. (Retained-subset selection + interleave stay in the code but
  are inert — Phase 3.)
- **time_embed:** **not** absolute `index_sincos`. Use per-segment
  `(gap_seconds, duration_seconds)` → `time_mlp` (`rel_gap_mlp`), or `none` for the
  first runs (lean on the SSD's implicit ordering).

#### Compression-variance training (the anti-collapse regularizer)

The fold is trained to be a good summariser at **any operating point it will meet
at inference** — `(fold depth N_u, unit size / count U)` — by randomising both
(`r_u` is Phase 3). Mechanism (as built — **not** the Phase-1 "pure function of the
frame count" rule):

- `compression_parts` is **one whole-video window**; `Phase2FoldDataset` draws only
  `U` (via a depth-class `N̄_u`) and a `compression_seed`, both seeded by
  `(epoch, dataset_index)`.
- The **model** does the segment cut *and* the `U−1` unit-boundary placement
  itself, under a **seeded** Gumbel keyed on `compression_seed` (`compress_windows`
  / `stage1.forward`), and returns the realised `unit_meta` (per-unit `n_out`,
  `pos_offsets`, `unit_span`). The arch **conforms** to `unit_meta` — it does not
  independently recompute a token count.
- Reproducible on resume via the seed; a clip gets a new partition each epoch
  because `_epoch` feeds the seed. `_epoch` / `_progress` are written by a
  `TrainerCallback` in the main process; `Phase2FoldDataset` backs them with
  fork-shared `multiprocessing.Value`s so `dataloader_num_workers > 0` /
  `dataloader_persistent_workers` still see every write (a plain attribute does
  not — that silently froze the seed + the variance curriculum). Assumes the
  `fork` start method (Linux / torch default).

- **Homogeneity axis = `max_u N_u` only.** The SSD recurrence runs `N_u·K` steps,
  so backward-graph depth / gradient scale ∝ the deepest fold in the batch; that
  **should** be ~minibatch-homogeneous. `U` and the unit boundaries do **not**
  touch recurrence depth. NOTE: `N_u` is set by boundary placement and is only
  soft-bounded (`min_gap ≈ N/(2U)`); the `[4,16]` clamp in the vocab table is
  **not** enforced in `_place_unit_boundaries`, so `max_u N_u` can still vary
  within a depth-class megabatch — tighten the sampler grouping or add the clamp if
  gradient-scale variance shows up.

- **`U` via depth-class bucketing.** Give each clip a class from its total segment
  count `N` (`= ⌊T/4⌋+1`, known from the annotation, no decode); a grad-accum
  window is filled from **one class**. A mean segments-per-unit `N̄_u` is drawn from
  the class's *narrow* range and `U = round(N / N̄_u)`, clamped to
  `[1, min(stage2_max_units, ⌊N/4⌋)]`:

  | class | `N` range | `N̄_u` draw |
  |---|---|---|
  | shallow | `N < 23` | `U[4, 8]` |
  | mid | `23 ≤ N < 46` | `U[8, 12]` |
  | deep | `N ≥ 46` | `U[12, 16]` |

  Reuse `LengthGroupedSampler` with `lengths = N` (`--group_by_compression_depth`,
  needs `durations_json`); duration-homogeneity then comes for free.

- **Unit boundaries: content-aware + Gumbel jitter.** Place the `U − 1` cuts at the
  largest **inter-segment** `diff`, computed on the **stage-1 qbase segment
  tokens** — each segment's `K` qbase tokens mean-pooled to one vector — **not** the
  raw frozen-encoder feature. The fold consumes qbase tokens, so units are grouped
  by what the fold will actually see; the raw encoder feature's ~0.94 common
  component would dominate the cosine. z-normalised, Gumbel-top-k jitter (`τ ≈ 0.5`,
  seeded, so cuts move each epoch), `min_gap = max(1, ⌊N/(2U)⌋)` spacing,
  uniform-split fallback when `U−1` valid cuts can't be placed. Units are then
  semantic chunks (a scene / an activity). `Σ N_u = N` exact.

- **Retained `r_u` — deferred to Phase 3.** Every Phase-2 unit emits exactly `M`
  readout tokens (`n_out = M`). `Phase2FoldDataset` passes `r_u ≡ 0`;
  `compress_windows` keeps the `r_u`-shortest-segment pick + slot interleave behind
  an `r_u > 0` guard. Rationale in §1: with `M = K` a moderate `E[r_u]` lets the
  raw-qbase tokens outnumber the fold output and the LLM leans on them. Phase 2
  keeps the raw-qbase path alive via the pure-qbase replay stream instead.

- **Fixed for the first runs — one variance family at a time.**
  `segment_target_frames = 4` (no `k` jitter — the segment *count* `N` stays a pure
  function of the frame count; only the segment *positions* move), `M = 64`,
  fps = 1, dynamic HW. No frame-drop augmentation. **Never shuffle segment order**
  (trains order-invariance — the banned failure mode).

- **Segment-position jitter: ON.** `Phase2ModelArguments` defaults
  `segment_sample_tau = 0.5` (the Phase-1 qbase pretrain leaves it 0). Both the
  segment cut *and* the `U−1` unit boundaries are re-drawn each epoch by a
  Gumbel-top-k keyed on `compression_seed` — a clip meets a new partition every
  epoch, which is the whole point of the fold-variance thesis (§ *the thesis* is in
  the memory / `docs` intro). `compress_windows` gates the jitter on **this
  wrapper's** train state + `segment_sample_tau` (not `stage1.training`), so
  freezing the qbase (`--qbase_lr 0`) no longer silently kills the segment
  re-draw. Set
  `--segment_sample_tau 0` for a fixed-partition ablation — it now freezes the
  segment cut *and* the `U−1` unit boundaries.

- **Variance curriculum.** During the cold-fold window (`variance_cold_frac`,
  default first 15 %): `U = min(3, ⌊N/4⌋)`. After it: `U = round(N / N̄_u)` with
  `N̄_u` from the full depth-class range. The cold window narrows **`U` only** — the
  unit boundaries are still Gumbel-jittered on it (memory-thesis: even the cold
  window's cuts land on `diff` peaks). The old `r_u` ramp is gone with retained;
  `variance_ramp_frac` was removed. Do not hit a cold module with max entropy on
  step 1.

#### Readout regularization

Two defenses, co-equal: the scale/distribution match (below) and the
compression-variance training (above). §2.1's collapse was mis-scale + `final_norm`
pinning + no variance; each half covers what the other cannot.

- **Option A** (`_match_encoder_scale`) with `ref` = **that unit's** encoder tokens
  (`win[fstart[a]·hw : fstart[b]·hw]` in `compress_windows`, per-unit — not the
  global `kv`), applied **after** the fold's `output_proj`.
- **Option B** (`_distribution_match_loss`), wired into the trainer for the fold
  (`_last_distr_loss` on the `TwoStageCompressor` wrapper, summed with stage-1's
  term).
- `final_norm` before `output_proj` made configurable (RMSNorm / LayerNorm /
  scale-only) — RMSNorm is what pinned `cc_frac → 1.0` in the superseded run;
  Option A after `output_proj` should undo it, keep the knob to ablate.
- **No separate fold projector.** The fold works in `stage2_d_model` (default
  1024, a bottleneck below the compressor hidden 1152); `output_proj` decodes the
  M readout tokens back to 1152 → the *one* frozen-then-unfrozen `mm_projector`
  the qbase tokens also use (Option A/B align them). The SSM **state** stays its
  own `(nheads, headdim, d_state)` object — distinct from the readout, which is
  `C·state` decoded, not the state itself.
- `weight_decay 0.03–0.05` (standard `--weight_decay`); fold `dropout` via
  `--stage2_dropout` (`Phase2ModelArguments`, default `0.1`, threaded through
  `_build_phase2_token_compressor_config`).
- **No cross-sample / VICReg decorrelation loss.** (An earlier plan added a
  variance/covariance term on the readout to defend `struct_ρ` — §2.1 shows A + B
  alone do not. Dropped: `struct_ρ` is now a hard **gate** metric only, and the
  compression-variance training + `rel_gap_mlp` time embed are the structural
  defenses. If the Phase-2 gate fails on `struct_ρ`, revisit this.)

#### Trainable + gate

- **Trainable:** qbase + fold + `mm_projector`, LLM frozen. A single knob,
  `--qbase_lr`, controls the qbase: **`> 0` trains it** (its own optimizer group
  at that LR, jointly with the fold from step 1 — the shell wrapper defaults it
  to `$MAMBA_LR`); **`<= 0` freezes it** (`requires_grad=False`; there is no
  separate freeze flag). Launch must set `--mm_projector_lr > 0` (unfreeze alone
  ≠ an optimizer group). **The cold-fold qbase stagger is NOT automatic:** to
  stagger you run the first ~15 % with `--qbase_lr 0` then resume with
  `--qbase_lr <small>` (mamba/fold LR 3–10× the qbase LR, set via `--mamba_lr`;
  `create_optimizer` has the qbase/rest split, active only when `--qbase_lr > 0`).
  The only progress-based auto-anneal is the variance curriculum's
  `variance_cold_frac` window.
- **Gate to Phase 3 — checked at fixed operating points**, not one: `N̄_u ∈
  {shallow, mid, deep} × U ∈ {2, 3, 5}` (`r_u = 0` — retained is off in Phase 2).
  At **every** point: fold `struct_ρ` ≥ ~0.9 × qbase's; `xcos_cen` ≈ encoder's;
  **`cc_frac` off 1.0 (near ~0.94)**. Plus: caption metrics plateaued; the U-cap
  budget ablation (capped vs. uncapped) shows the cap is nearly free; and the
  pure-qbase replay caption score holds vs. the Phase-1 qbase baseline (projector
  not drifted). **NOTE:** `struct_probe.py` currently only drives the whole-video
  single-fold path (`retained_counts=None`) — the `--n-units / --segs-per-unit`
  sweep is not yet implemented.

### Phase 3 — deep folds; **420–1200 s**

- **Data:** 420–1200 s (~451k) + layered replay (≈ 60 % long / 25 % mid / 15 %
  short). Drop the 98 clips > 1200 s.
- **Re-enable the retained set here.** Have `Phase2FoldDataset` draw `r_u` again
  (`~{0,1,2,3}`, weights `≈ [0.15, 0.35, 0.35, 0.15]`; the `compress_windows`
  `r_u > 0` path + slot interleave are already there); `r_u`-shortest segments,
  tie-break on `seg_diff`, Gumbel over the shortest `2·r_u`. This is where a finite
  fold state actually floods (long uniformly-high-action clips). Watch the
  retained-vs-fold token ratio (`M = K` ⇒ each retained segment costs a whole
  unit's readout) — consider `K > M` or a lower `E[r_u]` if the LLM leans on
  retained-K.
- **Trainable:** same as Phase 2. Deeper folds. **TBPTT** over a window of recent
  segments once the offline `[N·K ; M]` sequence stops fitting (older segments
  detached, forward streamed).
- **Length curriculum inside the phase** — grow the max frame count gradually; do
  not start at 1200 s.
- **Gate:** needle / frame-order probes on long video; frozen-LLM caption eval
  holds; `struct_ρ` holds at depth.

---

## 5. Implementation requirements (code changes)

| # | change | where | GPU? |
|---|---|---|---|
| 1 | **Option A/B on the fold readout** — after `SegmentAggregator.output_proj`, apply `_match_encoder_scale` (ref = **that unit's** encoder-token slice `win[fstart[a]·hw : fstart[b]·hw]`, per-unit) and stash `_last_distr_loss` from `_distribution_match_loss` on the `TwoStageCompressor` wrapper. **Done.** | `compressor.py::TwoStageCompressor.compress_windows` | no |
| 2 | **Trainer wiring fix** — `_compressor_distr_loss` does `getattr(comp, "distr_loss_weight", 0.0)` on the `TwoStageCompressor` wrapper → 0. Sum `_last_distr_loss` from `.stage1` **and** `.stage2`. | `videollama3_trainer.py` | no |
| 3 | **Fixed-count adaptive causal segmenter** — model-side, non-learned. `N = ⌊T/4⌋+1` from the frame count (collator + model compute it identically ⇒ no dynamic-length plumbing, no §5 item 5 refactor for Phase 1). Forced boundary every 8 f + remaining budget to top-`diff` positions on the per-frame encoder mean feature; Gumbel-top-k draw (z-normalised `diff`, `τ≈0.5`) for per-epoch augmentation; `[1,8]` clamp (`--segment_force_every`). Validated: `eval_ablation/segmenter_validate.py` (+ `_hist.py`). Implemented: `compressor.py::adaptive_segment_count` / `adaptive_segment_lengths`, `TransformerDecoderFlatCompressor` `output_len_for` + `forward` (`--adaptive_segmentation`); `videollama3_arch.py::_compressed_len` uses `output_len_for` for the placeholder count. | `compressor.py`, `videollama3_arch.py` | no |
| 4 | **Chunked encoder forward** — ≤ 8-frame groups under `no_grad` (encoder frozen, no cross-frame attention). Required at Phase-3 lengths; **not yet implemented** (Phase 1 runs the single-shot forward). | `videollama3_arch.py::encode_images` | no |
| 5 | **Retained-subset selection** — **Phase 3 only.** Code present in `compress_windows` (guarded `r_u > 0`: `r_u`-shortest segments + Gumbel over the shortest `2·r_u`, tie-break `seg_diff`, rows emitted in ascending slot order). Phase 2's `Phase2FoldDataset` passes `r_u ≡ 0`; re-enable the draw for Phase 3 (§4 Phase 3). | `compressor.py::compress_windows`, `phase2_pretrain_fold.py` dataset | no |
| 6 | **Compression-variance training** (§4 Phase 2) — **Done.** `Phase2FoldDataset` draws, per `(epoch, index)` seed: depth class from `N` → `N̄_u` from the class's narrow range → `U = round(N/N̄_u)` (cold window: `U = min(3, ⌊N/4⌋)`). Segment + `U−1` unit boundaries are placed **model-side** under a seeded Gumbel (`compression_seed`), on the **qbase segment tokens'** inter-segment cosine. `r_u` deferred to Phase 3. Depth-class **grouped sampler** (`LengthGroupedSampler` on `N`, `--group_by_compression_depth` + `durations_json`; `SubsetWithLengths` forwards `compression_depths` so it survives a val split). `_epoch` / `_progress` are fork-shared `mp.Value`s so the seed + curriculum reach dataloader workers. `variance_ramp_frac` removed. | `Phase2FoldDataset`, `videollama3_trainer.py` sampler, `train/data/compressor.py` | no |
| 7 | **time_embed** — `rel_gap_mlp` (per-segment gap + duration seconds, built model-side from the adaptive cut) or `none` (Phase-2 first-runs default); `index_sincos` stays in the code but not for the fold. **Done.** | `segment_aggregator.py` config, `compress_windows` | no |
| 8 | **K/M knob** — default K = M = 64. Decoupling (K > M, e.g. 128/64) only needs `TwoStageCompressor.__init__` to stop default-tying `n_summary_tokens` to `tokens_per_segment` (`SegmentAggregatorConfig` already has them separate). Open decision §6. | `compressor.py` | no |
| 9 | **`struct_ρ` / `xcos_cen` probe** — centered cross-video cosine + encoder-similarity Spearman + `cc_frac`, on the pre-projector compressor output vs. the raw encoder. Implemented: standalone `eval_ablation/struct_probe.py` (whole-video part, works on the Phase-1 flat-adaptive checkpoint) and mirrored into `grounding_probe.py`'s aggregate/verdict/table. | `eval_ablation/` | yes (eval) |
| 10 | **LR groups** — `create_optimizer` `stage1`/`stage2` split (present) + `mm_projector` group (present). No change, just enable. | — | — |
| 11 | **Strided `position_ids` for the compressed region** (§1) — **Done.** `compress_windows` returns per-output-token slot offsets relative to the unit start: readout token `m` at `round(m·N_u·K/M)` (Phase 3 also: a retained segment's `K` tokens at their real qbase offset, stride 1, rows emitted in ascending slot order). The arch `position_ids` loop advances `cur` by `unit_span` (`= N_u·K`) per unit and stays stride-1 elsewhere, so the compressed span ≈ `N·K` slots (+ the per-unit `Time:`/`<cs>`/`<ce>` text). Sample-start detection (`position_id == 0`) unaffected. Identical formula train/inference. Phase-3 scale knob (`stage2_rope_slot_scale`): §7. | `compressor.py::compress_windows`, `videollama3_arch.py::prepare_inputs_labels_for_multimodal` | no |
| 12 | **qbase-only replay path** — meta-JSON `"qbase_only": true` → `Phase2FoldDataset` emits `compression_qbase_only=[True]`; `compress_windows` returns stage-1's `N·K` qbase tokens as one `unit_meta` entry (stride-1 slots, no unit split, no `SegmentAggregator` call). Threaded `dataset → DataCollatorWithCompressor → prepare_inputs_labels_for_multimodal → encode_images → compress_visual_tokens_with_compressor → compress_windows`. Keeps `mm_projector` / an unfrozen qbase anchored to the raw-qbase manifold. **Done.** | `compressor.py`, `videollama3_arch.py`, `videollama3_qwen2.py`, `train/data/{compressor,global_compressor}.py`, `phase2_pretrain_fold.py` | no |

**Keep, do not touch:** no feature cache; CE only; **no reconstruction / MSE /
masked-feature** (collapses to order-invariant — Option A/B are distribution
moments, not per-token targets, and are allowed);
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

- **K = M vs K > M.** Default K = M = 64 (matches the pretrained qbase). K > M
  (e.g. 128 / 64) makes qbase genuinely higher-resolution and the fold a real 2:1
  reduction. With retained re-enabled in Phase 3 this also matters for the
  **retained-vs-fold token ratio** — at `M = K` one retained segment costs a whole
  unit's readout, so `K > M` (or a low `E[r_u]`) stops retained-K from dominating
  the compressed sequence.
- **Segmenter criterion — RESOLVED (§4 Phase 1, measured):** fixed count
  `N = ⌊T/4⌋+1`, forced boundary every 8 f + remaining budget to top-`diff`
  positions (Gumbel-top-k for augmentation). Bare threshold-τ / bare top-k
  rejected. Still ablatable: the `diff` score (raw vs common-component-removed vs
  RGB shot boundary) as the *non-forced* ranking.
- **Retained-subset — DEFERRED to Phase 3.** Phase 2 runs `r_u ≡ 0` (see §1: with
  `M = K` a moderate `E[r_u]` lets raw-qbase tokens dominate). Heuristic when
  re-enabled: per-unit draw `r_u ∈ {0,1,2,3}` (weights `≈ [.15,.35,.35,.15]`),
  segments = the `r_u` **shortest by frame count** + Gumbel over the shortest
  `2·r_u`, tie-break `seg_diff`. Still ablatable: the distribution shape, the
  `clip(round(N_u/6),0,3)` alternative, and `K > M` to shrink the retained cost.
- **Unit-boundary feature — RESOLVED (§4 Phase 2):** inter-segment cosine on the
  **stage-1 qbase segment tokens** (mean-pooled `K` per segment), not the raw
  frozen-encoder feature. Still ablatable: raw-encoder feature;
  common-component-removed encoder feature; RGB shot boundary.
- **RoPE-index scale for the compressed region — OPEN (§1, §5 item 11).** Default:
  readout stride `N_u·K / M` ⇒ region footprint `≈ N·K` slots (`≈ 16·T`; matches
  the Phase-1 all-qbase footprint, and — with retained off in Phase 2 — the
  readout-only set `{m·N_u}` is exactly collision-free at `K = M = 64`). Phase 2
  (`T ≲ 320` ⇒ `N·K ≲ 5.2k`) is fine on the default. Phase 3's 420–1200 s clips
  push `N·K` to ~19k slots + surrounding text → near Qwen2's 32k; fallback is a
  seconds scale (`stage2_rope_slot_scale = S` slots/s, `S ≈ 4–8`) accepting
  coincident slots (retained-K, once on, can already share a readout slot).
- **Phase-2 frame cap — OPEN.** `U ≤ 5 × N_u ≤ 16 ⇒ ≤ 80 segments ⇒ T ≲ 320`, but
  the mid bucket is 180–420 s. Pick: (a) `--max_frames ~320` (drops the tail of
  each long clip), (b) a per-epoch contiguous sub-span of the full clip, (c) raise
  `N_u` / `U` caps for long clips only. The §4 variance design assumes ≲ 320;
  leaning (b) so long clips still contribute their later content across epochs.
- **Learned segmenter** — Phase 1 keeps the fixed-count `diff` rule; a learned
  causal boundary head is a later option (would have to keep the fixed output
  count). Same question for the **unit** boundaries (§4 Phase 2 uses a non-learned
  `diff`-at-segment-granularity rule).
- **`mamba_ssm` varlen packed scan** vs. the current per-unit Python loop
  (throughput only; same math).
- **Streaming budget controller** (`merge` of adjacent oldest unit states to hold
  U ≤ 5 on an unbounded stream) — deferred; needed for real streaming inference,
  not for the pretrain.

---

## 8. Build order

1. **Code (no GPU):** items 1–2, 6–8, 10 of §5 — Option A/B on the readout
   (after `output_proj`, + configurable `final_norm`), trainer distr-loss fix,
   compression-variance training (`Phase2FoldDataset` draws + depth-class grouped
   sampler), `time_embed` swap, K/M knob, LR groups.
   Then **Phase 1 Phase-1→2 gate is already met** (§ analysis: `struct_ρ ≈ 1.0`,
   `xcos_cen = encoder`, `cc_frac ≈ encoder` on InternVid + VCG; caption better on
   InternVid, mixed on VCG — one out-of-domain hallucination, a qbase style-transfer
   artifact, carried into Phase 2).
2. **Code:** item 3 (fixed-count adaptive segmenter + `output_len_for`, **done**) +
   item 4 (chunked encoder forward, **still pending** — Phase 1 runs the single-shot
   forward). Item 3's fixed `N = ⌊T/4⌋+1` means Phase 1 does **not** wait on the
   §5 item 5 arch refactor — collator and model agree on the length.
3. **Phase 1 run** on short InternVid → gate on the §2.3 probe
   (`eval_ablation/struct_probe.py`: `struct_ρ`, `xcos_cen`, `cc_frac`) +
   frozen-LLM caption eval.
4. **Code (done):** item 11 (`prepare_inputs_labels_for_multimodal` consumes
   model-returned `unit_meta` + baked `Time:` + strided `position_ids`) + item 6
   (model-side seeded unit boundaries on the qbase segment tokens) + item 12
   (qbase-only replay path). Item 5 (retained subset) stays guarded / inert for
   Phase 3.
5. **Phase 2 run** → gate (`struct_ρ`, `cc_frac` back off 1.0, caption plateau,
   U-cap budget ablation, pure-qbase replay caption holds).
6. **Code:** re-enable the item 5 `r_u` draw for Phase 3. **Phase 3 run** +
   long-video needle / frame-order probes + length curriculum.

---

## Load-bearing facts (unchanged)

- **LLM cross-entropy only.** No reconstruction / MSE / masked-feature objective —
  every such variant collapses the bottleneck to be perfectly order-invariant
  (`cos(real, shuffled) = 1.0000`). Option A (affine mean/std match) and Option B
  (CORAL centroid + covariance + norm) operate on **distribution moments**, not
  per-token targets, and are allowed.
- **No pre-extracted vision-feature cache** anywhere. Frames decoded + encoded on
  the fly every step; the encoder has no cross-frame attention so it *can* run in
  ≤ 8-frame chunks (§5 item 4 — not yet wired; Phase 1 uses the single-shot forward).
- **Gate on the grounding probe** — centered cross-video cosine (`xcos_cen`) and
  `struct_ρ` vs the encoder — **not** raw `content_collapse`, which is dominated by
  an inherent SigLIP-NaViT common component (`cc_frac ≈ 0.94`, `content_collapse ≈
  0.87` on the raw encoder itself).
- **LLM frozen every phase; `mm_projector` unfrozen.** Every checkpoint is also
  evaluated as *compressor + the original frozen LLM* against a no-compression
  baseline, so a gain is attributable to compression and not to anything downstream.
