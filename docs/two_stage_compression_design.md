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

Per video, in time order, the LLM sees for each unit `u` (U ≤ 5 contiguous units),
the same format in every phase:

```
[ Time:{a}s-{b}s: ] <|compression_start|>  unit u's M readout tokens  <|compression_end|>
```

- **Vision-token cost to the LLM** = `U·M` (`≤ 5·64 = 320`) — bounded and
  predictable for any video length, the same formula every phase.
- **RoPE-index footprint — sparsified readout.** The compressed tokens are **not**
  given consecutive `position_ids`. Readout token `m` of unit `u` is placed at
  stride `N_u·K / M` (the unit's compression ratio; `= N_u` at `K = M = 64`), so
  unit `u` spans `N_u·K` position slots and the whole compressed region spans `N·K`
  — its uncompressed qbase footprint — regardless of how many tokens it actually
  holds. Every phase then places the video on an identical positional scale.
  (Approximation: the region also carries the per-unit `Time:` / `<cs>` / `<ce>`
  text at stride 1, so its true footprint is `Σ N_u·K` + those few text tokens.)
  `compress_windows` **emits each unit's readout rows in ascending slot order**,
  with `pos_offsets` carrying that order. The frozen LLM reads relative distance
  from RoPE; `Time:{a}s-{b}s:` is only a discrete hint. Same formula at train and
  inference so streaming readout positions track real elapsed time. Scale for very
  long clips: §7.

**Phase 1 is the degenerate case:** no fold, no units — every segment's qbase
tokens pass straight through ⇒ the LLM sees `N·K` tokens (qbase per semantic
segment, straight to the projector). Phase 2's **pure-qbase replay** stream (§4)
reuses exactly this path.

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
3. **The fold readout collapsed.** `cc_frac` went to exactly 1.000 and `struct_ρ`
   crashed (0.999 → 0.39 on videoxl) — the surviving signal no longer tracks which
   video is which.

### 2.2 Root cause

**Not a scale/regularization issue.** An earlier version of this section
attributed the collapse to fold-output mis-scale, an RMSNorm-pinning effect on
`final_norm`, CE giving too little gradient, and a handful of other training
hyperparameters — that explanation was wrong and has been removed (2026-09-19).
**The actual root cause: the qbase was pretrained on very sparse input.**
(Open item — the specifics of that sparsity, and exactly how it propagates into
the fold's cross-video collapse, still need writing up here.)

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
  **not** fix this (it carries no information a deep `append` would not). This is
  an **open problem for Phase 3** (an earlier draft's answer — a per-unit raw-qbase
  retained set — was tried and dropped, see §4 Phase 2; not to be reintroduced).

### Vocabulary

| sym | meaning | set by | typ. |
|---|---|---|---|
| **T** | decoded frames (fps = 1 ⇒ ≈ seconds) | `load_video`, `--max_frames` | ~180–320 (Ph2) |
| **N** | **segments** the video is cut into (qbase does not change this — it changes *tokens per segment*) | `⌊T/4⌋+1` (fixed count, adaptive placement); each seg 1–8 f | ~46–81 (Ph2) |
| **K** | tokens qbase emits **per segment** — fixed | `num_queries` (pretrained) | 64 |
| **M** | tokens **one** fold readout emits **per unit** | `stage2_n_summary_tokens` | 64 |
| **U** | readout units (independent folds), **capped** | `round(N / N̄_u)`, clamp `[1, min(stage2_max_units, ⌊N/4⌋)]` | ≤ 5 |
| **N_u** | segments in unit `u` = that fold's depth; `Σ_u N_u = N` | boundary placement (§4 Phase 2), grows with length at inference | soft-bounded by `min_gap ≈ N/(2U)`; no hard `[4,16]` clamp |
| — | frames per segment | segmenter, clamp **[1, 8]** | ~1–8 |
| — | vision tokens to the LLM | `U·M` (same formula every phase) | ≤ 320 |
| — | `position_ids` slots the compressed region spans (RoPE footprint) | readout stride `N_u·K/M` | `≈ N·K` (= Phase-1 footprint; + the per-unit `Time:`/`<cs>`/`<ce>` text tokens) |

Chain (fps = 1, **Phase 2**): `T frames → N = ⌊T/4⌋+1 segments → N·K qbase tokens →
U units (Σ N_u = N) → per unit: fold N_u segments → M readout tokens`. Worked
example, T = 240: N = 61; draw `N̄_u = 13` ⇒ U = 5; partition e.g.
`N_u = [13,12,13,11,12]`; the LLM sees `5·64 = 320` compressed tokens across
`≈ N·K = 3904` `position_ids` slots. Same chain, same formula in Phase 3.

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
  `prepare_inputs_labels_for_multimodal` refactor and no dynamic-length
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
  projector in the Phase-1 layout (stride-1 slots). This is what keeps the
  raw-qbase → projector → LLM path exercised: the unfrozen `mm_projector` (and,
  once unfrozen, the qbase) must not drift onto
  the fold-readout manifold and forget how to read a plain qbase sequence. The
  30 % fold-replay above still supplies the shallow-fold signal — the two replay
  streams are separate.
- **Working frame cap ≈ 320.** With `U ≤ 5` and `N_u` typically `≲ 16` a training
  sample folds `⌊T/4⌋+1 ≲ 80` segments ⇒ `T ≲ 320`. So the 180–420 s bucket is
  either frame-capped (`--max_frames ~320`) or a contiguous sub-span is taken per
  epoch. (Open decision §7 — the variance design below assumes ≲ 320.)
- **New machinery:** unit grouping (U ≤ 5 contiguous), SSD fold per unit (fresh
  `init_state`), readout M = 64, the `qbase_only` passthrough branch in
  `compress_windows`.
- **time_embed:** **not** absolute `index_sincos`. Use per-segment
  `(gap_seconds, duration_seconds)` → `time_mlp` (`rel_gap_mlp`), or `none` for the
  first runs (lean on the SSD's implicit ordering).

#### Compression-variance training (the anti-collapse regularizer)

The fold is trained to be a good summariser at **any operating point it will meet
at inference** — `(fold depth N_u, unit size / count U)` — by randomising both.
Mechanism (as built — **not** the Phase-1 "pure function of the
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

- **No retained set.** Every unit emits exactly `M` readout tokens (`n_out = M`) —
  the LLM never sees a segment's raw qbase `K` tokens once it is inside a unit. An
  earlier draft reserved a per-unit raw-qbase escape hatch (`r_u`) for Phase 3;
  dropped, not deferred: with `M = K` a moderate `E[r_u]` let the raw-qbase tokens
  outnumber the fold output and the LLM leaned on them instead of the fold summary,
  undermining the reason the fold exists. Do not reintroduce it. The raw-qbase path
  stays exercised a different way — the **pure-qbase replay** stream, a separate
  whole-video no-fold pathway (above), not a per-unit mechanism.

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
  window's cuts land on `diff` peaks). `variance_ramp_frac` was removed (there is
  no retained-set draw to ramp). Do not hit a cold module with max entropy on
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
  {shallow, mid, deep} × U ∈ {2, 3, 5}`. At **every** point: fold `struct_ρ` ≥ ~0.9
  × qbase's; `xcos_cen` ≈ encoder's; **`cc_frac` off 1.0 (near ~0.94)**. Plus:
  caption metrics plateaued; the U-cap budget ablation (capped vs. uncapped) shows
  the cap is nearly free; and the pure-qbase replay caption score holds vs. the
  Phase-1 qbase baseline (projector not drifted). **NOTE:** `struct_probe.py`
  currently only drives the whole-video single-fold path (`unit_counts=None`) —
  the `--n-units / --segs-per-unit` sweep is not yet implemented.

### Phase 3 — deep folds; **420–1200 s + budget-driven length-adaptive fold (draft, 2026-09-18)**

**Status: built + data ready 2026-09-20, run not started.** Items 4, 13, 14, 16,
17 are implemented and self-checked (see each row in §5); the entrypoint is
`videollama3/train/phase3_pretrain_budget.py` (launcher `shell/pretrain_phase3_budget.sh`),
a thin wrapper over `phase2_pretrain_fold.py` via the same injection hooks.
Supersedes nothing in Phase 1/2 — their fixed formulas (`N=⌊T/4⌋+1`, the `U≤5`
depth-class heuristic) are unchanged and still the default code path; this is
Phase 3's own recipe, warm-started from a Phase 2 checkpoint. Two deviations
from the draft below, both recorded where they bite: the DP is **deterministic**
(no unit-level Gumbel — §5 item 16), and `variance_cold_frac` no longer narrows
`U` (nothing to narrow under a budget-derived `U`); it now scopes an optional
`B` ramp, `--token_budget_cold`, which is the "warm-start distribution shift"
answer problem #8 asked for.

**Pre-flight audit (2026-09-20).** A 4-step run exercises the whole path; three
things it turned up, all fixed:

1. **`--pretrained_compressor_path` silently loaded nothing.** The loader behind
   it was the qbase-only one (`stage1.` stripped, `stage2.*` dropped), so a
   Phase-2 run dir left **all 194 parameters missing** behind a `[WARN]` and
   Phase 3 would have trained a randomly-initialised qbase *and* fold. Replaced
   by `compressor.py::load_pretrained_compressor`, which adapts between the
   single- and two-stage key layouts (two-stage→two-stage verbatim,
   single-stage→`stage1.*` with the fold at init, two-stage→qbase dropping
   `stage2.*`) and **raises** when the checkpoint and the model share no
   parameter names. Now `loaded 196/196, missing=0`.
2. **No long data.** `phase3_blend.json` held only the `unified`/`vcd` blend
   (70,692 rows, sampled p50 40 s, **0 % ≥ 420 s**): mean `N_u` 2.66, max 4.0 —
   *shallower than Phase 2*, so the depth cost, the fold bucketing and the
   chunked encoder would never have been exercised. Rebuilt with the InternVid
   420–1200 s bucket merged in (`build_phase2_manifest.py --mid_lo 420
   --mid_hi 1200 --prefix internvid_qwen3vl_p3` → `anno_data/phase3_internvid.json`,
   then `build_phase3_blend.py --merge`): **15 subsets / 712,910 rows** (439,482
   long + 160,963 `<420 s` fold replay + 41,773 `qbase_only` replay + 70,692
   blend), held-out **1,543 videos** across 15 tasks, train × held-out overlap 0.
   Pool-wide `N_u` at the curriculum points (row-weighted, real durations,
   p50 513 s):

   | `--max_frames` | all-bypass | mean `N_u` | p50 | p90 | p99 | max | budget fill |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | 420 | 1.4 % | 5.51 | 6.56 | 6.56 | 6.56 | 6.56 | 99.8 % |
   | 560 | 1.4 % | 6.64 | 8.06 | 8.69 | 8.69 | 8.69 | 99.8 % |
   | 720 | 1.4 % | 7.53 | 8.06 | 11.19 | 11.19 | 11.19 | 99.8 % |
   | 1200 | 1.4 % | 8.38 | 8.06 | 14.94 | 18.31 | 18.69 | 99.8 % |

3. **`durations_json` did not cover the blend.** Pointed at
   `internVid_durations.json`, every `unified`/`vcd` clip missed and
   `compression_depths` fell back to its constant `32`, putting the whole pool in
   one depth class — `--group_by_compression_depth` silently off. New
   `anno_data/phase3_durations.json` merges the InternVid ffprobe scan with a
   fresh scan of the 21,641 `unified`/`vcd` clips (999,062 entries, 100 % row
   coverage on both the train and held-out metas); it is the launcher's default.

   Also: `build_phase2_manifest.py`'s registry keys were hardcoded to the
   `180`/`420` defaults, so a 420–1200 build emitted key names contradicting the
   files they pointed at. They now track `--mid_lo`/`--mid_hi`.

4. **A no-fold window starved `stage2` of gradient and hung the allreduce.**
   Only multi-rank runs can see this, so every single-GPU check above passed
   while an 8-rank run died on **step 1**: a 600 s NCCL timeout on a
   250,877,952-element `ALLREDUCE` inside DeepSpeed's
   `independent_gradient_partition_epilogue`, surfacing as
   `torch.AcceleratorError: Invalid access of peer GPU memory over nvlink or a
   hardware error`. **Not hardware** — an isolated 8-GPU allreduce of exactly
   that size runs in 4.0 ms at 443 GB/s busbw, and decoding a 1200 s InternVid
   clip off weka takes 1.6 s, so neither NVLink nor the dataloader was the
   straggler.

   Cause: `compress_windows` calls `self.stage2` only when the window produced at
   least one fold unit (`if fold_idx:`). A window can legitimately produce none —
   a `qbase_only` replay window (5.9 % of the pool), or a Phase-3 clip short
   enough that `N <= U` makes every unit a single-segment raw bypass (1.4 %, and
   **only reachable because `--stage2_bypass_singletons` is a Phase-3 addition**;
   Phase 2 always folded). That rank's 29.9 M fold parameters then get no
   gradient at all, so ZeRO's flattened gradient buffer is a different size there
   than on ranks that folded, and the matching collective never completes.
   `out_gamma`/`out_beta` (Option A on the wrapper, applied in the fold branch
   only) have the same exposure.

   This is precisely the hazard `videollama3_arch._apply_mm_projector`'s docstring
   already documents for the three `mm_projector` copies — and warns is "a NCCL
   hang / invalid peer GPU memory access waiting to happen" — solved there by
   running all three unconditionally. `stage2` was the gap. Fixed the same way:
   when `fold_idx` is empty and stage2 is trainable, a one-segment **zero** input
   goes through the fold and its output enters the graph at coefficient `0.0`
   (added to `compressed`, which leaves every value bit-identical). Gradients are
   exactly zero; the backward hooks fire, so the reduction matches. Verified:
   all-bypass windows go from **0/47** to **47/47** stage2 tensors with a gradient
   (all numerically zero), `out_gamma` likewise, and mixed / all-fold windows are
   untouched (47/47 non-zero, same values).

   **Why `group_by_compression_depth` makes this worse, not better:** it
   deliberately fills each grad-accum window with same-depth samples, so "every
   sample on this rank this step is a bypass / `qbase_only`" is a *correlated*
   event, not `p^(grad_acc)`. A large `GLOBAL_BATCH` shrinks the odds but does not
   remove them — which is why Phase 2 (`GLOBAL_BATCH 512`, no bypass path) never
   tripped it.

5. **OOM at 79 GiB: `grad_acc` batches of `pixel_values` parked in VRAM.** At the
   real `GLOBAL_BATCH 512` the run died on step 2 with ~75 GiB live. It is not the
   model and not activations — measured, in order:

   | point | live |
   |---|---:|
   | before `trainer.train()` (model still on host) | 1.25 G |
   | after deepspeed init (params 15.41 G bf16) | 15.84 G |
   | **first forward, before `encode_images`** | **48.63 G** |
   | after vision encoder / compressor / projector | +0.14 / +0.64 / +0.05 G |

   The whole vision path costs **0.83 GiB**; DeepSpeed's engine costs 0.43 GiB;
   `peak − live` is 3.7 GiB. The 32.8 GiB is there *before the first forward runs*
   and is flat across all 64 micro-steps. A `gc` walk of live CUDA tensors named it:
   **53 tensors >0.5 G totalling 30.19 G, almost all `(250880, 588) float32` —
   `pixel_values`, one per prefetched batch.**

   Cause: `Trainer.get_batch_samples` (`transformers/trainer.py`) pulls
   `gradient_accumulation_steps` batches into a list in one go, and accelerate's
   `DataLoaderShard.__iter__` has already `send_to_device`'d each. At
   `GLOBAL_BATCH 512` on 8 GPUs that is **64 × 0.55 GiB ≈ 32 GiB per rank**, held
   for the whole window. Note the samples-in-flight per rank is
   `GLOBAL_BATCH / NGPU` regardless of `per_device_train_batch_size` — raising it
   lowers `grad_acc` but puts proportionally more samples in each batch, so it does
   not help.

   Fix: `VideoLLaMA3Trainer.get_train_dataloader` clears `dataloader.device`.
   `DataLoaderShard` skips the transfer when it is `None`, and
   `Trainer.training_step` already calls `_prepare_inputs`, which moves **and**
   (under DeepSpeed) casts each batch to bf16 as its micro-step runs — so the
   window costs one batch at 0.27 GiB instead of 64 at 0.55 GiB.
   `VL3_KEEP_BATCHES_ON_GPU=1` restores the stock behaviour. Measured after the
   fix, `GLOBAL_BATCH 512` unchanged: first forward **16.12 G** (was 48.63),
   peak live over 132 micro-steps **20.09 G**, peak reserved **26.85 G**, real GPU
   use **29–35 GiB** (was 79 → OOM); 3 optimizer steps green at 200 s/step.

   **Diagnostic that found it, and the lesson:** `VL3_LOG_MEM=1` adds per-micro-step
   `live`/`peak`/`reserved` plus an engine-init bracket, and `VL3_MEM_DUMP=1` walks
   `gc` once and names every live CUDA tensor >0.5 G. The `gc` dump is what actually
   identified it and should be the FIRST step next time — four successive
   from-first-principles guesses (compressor activations, an fp32 model copy,
   gradient-accumulation hoarding, ZeRO buffers) were all wrong, and
   `get_batch_samples` was in the top frames of the very first traceback.

#### Why InternVid long clips alone are not enough

The Qwen3-VL re-caption pipeline (`dataset_util/recaption_videos_vllm.py`)
samples a FIXED frame count per clip regardless of duration — its own comment:
"fps sampling with min_frames == max_frames, which pins every clip to an
[max_frames] frame budget regardless of length." A 420–1200 s InternVid clip's
caption is therefore generated from the same sparse frame sample as a 60 s
clip's — there is no CE gradient pressure to preserve any temporal detail finer
than that sampling density, independent of how good the fold is. Training
Phase 3 on InternVid captions alone chases a ceiling the *data* imposes, not a
compressor limitation.

**Fix: blend `../datasets/unified` + `../datasets/vcd`** (~278k clips,
consolidated activitynet / llava-video / videoxl / videoxl_pro / vcd — see
their own `README.md` / `VIDEO_DURATION_REPORT.md`) directly into the
length-bucketed curriculum (not a separate fine-tune stage — see §6). These
datasets are overwhelmingly ≤ 3 min (InternVid remains the only source of
genuine 420–1200 s length coverage); they fix the training *signal*, not the
length gap — real multi-event timestamp annotation, precise-event VQA, and
multi-granularity captioning all give CE loss actual pressure to preserve
detail the InternVid captions structurally cannot reward (see §6 for the
per-dataset breakdown).

#### Budget-driven adaptive fold (the mechanism)

New length-adaptation knob: a per-video **token budget `B`** (design default
1024) that every video tries to reach without exceeding, replacing Phase 2's
fixed `U ≤ 5` depth-class heuristic for Phase 3. `K = M = 64` (unchanged, tied
to the warm-started qbase/fold weights) makes the token cost of a "unit" — raw
or folded — identical either way, so the mechanism reduces to choosing how many
final units there are:

```
U = min(N, ⌊B / M⌋)        # target final unit count — 16 for B=1024, M=64
k = N − U                  # segments that must be merged away
```

- **`k = 0`** (video short enough that `N ≤ U`): no merging — every segment
  stays its own unit. This is the "one frame, one qbase compressor" extreme,
  reached automatically whenever `N ≤ ⌊B/M⌋` — which means it matters that the
  qbase segmenter's `target_frames` is fine enough for short clips to actually
  get here (see the open question in §7).
- **`k > 0`:** `_place_unit_boundaries` (unchanged) already selects its `U−1`
  cuts by taking the *highest*-`diff` (least-similar) boundaries out of the
  `N−1` candidates, subject to `min_gap` spacing. Given the SAME `U`, choosing
  which `U−1` boundaries are cuts is mathematically the complement of choosing
  which `k = N−U` boundaries are *merge points* (highest-similarity, lowest
  `diff`) — **no new boundary-selection code needed**, only the upstream `U`
  formula changes (budget-derived instead of the depth-class draw), and the
  existing training-time Gumbel jitter on that same ranking carries over
  unchanged.
- **Per resulting unit `u` (size `N_u`):**
  - `N_u == 1` → **raw bypass** (new): emit that segment's own `K=64` qbase
    tokens unchanged; `SegmentAggregator` is never called for it. Today every
    unit, even `N_u=1`, goes through the fold — this is the one genuinely new
    code path.
  - `N_u > 1` → fold via `SegmentAggregator` as today, `M=64` output.
  - Either way the unit costs exactly `64` tokens, so total = `U · 64 ≤ B`,
    maximal for the given `B` (pick `B` as a multiple of `64` so nothing is
    left on the table).

**This is NOT the banned retained-K.** Retained-K sent a unit's raw `K` tokens
*alongside* its own fold summary — the same content in both forms, in the same
forward pass, which is exactly the redundancy that let the LLM learn to ignore
the fold (CLAUDE.md load-bearing facts, [[planx-phase3]]). Here a segment is
*either* raw *or* absorbed into some other unit's fold — never both — so there
is no duplicate representation of the same content to shortcut through. Do not
conflate the two when revisiting this section.

Worked example, `T=1200`, `B=1024`: `U=16` regardless of `N`. At the qbase
segmenter's finest (`target_frames=1`, `N=1200`), the algorithm picks the
`1200−16=1184` most-similar boundaries as merge points — in practice these
concentrate wherever the clip has long low-motion/redundant runs, so `N_u` is
very uneven: near-`1` (raw bypass) for genuinely novel segments, very deep for
redundant runs. `Σ N_u = N = 1200`, `U=16` units, `1024` tokens to the LLM
either way.

#### Potential problems (analysis, 2026-09-18)

1. **Fold-depth flooding, actively produced rather than passively risked.**
   `_place_unit_boundaries`'s `min_gap` spacing already guards against cuts
   clustering too close together, but nothing bounds a unit from the OTHER
   side — a long enough redundant run can pull most of the `k` merge points
   into one place, producing a unit with `N_u` far deeper than anything
   validated in Phase 2 (`N_u` there tops out around 10–16). Because this
   mechanism explicitly *seeks out* the most-mergeable stretches, it
   concentrates depth more aggressively than Phase 2's heuristic-drawn `U`
   ever did — the same "fold state flooding on a long, uniformly-high-action
   clip" open problem the doc already flags (§3, §4 Phase 2), now something
   the algorithm can actively produce. Beyond the engineering side (sequence
   length, memory), the Mamba-2 scalar decay itself has a finite *effective*
   memory horizon set by `A_init_range` — a segment near the start of a
   100+-segment merged unit may be decayed to near-nothing by the time the
   `M` readout queries run, regardless of how well-trained the module is; this
   is a property of the recurrence, not just a compute problem. See §7 for
   the not-yet-decided safety-valve options.
   **Measured 2026-09-20 — largely closed.** Real held-out video does reach
   `N_u = 196` under pure-similarity selection, but `λ · depth_cost` with a knee
   scaled to `N/U` bounds it to ~34 (see "Measured: depth, padding and the
   `N_u_soft` knee"), and the target-`N` segmentation rule never enters that
   regime at all (`N_u` p99 ≤ 18.2). The *padding* half of the cost is fixed
   separately by length bucketing (§5 item 15). What remains is the
   `A_init_range` effective-memory-horizon argument above, which no partition
   rule addresses.
2. **The local pairwise-`diff` signal cannot see slow drift.** `diff[i]`
   compares only segment `i` to segment `i+1`. A long span that changes
   gradually (a slow pan, a fading expression) has low `diff` at every single
   adjacent pair even though the span's start and end look very different —
   the whole span reads as "safe to merge" even though merging it loses a real
   arc. This is the classic greedy pairwise-comparison blind spot; nothing in
   the current design detects it.
3. **Visual similarity is not the same thing as task relevance — likely the
   most fundamental risk here.** "Different from its neighbor" does not imply
   "important for the caption/answer," and "similar to its neighbor" does not
   imply "safe to discard" — e.g. a sustained but visually static action (the
   literal subject of many VQA questions) looks exactly like the "redundant,
   mergeable" case this heuristic is designed to fold away. The selection has
   no access to what the downstream task actually needs.
4. **The selection is entirely non-learned, so a wrong call never improves.**
   `seg_feat` is computed under `torch.no_grad()` (verified in
   `compress_windows`), so gradient cannot flow through the merge/raw
   decision — this also means the qbase has no way to game the signal by
   learning to make adjacent segments falsely similar (checked, not a live
   risk), but the flip side is the heuristic is frozen forever: if problem 3
   above is real for some content, no amount of training data fixes it.
5. **Per-epoch Gumbel jitter is in tension with the "protect informative
   content" goal.** The existing jitter re-draws the boundary ranking every
   epoch (compression-variance training thesis: the fold must be robust to
   *any* partition it meets). Applied unchanged to this mechanism, the same
   segment can be a protected raw-bypass unit in one epoch and get merged away
   in the next, purely from Gumbel noise — undermining the fidelity guarantee
   that motivated keeping it raw in the first place. The two goals (fold
   robustness to arbitrary partitions vs. reliable protection of informative
   content) pull in different directions and are not yet reconciled.
6. **Mixed-kind windows are a new, unvalidated combination.** Today `kind`
   (which of the three `mm_projector_*` copies a row uses) is a whole-window
   property — a window is either entirely `qbase_only` (kind 1) or entirely
   unit/fold (kind 2, even a trivial `N_u=1` unit goes through the fold today).
   This mechanism produces windows where SOME rows are raw-bypass (kind 1) and
   OTHERS are folded (kind 2), interleaved in one sequence the LLM attends to
   jointly — mechanically `_apply_mm_projector`'s per-row `torch.where` should
   handle it, but a sequence mixing tokens from two independently-drifting
   projector copies has never been exercised or evaluated.
   **Checked 2026-09-20 — mechanically safe, but this is the NORM, not an edge
   case.** `_apply_mm_projector` runs all three projectors on the full row set
   unconditionally and selects per-row, deliberately (its docstring: a
   skip-if-absent would desynchronise ZeRO's flat gradient buffer across ranks),
   so mixed windows carry no hang risk and need no new code. What is new is the
   frequency: see problem 10 — under the budget rule most short clips are
   *entirely* bypass rows, which skews the two projectors' training exposure,
   not just their output distributions.
7. **No slack — merging triggers on any overage, however marginal.** `U =
   min(N, ⌊B/M⌋)` starts merging the instant `N` exceeds `⌊B/M⌋` by even one
   segment (e.g. `N=17` vs. the `⌊B/M⌋=16` threshold triggers exactly the same
   kind of merge as `N=1200`). Whether a small, cheap overage should just be
   allowed to slightly exceed `B` instead of forcing a merge is an open
   framing question, not just an implementation detail.
8. **Warm-start distribution shift.** Phase 2's fold has only ever seen
   `N_u` drawn from the depth-class heuristic (`N̄_u ∈ [4,16]`, `U≤5`) — a
   fairly narrow, moderate range. This mechanism can hand a Phase-3-warm-
   started fold `N_u` anywhere from `2` to `100+` in the same batch. The
   existing "grow the max frame count gradually" curriculum bullet should
   probably also ramp the *merge depth* (e.g. via `B` itself, starting smaller
   and growing), not just raw frame count `T` — not yet specified.
9. **Tension with §1's own stated design goal.** §1 opens with "compression
   must be dynamic and non-uniform... a spare token budget spent on keeping
   resolution, not on a fixed ratio." This mechanism instead has every video
   *reach for* the same fixed ceiling `B` regardless of how little information
   it actually contains — which is itself a fixed target, just denominated in
   tokens rather than a ratio. This was an explicit choice in this
   conversation (the user's framing: "想辦法去達到上限但是又不可以超過" — reach
   the ceiling, don't exceed it), recorded here as a deliberate trade-off
   against §1's original framing, not an oversight — worth revisiting if the
   goal shifts from "maximize information under a fixed compute budget" back
   toward "minimize tokens for genuinely simple content."
10. **The budget rule starves the fold of gradient on short clips (measured,
    2026-09-20).** `U = min(N, ⌊B/M⌋)` means `k = N − U = 0` whenever
    `N ≤ ⌊B/M⌋`: no merging, every segment its own unit, every unit a raw
    bypass, `SegmentAggregator` never called — **zero fold gradient from that
    clip**. At `target_frames = 4, B = 1024` that is every clip under 64 s,
    which is **69.0 % of the `unified`/`vcd` blend** (median 32 s) and 38.1 % of
    the InternVid `< 180 s` replay pool. So the §6 data blend — added precisely
    because InternVid captions cannot reward temporal detail — would contribute
    almost nothing to training the fold, and the mechanism added to reach the
    token budget is what prevents it. Two knock-on effects: `mm_projector_fold`
    sees far less of the stream than `mm_projector_qbase` (an exposure
    imbalance, cf. problem 6), and a training stream that is mostly raw qbase
    tokens teaches the LLM that vision tokens *look like raw qbase*, making the
    fold readout a minority dialect it rarely has to read — structurally where
    retained-K failed, arrived at from the dataset level rather than per-unit.
    **Resolved by the target-`N` segmentation rule** (next subsection): 69.0 % →
    20.4 %, the remainder being clips genuinely too short to fill the budget.
    Note the inference-side reading is different and benign: a 30 s clip passing
    through entirely raw at full qbase resolution is exactly §1's "spare budget
    spent on keeping resolution". This is a *training-curriculum* problem only.

#### Alternative partitioning strategies (2026-09-18, not decided against — recorded for comparison)

- **(A) Agglomerative merge instead of a single-shot global top-`k` ranking.**
  Repeatedly merge the currently-most-similar ADJACENT pair, re-pooling the
  merged group's feature and re-comparing to its (now different) neighbors,
  until `U` groups remain. Costs more than one sort, but a slowly-drifting
  span (problem 2 above) tends to get merged pairwise in local sequence
  instead of having a single global ranking silently skip over its
  interior — more explicitly tracks local coherence at each step.
- **(B) Bounded-depth split — cap `N_u_max` directly instead of letting depth
  float freely.** A much larger `min_gap` (or an explicit max-run-length
  constraint on `_place_unit_boundaries`) forces an extra cut inside an
  overlong redundant run even when the similarity signal says "keep merging."
  Directly neutralizes problem 1 (flooding) at the cost of sometimes not
  reaching `U = ⌊B/M⌋` exactly (a few units end up smaller than the budget
  would allow). This is essentially §7's safety-valve option (a), stated as a
  concrete partitioning rule rather than a caveat.
- **(C) Learned importance/gating head instead of a hand-coded `diff`
  heuristic.** A small linear head on each segment's qbase feature predicts a
  "keep-raw" score; a Gumbel-softmax / straight-through estimator makes the
  discrete raw-vs-fold choice differentiable, trained end-to-end through the
  same downstream CE loss (not a new reconstruction target — stays inside the
  "LLM cross-entropy only" rule). Directly addresses problems 3 and 4 (the
  decision becomes task-aware and can improve with data) at the cost of a new
  trainable module and a harder, noisier training signal (discrete-selection-
  through-Gumbel-softmax is notoriously finicky to tune). The most ambitious
  option here.
- **(D) Fixed per-unit depth cap, let `U` (and total tokens) grow with `N`
  instead of capping at `B`.** Essentially Phase 2's existing depth-class
  scheme, unchanged. Included as the contrast case: it does NOT meet the
  "reach a hard ceiling `B`" requirement this whole design started from, but
  has none of problems 1, 7, 8 above — useful as a reminder of exactly what is
  being traded away to get the fixed-budget property.
- **(E) Guaranteed primacy/recency window + adaptive middle.** Always keep the
  first and last few segments raw regardless of what the similarity signal
  says (a fixed "attention-sink"-style policy, cf. StreamingLLM), and apply
  the content-adaptive merge only to the interior. Hedges directly against
  problem 3 for the common case where captions/QA reference "at the start" /
  "at the end" (e.g. the `nextqa` example in §6), without needing a learned
  module — a smaller, safer version of (C).
- **(F) Hierarchical fold-of-folds via the already-implemented (but unused)
  SSD `merge` operator.** §3 already defines `merge: state ⊕ state` via the
  SSD associative-scan operator and §7 already lists it as implemented but
  "deferred... needed for real streaming inference, not for the pretrain." A
  single very-deep unit could instead be built as a binary-tree of shallower
  `append` scans combined with `merge`, bounding the effective single-scan
  depth at any level while producing the same final `M`-token readout and the
  same total `U·M` budget — this reuses a currently-dormant primitive rather
  than adding new machinery, and is the most direct answer to problem 1 that
  does not sacrifice hitting the budget the way (B) does.

#### Depth-aware boundary cost — DP + PELT-streaming draft (2026-09-19/20, discussion, not implemented; `internal_cost` replaced 2026-09-20 after an empirical check)

A formalization of alternative (B) above, worked out in more detail: instead
of a hard `N_u_max` cap, fold depth is penalized as a *soft* term inside a
DP-optimized boundary search (Phase 3 only — Phase 2's own greedy
`_place_unit_boundaries` ranking is unchanged and already validated, see §7
"Unit-boundary feature — RESOLVED"), so similarity and depth are optimized
jointly rather than similarity-first with a depth repair-pass bolted on
afterward.

**Cost.** For a candidate unit spanning segments `(a, b]`, with `x_i` the
per-segment qbase feature (unit-normalized):

```
cost(a, b) = internal_cost(a, b) + λ · depth_cost(b - a)
internal_cost(a, b) = Σ_{i=a}^{b-1} ‖x_i − c(a,b)‖²,   c(a,b) = mean({x_a, ..., x_{b-1}})
depth_cost(N_u)     = max(0, N_u - N_u_soft)²          # 0 inside Phase 2's validated range, convex beyond it
```

`internal_cost` is the unit's within-cluster sum of squared distances to its
own centroid (WCSS) — the classical mean-shift changepoint cost, exactly what
PELT was originally designed to detect. `N_u_soft` ≈ Phase 2's validated
depth range (~10–16), unchanged.

**This replaces an earlier draft** where `internal_cost(a,b) = Σ_{j=a+1}^{b-1}
diff[j]` (sum of adjacent-pair `1-cos`, the same signal Phase 2's
`_place_unit_boundaries` ranks on) — dropped after the empirical check below
found it structurally blind to gradual drift, common enough in real InternVid
clips to matter, and worsening exactly in the direction Phase 3 pushes
(deeper units).

**Why the sum-of-adjacent-diff version was replaced (measured, 2026-09-20).**
`Σdiff[j]` only measures step-to-step motion: a unit that drifts slowly and
steadily — each adjacent pair looks similar, but the first and last segment
end up looking very different — scores as "cheap to merge" under this cost
even though a real arc is lost, because the cost never looks past one hop
(problem #2 above). Checked on
`eval_ablation/manifest_internvid_true_heldout_420_550.json` (8 InternVid
clips; verified by basename cross-reference to have **zero** overlap with the
actual short/mid-bucket training files `anno_online/internvid_qwen3vl_lt180.json`
/ `_mid_180_420.json` — the previously-used `manifest_internvid_heldout.json`
turned out to be fully in-pool with the short-bucket training set and was
retired), sliding fixed-length windows (`L=5,10,20`, matching the `N_u` range
already observed in production, up to 34) over both the frozen-encoder and
qbase per-segment features, comparing `Σdiff` against `WCSS` and against a
direct `endpoint_diff = 1 - cos(x_a, x_{b-1})` proxy:

| L | Spearman(Σdiff, WCSS) enc / qbase | Spearman(Σdiff, endpoint_diff) enc / qbase | blind-spot rate (Σdiff≤p33 & endpoint_diff≥p67) enc / qbase |
|---|--:|--:|--:|
| 5  | 0.90 / 0.89 | 0.51 / 0.50 | 1.3% / 1.6% |
| 10 | 0.76 / 0.79 | 0.27 / 0.28 | 7.5% / 5.0% |
| 20 | 0.62 / 0.76 | 0.23 / 0.38 | 9.7% / 6.7% |

`Σdiff` agrees less and less with actual head-to-tail displacement as the
window grows — exactly the regime Phase 3's budget-driven merge pushes units
into (production `N_u` already reaches 34 under the existing `U≤5`
depth-class draw, §4 Phase 2's mechanism, before Phase 3's own budget
mechanism is even applied). At `L=20`, ~1 in 10 candidate windows would be
ranked "cheap" by `Σdiff` while its own endpoints are in the top third of
most-different pairs at that length. Concrete case (`R74oZMEWpzk.mp4`,
encoder feature, `start=54, L=10`): `Σdiff=0.157` (8th percentile — among the
"safest to merge") while `endpoint_diff=0.174` (97th percentile) and
`WCSS=0.844` (correctly elevated) — a ~10-segment span with a real, sustained
drift the old cost would have merged away first.

**Checked the reverse direction too, before committing to a full replacement
rather than a blend.** The concern: `WCSS` is invariant to the *order* of
points within the window (it can't distinguish "steady drift" from "noisy
jitter that returns near its start"), so it could in principle miss content
that jitters a lot step-to-step but never travels far overall (camera shake,
flicker) — exactly the case `Σdiff` is naturally suited to catch. Measured
rate of `Σdiff≥p67 & WCSS≤p33`: 0.0–0.7% across all `L` on the encoder
feature, and **0.0% at every `L` on the qbase feature** — the one
`_place_unit_boundaries` actually operates on. The handful of encoder-side
hits (e.g. `mdieXloRC4s.mp4, start=58, L=10`: `Σdiff` at the 78th percentile
but `WCSS` at the 31st) all have `endpoint_diff≈0` — genuine
oscillation-back-to-baseline content, where `WCSS`'s "cheap to merge" call is
the *correct* one and `Σdiff`'s "expensive" call was the false positive. No
case was found where `Σdiff` correctly flags danger that `WCSS` misses. This
is why the formula above is a straight replacement, not `α·WCSS +
(1-α)·Σdiff` — a blend would add a tuning knob with no measured benefit.

**Computation.** `WCSS(a,b) = Σ‖x_i‖² − ‖Σx_i‖²/(b−a)`, maintained via running
prefix sums of `Σx_i` and `Σ‖x_i‖²` — `O(1)` amortized per candidate span,
same as the `Σdiff` version, so the DP's `O(N²U)` bound below is unaffected.
The identity `WCSS(a, b+1) = WCSS(a,b) + (n/(n+1))·‖x_{b+1} − mean(a,b)‖²`
(Welford-style) confirms `WCSS` is monotonically non-decreasing in window
length — re-checked for this formula specifically, since the streaming
section's PELT-pruning argument depends on it and does not carry over
automatically from the old formula's version of the same check. Secondary
benefit for the streaming path: `WCSS`'s sufficient statistics are exactly
`(Σx, Σ‖x‖², n)`, cheaper to maintain online than the raw per-segment history
`Σdiff` would need to recompute over an arbitrary candidate range.

**Training (offline, whole video in hand): exact DP, `U` fixed.**

```
dp[u][i] = min_a [ dp[u-1][a] + cost(a, i) ]
```

Solve for `dp[U][N]`, backtrack for the `U-1` cut positions. `O(N²U)`
(`N ≲ 1200`, `U ≲ 16` at Phase-3 scale) is cheap — a single CPU/numpy pass,
negligible next to one training step. Replaces the greedy top-k + `min_gap`
heuristic with a global optimum under the joint cost; `min_gap`'s role (avoid
degenerate near-zero-length units) can stay as a hard floor on top of the soft
`depth_cost` ceiling. The existing per-epoch Gumbel jitter (§4 Phase 2,
"Segment-position jitter") is intended to still apply, but its exact form
needs restating for this cost — the old version injected noise directly into
`diff[j]`; under `WCSS` the natural analogue is jittering each `x_i` (or the
`cost` lookup table) before the DP runs, not yet worked out in detail (open
item below).

*(The original synthetic 6-segment worked example for this subsection used
`Σdiff` arithmetic and no longer matches the formula above; removed rather
than re-derived, since the measured `R74oZMEWpzk.mp4` example above serves
the same illustrative purpose with real data.)*

**Inference (streaming): does NOT transfer as-is — splits into two separate
mechanisms.** The exact DP above needs the whole per-segment feature array up
front (non-causal) and needs raw per-segment features (or their sufficient
statistics) to (re)compute `internal_cost` over any candidate range — once
segments are folded into an SSM state, neither is available. Maps onto the
`lift`/`append`/`merge` monoid ops already sketched in §3:

- **append vs. cut ("fold + feature" — does the currently-open unit keep
  growing) = online changepoint detection.** PELT / Optimal Partitioning:
  drop the fixed `U`, pay a fixed penalty `β` per opened unit instead (`β` is
  the streaming analogue of `(U, λ)` — needed because stream length is
  unknown in advance):

  ```
  F(t) = min_s [ F(s) + cost(s, t) + β ],   s ranging over a maintained candidate list R_t
  ```

  Same `cost(s,t)` as above — only the recursion's indexing changes, not the
  cost itself. `R_t` is the "maintained list": PELT prunes any candidate `s`
  that is provably dominated for all future `t' > t` and can never win again.
  This pruning is *exact* here (not an approximation) specifically because
  both `internal_cost` and `depth_cost` are monotonically non-decreasing in
  the window length — re-checked for the `WCSS` formula above (the Welford
  identity in "Computation"), holds for this cost. Caveat: `F(t)`'s current
  best changepoint `cp(t)` can still be revised as more data arrives — PELT
  alone gives a rolling value function, not a hard commit; genuine
  low-latency streaming needs an added commit/decision-lag rule (e.g. treat
  `cp(t)` as final once unchanged for `L` steps, or a hard max-lag) layered on
  top, which PELT does not provide on its own.
- **merge ("fold + fold" — freeing a slot among already-closed unit states) =
  a different, much simpler mechanism**, not PELT-related. By the time a
  merge is needed, raw features are gone — only a small bounded list of
  currently-materialized units' `(state, N_u, a cheap readout-level
  representative vector)` survives (bounded by the streaming unit cap).
  Merge candidates are restricted to **adjacent** pairs only — the SSD
  associative combine `(A₁,b₁)∘(A₂,b₂) = (A₁A₂, A₁b₂+b₁)` (§3) is
  order-sensitive, not commutative, so only temporally adjacent states are
  valid to fuse. Pick the adjacent pair minimizing a
  similarity(readouts)-minus-depth-penalty score, `O(U_max)` per decision.
  This is §3's already-sketched "streaming budget controller" (merge of
  adjacent oldest states) with a concrete selection rule attached — checked
  against `segment_aggregator.py` (2026-09-19): only `init_state` / `update` /
  `readout` / `reduce` exist today, no `merge` method, so despite the (F)
  bullet above calling it "already-implemented," this would be genuinely new
  code, not wiring an existing one in.

**Open items, not resolved:**
- `β` (streaming) and `(U, λ)` (training) are different parameterizations of
  the same "how aggressively to open new units" knob and are not yet
  calibrated against each other — a mismatch here reproduces the same category
  of train/inference distribution-shift risk §2 flags generally (independent
  of the specific root-cause story for the superseded collapse). Suggested
  validation before trusting the online path: run both the offline DP and the
  online PELT+merge simulation on the same full offline video and compare the
  resulting partitions / depth distributions.
- `λ` / `N_u_soft` / `β` all need calibrating against real `WCSS`
  distributions (extend `eval_ablation/unit_segmenter_probe.py`-style
  measurement to the new cost) before they can be trusted as more than
  placeholders — not yet done; the swap from `Σdiff` to `WCSS` changes the
  numeric scale `internal_cost` operates on, so no prior calibration
  intuition transfers.
- Gumbel-jitter analogue for `WCSS` (perturbing `x_i` vs. perturbing the
  `cost` lookup table directly) — not yet worked out, see the "Training"
  paragraph above.
- This is a refinement of alternative (B), not a replacement for (C)/(E)/(F)
  above — task-relevance blindness (problem 3) and the mechanism's
  non-learned-ness (problem 4) are untouched by this proposal either.

#### Measured: depth, padding and the `N_u_soft` knee (2026-09-20)

`eval_ablation/phase3_dp_depth_probe.py`, 8 clips from
`manifest_internvid_true_heldout_420_550.json` (the verified-disjoint held-out
set), at the Phase-3 budget `U = min(N, ⌊1024/64⌋) = 16`, sweeping the qbase
segmenter granularity, `λ` and `N_u_soft` — 144 configurations. Raw rows in
`work_dirs/user_eval_0920/phase3_dp_{depth,soft}/rows.json`.

**1. `depth_cost` bounds the fold depth, and `λ` is not a sensitive knob.**
Pure similarity (`λ = 0`) produces genuinely deep units on real video — worst
case `N_u = 196` (`RVxDPtxMxmg.mp4`, finest segmentation), partition
`[196, 68, 59, 57, 46, 39, 15, 11, 8, 8, 8, 7, 7, 6, 5, 3]`. Any `λ ≥ 0.01`
collapses `max_u N_u` to ~36 / ~21 / ~17 at the three granularities tested, and
`λ = 0.01` vs `λ = 0.1` differ by 1–2 segments — so `λ` needs an order of
magnitude, not tuning.

**2. `N_u_soft` must scale with `N/U`, or the DP degenerates to a uniform
split.** `depth_cost = max(0, N_u − N_u_soft)²` is zero below the knee. Once
`N_u_soft < N/U` *every* unit pays the quadratic, the convex term swamps the
`WCSS` term, and the optimum is equal-sized units — the content-adaptive
placement does nothing. Measured spread (`mean(N_u_max − N_u_min)`, ≈0 = uniform):

| `N` | mean `N_u` = N/U | `s16` | `s24` | `s32` | `s48` | `s64` |
|---:|---:|---:|---:|---:|---:|---:|
| 121 | 7.6 | 13.9 | 19.1 | 21.5 | 21.5 | 21.5 |
| 241 | 15.1 | **7.0** | 20.6 | 28.1 | 39.4 | 46.5 |
| 480 | 30.0 | **1.1** | **1.1** | 13.1 | 39.6 | 53.9 |

Rule of thumb: **`N_u_soft ≈ 1.5–2 × N/U`**. A fixed `N_u_soft = 16` (Phase 2's
validated range, the original draft's suggestion) is on the wrong side of this
for any clip with `N/U > 16` — e.g. a 1200 s clip at `target_frames = 4`
(`N = 301`, `N/U = 18.8`). Far above the mean the term is simply inert
(`s32`–`s64` at `N = 121` are identical to `λ = 0`).

**3. The greedy placement cannot be reused — `_place_unit_boundaries` is worse
than the DP on depth, and structurally cannot emit a raw-bypass unit.** At the
finest granularity greedy gives `N_u_max = 75` (spread 49) against the DP's
34–36, and it produced **zero** `N_u == 1` units in all 144 configurations —
`min_gap = max(1, N // (2U))` is a floor on every unit's size (9 at
`N = 301, U = 16`), so the bypass branch can never fire under it. §5 item 13's
"no change to `_place_unit_boundaries` itself" is therefore wrong on both
counts; Phase 3 needs its own placement method (leaving Phase 2's greedy
untouched for reproducibility, and `eval_ablation/unit_segmenter_probe.py`'s
instrumented copy of it valid).

**4. Padding waste is a batching problem, and it is solved.** The batched fold
front-pads every unit to its group's deepest `N_u`, so the `λ = 0` partition
above costs 5.78× its real token count in one padded call. Length bucketing
(§5 item 15, implemented) drops that to 1.18×, and across all 144
configurations — greedy, `λ = 0`, every granularity — bucketed padding never
exceeds **1.46×**. No varlen/packed rewrite is needed. What bucketing does *not*
fix is a single deep unit's own sequence length (`N_u = 196` ⇒ 12,608 tokens
through the fold); that is problem #1, and #1 is what `depth_cost` addresses.

#### qbase segmentation: **target `N` + clamp**, not `target_frames` (measured, 2026-09-20)

The first-level cut is parameterised in FRAMES
(`N = ⌊T/segment_target_frames⌋ + 1`, `target_frames = 4`) while the Phase-3
budget is stated in TOKENS (`U = min(N, ⌊B/M⌋)`). That is the wrong control
variable, and it is the common cause of two separate Phase-3 failures: short
clips never reach the budget (and never call the fold at all), long clips blow
up `N·K`. Phase 1's actual requirement was only that `N` be a pure function of
the frame count — which `N = B_q/K` satisfies trivially — so nothing ever forced
a frame denomination.

**Two budgets, only one of them named.** `U·M` is what the LLM sees (`B`, stated
in tokens). `N·K` is the fold's input length, the RoPE footprint (§7), and the
actual LLM token count on the bypass / qbase-only-replay paths — controlled by
`target_frames`, never denominated in tokens.

**Rule.**

```
N        = clip(c · U_cap, N_min(T), T),     U_cap = ⌊B/M⌋
N_min(T) = ⌈ n_forced / (1 − min_adapt) ⌉ + 1,   n_forced = ⌊(T−1)/force_every⌋
```

`min_adapt` is the minimum fraction of cuts that must stay discretionary. Both
floors are load-bearing:

- **`N ≥ n_forced + 1` is hard.** `adaptive_segment_lengths` re-inserts cuts to
  keep every segment ≤ `force_every` frames, but `adaptive_segment_count` does
  not know that, so asking for fewer segments **silently breaks the
  collator/model length contract** — verified: `T = 1200, target_frames = 12`
  ⇒ count says 101, the cut returns 150. Needs an assert (§5 item 17); today
  `target_frames = 4 < force_every = 8` is why nobody has hit it.
- **At exactly `n_forced + 1` the segmenter is inert** (`budget = n_cuts −
  n_forced = 0`): all cuts are forced, the split is uniform `force_every`-frame,
  and the content-adaptive placement does nothing.

**`target_frames = 4` is this rule in disguise.** `N − 1 ≥ n_forced/(1−min_adapt)`
with `force_every = 8` gives

```
target_frames = force_every × (1 − min_adapt)     ⇒  fe=8, min_adapt=0.5 ⟺ tf=4
```

i.e. the undocumented constant 4 is exactly "keep half the cuts adaptive".

**Sweep** (`eval_ablation/phase3_segment_budget_sweep.py`, pure duration
arithmetic over both Phase-3 sources; `B=1024, M=K=64, force_every=8`;
`bypass100%` = clips with `N ≤ U_cap`, i.e. every unit a single segment and
`SegmentAggregator` never called — **zero fold gradient from that clip**;
`fill%` = mean `(U·M)/B`; `adapt%` = mean discretionary-cut fraction):

| | rule | N_med | adapt% | bypass100% | fill% | `N_u` mean | `N_u` p99 | `N·K` p90 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| InternVid 977k | tf=4 (today) | 92 | 51.2% | 10.3% | 96.4% | 6.8 | 18.2 | 14400 |
| (median 366 s) | N=2U, ma=0 | 46 | 19.1% | 1.1% | 99.9% | 3.7 | 9.1 | 7168 |
| | **N=4U, ma=0.5** | 91 | 59.3% | 1.1% | 99.9% | 7.3 | 18.2 | 14272 |
| unified+vcd 277k | tf=4 (today) | 9 | 54.9% | **69.0%** | 61.8% | 1.4 | 4.1 | 2432 |
| (median 32 s) | N=1U, ma=0.5 | 16 | 70.6% | 69.2% | 93.3% | 1.3 | 2368 | 2368 |
| | N=2U, ma=0.5 | 32 | 77.7% | 20.4% | 93.3% | 1.7 | 4.1 | 2368 |
| | **N=4U, ma=0.5** | 32 | 84.4% | **20.4%** | 93.3% | 2.4 | 4.1 | 4096 |

- **`c` and `min_adapt` are decoupled.** `c·U_cap` is below the floor for
  anything long, so `c` only moves short clips and `min_adapt` only moves long
  ones. Tune them independently.
- **`min_adapt = 0.5` leaves long-clip behaviour identical to today** (InternVid
  `N_med` 92→91, `adapt%` 51.2→54.8, `N_u` 6.8→6.9, `N·K` p90 14400→14272) — so
  the operating point Phase 1/2 validated, and the Phase-2 checkpoint Phase 3
  warm-starts from, are not disturbed.
- **Short clips go from 69.0% to 20.4% zero-fold-gradient**, budget fill 61.8% →
  93.3%. The residual 20.4% is irreducible: a `T ≤ 16 s` clip has at most 16
  one-frame segments and cannot exceed `U_cap = 16`. Those clips being entirely
  raw is the correct answer, not a defect.
- **`c = 1` is useless** (`N = U_cap` sits exactly on the all-bypass boundary,
  85.1%). **`c = 4` beats `c = 2`**: same bypass rate and fill, but `N_u` 1.7 →
  2.4, i.e. ~40 % more fold gradient from the short-clip blend, for `N·K` p90
  2368 → 4096 (negligible). Since §6's whole reason for adding `unified`/`vcd`
  is fold training signal, take the larger `c`.
- **`min_adapt = 0` would halve the RoPE footprint** (`N·K` p90 14272 → 7168,
  `N·K > 16k` 5.0% → 0.0%, which would close §7's RoPE concern outright) at the
  cost of `adapt%` 54.8% → 19.1% — the adaptive segmenter mostly stops working.
  Keep the adaptivity; handle the 5 % with `stage2_rope_slot_scale`.
- **Fold depth is a non-issue under this rule.** `N_u` p99 ≤ 18.2 (InternVid) /
  4.1 (unified+vcd), inside Phase 2's validated band. The `N_u > 100` cases in
  the DP probe above only arise at `target_frames = 1` *with* `λ = 0`; the
  target-`N` rule never reaches that regime.

**Decision: `c = 4`, `min_adapt = 0.5`.** Strictly better than `target_frames = 4`
on short clips, identical on long ones.

- **Trainable / warm-start:** same as Phase 2 — qbase + fold + `mm_projector`,
  LLM frozen, warm-started from a Phase 2 checkpoint. No module shape changes:
  `M=64` stays a fixed learned `summary_tokens` parameter, `K=64` stays the
  qbase's `num_queries` — this mechanism only changes how `U`/`unit_counts` is
  computed and adds the `N_u==1` bypass branch in `compress_windows`.
- **No retained set beyond the raw-bypass above** — see the callout above. Do
  not reintroduce per-unit `r_u` (redundant raw-alongside-fold) in any form.
- **Length curriculum inside the phase** — grow the max frame count gradually;
  do not start at 1200 s. (Unchanged from the original sketch.)
- **Gate:** needle probes (content recall at specific timestamps) on long
  video; frozen-LLM caption eval holds; `struct_ρ` holds at depth; **and now
  also**: raw-bypass units should be measurably more faithful than folded ones
  (better recall on needle-probe content that lands in a bypass unit vs. a
  folded one) — if not, the novelty-score selection isn't actually finding the
  informative segments and needs revisiting. Frame-order / temporal-blindness
  metrics stay a diagnostic, not a blocking gate (unchanged rationale, see
  CLAUDE.md load-bearing facts).

---

## 5. Implementation requirements (code changes)

| # | change | where | GPU? |
|---|---|---|---|
| 1 | **Option A/B on the fold readout** — after `SegmentAggregator.output_proj`, apply `_match_encoder_scale` (ref = **that unit's** encoder-token slice `win[fstart[a]·hw : fstart[b]·hw]`, per-unit) and stash `_last_distr_loss` from `_distribution_match_loss` on the `TwoStageCompressor` wrapper. **Done.** | `compressor.py::TwoStageCompressor.compress_windows` | no |
| 2 | **Trainer wiring fix** — `_compressor_distr_loss` does `getattr(comp, "distr_loss_weight", 0.0)` on the `TwoStageCompressor` wrapper → 0. Sum `_last_distr_loss` from `.stage1` **and** `.stage2`. | `videollama3_trainer.py` | no |
| 3 | **Fixed-count adaptive causal segmenter** — model-side, non-learned. `N = ⌊T/4⌋+1` from the frame count (collator + model compute it identically ⇒ no dynamic-length plumbing, no arch refactor for Phase 1). Forced boundary every 8 f + remaining budget to top-`diff` positions on the per-frame encoder mean feature; Gumbel-top-k draw (z-normalised `diff`, `τ≈0.5`) for per-epoch augmentation; `[1,8]` clamp (`--segment_force_every`). Validated: `eval_ablation/segmenter_validate.py` (+ `_hist.py`). Implemented: `compressor.py::adaptive_segment_count` / `adaptive_segment_lengths`, `TransformerDecoderFlatCompressor` `output_len_for` + `forward` (`--adaptive_segmentation`); `videollama3_arch.py::_compressed_len` uses `output_len_for` for the placeholder count. | `compressor.py`, `videollama3_arch.py` | no |
| 4 | **Chunked encoder forward** — groups of `--vision_encoder_chunk_frames` frames under `no_grad` (skipped only if some encoder param is trainable), one row `(t,h,w)` split into `(t_chunk,h,w)` rows. The encoder has no cross-frame attention (per-frame `cu_seqlens`, 2-D spatial RoPE repeated per frame), so this is a padding/layout change only — **verified bit-exact** vs. the single-shot forward at chunk ∈ {1,8,16,64,128} on the real SigLIP-NaViT config, including a packed multi-row batch; peak allocation scales with the chunk. **Done** (`0` = single shot, Phase 1/2 unchanged; Phase 3 defaults to 8). | `videollama3_arch.py::_run_vision_encoder` | no |
| 6 | **Compression-variance training** (§4 Phase 2) — **Done.** `Phase2FoldDataset` draws, per `(epoch, index)` seed: depth class from `N` → `N̄_u` from the class's narrow range → `U = round(N/N̄_u)` (cold window: `U = min(3, ⌊N/4⌋)`). Segment + `U−1` unit boundaries are placed **model-side** under a seeded Gumbel (`compression_seed`), on the **qbase segment tokens'** inter-segment cosine. Depth-class **grouped sampler** (`LengthGroupedSampler` on `N`, `--group_by_compression_depth` + `durations_json`; `SubsetWithLengths` forwards `compression_depths` so it survives a val split). `_epoch` / `_progress` are fork-shared `mp.Value`s so the seed + curriculum reach dataloader workers. `variance_ramp_frac` removed. | `Phase2FoldDataset`, `videollama3_trainer.py` sampler, `train/data/compressor.py` | no |
| 7 | **time_embed** — `rel_gap_mlp` (per-segment gap + duration seconds, built model-side from the adaptive cut) or `none` (Phase-2 first-runs default); `index_sincos` stays in the code but not for the fold. **Done.** | `segment_aggregator.py` config, `compress_windows` | no |
| 8 | **K/M knob** — default K = M = 64. Decoupling (K > M, e.g. 128/64) only needs `TwoStageCompressor.__init__` to stop default-tying `n_summary_tokens` to `tokens_per_segment` (`SegmentAggregatorConfig` already has them separate). Open decision §6. | `compressor.py` | no |
| 9 | **`struct_ρ` / `xcos_cen` probe** — centered cross-video cosine + encoder-similarity Spearman + `cc_frac`, on the pre-projector compressor output vs. the raw encoder. Implemented: standalone `eval_ablation/struct_probe.py` (whole-video part, works on the Phase-1 flat-adaptive checkpoint) and mirrored into `grounding_probe.py`'s aggregate/verdict/table. | `eval_ablation/` | yes (eval) |
| 10 | **LR groups** — `create_optimizer` `stage1`/`stage2` split (present) + `mm_projector` group (present). No change, just enable. | — | — |
| 11 | **Strided `position_ids` for the compressed region** (§1) — **Done.** `compress_windows` returns per-output-token slot offsets relative to the unit start: readout token `m` at `round(m·N_u·K/M)`. The arch `position_ids` loop advances `cur` by `unit_span` (`= N_u·K`) per unit and stays stride-1 elsewhere, so the compressed span ≈ `N·K` slots (+ the per-unit `Time:`/`<cs>`/`<ce>` text). Sample-start detection (`position_id == 0`) unaffected. Identical formula train/inference. Phase-3 scale knob (`stage2_rope_slot_scale`): §7. | `compressor.py::compress_windows`, `videollama3_arch.py::prepare_inputs_labels_for_multimodal` | no |
| 12 | **qbase-only replay path** — meta-JSON `"qbase_only": true` → `Phase2FoldDataset` emits `compression_qbase_only=[True]`; `compress_windows` returns stage-1's `N·K` qbase tokens as one `unit_meta` entry (stride-1 slots, no unit split, no `SegmentAggregator` call). Threaded `dataset → DataCollatorWithCompressor → prepare_inputs_labels_for_multimodal → encode_images → compress_visual_tokens_with_compressor → compress_windows`. Keeps `mm_projector` / an unfrozen qbase anchored to the raw-qbase manifold. **Done.** | `compressor.py`, `videollama3_arch.py`, `videollama3_qwen2.py`, `train/data/{compressor,global_compressor}.py`, `phase2_pretrain_fold.py` | no |
| 13 | **Budget-driven Phase-3 unit count** (§4 Phase 3) — `U = min(N, ⌊B/M⌋)` replaces the depth-class `_draw_units` heuristic; `N_u==1` units bypass `SegmentAggregator` (raw qbase passthrough) instead of folding a singleton, emitting `unit_meta` with `kind="qbase"` (already routes to `mm_projector_qbase`, and at `K==M` the existing `round(m·N_u·K/M)` slot formula already yields stride-1 — no RoPE special case). Also needs `stage2_max_units` 5 → `⌊B/M⌋` and removal of `_draw_units`' `u_cap = min(max_units, N//4)` clamp, which caps short clips below the budget. **CORRECTION (2026-09-20): `_place_unit_boundaries` CANNOT be reused** — `min_gap = max(1, N//(2U))` is a floor on every unit's size, so it emitted zero `N_u==1` units in all 144 measured configurations, and its greedy ranking is also worse than the DP on depth (`N_u_max` 75 vs 34). Phase 3 needs its own placement (item 16), leaving Phase 2's method untouched. **Done** — `budget_unit_count` + `Phase3BudgetDataset._draw_units` (no depth-class draw, no `N//4` clamp, no `stage2_max_units`: the cap IS `⌊B/M⌋`), `--stage2_bypass_singletons` emits the `N_u==1` unit's `K` rows with `kind="qbase"` and a stride-1 slot span via the shared `_unit_slot_span`. | `compressor.py::TwoStageCompressor.compress_windows`, `train/phase3_pretrain_budget.py` | no |
| 14 | **`unified`/`vcd` → Phase-3 meta-JSON converter + per-task held-out split** (§6) — **Done**: `dataset_util/build_phase3_blend.py`. activitynet's dense captions (`{vid: {duration, timestamps, sentences}}`, the only non-LLaVA-shape source) become one human/gpt pair whose answer is the event list with its real `{a}s-{b}s:` ranges; the llava-video `nextqa`/`perceptiontest`/`activitynetqa` families are already LLaVA-shape and are only path-verified and registered. Measured on disk: **14,926 activitynet clips, 0 missing media**; ~57k llava-video VQA entries across 8 subsets, every sampled path present. **`vcd VDC_1k` IS included** (1,027/1,027 clips, five caption turns each), built from the original `VDC_1k.jsonl` whose `video_id` is the on-disk stem. It also carves the **per-task held-out slices** the eval policy above requires (split by video, union across tasks, disjointness asserted) and writes the train registry, the eval registry and the eval manifest in one pass; `--merge` folds in the InternVid meta so one file drives `--multi_dataset`; `--probe_durations` writes the `durations_json` the depth-grouped sampler wants. | `dataset_util/build_phase3_blend.py` | no |
| 15 | **Length-bucketed batched fold** — the fold's pass 2 front-pads every unit to the batch's deepest `N_u`, which the Phase-3 budget mechanism's deliberately-uneven `N_u` turns into a >5× blow-up. `_bucket_fold_tasks` sorts by depth and cuts a new group when the next unit is `>stage2_fold_bucket_ratio` (2.0) times shallower, or when `group_size × N_max` would pass `stage2_fold_bucket_max_segments` (512); a single over-deep unit still gets its own group. Batch rows never interact in the SSD scan, so this is a padding-layout change only — verified equal to the single-call result to fp32 noise (max \|Δ\| 4.8e-07), and a Phase-2-like homogeneous `N_u` still yields one call. Measured: worst real partition 5.78× → 1.18×; ≤1.46× across all 144 probe configurations. **Done.** | `compressor.py::TwoStageCompressor._bucket_fold_tasks` + `compress_windows` pass 2 | no |
| 16 | **Phase-3 unit placement: exact DP on `WCSS + λ·depth_cost`** (§4 Phase 3) — new method alongside `_place_unit_boundaries` (which stays for Phase 2). `O(N²U)` numpy/torch pass; `min_gap` floor drops to 1 so raw-bypass units can exist. `λ ≥ 0.01` (insensitive above that); **`N_u_soft ≈ 1.5–2 × N/U`, NOT a fixed 16** — below `N/U` the quadratic swamps `WCSS` and the DP degenerates to a uniform split. Reference implementation + probe: `eval_ablation/phase3_dp_depth_probe.py`. **Done** — `TwoStageCompressor._place_unit_boundaries_dp` (float64 prefix-sum `WCSS`, `N_u_soft = stage2_dp_soft_ratio · N/U` as a **float**, no `min_gap`), selected by `--stage2_unit_placement dp`; verified to reproduce the probe's partition exactly on 6 `(N, U, λ, ratio)` points. The knee being a float and not the probe's rounded int is load-bearing at small `N` (it moved the partition at `N=121`). **Deterministic — no Gumbel**: Phase-3 per-epoch partition variance comes from the segment-level jitter (`--segment_sample_tau`) moving the DP's input; the WCSS-space jitter analogue stays an open item. | `compressor.py`, `train/phase3_pretrain_budget.py` | no |
| 17 | **Target-`N` qbase segmentation + clamp** (§4 Phase 3) — replace the fixed `segment_target_frames` with `N = clip(c·⌊B/M⌋, N_min(T), T)`, `N_min` from the `force_every` hard floor and the `min_adapt` adaptivity floor; decision `c = 4, min_adapt = 0.5` (identical to `tf=4` on long clips, 69.0 % → 20.4 % zero-fold-gradient on short ones). Separately **add an assert in `adaptive_segment_count`**: a requested `N < ⌊(T−1)/force_every⌋ + 1` silently disagrees with what `adaptive_segment_lengths` actually cuts (`T=1200, tf=12` ⇒ 101 vs 150), corrupting the arch's placeholder count. Sweep: `eval_ablation/phase3_segment_budget_sweep.py`. **Done** — `target_segment_count` / `min_segment_count` / `n_forced_cuts` + `_assert_segment_count_contract` (raises on the `tf=12` case, inert at `tf=4 < force_every=8`), `adaptive_segment_lengths(n_segments=…)`, `TransformerDecoderFlatCompressor.segment_count_for` behind `--segment_count_rule` (`frames` default keeps Phase 1/2 byte-identical; Phase 3 sets `target_n`). Verified equal to the sweep's `plan()` for every `T ∈ [1, 1400)`, and the realised cut matches the promised count with all lengths ≤ `force_every`. | `compressor.py::adaptive_segment_count` / `adaptive_segment_lengths` / `output_len_for`, `train/phase3_pretrain_budget.py` | no |

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
### Held-out policy — **per-task slices, carved from the training pool**

**The rule (user decision, 2026-09-20): every task keeps its own small isolated
eval slice. No whole benchmark is reserved wholesale as eval, and no eval is ever
drawn from the training pool.** Concretely, `dataset_util/build_phase3_blend.py`
splits **each registered sub-dataset** into a training annotation and a held-out
one before anything is trained on:

- **Split by *video*, not by row.** llava-video carries several QA rows per clip
  and VDC five caption turns per clip; a row-level split would put the same
  footage on both sides.
- **The held-out set is the union across tasks.** A clip held out for one task is
  excluded from *every* task's training — activitynet's videos reappear inside
  llava-video's `*_activitynetqa` subsets, so a per-task-only split would train
  on footage it is about to be evaluated on. Identity is the file **stem**, since
  the same clip has different relative paths in different sub-datasets.
- **Size:** `--heldout_frac 0.02` of each task's videos, floor `--heldout_min 20`,
  cap `--heldout_max 300`, and a `--heldout_cap_frac 0.2` ceiling so the floor
  cannot swallow a small task. Seeded by `--seed`, so the split is reproducible
  and re-runnable.
- **Outputs:** `anno_data/phase3_blend.json` (train) /
  `anno_data/phase3_heldout.json` (eval) — two `--multi_dataset` registries — plus
  `eval_ablation/manifest_phase3_heldout.json`, the per-task video manifest the
  `eval_ablation/` probes take. The builder asserts the two sides are disjoint by
  stem and refuses to finish if they are not.
- **Measured (2026-09-20 build):** 523 videos held out across 12 tasks; train ×
  held-out overlap **0**.

This is what makes **VDC usable as training data**: it goes into the blend like
any other task, and it is evaluated on its own held-out slice rather than on the
benchmark in bulk. The same applies to any InternVid registry passed through
`--merge` (split unless `--no_split_merged`) — note the Phase-1/2 runs trained on
the whole `< 420 s` pool, so only a **long-bucket** InternVid held-out slice is
genuinely clean for a Phase-3 checkpoint.

- **Phase 3 also blends `../datasets/unified` + `../datasets/vcd`** (~278k
  clips) directly into the length-bucketed curriculum — see §4 Phase 3 for why
  (the InternVid re-caption pipeline's fixed frame-sample budget caps how much
  temporal detail CE can ever reward, independent of duration). Per-dataset
  signal:
  - **activitynet** (19,994 clips, 648 h, `unified/annotations/activitynet/`):
    real human multi-event annotation, `timestamps: [[s,e], ...]` +
    `sentences` — trains `Time:{a}s-{b}s:` against genuine localization, not a
    frame-capped VLM guess.
  - **llava-video `nextqa` / `perceptiontest` / `activitynetqa`**
    (`unified/annotations/llava-video/`): precise-event VQA (e.g. "what did the
    man in blue do at the end of the video?") — penalizes averaging away
    order/recency in a way a generic caption never does.
  - **vcd `VDC_1k`** (**1,027 clips, all resolved 2026-09-20**): multi-turn,
    multi-granularity detailed captioning (camera work / background /
    main-object / short summary / full detailed caption — five human/gpt turns
    per clip) — far denser supervision than a single InternVid sentence. **It is
    training data**, evaluated on its own held-out slice (policy above). Build it
    from the **original `../datasets/vcd/VDC_1k.jsonl`**, not
    `unified/annotations/vcd/VDC_1k.json`: the unified copy re-ids every clip to
    a UUID whose `vcd/videos/<uuid>.mp4` path does not exist, while the jsonl's
    `video_id` is the on-disk file stem. The ego4d ids live in
    `vcd/videos/videos_3.tar.gz` — extracted 2026-09-20 (35 GB, 200 clips), which
    took resolution from 827/1027 to 1027/1027.
  - Duration profile (`unified/VIDEO_DURATION_REPORT.md`): 277,024 clips total,
    mean 57.8 s, p90 149.6 s — overwhelmingly short/mid, only activitynet's
    tail and a sliver of videoxl reach into the 5–60 min range. Mix these into
    the existing short/mid buckets above; they do not supply new long-bucket
    coverage.

---

## 7. Open decisions

- **K = M vs K > M.** Default K = M = 64 (matches the pretrained qbase). K > M
  (e.g. 128 / 64) makes qbase genuinely higher-resolution and the fold a real 2:1
  reduction.
- **Segmenter criterion — RESOLVED (§4 Phase 1, measured):** fixed count
  `N = ⌊T/4⌋+1`, forced boundary every 8 f + remaining budget to top-`diff`
  positions (Gumbel-top-k for augmentation). Bare threshold-τ / bare top-k
  rejected. Still ablatable: the `diff` score (raw vs common-component-removed vs
  RGB shot boundary) as the *non-forced* ranking.
- **Unit-boundary feature — RESOLVED (§4 Phase 2):** inter-segment cosine on the
  **stage-1 qbase segment tokens** (mean-pooled `K` per segment), not the raw
  frozen-encoder feature. Still ablatable: raw-encoder feature;
  common-component-removed encoder feature; RGB shot boundary.
- **RoPE-index scale for the compressed region — OPEN (§1, §5 item 11).** Default:
  readout stride `N_u·K / M` ⇒ region footprint `≈ N·K` slots (`≈ 16·T`; matches
  the Phase-1 all-qbase footprint; the readout-only set `{m·N_u}` is exactly
  collision-free at `K = M = 64`). Phase 2 (`T ≲ 320` ⇒ `N·K ≲ 5.2k`) is fine on
  the default. Phase 3's 420–1200 s clips push `N·K` to ~19k slots + surrounding
  text → near Qwen2's 32k; fallback is a seconds scale (`stage2_rope_slot_scale =
  S` slots/s, `S ≈ 4–8`), accepting coincident readout slots.
  **Quantified 2026-09-20:** under the adopted `c=4, min_adapt=0.5` rule only
  **5.0 %** of InternVid clips exceed 16k `N·K` slots (p90 = 14,272), so the
  seconds-scale fallback is needed for a tail, not the bucket. `min_adapt = 0`
  would take it to 0.0 % (p90 7,168) but costs most of the segmenter's
  adaptivity (`adapt%` 54.8 → 19.1) — rejected.
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
- **Phase-3 fold-depth safety valve — RESOLVED (2026-09-20, measured).** Take
  alternative (B) in its soft form: the DP on `WCSS + λ·depth_cost` (§5 item
  16), `λ ≥ 0.01`, `N_u_soft ≈ 1.5–2 × N/U`. Measured on held-out video this
  bounds `max_u N_u` to ~34 where pure similarity reaches 196, and under the
  target-`N` segmentation rule below the regime never arises at all (`N_u`
  p99 ≤ 18.2). The *padding* cost that motivated much of the concern is a
  separate, now-fixed batching issue (§5 item 15, ≤1.46×). (F) hierarchical
  fold-of-folds is no longer needed for depth and stays where it belongs — the
  streaming path. Still unaddressed: the `A_init_range` effective-memory-horizon
  argument in "Potential problems" #1, which is a property of the recurrence,
  not of the partition.
- **Phase-3 qbase segmentation granularity vs. the budget mechanism —
  RESOLVED (2026-09-20, measured).** `target_frames` was the wrong control
  variable: the budget is denominated in tokens, the cut in frames. Replaced by
  `N = clip(c·⌊B/M⌋, N_min(T), T)` with `c = 4, min_adapt = 0.5` (§4 Phase 3,
  "qbase segmentation: target `N` + clamp"; §5 item 17). The feared trade-off —
  finer cuts for short clips growing `N` and worst-case depth at the long end —
  does not materialise, because `c·⌊B/M⌋` sits below the `force_every` floor for
  anything long: `c` moves only short clips, `min_adapt` only long ones, and
  `min_adapt = 0.5` reproduces today's `tf = 4` exactly
  (`target_frames = force_every × (1 − min_adapt)`).

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
   forward). Item 3's fixed `N = ⌊T/4⌋+1` means Phase 1 does **not** wait on an
   arch refactor — collator and model agree on the length.
3. **Phase 1 run** on short InternVid → gate on the §2.3 probe
   (`eval_ablation/struct_probe.py`: `struct_ρ`, `xcos_cen`, `cc_frac`) +
   frozen-LLM caption eval.
4. **Code (done):** item 11 (`prepare_inputs_labels_for_multimodal` consumes
   model-returned `unit_meta` + baked `Time:` + strided `position_ids`) + item 6
   (model-side seeded unit boundaries on the qbase segment tokens) + item 12
   (qbase-only replay path).
5. **Phase 2 run** → gate (`struct_ρ`, `cc_frac` back off 1.0, caption plateau,
   U-cap budget ablation, pure-qbase replay caption holds).
6. **Code (no GPU) — DONE 2026-09-20:** item 15 (length-bucketed batched fold) →
   item 17 (target-`N` segmentation + the `adaptive_segment_count` assert) → item 16
   (DP on `WCSS + λ·depth_cost`) → item 13 (budget-driven `U`, `N_u==1` bypass,
   `stage2_max_units` / `u_cap` clamp removal) → item 14 (`unified`
   converter; `vcd` excluded, see item 14). Items 15–17 are each independently checkable without a training
   run: `eval_ablation/phase3_dp_depth_probe.py` (depth + padding on real
   held-out video) and `eval_ablation/phase3_segment_budget_sweep.py`
   (segmentation rule, pure duration arithmetic).
   The order is not arbitrary — the items sit on the two different cuts of the
   pipeline, outer first:

   ```
   T frames ──[item 17]── N segments ──[item 16]── U units ──[item 13]── K or M tokens/unit
              target N + clamp        DP on WCSS+λ·depth_cost   N_u==1 → raw bypass
   ```

   **17 must land before 16**: item 16's only real knob, `N_u_soft`, is defined
   relative to `N/U`, so `N` has to be settled first — and the depth results
   that justify item 16 (`N_u` p99 ≤ 18.2, never the `N_u > 100` regime) were
   measured *under* the target-`N` rule, not under `target_frames = 4`.
   Item 13 then joins the two: the budget-derived `U`, the bypass branch, and
   removal of the Phase-2 caps (`stage2_max_units = 5`, `_draw_units`'
   `u_cap = min(max_units, N//4)`) that would otherwise hold short clips below
   the budget.
7. **Code — DONE 2026-09-20:** item 4 (chunked encoder forward,
   `--vision_encoder_chunk_frames 8`) — no longer optional at Phase-3
   lengths — plus the length settings it forces, now the launcher's defaults:
   `--vision_max_tokens 65536` (up from the per-video 16,384 default, which
   crushes a 1200-frame clip to ~8 tokens/frame) and `--model_max_length 131072`,
   above the resulting *pre-compression* `input_ids` length, which
   `Phase2FoldDataset.__getitem__` checks and otherwise rejects every long clip.
   The LLM never sees that length — the rewrite happens before `embed_tokens` —
   so this only widens a guard.
8. **Phase 3 run** (no code item left from the retained-set idea — see §4 Phase 3)
   + long-video needle probes + length curriculum. A 3-step smoke run of the
   full chain (Phase-2 warm start → target-`N` cut → budget `U` → DP → mixed
   bypass/fold rows → CE) is green; the real run has not been started.

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
- **Eval is a per-task held-out slice carved out of the training pool** (§6
  "Held-out policy"), never a whole benchmark reserved wholesale and never an
  in-pool sample. Split by video, union across tasks, disjointness asserted by
  `dataset_util/build_phase3_blend.py`. This is why VDC can be training data.
- **LLM frozen every phase; `mm_projector` unfrozen.** Every checkpoint is also
  evaluated as *compressor + the original frozen LLM* against a no-compression
  baseline, so a gain is attributable to compression and not to anything downstream.
- **No retained set, in any phase.** A per-unit raw-qbase escape hatch (`r_u`) was
  tried and dropped (§4 Phase 2) — do not reintroduce it.
- **Frame-order / temporal-blindness metrics are a diagnostic, not a gate.** Every
  fold checkpoint evaluated so far shows flat frame-order insensitivity regardless
  of time-embed choice or qbase-unfreeze — read as a property of CE-on-generic-
  captions training, not a compressor defect to engineer away.
