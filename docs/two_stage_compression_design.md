# Dynamic video compression — qbase curriculum + SSD fold

**This document supersedes** the earlier plan (kept in git history): the qbase
pretrained on whole-video windows against a frozen LLM, then an SSD fold bolted on and
trained separately on videoxl caption sets, then the qbase unfrozen. That plan *ran to
completion* and its fold readout **collapsed cross-video** — worse caption quality than
the qbase alone despite 2× the tokens. §2 has the measured post-mortem.

## Status (2026-10-04)

**Curriculum (renumbered 2026-10-02, user decision): Phase 1 → Phase 2, both qbase-only,
LLM frozen.** Phase 1 pretrains the qbase (InternVid `<180 s` + still images); **Phase 2 is
qbase enhance** — qbase-only CE on the `/root/datasets` blend + still images, mixed 1–4
frames/segment. Phase 2 is a required step, not an optional polish: the Phase-1 qbase is
not capable enough to build on — at one frame per segment it lost to plain 4×4 feature
pooling at equal tokens (`eval_ablation/qbase_vs_pool.py`, 2026-09-29).

The two fold stages that used to be Phase 2 and Phase 3 are **moved out of the phase
numbering** into the parked fold line (§4b) — the **unit fold** (formerly Phase 2) and the
**budget fold** (formerly Phase 3). Where they re-enter the curriculum is undecided. Both
ran. Run history:

| run (`work_dirs/`) | stage | entrypoint | data | trains | outcome |
|---|---|---|---|---|---|
| `phase1_qbase_internvid` (on disk) | Phase 1 | `compressor_pretrain_with_videollama3.py` | `phase1_blend.json`: InternVid `<180 s` + 40 % still images | qbase + `mm_projector` | 3,441 steps / 3 epochs. Images: −0.561 nats/token on unseen images vs the warm start, but read as LLM-interface adaptation — OCR gap vs uncompressed still +1.0–1.2. Epochs 2–3 ran under the DeepSpeed GA-boundary bug (fixed 2026-09-28). |
| `phase2_fold_frozen_qbase` (+ `_ep2fix`) | unit fold | `phase2_pretrain_fold.py` | `phase2_internvid.json`: InternVid 180–420 s, 273k | fold + split projectors, **qbase frozen** | `checkpoint-500` is the best held-out point; a correctly re-run epoch 2 lowers train loss but worsens held-out CE by +0.0136 nats/token [+0.0118, +0.0156]. |
| `phase3_budget_fold` | budget fold | `phase3_pretrain_budget.py` | the 2026-09-21 blend: 419,652 rows, /root/datasets 65 % + InternVid 35 % | qbase (then-default 3e-6) + fold + projectors | 2 epochs; epoch 2 flat (and under the GA bug). Warm-started from an older qbase-unfrozen unit-fold run. |
| `qbase_enhance` | Phase 2 (first run) | `qbase_enhance_pretrain.py` | `phase3_blend_img.json` | qbase + `mm_projector` | +2.0–3.6 pp over the budget-fold qbase on Video-MME, still 3.7–4.4 pp under the best raw dynamic-HW arm at equal budget; long video: Video-MME long −1.5 pp [−2.9, −0.0], LongVideoBench +0.2 pp (n.s.). Segmentation ablation ruled out partitioning as the cause. |
| `qbase_enhance_mlvuclean` (on disk) | Phase 2 (latest checkpoint) | `qbase_enhance_pretrain.py` | `phase3_blend_img_mlvuclean.json` | qbase + `mm_projector` | Clean retrain after the MLVU overlap was found; finished 2026-10-03 (1,412 steps / 2 epochs). Full benchmarks, equal budget, best vs best: Video-MME −2.1 pp, LongVideoBench 0.0, MLVU +2.0 — the loss is per-frame local detail (Video-MME Object Reasoning −5.5). §4 "MLVU-clean Phase 2". |
| `qbase_qa_mlvuclean` (on disk) | H1 test, one epoch past Phase 2 | `qbase_enhance_pretrain.py` | `qa_qbase_mlvuclean.json` | qbase + `mm_projector` | Did the caption-dominant objective hide what the qbase could keep? QA-only CE moved nothing: Video-MME 880 −0.4 / +0.5 pp, held-out per-turn CE within ±0.003 nats/token, all n.s. — rejected. |

Every run dir except the three marked "on disk" is no longer under `work_dirs/`; the numbers
above are kept as the record. Launcher defaults that still point at the missing dirs are
listed in `CLAUDE.md`.

**Where it stands (2026-10-04).** The MLVU-clean Phase-2 qbase meets the raw-frame bar (§4
"Bar") on LongVideoBench and MLVU but not on Video-MME; the gap is per-frame local detail
(on-screen text, small objects, binding a name to a person), and switching the supervision to
QA did not move it (H1, §4). The same-day inference-only follow-ups (§4 "Inference-time
follow-ups") localise it further: on its robust failures the qbase gets the gist right and
misses a detail; 8 frames per segment cost nothing against 1–4; and an inference hybrid —
raw keyframes next to the qbase segments (`lowraw`) — ties the best raw arm at B = 2048 but
does not beat it, stops gaining after B = 4096 while raw keeps rising (long video falls
behind), and is at parity on short clips in a pre-registered confirmation.

**Naming.** Training-phase vocabulary is **Phase 1 / Phase 2** (§4); the fold stages are
named, not numbered (§4b). Code, launcher and data names keep the numbering of the time
they were written: `phase2_pretrain_fold.py` / `shell/pretrain_phase2_internvid.sh` /
`phase2_internvid.json` = the **unit fold**; `phase3_pretrain_budget.py` /
`shell/pretrain_phase3_budget.sh` = the **budget fold**; `phase3_blend*.json` / `phase3_heldout.json` /
`manifest_phase3_heldout*.json` = the `/root/datasets` blend Phase 2 trains on and its
held-out slice; Phase 2 itself runs `qbase_enhance_pretrain.py` /
`shell/pretrain_qbase_enhance.sh`. Older dated text — §4b, §5's item descriptions, the
dated subsections of §6, the early steps of §8 — also keeps the old numbers: there
"Phase 2" = unit fold, "Phase 3" = budget fold.

In code the two modules are `TwoStageCompressor.qbase` / `.fold`, configured by the
`fold_*` config fields / `--fold_*` flags (renamed from `stage1` / `stage2` / `stage2_*` on
2026-10-04); the per-group LR split and the warm-start loaders key on the `qbase` / `fold`
names. The per-group LR CLI flags: `--qbase_lr` and `--mamba_lr` (overrides
`--compressor_lr` for the fold group when set).

**What changed vs. the superseded plan**

| | superseded | current |
|---|---|---|
| qbase / fold trained | separately (qbase frozen while fold trains) | the curriculum trains the qbase alone (Phases 1–2); in the parked fold line the unit fold kept the qbase frozen, the budget-fold launcher trains both jointly (`QBASE_LR=1e-5`) |
| pretrain data | videoxl caption sets (~73k) | Phase 1: **InternVid** Qwen3-VL re-captions (`anno_online/internvid/pool_all_qwen3vl.json`, 977,422) `<180 s` + still images; Phase 2: the `/root/datasets` blend + 20 % images (§6); unit fold: InternVid 180–420 s |
| video length | ≤160 f, caps most clips | ≤180 frames at 1 fps in Phases 1–2; unit fold 180–420 s (≤320 frames); the 420–1200 s budget-fold curriculum was retired with InternVid's long bucket |
| qbase segments | whole-video window, or uniform 4-frame (superseded runs) | **fixed count `⌊T/4⌋+1`, adaptively placed** 1–8 frame segments (forced-8 + top-`diff`); Phase 2 draws `tf ∈ {1..4}` per sample; the budget fold uses the target-`N` rule |
| LLM | frozen | frozen in Phases 1–2 and the fold stages |
| `mm_projector` | frozen | **unfrozen**; a two-stage (fold) model gives qbase rows and the fold readout their own copy each (`mm_projector_qbase` / `mm_projector_fold`) instead of forcing one projector to track both moving manifolds |
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

The design gives the LLM **two levels of detail** per video (the curriculum, Phases 1–2,
currently trains only the first; the fold is the parked line of §4b):

- **qbase (hi-res).** The video is cut into `N = ⌊T/4⌋+1` **1–8 frame segments** —
  a *fixed count* (so the compressed length is predictable, §4), *adaptively
  placed*: boundaries land on the largest frame-to-frame encoder-feature changes,
  with a forced cut every 8 frames. Each segment → **K = 64** LLM-readable tokens.
  Dense action ⇒ 1-frame segments there; a calm span ⇒ 8-frame segments — same
  total, redistributed.
- **fold (coarse / gist).** An SSD associative fold folds a *unit*'s segment
  summaries into **M = 64** tokens with an **O(1) recurrent state** — depth grows
  with video length, token count does not.

Per video, in time order, the LLM sees for each unit `u` (`U` contiguous units:
`U ≤ 5` in the unit fold, `U = min(N, ⌊B/M⌋)` in the budget fold), the same format in every stage:

```
[ Time:{a}s-{b}s: ] <|compression_start|>  unit u's M readout tokens  <|compression_end|>
```

- **Vision-token cost to the LLM** = `U·M` (`≤ 5·64 = 320` in the unit fold, `≤ B`
  in the budget fold) — bounded and predictable for any video length. In the budget fold a unit holding a
  single segment emits that segment's `K = M` raw qbase tokens instead of a fold
  readout (raw bypass, routed to `mm_projector_qbase`), so the cost formula is
  unchanged.
- A still image (`compression_is_image`) gets the `<cs> … <ce>` block with **no**
  `Time:` string.
- **RoPE-index footprint — sparsified readout.** The compressed tokens are **not**
  given consecutive `position_ids`. Readout token `m` of unit `u` is placed at
  stride `N_u·K / M` (the unit's compression ratio; `= N_u` at `K = M = 64`), so
  unit `u` spans `N_u·K` position slots and the whole compressed region spans `N·K`
  — its uncompressed qbase footprint — regardless of how many tokens it actually
  holds. Every stage then places the video on an identical positional scale.
  (Approximation: the region also carries the per-unit `Time:` / `<cs>` / `<ce>`
  text at stride 1, so its true footprint is `Σ N_u·K` + those few text tokens.)
  `compress_windows` **emits each unit's readout rows in ascending slot order**,
  with `pos_offsets` carrying that order. The frozen LLM reads relative distance
  from RoPE; `Time:{a}s-{b}s:` is only a discrete hint. Same formula at train and
  inference so streaming readout positions track real elapsed time. Scale for very
  long clips: §7.

**Phases 1 and 2 are the degenerate case:** no fold, no units — every segment's qbase
tokens pass straight through ⇒ the LLM sees `N·K` tokens (qbase per semantic
segment, straight to the projector) behind one `Time:{first}s-{last}s:` range. The unit
fold's **pure-qbase replay** path (§4b, no longer in any registry) reuses this layout
inside a two-stage model.

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
  an **open problem for the budget fold** (an earlier draft's answer — a per-unit raw-qbase
  retained set — was tried and dropped, see §4b unit fold; not to be reintroduced).

### Vocabulary

| sym | meaning | set by | typ. |
|---|---|---|---|
| **T** | decoded frames (fps = 1 ⇒ ≈ seconds) | `load_video`, `--max_frames` | ~180–320 (unit fold) |
| **N** | **segments** the video is cut into (qbase does not change this — it changes *tokens per segment*) | `⌊T/4⌋+1` (fixed count, adaptive placement); each seg 1–8 f. Phase 2: `⌊T/tf⌋+1`, `tf ∈ {1..4}` per sample; budget fold: `clip(c·⌊B/M⌋, N_min(T), T)` | ~46–81 (unit fold) |
| **K** | tokens qbase emits **per segment** — fixed | `num_queries` (pretrained) | 64 |
| **M** | tokens **one** fold readout emits **per unit** | `fold_n_summary_tokens` | 64 |
| **U** | readout units (independent folds), **capped** | unit fold: `round(N / N̄_u)`, clamp `[1, min(fold_max_units, ⌊N/4⌋)]`; budget fold: `min(N, ⌊B/M⌋)` | ≤ 5 (unit fold), ≤ 16 (budget fold, B=1024) |
| **N_u** | segments in unit `u` = that fold's depth; `Σ_u N_u = N` | boundary placement (§4b unit fold), grows with length at inference | soft-bounded by `min_gap ≈ N/(2U)`; no hard `[4,16]` clamp |
| — | frames per segment | segmenter, clamp **[1, 8]** | ~1–8 |
| — | vision tokens to the LLM | `U·M` (same formula every fold stage) | ≤ 320 |
| — | `position_ids` slots the compressed region spans (RoPE footprint) | readout stride `N_u·K/M` | `≈ N·K` (= the Phase-1 footprint; + the per-unit `Time:`/`<cs>`/`<ce>` text tokens) |

Chain (fps = 1, **unit fold**): `T frames → N = ⌊T/4⌋+1 segments → N·K qbase tokens →
U units (Σ N_u = N) → per unit: fold N_u segments → M readout tokens`. Worked
example, T = 240: N = 61; draw `N̄_u = 13` ⇒ U = 5; partition e.g.
`N_u = [13,12,13,11,12]`; the LLM sees `5·64 = 320` compressed tokens across
`≈ N·K = 3904` `position_ids` slots. Same chain, same formula in the budget fold.

---

## 4. Training curriculum — Phase 1 → Phase 2, **LLM frozen**

**Common to every phase**

- Data: Phase 1 trains on InternVid Qwen3-VL-8B re-captions
  (`anno_online/internvid/pool_all_qwen3vl.json`, 977,422 clips; the re-caption run is
  complete) `< 180 s` + still images; Phase 2 on the `/root/datasets` blend + 20 % still
  images (§6). fps = 1 always. Dynamic HW (native aspect ratio, per-video token budget;
  `--force_image_size` **not** set). On-the-fly decode + frozen-encoder forward, **no
  feature cache**.
- Single CE forward, `B == 1` flattened collator, `use_dual_forward=False`. **ZeRO-1 in
  every run** (`shell/zero1.json`). ZeRO-2 was tried for the budget fold and NaN'd the trainable
  `embed_tokens` partition on the first optimizer step; it is banned.
- **`mm_projector` unfrozen** every phase. Phases 1–2 have one compressed-token
  manifold (qbase), so they keep the single shared `mm_projector`. The fold stages have
  two independently-moving manifolds (qbase rows, fold readout) sharing
  the vision-encoder scale but not the same distribution, so `TwoStageCompressor`
  models get `mm_projector_qbase` / `mm_projector_fold` instead. They are seeded from the
  run that trained the qbase (`_warmstart_projectors_and_compression_embeds`, §5 item 19)
  and are free to diverge. All three projectors run on every row every step and
  `_apply_mm_projector` selects per row (a skip-if-absent desynchronises ZeRO's gradient
  buffer across ranks); a raw/uncompressed row still uses the plain shared
  `mm_projector`. **LLM frozen** every phase.
- **Option A + Option B on every compressor output** — the qbase output and, on a
  `TwoStageCompressor`, each fold unit's readout (§5 item 1).
- Length buckets from `anno_data/internVid_durations.json` (built by the full
  ffprobe scan): **< 180 s ≈ 264k · 180–420 s ≈ 262k · 420–1200 s ≈ 451k**
  (> 1200 s: 98 clips, dropped). Phase 1 trains the first, the unit fold trained the
  second; the third was trained only by the first budget-fold run (§6).
- **No LLM warm-up / adaptation stage** inside the curriculum. (The earlier "Phase 0: finetune
  the LLM on Qwen3-VL-style captions" is dropped — it lowers the CE floor and *weakens*
  the compressor gradient, and it muddies the frozen-LLM control.) The residual
  style-mismatch in CE is not compressor signal; that is why the gate is the
  representation-level probe (§2.3), and why caption eval always also runs
  **compressor + the original frozen LLM**.

### Phase 1 — qbase only, content-adaptive segments, **NO fold**

**Still images are the degenerate case of this same path** and are the densest
supervision it can get. `T = 1` ⇒ `adaptive_segment_count(1) = 1` segment ⇒ `K = 64`
tokens, with the whole per-sample vision budget spent on that one frame (bunny's
native grids are p50 216–999 tokens ⇒ a 3.4–15.6× squeeze). InternVid captions are
generated from a FIXED 64-frame sample regardless of duration
(`shell/recaption_vllm.sh`), so nothing in the video stream rewards detail finer
than that sampling density — which is exactly §2.2's "the qbase was pretrained on
very sparse input". Images supply the missing pressure: OCR strings, attributes,
counts, spatial relations, all in one frame. Three constraints, all measured:

- **No `Time` marker.** An image has no timeline.
  `_rewrite_image_block_as_single_frame_video` drops its timestamp so the chat
  template renders a bare `<image>` (no `Time X.0s:` prefix), `build_range_ts_info`
  returns `(0, [])` so nothing is cut back, and the datasets set
  `compression_is_image=[True]` so the two-stage path does not emit a degenerate
  `Time:0s-0s:` range either (§5 item 18).
- **An image must hold ≥ K vision tokens.** It is ONE segment, so the compressor
  emits `K` rows into a part that holds only that image's own tokens; anything
  smaller overflows the part (the assert in
  `compress_visual_tokens_with_compressor`). Enforced twice: the builder drops
  sub-K images *at native resolution* (upscaling a thumbnail invents no detail
  while still costing a full segment), and `--vision_min_tokens 64` is the
  processor-level guard.
- **Mix ratio is the row count.** The meta format has no per-entry weight, so
  `--image_share` subsamples the image tasks. Too high and the qbase becomes a
  single-frame expert and loses the 1–8 frame segment behaviour it is actually
  deployed with; the default is 0.40.

- **Data:** every clip **< 180 s** (~264k) — **including the very short ones**.
  Short clips are useless for the fold but perfectly good qbase signal; do not drop
  them. **Plus a still-image stream (added 2026-09-26, §6 "Phase-1 image
  stream")**: 176k samples at a 40 % row share, `anno_data/phase1_blend.json`.
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
- **Path:** frozen encoder forward (single-shot — the Phase-1 launcher leaves
  `--vision_encoder_chunk_frames` at 0; chunking (§5 item 4) is only needed at longer
  lengths) → segmenter → qbase per segment → K = 64 tokens/segment → **all** segments'
  tokens → `mm_projector` (trainable) → frozen LLM → CE. The trainer logs per-stream CE
  (`loss_img` / `loss_vid` / `frac_tok_img`) for the image + video blend.
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
- **Run (2026-09-27): `work_dirs/phase1_qbase_internvid`** — `phase1_blend.json`,
  3,441 steps / 3 epochs / ~14 h, clean log. `token_compressor` moved ~9 % from the
  warm start (cross-attention most). Epochs 2–3 ran under the DeepSpeed GA-boundary bug
  (§5 item 23), fixed only afterwards. What the images bought, measured answer-grounded
  (`eval_ablation/phase1_per_task_ce.py`, vs the never-saw-an-image warm start): better
  on all 6 image tasks, and on 2,866 never-seen `pretrain.json` images −0.561 nats/token
  [−0.668, −0.456] with no memorisation signal. But the compressed-minus-uncompressed gap
  is NEGATIVE (−0.54), i.e. the gain is an easier format for the frozen LLM, not retained
  detail — OCR tasks still cost +1.0–1.2 nats/token vs the uncompressed path, and the
  qbase's cross-attention does not track edge energy at all (a near-uniform learned
  pooling). This checkpoint is the warm start of every later qbase run.

### Phase 2 — qbase enhance: qbase-only CE on the `/root/datasets` blend, mixed 1–4 frames/segment

Entrypoint `videollama3/train/qbase_enhance_pretrain.py`, launcher
`shell/pretrain_qbase_enhance.sh`. Single-stage qbase CE pretrain
(`transformer_decoder_flat` + adaptive segmentation, no fold), LLM frozen, warm-started
from Phase 1 (qbase + its trained projector + the two compression-token rows).

**Why it is a phase (2026-10-02).** The qbase has to go through this step to be capable
enough to build anything on. Phase 1 trains it on captions written from a fixed 64-frame
sample, i.e. sparse targets, plus still images that bought an easier LLM interface rather
than retained detail (§4 Phase 1). Measured with segmentation and the fold removed, that
qbase loses to plain 4×4 feature pooling and to downscaling by 4.5–6.8 pp on Video-MME at
equal tokens (`qbase_vs_pool.py`, §4b). Phase 2 retrains it on the densest supervision on
the machine — GPT-4o dense captions, open-ended and multiple-choice QA, multi-event
timestamps, multi-granularity captions — at the 1–4 frames/segment operating points it is
deployed at.

- **Mixed granularity.** Each `__getitem__` draws a fresh random `compression_seed`; the
  sample's frames/segment is `tf = choices[seed % len(choices)]` with
  `--segment_target_frames_choices 1,2,3,4` (`compressor.py::pick_target_frames`, called
  with the same seed by the dataset and the model, so the placeholder count and the cut
  agree); `N = ⌊T/tf⌋ + 1`. `tf = 1` is `qbase_vs_pool.py`'s operating point, 2–4 keep the
  multi-frame segments a fold would need. Without a seed (eval / inference) the model uses
  `--segment_target_frames` (4). Every choice must be `< segment_force_every`.
- **Data.** The `/root/datasets` video blend (`phase3_blend.json`, §6) + 20 % still images
  (`phase3_blend_img.json`; since 2026-10-02 the MLVU-clean variant). Caption rows that tag a clip `<image>` are
  normalised to `<video>`.
- **Geometry.** `MAX_FRAMES=180` (uniform subsample above), `VISION_MAX_TOKENS=65536` per
  video with a per-frame cap `--vision_max_tokens_per_frame 2048` (stops a short high-res
  clip pouring the whole budget into a few frames; the encoder's per-frame attention is
  quadratic), `VISION_MIN_TOKENS=64`, one encoder call per video
  (`VISION_CHUNK_FRAMES=$MAX_FRAMES`), compressor activation checkpointing on.
- **Optimisation.** `COMPRESSOR_LR=2e-5`, `MM_PROJECTOR_LR=1e-6`, `GLOBAL_BATCH=480`,
  2 epochs, Option A/B on, `NUMPY_MADVISE_HUGEPAGE=0` (data workers otherwise stall in
  kernel page compaction on this host).
- **Rank load balance** (`--group_by_encode_cost` + `--media_geometry_json`, §5 item 21):
  each optimizer step's uniformly drawn sample set is dealt into equal-estimated-cost
  micro-batches, so no rank idles at the step-boundary allreduce.
- **Result, first run (`qbase_enhance`, 1,766 steps).** Video-MME (n = 880, 1/3 of the
  clean set), accuracy %, equal budget:

  | B | Phase 2 qbase | budget-fold qbase | fixed448 | resize | best raw dynamic-HW | raw full (4×B, not budget-legal) |
  |---|---:|---:|---:|---:|---:|---:|
  | 1024 | 56.9 | 54.9 | 54.4 | 60.1 | 60.8 (32 f) | 63.3 |
  | 2048 | 58.4 | 54.8 | 57.2 | 61.5 | 62.2 (64 f) | 64.9 |
  | 4096 | 60.8 | 57.4 | 63.3 | 65.0 | 65.2 (64 f) | 64.3 |

  Long video (`long_mcq_budget.py`): Video-MME long (900 q) −1.5 pp [−2.9, −0.0]
  frame-matched — reasoning −1.9, perception +0.3, the loss concentrated in Object
  Reasoning; LongVideoBench ≈ parity. The qbase only "wins" where raw is starved to
  16 tokens/frame. `long_mcq_segment.py` ruled out segmentation (uniform cuts, or
  training-like local 1-fps segments, change nothing): the deficit is the per-segment 64
  qbase tokens being worth less than a 64-token raw frame. Epoch 2 bought < 1 pp.
- **Bar.** The best budget-legal raw arm with dynamic HW and a frame-count sweep
  (`qbase_budget_ceiling.py`), with `qbase_vs_pool.py` as the per-frame check. Pick
  checkpoints on the held-out caption CE + held-out MCQ slices, not on Video-MME (the
  launcher header's rule); external benchmarks are for the final verdict only. The fold
  line stays parked until a Phase-2 qbase clears the bar. As of 2026-10-04 the clean
  Phase-2 qbase clears it on LongVideoBench and MLVU, not on Video-MME (next subsection).

### MLVU-clean Phase 2 (2026-10-02, finished 2026-10-03)

`dataset_util/frame_hash_overlap.py` found 204 training identities whose content matches
MLVU dev MC videos (`anomaly_reco` 165/200 clips are UCF-Crime clips already in
`vxl_anomaly_det`; 86 `order` videos reuse ActivityNet training clips), which an id-based
check misses because MLVU renames clips. `dataset_util/exclude_overlap.py` wrote
`anno_data/{phase3_blend,phase3_blend_img}_mlvuclean.json`. The clean chain is
Phase-1 checkpoint → Phase 2 on `phase3_blend_img_mlvuclean.json`
(`work_dirs/qbase_enhance_mlvuclean`, finished 2026-10-03, 1,412 steps / 2 epochs).

**Evaluation of the clean checkpoint (2026-10-03/04).** Same checkpoint throughout, raw arms
on the original base model (LLM asserted identical), contamination-filtered question lists.

- **Equivalent to the first run.** Untouched `qbase_budget_ceiling.py` on the same 880
  Video-MME questions (`work_dirs/qbase_ceiling_mlvuclean/`): qbase 57.5 / 58.6 / 59.2 at
  B = 1024 / 2048 / 4096 vs the first run's 56.9 / 58.4 / 60.8; the raw arms reproduce the
  09-30 numbers within ±0.7 pp, so the environment did not move.
- **Short clips at strict 1 fps** (`short_mcq_1fps.py`, Video-MME short, 873 q): qbase ≈ raw
  at the same frames and tokens (tf 1 / 2 / 4: −1.0 / −1.4 / +2.2 pp, n.s.), 0.7–2.6 pp
  under the best raw frame count, which prefers 20–40 sharper frames. Inference hybrid
  (`short_mcq_hybrid.py`, each segment's 64 qbase rows + one raw keyframe): parity with the
  best raw at equal budget (−1.1 … +0.5 pp, n.s.); splitting a budget between qbase and
  keyframes beats spending it all on qbase (+2.3 pp [+0.3, +4.4] at ~5k tokens).
- **Long-video hybrid** (`long_mcq_hybrid.py`, Video-MME long 900 q + LongVideoBench 1,322 q):
  on Video-MME long every qbase-containing arm trails the best raw arm by 2.7–4.4 pp, and
  keyframes alone (half the tokens) beat the hybrid by 1.9–3.1 pp — but the keyframes-only
  arm carries per-frame `Time` stamps while the hybrid's keyframes sit inside one `<cs>`
  block behind one range (format confound, untested). LongVideoBench: parity.
- **Full-benchmark profile** (`long_mcq_budget.py --raw_divs 256,128,64,32,16 --tf 1,2,4`
  over all of Video-MME (2,619 q), LongVideoBench val (1,322 q) and MLVU dev MC (2,173 q),
  0 error rows → `work_dirs/full_bench/<bench>/profile.txt` via `long_mcq_profile.py`).
  qbase − raw, pp, averaged over B, 95 % cluster-bootstrap CI by video. "Same frames" pairs
  qbase tf 1 / 2 / 4 with raw at 64 / 32 / 16 tok/frame on literally the same frames; "best
  vs best" picks each side's best arm per B on the full set (optimistic for both):

  | | Video-MME | LongVideoBench | MLVU |
  |---|---:|---:|---:|
  | same frames, 64 tok/frame | −2.9 [−3.9, −1.8] | +0.4 [−1.0, +1.9] | +0.1 [−0.9, +1.1] |
  | same frames, 32 tok/frame | −2.2 [−3.3, −1.2] | +1.7 [+0.3, +3.4] | +1.8 [+0.8, +2.9] |
  | same frames, 16 tok/frame | +0.8 [−0.3, +2.0] | +4.7 [+3.0, +6.4] | +4.4 [+3.1, +5.6] |
  | **best vs best** | **−2.1 [−3.2, −1.0]** | **0.0 [−1.7, +1.7]** | **+2.0 [+0.8, +3.2]** |

  Losses: Video-MME Object Reasoning −5.5 best vs best (−4.9 same frames at 64 tok/frame);
  at 64 tok/frame OCR −5.1, Attribute Perception −4.6, Object Recognition −3.4 (all three
  turn positive against 16-tok/frame raw); LongVideoBench 8–15 s clips −6.0 best vs best (raw
  spends the budget on a few sharp frames; the qbase's 64 tokens per segment cannot — note
  the decode grid there also puts the qbase far above its 1 fps training rate). Wins: MLVU
  `count` +8.9, `order` +6.4 best vs best; MLVU without `anomaly_reco` +2.4. The 880-question
  "Counting +10.6" seen on Video-MME earlier did not replicate (full set −0.3). Report:
  https://claude.ai/artifact/MuxycAut7CKeeG7mktbDdg (2026-10-04).
- **qbase vs pooling on this checkpoint** (`qbase_vs_pool.py`, the same 880 Video-MME
  questions, 16 / 32 / 64 frames at 64 tok/frame, per-type split by
  `qbase_vs_pool_split.py`): 4×4 pooling beats the same-pixel qbase arm by 3.0 pp
  [1.0, 5.0] and the training-geometry arm (`qbase_native`, more pixels than pool) by 2.1
  [0.2, 4.1]; resize beats them by 3.7 / 2.8. The Phase-1 checkpoint on the same questions:
  −8.5 / −8.6 vs pool. Caveat: pool / resize run in the base model's format (per-frame
  `Time X.0s:`, no `<cs>`/`<ce>`) while the qbase rows sit in one block behind one range, so
  this compares operator and format together.
- **H1 — "the caption-dominant objective hides what the qbase could keep" — rejected.**
  QA turns are 43 % of the Phase-2 blend's turns but 3.4 % of its answer characters (§6), so
  per-token CE barely rewards what an MCQ needs. Test: the Phase-2 checkpoint + one epoch
  (440 steps) of qbase-only CE on `qa_qbase_mlvuclean.json` (§6), everything else as the
  Phase-2 launcher (`META_PATH=anno_data/qa_qbase_mlvuclean.json
  PRETRAINED_COMPRESSOR_PATH=work_dirs/qbase_enhance_mlvuclean
  OUTPUT_DIR=work_dirs/qbase_qa_mlvuclean EPOCHS=1 shell/pretrain_qbase_enhance.sh`);
  pre-registered win: Video-MME +2 pp. Result, paired against the Phase-2 checkpoint:
  Video-MME 880 −0.4 pp [−1.4, +0.7] (same-pixel arm) / +0.5 [−0.7, +1.6] (training
  geometry); held-out CE per answer turn (`heldout_turn_ce.py`, 24 tasks / 3,777 turns)
  caption +0.0003, open-ended −0.0024, MCQ −0.0001 nats/token, every CI covering 0 — not
  even the QA it trained on improved on unseen clips.

### Inference-time follow-ups on the clean checkpoint (2026-10-04)

Eval only — nothing was trained. Same checkpoint (`qbase_enhance_mlvuclean`), raw arms on the
base model (LLM asserted identical), letter log-likelihood, 95 % cluster bootstrap by video.
"Detail types" = Video-MME OCR, Attribute Perception, Object Recognition, Counting, Spatial
Perception (1,001 q). Covered by the report artifact above.

**Where the code is.** None of these drivers is in `eval_ablation/`. Each lives with its rows
and `report.txt` in its own gitignored `work_dirs/<name>/` (`<name>.py` + `report.py`), imports
the `eval_ablation` helpers (`long_mcq_budget`'s decode / memo, `long_mcq_segment`'s
forced-segmentation hook) and pairs its rows by qid with `work_dirs/full_bench`. The hybrids
patch the compressor at runtime: `work_dirs/lowraw/lowraw.py::GatedHybrid` wraps
`output_len_for` / `forward` / `_apply_mm_projector`, and `budget_scale.py` / `confirm_short.py`
import it from that directory via `sys.path`. **Hybrid format** (every hybrid arm): a raw
keyframe is the frozen encoder's merge-2 rows for the group's middle frame at an
aspect-closest grid (~60 / ~120 tokens for the nominal 64 / 128), sent through the BASE
projector, placed before (or instead of) its segment's qbase rows inside the one `<cs>…<ce>`
block behind one `Time:` range, with no per-frame timestamp — a layout the LLM never trained on.

- **Free-form answers on robust failures** (`freeform_fail/`). From the full-bench rows: qbase
  best arm wrong at all three B and best raw right at ≥ 2 (111 q), and the reverse as the
  control (74 q). Both sides at the B = 2048 best-vs-best setup (qbase tf4 on 128 frames vs
  raw_32 at 64 tok/frame), greedy, the question without options, plus one free description per
  video. Judged by hand, one rater; `sheet_{qfail,rawfail}.txt` hold the answers, the
  per-question labels were not saved (only the counts below).

  | | qbase fails (111) | raw fails (74) |
  |---|---:|---:|
  | free answers the same on both sides | 75 | 50 |
  | right topic, the failing side misses a detail | 24 | 18 |
  | the failing side's gist is wrong | 3 | — |
  | the failing side's free answer is the better one | 3 | 5 |
  | no visual evidence | 6 | 1 |

  The qbase's 24 detail misses: on-screen text / numbers 10, fine identity 7, colour / shape 4,
  count 2, peripheral action 1 — all near-misses (lexanimma.com for lexanimata.com, 270 for
  200). In the control the qbase reads what raw misses (a jersey number, a score of 160). When
  wrong, the correct option ranks second 51.9 % (qbase) vs 50.6 % (raw). Free form explains ~11
  of the 37-question net MCQ gap; the rest sits where both free answers agree — given the
  options, raw checks them against fine cues better. Some information is in the qbase tokens
  but unused: its description of 821-2 names the hot glue gun and of 067-1 names Ireland, yet it
  picks the wrong option on both.
- **8 frames in one segment** (`static_seg/`). Video-MME full (2,619 q), B = 1024: 16 anchors at
  whole-video bin centres, each 8 consecutive 1 fps frames; every qbase arm at the same
  512 tok/frame input, only the grouping changes; raw arms at 64 tok/frame. Change = mean
  `1 − cos` of consecutive per-frame mean encoder features (the segmenter's `diff`), terciles.

  | accuracy % | 1 frame | 4 merged | 8 merged | 8 separate (8× tok) | raw 1 frame | raw 8 frames (8× tok) |
  |---|---:|---:|---:|---:|---:|---:|
  | all | 55.2 | 56.4 | 56.6 | 58.3 | 57.4 | 61.4 |
  | low change | 58.9 | 58.4 | 58.8 | 60.5 | 60.3 | 63.3 |
  | high change | 52.1 | 54.5 | 55.4 | 58.2 | 54.8 | 59.1 |

  8 merged − 1 frame (same tokens) +1.4 [+0.1, +2.7]: −0.1 on low change, +3.3 [+1.0, +5.5] on
  high. 8 merged − 4 merged +0.3 n.s. 8 merged − 8 separate −1.7 [−3.0, −0.4] (low change −1.7
  [−4.0, +0.6]). Raw 8 − raw 1 frame +4.0 [+2.6, +5.3], still +3.1 on low change (8 blurry views
  add detail; the qbase already sees each frame sharp). The limit is 64 tokens per segment, not
  frames per segment; beyond `force_every = 8` is untested.
- **qbase + raw keyframe hybrids** (`gated_hybrid/`, `lowraw/`). Video-MME full, B = 2048, the
  full-bench 128-frame grid (raw arms paired from `work_dirs/full_bench`, same frames), 16
  groups of 8 frames; a video's lower-change half of its groups is "static".

  | arm | static half | dynamic half | tokens | acc % |
  |---|---|---|---:|---:|
  | q4 | 2 × qbase 4-frame | 2 × qbase 4-frame | 2,048 | 58.0 |
  | h8 | qbase 8 + raw 64 | qbase 8 + raw 64 | ~1,984 | 60.2 |
  | adapt | qbase 8 + raw 64 | 2 × qbase 4 | ~2,016 | 59.6 |
  | adapt_inv | 2 × qbase 4 | qbase 8 + raw 64 | ~2,016 | 59.1 |
  | **lowraw** | raw 128 only | qbase 8 + raw 64 | ~2,016 | **60.3** |
  | lowraw_inv | qbase 8 + raw 64 | raw 128 only | ~2,016 | 59.6 |
  | lowraw_q | raw 128 only | 2 × qbase 4 | ~2,048 | 60.1 |
  | raw_16 / 32 / 64 (base model) | — | — | ≤ 2,048 | 60.1 / 60.2 / 59.9 |

  h8 − q4 +2.3 [+1.0, +3.6]; lowraw − raw_32 +0.1 [−1.4, +1.5]; adapt − adapt_inv +0.5 n.s.;
  lowraw − lowraw_inv +0.6 n.s. overall, +2.8 [+0.6, +5.0] on detail types. Every reasonable mix
  plateaus at ~60 = the best raw arm: a qbase token trades roughly 1:1 with a raw token, and
  placing by change helps detail questions only. Not recovered: Object Reasoning (lowraw 53.6 vs
  raw_32 57.3, −3.6 [−7.0, −0.4]) and Video-MME long (−2.7 [−5.1, −0.1]). lowraw was picked from
  these six configs on Video-MME, so its Video-MME number is optimistic. The 111 / 74
  robust-failure subsets swing hard under any hybrid (h8 − q4 +27.9 on the qbase-fail set,
  −20.3 on the raw-fail control: selection bias both ways) — quote the full set.
- **Budget scaling** (`budget_scale/`). Video-MME 1/3 sample (912 q / 304 videos,
  `--frac 0.34`), B = 4096 / 8192 run fresh (512-frame decode, T = B/16, B/128 groups of 8),
  B = 2048 paired from the rows above; raw at 128 / 64 / 32 tok/frame, best picked per B.

  | accuracy % | B = 2048 | 4096 | 8192 |
  |---|---:|---:|---:|
  | q4 | 58.6 | 61.0 | 60.4 |
  | h8 | 60.4 | 62.4 | 62.1 |
  | lowraw | 61.1 | 62.9 | 63.3 |
  | best raw | 61.2 | 62.6 | 64.3 |
  | short (321 q): lowraw / best raw | 70.4 / 71.3 | 74.1 / 72.9 | 75.1 / 73.8 |
  | long (291 q): lowraw / best raw | 53.3 / 55.0 | 53.3 / 55.3 | 51.9 / 55.7 |

  lowraw 8192 − 2048 +2.2 [+0.0, +4.6], best raw +3.1. At 8192: lowraw − best raw −1.0 n.s.,
  q4 − best raw −3.8 [−6.2, −1.4], long video lowraw −3.8 [−7.6, +0.0]. (The 8192 best raw is a
  64.3 tie between 64 and 128 tok/frame.) Candidate causes, not separated: the qbase input is
  capped at 65,536 vision tokens per video, so its per-frame input drops 512 → 256 tok/frame at
  T ≥ 256 while raw arms get more per frame as B grows; one `Time:` range over up to ~8k tokens;
  sparse long-video sampling puts several shots in one segment. Cells are ±2.5 pp.
- **Short-clip confirmation** (`confirm_short/`). lowraw frozen and the primary test fixed
  before the run: lowraw − best raw (of 128 / 64 / 32 tok/frame, picked per B on the same
  questions — optimistic for raw), averaged over B = 2048 / 4096 / 8192. LongVideoBench 8–60 s
  (361 q, never used to pick the config): −1.2 [−3.3, +0.9]. Video-MME short, all 873 q (its
  B = 2048 rows and ~1/3 of the rest were seen while picking): +1.15 [−0.15, +2.44]; the 552
  unseen: +1.0 [−0.8, +2.9]. **Parity, not a win**; lowraw beats q4 by 4–6 pp everywhere.
  Sub-groups: `report.txt` keeps the whole-set best raw (LVB 8–15 s −3.35 [−6.7, +0.2], 15–60 s
  +1.16 [−1.2, +3.5]); the report artifact re-picks best raw inside each sub-group (8–15 s −3.4,
  15–60 s +0.6) — the more conservative reading, computed ad hoc, no saved script. The decode
  spreads B/16 frames over the whole clip, so an 8–15 s clip at B = 8192 gets 512 frames with
  many duplicates, far above the 1 fps the qbase trained on.

**Reading.** The qbase keeps the gist; what it loses is detail recall and checking an offered
option against fine cues. As an inference-time hybrid with raw keyframes it substitutes for raw
frames at equal cost up to B ≈ 4096 but does not beat them, and on long video at high budget it
falls behind. Not tested: a `Time` stamp per segment.

---

## 4b. The fold line — parked, outside the phase numbering

Moved out of the curriculum 2026-10-02 (user decision). The two fold stages — the **unit
fold** (formerly Phase 2: fixed `U ≤ 5`, InternVid 180–420 s) and the **budget fold**
(formerly Phase 3: `U = min(N, ⌊B/M⌋)`, DP unit placement, raw bypass) — are kept here as
built and measured, unnumbered, until the qbase clears the raw-frame bar; where they
re-enter the curriculum is undecided. Every fold run so far sat on a pre-Phase-2 qbase
(Phase 1's, lightly co-trained with the fold in some runs). Code, launchers and
checkpoints keep their old names (`phase2_*` = unit fold, `phase3_*` = budget fold).

**Naming inside this section:** the subsections below were written when these stages
were Phase 2 and Phase 3 and keep that wording — "Phase 2" = unit fold, "Phase 3" =
budget fold, "Phase-2 / Phase-3 checkpoint" = the corresponding runs. "Phase 1" still means Phase 1.

### Why the fold line is parked — the qbase-vs-pooling probe (2026-09-29)

`eval_ablation/qbase_vs_pool.py` holds the frames and the 64 tokens/frame fixed and swaps
only the per-frame compressor, at one frame per segment with every unit a raw bypass (so
segmentation and the fold drop out): the qbase loses to plain 4×4 encoder-feature pooling
and to downscaling by **4.5–6.8 pp on Video-MME**. Fold work is parked until the qbase
alone clears that bar — which is what Phase 2 (§4) is for; the bar itself is the best raw dynamic-HW arm with a frame sweep
(`qbase_budget_ceiling.py`). Re-run on the clean Phase-2 checkpoint (2026-10-03) the pooling gap is
2–3 pp (§4 "MLVU-clean Phase 2"), and against the raw bar that qbase is at parity or better
on LongVideoBench / MLVU but −2.1 pp on Video-MME.

### Unit fold (formerly Phase 2) — add the fold; qbase + fold streams; **+ 180–420 s**

> **Superseded in part (2026-09-27).** Both replay streams are **gone** and the
> qbase is **frozen for the whole phase** (`QBASE_LR=0`). The data is a single
> stream: `internvid_mid_180_420`, 273,356 rows, 180–420 s. Rationale and measured
> composition in §6 *"Phase 2/3 data — replay removed, qbase frozen (2026-09-27)"*.
> The two bullets immediately below are kept because they record **why** the replay
> streams existed — they no longer describe the run.

- **Data:** 180–420 s (~273k, Qwen3-VL re-captions complete) **+ 30 % replay** of
  the Phase-1 `< 180 s` pool. The replay is not only short-video retention — it is
  where the **shallow-fold** training signal comes from (a 60 s replay clip has
  `N ≈ 16` ⇒ `U = 1–2`, `N_u ≈ 8–16`; a 30 s clip `N ≈ 8`).
- **Pure-qbase replay** (new, separate meta-JSON entry with `"qbase_only": true`).
  A small slice (~5–10 % of steps) of short clips routed **straight through
  qbase** — no unit split, no fold: the model emits the `N·K` qbase tokens to the
  projector in the Phase-1 layout (stride-1 slots). This is what keeps the
  raw-qbase → projector → LLM path exercised: the unfrozen `mm_projector` (and,
  once unfrozen, the qbase) must not drift onto
  the fold-readout manifold and forget how to read a plain qbase sequence. The
  30 % fold-replay above still supplies the shallow-fold signal — the two replay
  streams are separate.
- **Working frame cap ≈ 320.** With `U ≤ 5` and `N_u` typically `≲ 16` a training
  sample folds `⌊T/4⌋+1 ≲ 80` segments ⇒ `T ≲ 320`. So the 180–420 s bucket is
  either frame-capped (`--max_frames ~320`) or a contiguous sub-span is taken per
  epoch. **As built: the launcher frame-caps at `MAX_FRAMES=320`** (uniform subsample
  of the whole clip); the sub-span option (§7) was never implemented.
- **New machinery:** unit grouping (U ≤ 5 contiguous), SSD fold per unit (fresh
  `init_state`), readout M = 64, the `qbase_only` passthrough branch in
  `compress_windows`.
- **time_embed:** **not** absolute `index_sincos` (it is also rejected outright under
  the padded batched fold). Per-segment `(gap_seconds, duration_seconds)` → `time_mlp`
  (`rel_gap_mlp`) is the default (`Phase2ModelArguments` + launcher); the first runs used
  `none`, and switching to `rel_gap_mlp` bought caption quality but did not move the
  frame-order probes (see the load-bearing facts).

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
  / `qbase.forward`), and returns the realised `unit_meta` (per-unit `n_out`,
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
  `[1, min(fold_max_units, ⌊N/4⌋)]`:

  | class | `N` range | `N̄_u` draw |
  |---|---|---|
  | shallow | `N < 23` | `U[4, 8]` |
  | mid | `23 ≤ N < 46` | `U[8, 12]` |
  | deep | `N ≥ 46` | `U[12, 16]` |

  Reuse `LengthGroupedSampler` with `lengths = N` (`--group_by_compression_depth`,
  needs `durations_json`); duration-homogeneity then comes for free.

- **Unit boundaries: content-aware + Gumbel jitter.** Place the `U − 1` cuts at the
  largest **inter-segment** `diff`, computed on the **qbase segment
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
  wrapper's** train state + `segment_sample_tau` (not `qbase.training`), so
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
compression-variance training (above). (An earlier version justified this with a causal
story for §2.1's collapse — mis-scale + `final_norm` pinning + no variance; that story was
retracted 2026-09-19, see §2.2. The defenses stay; the justification is now simply that
the fold readout must sit on the scale the projector reads and must not see a single
partition.)

- **Option A** (`_match_encoder_scale`) with `ref` = **that unit's** encoder tokens
  (`win[fstart[a]·hw : fstart[b]·hw]` in `compress_windows`, per-unit — not the
  global `kv`), applied **after** the fold's `output_proj`.
- **Option B** (`_distribution_match_loss`), wired into the trainer for the fold
  (`_last_distr_loss` on the `TwoStageCompressor` wrapper, summed with qbase's
  term).
- `final_norm` before `output_proj` made configurable (`--fold_final_norm`:
  rmsnorm / layernorm / scale / none; default rmsnorm), kept as an ablation knob.
- **Own fold projector.** The fold works in `fold_d_model` (default 1024, a
  bottleneck below the compressor hidden 1152); `output_proj` decodes the M readout
  tokens back to 1152, Option A puts them on the encoder scale, and they go through
  **`mm_projector_fold`** — a split copy of the projector, separate from the
  `mm_projector_qbase` that qbase rows use. The SSM **state** stays its own
  `(nheads, headdim, d_state)` object — distinct from the readout, which is `C·state`
  decoded, not the state itself.
- `weight_decay 0.03–0.05` (standard `--weight_decay`); fold `dropout` via
  `--fold_dropout` (`Phase2ModelArguments`, default `0.1`, threaded through
  `_build_phase2_token_compressor_config`).
- **No cross-sample / VICReg decorrelation loss.** (An earlier plan added a
  variance/covariance term on the readout to defend `struct_ρ` — §2.1 shows A + B
  alone do not. Dropped: `struct_ρ` is now a hard **gate** metric only, and the
  compression-variance training + `rel_gap_mlp` time embed are the structural
  defenses. If the Phase-2 gate fails on `struct_ρ`, revisit this.)

#### Trainable + gate

- **Trainable:** qbase + fold + `mm_projector`, LLM frozen. A single knob,
  `--qbase_lr`, controls the qbase: **`> 0` trains it** (its own optimizer group
  at that LR, jointly with the fold from step 1); **`<= 0` freezes it**
  (`requires_grad=False`; there is no separate freeze flag). **The launcher default is
  `QBASE_LR=0` — the qbase is frozen for the whole phase** (2026-09-27: it protects the
  Phase-1 image gain outright, and unfreezing had bought nothing). Launch must set
  `--mm_projector_lr > 0` (unfreeze alone ≠ an optimizer group). **A cold-fold qbase
  stagger is NOT automatic:** run the first window with a tiny POSITIVE
  `--qbase_lr` (e.g. `1e-12`, not 0 — 0 drops the qbase out of the optimizer and the
  resume then fails on a param-group-count mismatch), then resume the same
  `OUTPUT_DIR` at the real rate; the trainer re-applies this run's per-group LRs after
  the DeepSpeed restore (§5 item 23). The only progress-based auto-anneal is the
  variance curriculum's `variance_cold_frac` window.
- **Gate to Phase 3 — checked at fixed operating points**, not one: `N̄_u ∈
  {shallow, mid, deep} × U ∈ {2, 3, 5}`. At **every** point: fold `struct_ρ` ≥ ~0.9
  × qbase's; `xcos_cen` ≈ encoder's; **`cc_frac` off 1.0 (near ~0.94)**. Plus:
  caption metrics plateaued; the U-cap budget ablation (capped vs. uncapped) shows
  the cap is nearly free; and the pure-qbase replay caption score holds vs. the
  Phase-1 qbase baseline (projector not drifted). **NOTE:** `struct_probe.py`
  currently only drives the whole-video single-fold path (`unit_counts=None`) —
  the `--n-units / --segs-per-unit` sweep is not yet implemented.
- **Run: `work_dirs/phase2_fold_frozen_qbase`** (launcher defaults: frozen qbase,
  `rel_gap_mlp`, `MAX_FRAMES=320`, `MAMBA_LR=3e-5`, `GLOBAL_BATCH=512`, 2 epochs). Its
  epoch 2 hit the DeepSpeed GA-boundary bug; a correct re-run of epoch 2 from
  `checkpoint-500` (`phase2_fold_frozen_qbase_ep2fix`) lowered train loss but made
  held-out CE worse (`eval_ablation/phase2_heldout_ce.py`, 643 held-out clips, no
  InternVid: +0.0136 nats/token [+0.0118, +0.0156] vs ckpt500, mostly caption-style drift
  toward InternVid). **`checkpoint-500` is the designated Phase-3 warm start.** Folding
  `N·K` down to `U·M` cost ~+0.15 nats/token vs the same clip through the qbase alone
  (that qbase arm ran through the base-initialised projector, so it is not a clean
  Phase-1 comparison).
  Structural gate clean across every fold checkpoint (`struct_ρ` 1.000, `xcos_cen` /
  `cc_frac` ≈ encoder's); frame-order probes flat (a diagnostic, not a gate).
  Phase-2 runs before 2026-09-29 seeded both split projectors from the BASE projector
  rather than Phase 1's — fixed in the warm-start path (§5 item 19); Phase 2 was not
  re-run, since its qbase projector was unused and the fold projector trained anyway.

### Budget fold (formerly Phase 3) — budget-driven length-adaptive fold (designed for 420–1200 s, 2026-09-18)

**Status (2026-10-02).** Built and self-checked 2026-09-20 (§5 items 4, 13, 14, 16, 17);
entrypoint `videollama3/train/phase3_pretrain_budget.py` (launcher
`shell/pretrain_phase3_budget.sh`), a thin wrapper over `phase2_pretrain_fold.py` via the
same injection hooks. Supersedes nothing in Phase 1/2 — their fixed formulas
(`N=⌊T/4⌋+1`, the `U≤5` depth-class heuristic) are unchanged and still the default code
path; this is Phase 3's own recipe, warm-started from a Phase-2 checkpoint. Two deviations
from the draft below, both recorded where they bite: the DP is **deterministic** (no
unit-level Gumbel — §5 item 16), and `variance_cold_frac` no longer narrows `U` (nothing
to narrow under a budget-derived `U`); it scopes an optional `B` ramp,
`--token_budget_cold` (off by default), the "warm-start distribution shift" answer
problem #8 asked for.

- **One run so far: `work_dirs/phase3_budget_fold`**, on the 2026-09-21 blend (419,652
  rows: `/root/datasets` 65 % + InternVid 35 %, §6), warm-started from the older
  qbase-unfrozen Phase-2 run, qbase at the then-default 3e-6, 2 epochs (epoch 2 flat and
  under the GA-boundary bug). It is what the budget-matched MCQ benchmarks measured.
- **The launcher has moved on since, and that configuration has not been run.** Data
  rebuilt 2026-09-27 (InternVid removed entirely, no replay; the 420 → 720 → 1200 length
  curriculum retired because the blend holds 150 clips in 420–1200 s), then 2026-09-29
  the user chose to **train the qbase jointly with the fold** ("以防qbase丟失細節能力")
  with a 20 % still-image share as its detail pressure. Current launcher defaults:
  `META_PATH=anno_data/phase3_blend_img.json`, `QBASE_LR=1e-5` = `MAMBA_LR`,
  `PRETRAINED_COMPRESSOR_PATH=work_dirs/phase2_fold_frozen_qbase/checkpoint-500` (no
  longer on disk) + `QBASE_PROJECTOR_PATH=work_dirs/phase1_qbase_internvid`,
  `TOKEN_BUDGET=1024`, `MAX_FRAMES=420`, `VISION_MAX_TOKENS=65536`,
  `VISION_CHUNK_FRAMES=32`, compressor activation checkpointing on, `EPOCHS=2`. (Its
  default `OUTPUT_DIR`, `phase3_budget_frozen_qbase`, still carries the pre-09-29 name.)
- **The fold line is parked** (2026-09-29) until the qbase alone clears the raw-pooling
  bar — see the top of §4b.

**Pre-flight audit (2026-09-20).** A 4-step run exercises the whole path; three
things it turned up, all fixed:

1. **`--pretrained_compressor_path` silently loaded nothing.** The loader behind
   it was the qbase-only one (`qbase.` stripped, `fold.*` dropped), so a
   Phase-2 run dir left **all 194 parameters missing** behind a `[WARN]` and
   Phase 3 would have trained a randomly-initialised qbase *and* fold. Replaced
   by `compressor.py::load_pretrained_compressor`, which adapts between the
   single- and two-stage key layouts (two-stage→two-stage verbatim,
   single-stage→`qbase.*` with the fold at init, two-stage→qbase dropping
   `fold.*`) and **raises** when the checkpoint and the model share no
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

4. **A no-fold window starved `fold` of gradient and hung the allreduce.**
   Only multi-rank runs can see this, so every single-GPU check above passed
   while an 8-rank run died on **step 1**: a 600 s NCCL timeout on a
   250,877,952-element `ALLREDUCE` inside DeepSpeed's
   `independent_gradient_partition_epilogue`, surfacing as
   `torch.AcceleratorError: Invalid access of peer GPU memory over nvlink or a
   hardware error`. **Not hardware** — an isolated 8-GPU allreduce of exactly
   that size runs in 4.0 ms at 443 GB/s busbw, and decoding a 1200 s InternVid
   clip off weka takes 1.6 s, so neither NVLink nor the dataloader was the
   straggler.

   Cause: `compress_windows` calls `self.fold` only when the window produced at
   least one fold unit (`if fold_idx:`). A window can legitimately produce none —
   a `qbase_only` replay window (5.9 % of the pool), or a Phase-3 clip short
   enough that `N <= U` makes every unit a single-segment raw bypass (1.4 %, and
   **only reachable because `--fold_bypass_singletons` is a Phase-3 addition**;
   Phase 2 always folded). That rank's 29.9 M fold parameters then get no
   gradient at all, so ZeRO's flattened gradient buffer is a different size there
   than on ranks that folded, and the matching collective never completes.
   `out_gamma`/`out_beta` (Option A on the wrapper, applied in the fold branch
   only) have the same exposure.

   This is precisely the hazard `videollama3_arch._apply_mm_projector`'s docstring
   already documents for the three `mm_projector` copies — and warns is "a NCCL
   hang / invalid peer GPU memory access waiting to happen" — solved there by
   running all three unconditionally. `fold` was the gap. Fixed the same way:
   when `fold_idx` is empty and fold is trainable, a one-segment **zero** input
   goes through the fold and its output enters the graph at coefficient `0.0`
   (added to `compressed`, which leaves every value bit-identical). Gradients are
   exactly zero; the backward hooks fire, so the reduction matches. Verified:
   all-bypass windows go from **0/47** to **47/47** fold tensors with a gradient
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
   clip" open problem the doc already flags (§3, §4b unit fold), now something
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
`eval_ablation/archive/manifest_internvid_true_heldout_420_550.json` (8 InternVid
clips; verified by basename cross-reference to have **zero** overlap with the
actual short/mid-bucket training files `anno_online/internvid/phase1_lt180.json`
/ `phase2_mid_180_420.json` — the previously-used `manifest_internvid_heldout.json`
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
depth-class draw, §4b unit fold's mechanism, before Phase 3's own budget
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
`depth_cost` ceiling. The existing per-epoch Gumbel jitter (§4b unit fold,
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
  distributions (extend `eval_ablation/archive/unit_segmenter_probe.py`-style
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

`eval_ablation/archive/phase3_dp_depth_probe.py`, 8 clips from
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
untouched for reproducibility, and `eval_ablation/archive/unit_segmenter_probe.py`'s
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

**Sweep** (`eval_ablation/archive/phase3_segment_budget_sweep.py`, pure duration
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
  Keep the adaptivity; handle the 5 % with `fold_rope_slot_scale`.
- **Fold depth is a non-issue under this rule.** `N_u` p99 ≤ 18.2 (InternVid) /
  4.1 (unified+vcd), inside Phase 2's validated band. The `N_u > 100` cases in
  the DP probe above only arise at `target_frames = 1` *with* `λ = 0`; the
  target-`N` rule never reaches that regime.

**Decision: `c = 4`, `min_adapt = 0.5`.** Strictly better than `target_frames = 4`
on short clips, identical on long ones.

- **Trainable / warm-start:** qbase + fold + the split projectors, LLM frozen
  (launcher: `QBASE_LR=1e-5`, `MAMBA_LR=1e-5`, `MM_PROJECTOR_LR=5e-6`). Warm start =
  a Phase-2 checkpoint via `--pretrained_compressor_path` (compressor + its
  `mm_projector_{qbase,fold}` + compression-token rows), with `mm_projector_qbase`
  overridden from the run that trained the qbase (`--qbase_projector_path`, Phase 1) —
  the raw-bypass rows go through it. No module shape changes:
  `M=64` stays a fixed learned `summary_tokens` parameter, `K=64` stays the
  qbase's `num_queries` — this mechanism only changes how `U`/`unit_counts` is
  computed and adds the `N_u==1` bypass branch in `compress_windows`.
- **No retained set beyond the raw-bypass above** — see the callout above. Do
  not reintroduce per-unit `r_u` (redundant raw-alongside-fold) in any form.
- **Length curriculum inside the phase — retired (2026-09-27).** Designed as
  `MAX_FRAMES` 420 → 720 → 1200 on one `OUTPUT_DIR`; with InternVid out, the blend holds
  150 clips in 420–1200 s, so `MAX_FRAMES=420` covers 99.9 % of it in one run. Restore it
  only if a long-video stream with trustworthy captions appears.
- **Gate:** needle probes (content recall at specific timestamps) on long
  video; frozen-LLM caption eval holds; `struct_ρ` holds at depth; **and now
  also**: raw-bypass units should be measurably more faithful than folded ones
  (better recall on needle-probe content that lands in a bypass unit vs. a
  folded one) — if not, the novelty-score selection isn't actually finding the
  informative segments and needs revisiting. Frame-order / temporal-blindness
  metrics stay a diagnostic, not a blocking gate (unchanged rationale, see
  CLAUDE.md load-bearing facts).

## 5. Implementation requirements (code changes)

| # | change | where | GPU? |
|---|---|---|---|
| 1 | **Option A/B on the fold readout** — after `SegmentAggregator.output_proj`, apply `_match_encoder_scale` (ref = **that unit's** encoder-token slice `win[fstart[a]·hw : fstart[b]·hw]`, per-unit) and stash `_last_distr_loss` from `_distribution_match_loss` on the `TwoStageCompressor` wrapper. **Done.** | `compressor.py::TwoStageCompressor.compress_windows` | no |
| 2 | **Trainer wiring fix** — `_compressor_distr_loss` reads `distr_loss_weight` / `_last_distr_loss` off the wrapper; the `TwoStageCompressor` now carries its own `distr_loss_weight` and its `_last_distr_loss` = mean of the fold-unit terms + qbase's term (when the qbase is trainable). The trainer divides it by GA (it has no token normalisation). **Done.** | `videollama3_trainer.py`, `compressor.py` | no |
| 3 | **Fixed-count adaptive causal segmenter** — model-side, non-learned. `N = ⌊T/4⌋+1` from the frame count (collator + model compute it identically ⇒ no dynamic-length plumbing, no arch refactor for Phase 1). Forced boundary every 8 f + remaining budget to top-`diff` positions on the per-frame encoder mean feature; Gumbel-top-k draw (z-normalised `diff`, `τ≈0.5`) for per-epoch augmentation; `[1,8]` clamp (`--segment_force_every`). Validated: `eval_ablation/archive/segmenter_validate.py` (+ `_hist.py`). Implemented: `compressor.py::adaptive_segment_count` / `adaptive_segment_lengths`, `TransformerDecoderFlatCompressor` `output_len_for` + `forward` (`--adaptive_segmentation`); `videollama3_arch.py::_compressed_len` uses `output_len_for` for the placeholder count. | `compressor.py`, `videollama3_arch.py` | no |
| 4 | **Chunked encoder forward** — groups of `--vision_encoder_chunk_frames` frames under `no_grad` (skipped only if some encoder param is trainable), one row `(t,h,w)` split into `(t_chunk,h,w)` rows. The encoder has no cross-frame attention (per-frame `cu_seqlens`, 2-D spatial RoPE repeated per frame), so this is a padding/layout change only — **verified bit-exact** vs. the single-shot forward at chunk ∈ {1,8,16,64,128} on the real SigLIP-NaViT config, including a packed multi-row batch; peak allocation scales with the chunk. **Done** (`0` = single shot, Phase 1's default; Phase 2/3 launchers use 32 — 8 left ~12 % of the step on the table because the GEMM is only `chunk·h·w` rows; Phase 2 uses one call per video). | `videollama3_arch.py::_run_vision_encoder` | no |
| 6 | **Compression-variance training** (§4b unit fold) — **Done.** `Phase2FoldDataset` draws, per `(epoch, index)` seed: depth class from `N` → `N̄_u` from the class's narrow range → `U = round(N/N̄_u)` (cold window: `U = min(3, ⌊N/4⌋)`). Segment + `U−1` unit boundaries are placed **model-side** under a seeded Gumbel (`compression_seed`), on the **qbase segment tokens'** inter-segment cosine. Depth-class **grouped sampler** (`LengthGroupedSampler` on `N`, `--group_by_compression_depth` + `durations_json`; `SubsetWithLengths` forwards `compression_depths` so it survives a val split). `_epoch` / `_progress` are fork-shared `mp.Value`s so the seed + curriculum reach dataloader workers. `variance_ramp_frac` removed. | `Phase2FoldDataset`, `videollama3_trainer.py` sampler, `train/data/compressor.py` | no |
| 7 | **time_embed** — `rel_gap_mlp` (per-segment gap + duration seconds, built model-side from the adaptive cut) or `none` (Phase-2 first-runs default); `index_sincos` stays in the code but not for the fold. **Done.** | `segment_aggregator.py` config, `compress_windows` | no |
| 8 | **K/M knob** — default K = M = 64. Decoupling (K > M, e.g. 128/64) only needs `TwoStageCompressor.__init__` to stop default-tying `n_summary_tokens` to `tokens_per_segment` (`SegmentAggregatorConfig` already has them separate). Open decision §7; never exercised — every run used K = M = 64. | `compressor.py` | no |
| 9 | **`struct_ρ` / `xcos_cen` probe** — centered cross-video cosine + encoder-similarity Spearman + `cc_frac`, on the pre-projector compressor output vs. the raw encoder. Implemented: standalone `eval_ablation/struct_probe.py` (whole-video part, works on the Phase-1 flat-adaptive checkpoint) and mirrored into `grounding_probe.py`'s aggregate/verdict/table. | `eval_ablation/` | yes (eval) |
| 10 | **LR groups** — `create_optimizer`: a compressor group at `--mamba_lr` (else `--compressor_lr`) that also holds `embed_tokens` when the LLM is frozen, a separate qbase group (`token_compressor.qbase.*`) when `--qbase_lr > 0`, an `mm_projector*` group, and LLM groups when `--llm_lr > 0`. **Done.** | `videollama3_trainer.py` | — |
| 11 | **Strided `position_ids` for the compressed region** (§1) — **Done.** `compress_windows` returns per-output-token slot offsets relative to the unit start: readout token `m` at `round(m·N_u·K/M)`. The arch `position_ids` loop advances `cur` by `unit_span` (`= N_u·K`) per unit and stays stride-1 elsewhere, so the compressed span ≈ `N·K` slots (+ the per-unit `Time:`/`<cs>`/`<ce>` text). Sample-start detection (`position_id == 0`) unaffected. Identical formula train/inference. Phase-3 scale knob (`fold_rope_slot_scale`): §7. | `compressor.py::compress_windows`, `videollama3_arch.py::prepare_inputs_labels_for_multimodal` | no |
| 12 | **qbase-only replay path** — meta-JSON `"qbase_only": true` → `Phase2FoldDataset` emits `compression_qbase_only=[True]`; `compress_windows` returns qbase's `N·K` qbase tokens as one `unit_meta` entry (stride-1 slots, no unit split, no `SegmentAggregator` call). Threaded `dataset → DataCollatorWithCompressor → prepare_inputs_labels_for_multimodal → encode_images → compress_visual_tokens_with_compressor → compress_windows`. Keeps `mm_projector` / an unfrozen qbase anchored to the raw-qbase manifold. **Done.** | `compressor.py`, `videollama3_arch.py`, `videollama3_qwen2.py`, `train/data/{compressor,global_compressor}.py`, `phase2_pretrain_fold.py` | no |
| 13 | **Budget-driven Phase-3 unit count** (§4b budget fold) — `U = min(N, ⌊B/M⌋)` replaces the depth-class `_draw_units` heuristic; `N_u==1` units bypass `SegmentAggregator` (raw qbase passthrough) instead of folding a singleton, emitting `unit_meta` with `kind="qbase"` (already routes to `mm_projector_qbase`, and at `K==M` the existing `round(m·N_u·K/M)` slot formula already yields stride-1 — no RoPE special case). Also needs `fold_max_units` 5 → `⌊B/M⌋` and removal of `_draw_units`' `u_cap = min(max_units, N//4)` clamp, which caps short clips below the budget. **CORRECTION (2026-09-20): `_place_unit_boundaries` CANNOT be reused** — `min_gap = max(1, N//(2U))` is a floor on every unit's size, so it emitted zero `N_u==1` units in all 144 measured configurations, and its greedy ranking is also worse than the DP on depth (`N_u_max` 75 vs 34). Phase 3 needs its own placement (item 16), leaving Phase 2's method untouched. **Done** — `budget_unit_count` + `Phase3BudgetDataset._draw_units` (no depth-class draw, no `N//4` clamp, no `fold_max_units`: the cap IS `⌊B/M⌋`), `--fold_bypass_singletons` emits the `N_u==1` unit's `K` rows with `kind="qbase"` and a stride-1 slot span via the shared `_unit_slot_span`. | `compressor.py::TwoStageCompressor.compress_windows`, `train/phase3_pretrain_budget.py` | no |
| 14 | **`unified`/`vcd` → Phase-3 meta-JSON converter + per-task held-out split** (§6) — **Done**: `dataset_util/build_phase3_blend.py`. activitynet's dense captions (`{vid: {duration, timestamps, sentences}}`, the only non-LLaVA-shape source) become one human/gpt pair whose answer is the event list with its real `{a}s-{b}s:` ranges; the llava-video `nextqa`/`perceptiontest`/`activitynetqa` families are already LLaVA-shape and are only path-verified and registered. Measured on disk: **14,926 activitynet clips, 0 missing media**; ~57k llava-video VQA entries across 8 subsets, every sampled path present. **`vcd VDC_1k` IS included** (1,027/1,027 clips, five caption turns each), built from the original `VDC_1k.jsonl` whose `video_id` is the on-disk stem. It also carves the **per-task held-out slices** the eval policy above requires (split by video, union across tasks, disjointness asserted) and writes the train registry, the eval registry and the eval manifest in one pass; `--merge` folds in the InternVid meta so one file drives `--multi_dataset`; `--probe_durations` writes the `durations_json` the depth-grouped sampler wants. | `dataset_util/build_phase3_blend.py` | no |
| 15 | **Length-bucketed batched fold** — the fold's pass 2 front-pads every unit to the batch's deepest `N_u`, which the Phase-3 budget mechanism's deliberately-uneven `N_u` turns into a >5× blow-up. `_bucket_fold_tasks` sorts by depth and cuts a new group when the next unit is `>fold_bucket_ratio` (2.0) times shallower, or when `group_size × N_max` would pass `fold_bucket_max_segments` (512); a single over-deep unit still gets its own group. Batch rows never interact in the SSD scan, so this is a padding-layout change only — verified equal to the single-call result to fp32 noise (max \|Δ\| 4.8e-07), and a Phase-2-like homogeneous `N_u` still yields one call. Measured: worst real partition 5.78× → 1.18×; ≤1.46× across all 144 probe configurations. **Done.** | `compressor.py::TwoStageCompressor._bucket_fold_tasks` + `compress_windows` pass 2 | no |
| 16 | **Phase-3 unit placement: exact DP on `WCSS + λ·depth_cost`** (§4b budget fold) — new method alongside `_place_unit_boundaries` (which stays for Phase 2). `O(N²U)` numpy/torch pass; `min_gap` floor drops to 1 so raw-bypass units can exist. `λ ≥ 0.01` (insensitive above that); **`N_u_soft ≈ 1.5–2 × N/U`, NOT a fixed 16** — below `N/U` the quadratic swamps `WCSS` and the DP degenerates to a uniform split. Reference implementation + probe: `eval_ablation/archive/phase3_dp_depth_probe.py`. **Done** — `TwoStageCompressor._place_unit_boundaries_dp` (float64 prefix-sum `WCSS`, `N_u_soft = fold_dp_soft_ratio · N/U` as a **float**, no `min_gap`), selected by `--fold_unit_placement dp`; verified to reproduce the probe's partition exactly on 6 `(N, U, λ, ratio)` points. The knee being a float and not the probe's rounded int is load-bearing at small `N` (it moved the partition at `N=121`). **Deterministic — no Gumbel**: Phase-3 per-epoch partition variance comes from the segment-level jitter (`--segment_sample_tau`) moving the DP's input; the WCSS-space jitter analogue stays an open item. | `compressor.py`, `train/phase3_pretrain_budget.py` | no |
| 17 | **Target-`N` qbase segmentation + clamp** (§4b budget fold) — replace the fixed `segment_target_frames` with `N = clip(c·⌊B/M⌋, N_min(T), T)`, `N_min` from the `force_every` hard floor and the `min_adapt` adaptivity floor; decision `c = 4, min_adapt = 0.5` (identical to `tf=4` on long clips, 69.0 % → 20.4 % zero-fold-gradient on short ones). Separately **add an assert in `adaptive_segment_count`**: a requested `N < ⌊(T−1)/force_every⌋ + 1` silently disagrees with what `adaptive_segment_lengths` actually cuts (`T=1200, tf=12` ⇒ 101 vs 150), corrupting the arch's placeholder count. Sweep: `eval_ablation/archive/phase3_segment_budget_sweep.py`. **Done** — `target_segment_count` / `min_segment_count` / `n_forced_cuts` + `_assert_segment_count_contract` (raises on the `tf=12` case, inert at `tf=4 < force_every=8`), `adaptive_segment_lengths(n_segments=…)`, `TransformerDecoderFlatCompressor.segment_count_for` behind `--segment_count_rule` (`frames` default keeps Phase 1/2 byte-identical; Phase 3 sets `target_n`). Verified equal to the sweep's `plan()` for every `T ∈ [1, 1400)`, and the realised cut matches the promised count with all lengths ≤ `force_every`. | `compressor.py::adaptive_segment_count` / `adaptive_segment_lengths` / `output_len_for`, `train/phase3_pretrain_budget.py` | no |
| 18 | **Still-image stream for Phase 1 + no `Time` marker on images** (§4 Phase 1, §6 "Phase-1 image stream") — **Done 2026-09-26.** Data: `dataset_util/build_phase1_image_blend.py` (media-verified, native-resolution `< K` filter, turn merging; per-task held-out by image realpath only if `--heldout_frac > 0` — the default is 0, no image held-out slice) → `anno_data/phase1_blend.json`. Code: `compression_is_image` threaded `dataset → DataCollatorWithCompressor → videollama3_qwen2.forward/generate → prepare_inputs_labels_for_multimodal`, mirroring `compression_qbase_only` (§5 item 12); the arch's two-stage branch emits `<cs> … <ce>` with **no** `Time:{a}s-{b}s:` for an image part, and `Phase2FoldDataset` no longer claims an `old_ts_len` for one (it would have cut that many tokens of real prompt text, since an image renders as a bare `<image>`). The single-stage Phase-1 path already emitted nothing (`build_range_ts_info` → `(0, [])`). `--vision_min_tokens` / `--vision_max_tokens` moved onto `base.DataArguments` so Phase 1 gets them too, with a launch-time warning when `min_tokens < K`. **Runtime fallback:** `GlobalCompressorLazySupervisedDataset._reject_if_too_small` warns and skips any sample whose compressed output would not fit in the vision tokens it replaces — `N·K` on the Phase-1 path, `M` (the `U=1` floor) on the two-stage one — because the arch writes the compressed rows back INTO the part (`replace_mask[ps : ps+n_out]`) and an overflow spills into the next part. Full remedy logged on the first occurrence per dataset, one line each after, every 100th beyond 20. The skip draws a replacement index through `_backup_sample`, now bounded by `MAX_SAMPLE_RETRIES = 50` and raising `TooManySampleRejections` (its own class, re-raised past the per-sample `except Exception`) so an all-thumbnail registry fails with one clear error instead of recursing until the stack dies. Verified: a 4-row all-sub-K registry at `min_tokens=16` warns then raises cleanly, the same registry at `--vision_min_tokens 64` upscales and passes (`compression_parts=[[0, 64]]`), and the real blend triggers **zero** rejections across all 7 tasks. Verified on GPU: an image is `962 → 64` tokens with zero `Time` tokens through both the Phase-1 single-stage and the Phase-3 two-stage path (the latter 320 → 312, the 8 removed being `Time:0s-0s:`), a video is unchanged (`14,994 → 1,600`, `Time:0.0s-97.0s:` intact), and a 4-step mixed image+video Phase-1 run is green. | `dataset_util/build_phase1_image_blend.py`, `train/data/{global_compressor,compressor}.py`, `train/phase2_pretrain_fold.py`, `train/compressor_pretrain_with_videollama3.py`, `model/videollama3_{arch,qwen2}.py` | no |
| 19 | **Warm starts carry the projectors and compression-token rows** — `compressor.py::load_pretrained_compressor` adapts single- ↔ two-stage key layouts and raises on an empty overlap (before 2026-09-20 a Phase-2 dir silently loaded nothing). `_warmstart_projectors_and_compression_embeds` (2026-09-29) then loads, in order: `--qbase_pretrained` HF dir → its `mm_projector` seeds both split projectors + the `<cs>`/`<ce>` rows; `--pretrained_compressor_path` HF dir → its `mm_projector_{qbase,fold}` (or, for a single-stage target, the source's qbase projector into `mm_projector`) + rows; `--qbase_projector_path` → `mm_projector_qbase` only. Rule: `mm_projector_qbase` comes from the run that trained the qbase. Before this every phase boundary re-copied the BASE projector and re-initialised the two rows. **Done.** | `compressor.py`, `train/compressor_pretrain_with_videollama3.py` | no |
| 20 | **Mixed-granularity qbase stream** (§4 Phase 2) — `segment_target_frames_choices` on the qbase config; `pick_target_frames(choices, seed, default)` picks `tf = choices[seed % len]`; the single-stage path threads the per-part `compression_seed` into `output_len_for` / `segment_count_for` (`videollama3_arch.py::_compressed_len`) so the arch's placeholder count matches the dataset's. No seed → `segment_target_frames`. **Done.** | `compressor.py`, `videollama3_arch.py`, `train/qbase_enhance_pretrain.py` | no |
| 21 | **Encode-cost rank balancing + per-frame token cap** — `--group_by_encode_cost`: the dataset's `encode_costs` (frozen-encoder `a·T·P + b·T·P²` + LLM `c·tokens`, constants fitted on profiled 8-GPU steps, native geometry from `--media_geometry_json` = `dataset_util/probe_media_geometry.py`) feed `LengthGroupedSampler` with one megabatch per optimizer step and an epoch-seeded `seed_fn`, so ranks get equal-cost micro-batches (the fastest rank idled 7–22 s of a ~50 s step before). `--vision_max_tokens_per_frame` caps each frame inside the per-video budget (`simple_batched_resize`). **Done.** | `train/qbase_enhance_pretrain.py`, `videollama3_trainer.py`, `videollama3_encoder/image_processing_videollama3.py` | no |
| 23 | **Trainer / infra fixes** — (a) DeepSpeed's grad-accumulation boundary pinned to HF's `sync_gradients` (2026-09-28; before it every multi-epoch run lost 15/16 of each step's gradient after epoch 1 when micro-batches/epoch % GA ≠ 0); (b) per-group LRs re-applied after a resume, live LR = `base·λ(step)` (DeepSpeed + HF restore the checkpointed LRs and silently overrode a changed `--qbase_lr`); (c) batches moved to GPU per micro-step instead of `grad_acc` at a time (Phase-3 pre-flight #5); (d) with `--dataloader_pin_memory False`, batches copied out of `/dev/shm` on arrival; (e) `--pixel_values_dtype bfloat16` for the worker → main handoff; (f) `--compressor_gradient_checkpointing` (`use_reentrant=False`, required because the compressor's inputs do not require grad); (g) per-stream CE logging `loss_img` / `loss_vid` / `frac_tok_img`; (h) `_load_durations_map` parsed once per process. **Known, not changed:** `compute_loss` divides CE by the global token count without multiplying by world size, so logged losses are ~1/(GA·W) of per-token CE and Option B's effective weight is W × configured. | `videollama3_trainer.py`, `train/data/common.py`, `train/phase2_pretrain_fold.py`, `compressor.py`, `segment_aggregator.py` | no |
| 24 | **Content-level benchmark-overlap check** — `dataset_util/frame_hash_overlap.py` (`hash` → `match_exact` → `verify`: ≥ 3 exact-dHash keyframes at one consistent offset AND median aligned-frame correlation > 0.8) and `dataset_util/exclude_overlap.py` (identity = YouTube id / Ego4D uuid / stem; writes `*_mlvuclean` registries, only changed tasks get new files). `check_benchmark_contamination.py` matches YouTube ids only. **Done 2026-10-02.** | `dataset_util/` | no |

**Keep, do not touch:** no feature cache; CE only; **no reconstruction / MSE /
masked-feature** (collapses to order-invariant — Option A/B are distribution
moments, not per-token targets, and are allowed);
`match_encoder_scale` + `distr_loss_weight=0.05` on qbase.

---

## 6. Data

What each run reads (row counts measured 2026-10-02; every registry is under
`anno_data/`, its annotation files under `anno_online/` — both have generated READMEs):

| registry | tasks | rows | what | read by |
|---|---:|---:|---|---|
| `phase1_blend.json` | 7 | 440,310 | InternVid `< 180 s` (264,186) + still images (176,124, 0.40 row share) | Phase 1 |
| `phase2_internvid.json` | 1 | 273,356 | InternVid 180–420 s, no replay | unit fold |
| `phase3_blend.json` | 28 | 271,080 | `/root/datasets` (unified + vcd), no InternVid | — (merged below) |
| `phase3_blend_img.json` | 34 | 338,851 | `phase3_blend` + 67,771 still images (0.20 row share) | **Phase 2** (first run); budget-fold launcher |
| `phase3_blend_img_mlvuclean.json` | 34 | 338,647 | the above minus MLVU-overlapping clips | **Phase 2** (latest run, `qbase_enhance_mlvuclean`) |
| `qa_qbase_mlvuclean.json` | 50 | 210,887 | QA-dominant blend over the TRAIN clips of the `/root/datasets` blend (`dataset_util/build_qa_sft_blend.py` output minus MLVU-overlapping clips), every row compressed (built by copying that registry minus its `raw_frames` flags; no builder script of its own) | H1 test (`qbase_qa_mlvuclean`, §4) |
| `phase3_heldout.json` | 28 | 5,396 | per-task held-out slices — the eval side | eval |

InternVid duration buckets (from `anno_data/internVid_durations.json`): `< 180 s` 264k
(Phase 1), 180–420 s 262k (unit fold), 420–1200 s 451k (trained only by the first
budget-fold run, removed 2026-09-27), `> 1200 s` 98 (dropped).

### Phase-1 image stream (2026-09-26)

`dataset_util/build_phase1_image_blend.py` → `anno_data/phase1_image_blend.json`
(train; no held-out registry or manifest since 2026-09-26 — see below), and with
`--merge anno_data/internvid_qwen3vl_lt180.json` the blended Phase-1 registry
`anno_data/phase1_blend.json` the launcher defaults to. The same builder, with every
output path redirected, produces the 20 % image share of Phase 2's blend (below).

Rationale in §4 Phase 1. Sources, all verified on disk 2026-09-26 (these three
files are the ones `build_phase3_blend.py` skips as "image-only"):

| source | raw rows | unresolvable | < K tokens native | → samples |
|---|---:|---:|---:|---:|
| `bunny_union` visual_genome | 316,040 | 10,691 | 58 | 96,031 |
| `bunny_union` coco_2017 | 206,244 | 20,064 | 54 | 68,343 |
| `bunny_union` ocrvqa | 79,949 | 27,049 | 801 | 52,099 |
| `bunny_union` open_images | 21,941 | 8,131 | 0 | 13,810 |
| `ocrvqa_19k` | 19,258 | 68 | 252 | 18,938 |
| videoxl `pretrain.json` (capped 60k of 2M) | 60,000 | 0 | 16,277 | 43,723 |

- **`bunny_union` is four corpora**, registered as four tasks so each gets its own
  share. Per-root resolution differs a lot
  (visual_genome 93.6 %, coco 89.0 %, ocrvqa 66.3 %, open_images 57.9 %), so every
  row is existence-checked rather than trusted. Mean answer 339 chars, p90 1,513 —
  a real dense-caption tail, 5.32 turns per sample.
- **`pretrain.json` is alt-text** (mean ~90 chars, "Piece of dark jeans fabric
  Royalty Free Stock Photography") and 27 % of it is thumbnails below `K` tokens:
  volume, not detail. Capped at `--vxl_pretrain_max 60000`.
- **Turn merging.** bunny ships ~2.1 rows per image; rows sharing an image merge
  into one multi-turn sample (`--max_turns 6`) so the frozen encoder runs once per
  image per epoch. The **longest** answer leads, so the dense-caption turn is the
  one that survives the cut.
- **No held-out slice (user decision 2026-09-26)** — `--heldout_frac` defaults to 0 and
  every resolvable row trains. Nothing measures this stream: the images are detail
  pressure on the qbase, and every reported number comes from the video held-out
  slices. `--heldout_frac > 0` re-enables a carve split by **image** (identity = media
  **realpath** — image stems are generic and the same file is reachable through two
  symlinks), union across tasks, disjointness asserted; the builder deletes stale
  held-out outputs when nothing is held out.
- **Measured blend**: 176,124 image samples + 264,186 InternVid rows ⇒ image share
  **0.400 by row, 0.295 by supervised token** (image answers are short; Visual Genome
  alone is ~88 % of the image stream's supervised tokens).

### InternVid

- Videos in `/share/dataset/internVid/` (977,423 clips, 1 missing on disk), stored at
  native 2 fps (decode stride-2 for fps = 1). The old video-list registry
  `anno_data/internVid.json` (empty conversations) was deleted 2026-09-21.
- **Captions:** original InternVid captions are ASR-derived and unusable. The
  Qwen3-VL-8B re-captions (`recaption/qwen3vl/captions.jsonl` →
  `dataset_util/captions_jsonl_to_stage1_anno.py`) are complete:
  `anno_online/internvid/pool_all_qwen3vl.json`, 977,422 rows, master of every InternVid
  bucket. `shell/recaption_vllm.sh` pins every clip to **64 frames** regardless of
  duration — a 1200 s clip is captioned at ~0.05 fps. That is why InternVid is
  training-only (never measured on) and why its long bucket left the budget fold.
- **Durations:** `anno_data/internVid_durations.json` (per-clip
  `duration_sec` / `fps` / `est_frames_1fps`, full ffprobe scan). Median ≈ 366 s,
  p90 ≈ 900 s. Bucket the manifest from this file.
- **Domain caveat:** InternVid (YouTube vlog / gameplay / how-to) ≠ the eval sets
  (ShareGPT4Video, cinepile, VCG). Keep **all evaluation on the target
  distribution**, and always also report **compressor + the original frozen LLM** so
  a gain cannot be the LLM memorising InternVid style. **Never measure on InternVid**
  (2026-09-23): a metric against a 64-frame caption measures the caption pipeline.

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
  on footage it is about to be evaluated on. Identity is the stem-or-realpath rule
  below, since the same clip has different relative paths in different sub-datasets.
- **Size:** `--heldout_frac 0.02` of each task's videos, floor `--heldout_min 20`,
  cap `--heldout_max 300`, and a `--heldout_cap_frac 0.2` ceiling so the floor
  cannot swallow a small task. Seeded by `--seed`, so the split is reproducible
  and re-runnable.
- **Outputs:** `anno_data/phase3_blend.json` (train) /
  `anno_data/phase3_heldout.json` (eval) — two `--multi_dataset` registries — plus
  `eval_ablation/manifest_phase3_heldout.json`, the per-task video manifest the
  `eval_ablation/` probes take. The builder asserts the two sides are disjoint by
  stem and refuses to finish if they are not.
- **Measured (2026-09-21 rebuild):** 3,642 videos held out across 29 tasks;
  train x held-out overlap **0**. Two rules the split now encodes:
  - **Identity** is the file *stem* when the stem is distinctive (>= 8 chars and
    not `video_<digits>`), the *realpath* otherwise. Stem alone merges
    `ShareGPT-4o/pvideo/video_10001.mp4` with
    `perception_test/videos/video_10001.mp4` (different videos, 1,511 such
    collisions); realpath alone splits NextQA's `1202/3264772244.mp4`, which the
    release ships as two separate physical copies under `0_30_s_nextqa/` and
    `0_30_s_academic_v0_1/` (identical bytes, different inodes).
  - **`--protect_from`** bars a clip that Phase 1/2 already trained on from being
    *chosen* as held-out. The 2026-09-20 build did not, and put 192/300 and
    209/300 of the InternVid replay held-out clips inside the Phase-2 training
    pool. With it, the InternVid `< 420 s` replay streams get **no** held-out
    slice at all -- the whole pool was trained on -- and only the 420-1200 s
    bucket has one.
- **Current split (2026-09-27 rebuild, InternVid out):** 28 tasks, **5,396** held-out
  videos; the 28 shared tasks' held-out sets are byte-identical to the 2026-09-21 split,
  only InternVid's 300-clip slice is gone. Every later blend (`phase3_blend_img`, the QA
  blend behind `qa_qbase_mlvuclean`, the `_mlvuclean` variants) draws from the same TRAIN side, so this split
  stays the eval side for all of them.
- **Exceptions / additions:** still-image streams keep no held-out slice (§6 image
  stream); InternVid is never measured on; external benchmarks are scored only on their
  contamination-filtered id lists (`anno_data/benchmarks/*_clean.json`) and need a
  content-level overlap check (§6 "Benchmark contamination").

This is what makes **VDC usable as training data**: it goes into the blend like
any other task, and it is evaluated on its own held-out slice rather than on the
benchmark in bulk.

### The `/root/datasets` blend -- PRIMARY, InternVid SECONDARY (2026-09-21; InternVid part superseded 2026-09-27)

Built as the budget fold's (then "Phase 3") data; since the 2026-10-02 renumbering it is
what **Phase 2** trains on (`phase3_blend.json`, + images → `phase3_blend_img.json`).

The 2026-09-20 build had InternVid at 90.1 % of rows and **98.7 % of the CE
token mass**, and its captions are the *least* dense annotation on the machine:
`shell/recaption_vllm.sh` pins every clip to `NUM_FRAMES=64` regardless of
duration, so the 678 s median of the long bucket is captioned at ~0.05 fps. At
the same time `/root/datasets` held 278k clips that were 8.6 % used --
llava-video at 4.2 % (and the 4 % kept was the letter-only MCQ, mean answer
10-18 chars, while the GPT-4o dense captions and open-ended QA were excluded by
the `LLAVA_FAMILIES` filter), videoxl + videoxl_pro at **0 %**.

`dataset_util/build_phase3_blend.py` now registers everything resolvable under
`--datasets_root` and demotes InternVid to `--merge_share` (default 0.35 of the
final rows, each stream keeping its relative proportion). Per-source signal:

  - **activitynet** (14,926 clips resolved of 19,994): human multi-event
    `timestamps` + `sentences` -> one event-list answer with real `{a}s-{b}s:`
    ranges. The only source that trains the `Time:{a}s-{b}s:` string the
    compressor emits per unit against genuine localization.
  - **llava-video, ALL subsets** (186,854 clips, 100 % on disk): the
    `*_cap_processed.json` files are GPT-4o recursive dense captions --
    147,460 rows at mean 3,010 chars (`youtube`) + 29,494 at 2,019
    (`academic`), i.e. 2.3x the density of an InternVid caption -- plus 952,396
    open-ended QA rows (mean ~410 chars) and 191,362 MCQ rows. The
    `nextqa`/`perceptiontest`/`activitynetqa` subsets stay for precise-event VQA.
  - **videoxl / videoxl_pro** (70,176 clips, media 100 % resolvable through
    `source/videoxl_pro/media/`): `sharegpt4v_38k` (37,769 ShareGPT4Video dense
    captions, mean 1,826 chars), `time_sft_opens`' ActivityNet slice (29,896
    rows of "the given query happens in a - b seconds" -- direct supervision for
    the per-unit time range), `baaicaption_10k`, `cinepine_union` + `VICO`
    (CinePile MCQ + event-ordering on 8,051 movie clips, p50 161 s),
    `vcg_union` (VideoChatGPT full-video descriptions on ActivityNet),
    `gpt4o_video_2k`, `ego_4d_0.7k`, `anomaly_det`. Skipped: `bunny_union` /
    `ocrvqa_19k` / `Pretraining/pretrain.json` (images), `finevideo_43k` (0 % of
    its media on disk), `_excluded/` and `_superseded_by_dedup/`.
  - **vcd `VDC_1k`** (1,027 clips, all resolved): multi-turn multi-granularity
    captioning (camera work / short / background / main-object / detailed).
    Build it from the **original `/root/datasets/vcd/VDC_1k.jsonl`**, not
    `unified/annotations/vcd/VDC_1k.json`: the unified copy re-ids every clip to
    a UUID whose `vcd/videos/<uuid>.mp4` path does not exist, while the jsonl's
    `video_id` is the on-disk file stem.

**Turn merging.** llava-video ships ~7 rows per clip across separate files, so a
row-per-QA registry would decode and vision-encode the same clip 7x per epoch.
Rows sharing a video are merged into ONE multi-turn sample (`--max_turns 6`,
dense caption first, `<video>` on the first human turn only): 1,320,712
llava-video rows become 177,834 samples with the same supervision.

**Measured composition (2026-09-21, `--merge_share 0.35`, `--max_turns 6`):**
31 train tasks / **419,652 samples** / 853M answer chars. *(Superseded 2026-09-27:
the InternVid secondary stream was removed outright — 28 tasks / 271,080 samples.
The table below is the last build that had it.)*

| | rows | CE token mass | vs. 2026-09-20 build |
|---|---:|---:|---|
| `/root/datasets` primary | 271,080 (64.6 %) | **78.3 %** | 70,692 rows / 1.3 % |
| InternVid secondary | 148,572 (35.4 %) | 21.7 % | 642,218 rows / 98.7 % |

Duration buckets over the train rows: `<60 s` 45.1 %, `60-180 s` 22.8 %,
`180-420 s` 7.9 %, `420-720 s` 13.7 %, `>720 s` 10.6 % -- so **24.3 % of steps
still land on a 420-1200 s clip** (was 61.6 %), all of it InternVid. Every
registered clip has a duration (`anno_data/phase3_durations.json`, 1,212,449 /
1,212,450 probed ok), so `--group_by_compression_depth` has a real depth class
for every sample. Biggest single CE contributors are now
`llava_2_3_m_youtube_v0_1` (19.4 %), `llava_0_30_s_youtube_v0_1` (16.9 %) and
`internvid_mid_420_1200` (15.7 %).

**Length is the one thing this cannot fix.** Sampled ffprobe over every pool:
cinepile / ShareGPT4Video / baaicaption / gpt4o_video / ego4d / time_sft 0 %
>= 420 s, llava-video 0 % (max 3-5 min, 309 clips), activitynet 0.1 %,
vcg_union 0.4 %, anomaly_det 4.8 % (~63 clips). `VIDEO_DURATION_REPORT.md`:
163 clips >= 5 min in the whole of `unified`. **Outside InternVid the machine
holds ~170 clips >= 420 s**, so InternVid stays as the sole long-bucket
supplier; `--merge_share` is the knob that trades fold depth against annotation
density.

### Unit-fold / budget-fold data (then "Phase 2/3") — replay removed, qbase frozen (2026-09-27)

**User decision, three parts:** delete every replay stream from Phase 2 and Phase 3;
keep the qbase **frozen for the whole of Phase 2** (extended to Phase 3 for
consistency — freezing in 2 and training in 3 gives back exactly what freezing
protects; **the Phase-3 half was reversed 2026-09-29**: the launcher trains the qbase
jointly, with a 20 % image share as the guard — §4b budget fold); and drop InternVid's `> 420 s` bucket because its annotation quality
cannot support it. The reasoning, and what it cost:

**Why freezing replaces replay.** The replay streams existed to keep a *trainable*
qbase (and the `mm_projector`) anchored to the raw-qbase manifold. A frozen qbase
cannot drift, so they defend nothing while costing 104k samples/epoch in Phase 2.
Freezing is also what protects the one measured win of the Phase-1 image stream:
**−0.561 nats/token vs the warm-start on 2,866 never-seen images, 95 % CI
[−0.668, −0.456]**, with `train − heldout = +0.009` (CI covers 0, so it is
generalisation, not in-sample fit — see the Phase-1 image stream section). Phase 2's
data is 100 % long-caption CE, the signal least likely to preserve that. And
unfreezing was already tried and bought nothing: the frame-order probes stayed flat
with the qbase unfrozen.

**Why InternVid `> 420 s` had to go.** `shell/recaption_vllm.sh` pins every clip to
`NUM_FRAMES=64` regardless of duration, so a 1200 s clip is captioned at **0.053 fps**
— one frame every 18.75 s. Training a compressor to predict that target is not merely
noisy: the target itself contains no detail, so it actively imposes a detail-discarding
ceiling. This is the same sparse-input family as the §2.2 collapse root cause.

**Measured after the rebuild.**

| | Phase 2 | Phase 3 |
|---|---:|---:|
| tasks | 3 → **1** | 31 → **28** |
| rows | 377,295 → **273,356** | 419,652 → **271,080** |
| `< 180 s` | 0 | 261,890 (96.6 %) |
| `180–420 s` | **273,356 (100 %)** | 9,015 (3.3 %) |
| `420–1200 s` | 0 | **150** |
| `> 1200 s` | 0 | 24 |

Signal composition per **turn** (not per task — a merged llava-video sample holds
1 caption + ~5.5 open-ended + ~1 MCQ turn, so a task-level classifier charges all of
them to caption):

| | CAPTION | SHORT-QA | MCQ |
|---|---:|---:|---:|
| Phase 2 turns / answer-chars | 100 % / 100 % | 0 | 0 |
| Phase 3 turns / answer-chars | 56.8 % / **96.57 %** | 32.4 % / 2.92 % | 10.7 % / **0.51 %** |

QA turns are **43.1 %** of Phase 3's turns but **3.4 %** of its answer characters —
a caption turn runs 272–1,849 chars against an MCQ turn's 12–27, and CE sums per
token. Removing InternVid raised the QA *turn* share (38.5 % → 43.1 %) and left the
*gradient* share essentially unchanged. The MCQ data is not scarce: 209,811 MCQ rows
exist on disk in llava-video alone (1,377,780 duration-bucketed rows total, ~7.53 per
clip before turn merging), and 129,920 MCQ turns are already in the blend. What is
scarce is their weight in a per-token loss.

**`--protect_from` must stay, even though Phase 3 no longer trains on InternVid.**
Phase 1 still trains 264k clips `< 180 s` and Phase 2 273k clips at 180–420 s, so any
Phase-3 clip sharing an identity with those pools must not become held-out. Dropping
the flag silently re-drew **166/170** of `vxlp_cinepile`'s and **262/320** of
`vxlp_sharegpt4video`'s held-out videos (they share YouTube stems with the InternVid
pool) — caught by diffing against the pre-rebuild split, 2026-09-27. The canonical
rebuild is:

    python dataset_util/build_phase3_blend.py --datasets_root /root/datasets --force \
      --durations_seed anno_data/internVid_durations.json anno_data/phase3_durations.json \
      --protect_from anno_data/phase2_internvid.json anno_data/internvid_qwen3vl_lt180.json

**Held-out is unchanged by policy decision.** It is 1.34 % of the pool (5,696 of
425,348 rows), so deleting it would buy +1.34 % training data while destroying the
only guaranteed-uncontaminated caption-side measurement. External
MCQ benchmarks cannot replace it: they score no caption metric, and Video-MME /
LongVideoBench share YouTube sources with the training pool (hence
`check_benchmark_contamination.py`). After the rebuild the 28 shared tasks'
held-out video sets are **byte-identical** to the pre-rebuild split; only InternVid's
300-clip slice is gone (5,696 → 5,396 videos). The real deficiency is the opposite of
too much held-out: only ~94 clips are actually read by any eval manifest.

**Length curriculum retired.** With 420–1200 s down to 150 rows, `MAX_FRAMES=420`
covers 99.9 % of the blend in one run and the 720 / 1200 runs have nothing to train
on. Deep-fold coverage comes from Phase 2's 180–420 s instead: the budget fold's
inference `N_u` measured 2.1–6.4, inside the trained 3.8–6.6. Restore the curriculum if a
long-video stream with trustworthy captions appears.

**Launcher defaults as of 2026-10-02** (`shell/`, gitignored; ZeRO-1 everywhere):

| | Phase 1 | Phase 2 (qbase enhance) | unit fold (parked) | budget fold (parked) |
|---|---|---|---|---|
| script | `pretrain_phase1_internvid.sh` | `pretrain_qbase_enhance.sh` | `pretrain_phase2_internvid.sh` | `pretrain_phase3_budget.sh` |
| data | `phase1_blend` | `phase3_blend_img_mlvuclean` | `phase2_internvid` | `phase3_blend_img` |
| warm start | `pretrained_models/compressor_pretrain_video_norm` | `work_dirs/phase1_qbase_internvid` | `work_dirs/phase1_qbase_internvid` | `phase2_fold_frozen_qbase/checkpoint-500`* + qbase proj. from Phase 1 |
| output | `phase1_qbase_internvid` | `qbase_enhance_mlvuclean` | `phase2_fold_frozen_qbase` | `phase3_budget_frozen_qbase` |
| qbase | 3e-5 | 2e-5 | frozen | 1e-5 |
| fold | — | — | 3e-5 | 1e-5 |
| projector | 5e-6 | 1e-6 | 5e-6 | 5e-6 |
| LLM | frozen | frozen | frozen | frozen |
| `MAX_FRAMES` | 180 | 180 | 320 | 420 |
| `VISION_MAX_TOKENS` | checkpoint (16384) | 65536 (+2048/frame cap) | 65536 | 65536 |
| `EPOCHS` / `GLOBAL_BATCH` | 3 / 384 | 2 / 480 | 2 / 512 | 2 / 512 |

\* no longer on disk — set the path explicitly before launching.

`VISION_MAX_TOKENS=65536` was **kept** after a false alarm: per-frame tokens rise to
~364 on a short clip, but the qbase operates per *segment*, and Phase 1 already
trained it on segments of 88–1,369 kv tokens (24.7 % of Phase-1 images exceed 364
tokens alone; `img_bunny_open_images` has a median of 999). A Phase-3 segment at
65536 is ~1,456 kv tokens — inside that range, not out of distribution. Note that
Phase 1's **images were never budget-limited at all** (every native grid is < 1,369,
far below 16384); only its video stream was (16384/T → 91–299 tok/frame), so raising
Phase 1's own `VISION_MAX_TOKENS` is the way to give the qbase higher-resolution
*video* input if that is ever wanted.

### Image share of the `/root/datasets` blend (2026-09-29)

Added when the budget fold (then Phase 3) switched to training the qbase jointly: the same still-image sources
as Phase 1, built by `build_phase1_image_blend.py` with **every** output path redirected
(the defaults would clobber Phase 1's stream): `anno_online/phase3_images/train/` +
`anno_data/phase3_image_blend.json` (67,771 rows), merged with `phase3_blend.json` into
`anno_data/phase3_blend_img.json` at `--image_share 0.2`. No image held-out slice.
`Phase2FoldDataset.compression_depths` gives image rows depth `_segment_count(1) = 1`
(they used to fall to the missing-duration default of 32). Phase 2 (qbase enhance) reads
this blend too. Row share is not gradient share: in `qbase_enhance_mlvuclean`'s first
1,521 logged micro-batches the image fraction of supervised tokens (`frac_tok_img`)
averaged 6.2 % (median 1.2 %).

### Benchmark contamination (2026-10-02)

`check_benchmark_contamination.py` matches YouTube ids, and reported MLVU clean. It was
not: MLVU renames clips. `dataset_util/frame_hash_overlap.py` (keyframe dHash + aligned-
frame correlation; rule ≥ 3 exact-hash frames at one consistent offset AND median
correlation > 0.8, validated 165/165 on `anomaly_reco`) found, against the 271k videos of
the `/root/datasets` blend: `anomaly_reco` 165/200 MLVU clips = UCF-Crime clips in `vxl_anomaly_det` (with the
class label as the answer), `order` 86 clips built from 35 training videos (mostly
ActivityNet), plus 3 `count`, 1 `ego`, 1 `needle`; 0 against the 264k InternVid Phase-1
pool. Removal by identity (YouTube id / stem / Ego4D uuid, so every copy goes) of 204
identities → `anno_data/mlvu_overlap_exclude.json` → `dataset_util/exclude_overlap.py` →
`anno_data/{phase3_blend,phase3_blend_img}_mlvuclean.json` (only the four
affected tasks get new files under `anno_online/mlvuclean/`). Any MLVU number from a
model trained on the unfiltered registries must exclude `anomaly_reco` (or report it as
contaminated). The base VideoLLaMA3's own pretraining data cannot be checked.

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
- **Unit-boundary feature — RESOLVED (§4b unit fold):** inter-segment cosine on the
  **qbase segment tokens** (mean-pooled `K` per segment), not the raw
  frozen-encoder feature. Still ablatable: raw-encoder feature;
  common-component-removed encoder feature; RGB shot boundary.
- **RoPE-index scale for the compressed region — OPEN (§1, §5 item 11).** Default:
  readout stride `N_u·K / M` ⇒ region footprint `≈ N·K` slots (`≈ 16·T`; matches
  the Phase-1 all-qbase footprint; the readout-only set `{m·N_u}` is exactly
  collision-free at `K = M = 64`). The unit fold (`T ≲ 320` ⇒ `N·K ≲ 5.2k`) is fine on
  the default. The budget fold's 420–1200 s clips push `N·K` to ~19k slots + surrounding
  text → near Qwen2's 32k; fallback is a seconds scale (`fold_rope_slot_scale =
  S` slots/s, `S ≈ 4–8`), accepting coincident readout slots.
  **Quantified 2026-09-20:** under the adopted `c=4, min_adapt=0.5` rule only
  **5.0 %** of InternVid clips exceed 16k `N·K` slots (p90 = 14,272), so the
  seconds-scale fallback is needed for a tail, not the bucket. `min_adapt = 0`
  would take it to 0.0 % (p90 7,168) but costs most of the segmenter's
  adaptivity (`adapt%` 54.8 → 19.1) — rejected.
- **Unit-fold frame cap — as built: (a).** `U ≤ 5 × N_u ≤ 16 ⇒ ≤ 80 segments ⇒ T ≲ 320`,
  but the mid bucket is 180–420 s. Options were (a) `--max_frames ~320`, (b) a
  per-epoch contiguous sub-span of the full clip, (c) raise `N_u` / `U` caps for long
  clips only. The launcher and the run used (a), `MAX_FRAMES=320` (a uniform subsample
  of the whole clip, so nothing is dropped from the tail — the clip is just sparser);
  (b) was never implemented.
- **Learned segmenter** — Phase 1 keeps the fixed-count `diff` rule; a learned
  causal boundary head is a later option (would have to keep the fixed output
  count). Same question for the **unit** boundaries (§4b unit fold uses a non-learned
  `diff`-at-segment-granularity rule).
- **`mamba_ssm` varlen packed scan** vs. the current per-unit Python loop
  (throughput only; same math).
- **Streaming budget controller** (`merge` of adjacent oldest unit states to hold
  U ≤ 5 on an unbounded stream) — deferred; needed for real streaming inference,
  not for the pretrain.
- **Budget-fold depth safety valve — RESOLVED (2026-09-20, measured).** Take
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
- **Budget-fold qbase segmentation granularity vs. the budget mechanism —
  RESOLVED (2026-09-20, measured).** `target_frames` was the wrong control
  variable: the budget is denominated in tokens, the cut in frames. Replaced by
  `N = clip(c·⌊B/M⌋, N_min(T), T)` with `c = 4, min_adapt = 0.5` (§4b budget fold,
  "qbase segmentation: target `N` + clamp"; §5 item 17). The feared trade-off —
  finer cuts for short clips growing `N` and worst-case depth at the long end —
  does not materialise, because `c·⌊B/M⌋` sits below the `force_every` floor for
  anything long: `c` moves only short clips, `min_adapt` only long ones, and
  `min_adapt = 0.5` reproduces today's `tf = 4` exactly
  (`target_frames = force_every × (1 − min_adapt)`).

---

## 8. Build order

In the order it happened. Steps 1–8 predate the 2026-10-02 renumbering: there "Phase 2" =
unit fold, "Phase 3" = budget fold.

1. **Code (no GPU):** items 1–2, 6–8, 10 of §5 — Option A/B on the readout
   (after `output_proj`, + configurable `final_norm`), trainer distr-loss fix,
   compression-variance training (`Phase2FoldDataset` draws + depth-class grouped
   sampler), `time_embed` swap, K/M knob, LR groups.
   Then **Phase 1 Phase-1→2 gate is already met** (§ analysis: `struct_ρ ≈ 1.0`,
   `xcos_cen = encoder`, `cc_frac ≈ encoder` on InternVid + VCG; caption better on
   InternVid, mixed on VCG — one out-of-domain hallucination, a qbase style-transfer
   artifact, carried into Phase 2).
2. **Code:** item 3 (fixed-count adaptive segmenter + `output_len_for`, **done**) +
   item 4 (chunked encoder forward, done in step 7; Phase 1 still runs the
   single-shot forward). Item 3's fixed `N = ⌊T/4⌋+1` means Phase 1 does **not** wait on an
   arch refactor — collator and model agree on the length.
3. **Phase 1 run** on short InternVid (+ still images) → gate on the §2.3 probe
   (`eval_ablation/struct_probe.py`: `struct_ρ`, `xcos_cen`, `cc_frac`) +
   frozen-LLM caption eval. **Done** (`phase1_qbase_internvid`, §4 Phase 1).
4. **Code (done):** item 11 (`prepare_inputs_labels_for_multimodal` consumes
   model-returned `unit_meta` + baked `Time:` + strided `position_ids`) + item 6
   (model-side seeded unit boundaries on the qbase segment tokens) + item 12
   (qbase-only replay path).
5. **Phase 2 run** → gate (`struct_ρ`, `cc_frac` back off 1.0, caption plateau,
   U-cap budget ablation, pure-qbase replay caption holds). **Done**
   (`phase2_fold_frozen_qbase`, qbase frozen, no replay; ckpt-500 kept).
6. **Code (no GPU) — DONE 2026-09-20:** item 15 (length-bucketed batched fold) →
   item 17 (target-`N` segmentation + the `adaptive_segment_count` assert) → item 16
   (DP on `WCSS + λ·depth_cost`) → item 13 (budget-driven `U`, `N_u==1` bypass,
   `fold_max_units` / `u_cap` clamp removal) → item 14 (`unified` + `vcd`
   converter). Items 15–17 are each independently checkable without a training
   run: `eval_ablation/archive/phase3_dp_depth_probe.py` (depth + padding on real
   held-out video) and `eval_ablation/archive/phase3_segment_budget_sweep.py`
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
   removal of the Phase-2 caps (`fold_max_units = 5`, `_draw_units`'
   `u_cap = min(max_units, N//4)`) that would otherwise hold short clips below
   the budget.
7. **Code — DONE 2026-09-20:** item 4 (chunked encoder forward,
   `--vision_encoder_chunk_frames`, launcher default now 32) — no longer optional at Phase-3
   lengths — plus the length settings it forces, now the launcher's defaults:
   `--vision_max_tokens 65536` (up from the per-video 16,384 default, which
   crushes a 1200-frame clip to ~8 tokens/frame) and `--model_max_length 131072`,
   above the resulting *pre-compression* `input_ids` length, which
   `Phase2FoldDataset.__getitem__` checks and otherwise rejects every long clip.
   The LLM never sees that length — the rewrite happens before `embed_tokens` —
   so this only widens a guard.
8. **Phase 3 run** (budget fold) — **done once** (`phase3_budget_fold`, on the 2026-09-21
   blend; §4b budget fold). The length curriculum was retired with InternVid's long
   bucket, and the needle probes were never built. The current launcher config (joint
   qbase, image share, ckpt-500 warm start) has not been run.
9. **qbase-vs-pooling probe** (2026-09-29) → fold line parked; §5 items 19–21 →
    qbase enhance, which became **Phase 2** in the 2026-10-02 renumbering (§4).
10. **MLVU-clean Phase 2** — §5 item 24 → `qbase_enhance_mlvuclean` (finished 2026-10-03).
11. **Evaluation round + H1 (2026-10-03/04)** — repro check, short 1 fps + hybrid, long
    hybrid, the full-benchmark profile (`long_mcq_budget.py --raw_divs …` →
    `long_mcq_profile.py`), qbase-vs-pooling on the clean checkpoint, and the H1 run
    `qbase_qa_mlvuclean` (rejected). §4 "MLVU-clean Phase 2".
12. **Inference-time follow-ups (2026-10-04)** — free-form answers on robust failures, 8 frames
    per segment, qbase + raw keyframe hybrids (`lowraw`), budget scaling, and the pre-registered
    short-clip confirmation; eval only, drivers in `work_dirs/<name>/`. §4 "Inference-time
    follow-ups".

---

## Load-bearing facts

- **LLM cross-entropy only.** No reconstruction / MSE / masked-feature objective —
  every such variant collapses the bottleneck to be perfectly order-invariant
  (`cos(real, shuffled) = 1.0000`). Option A (affine mean/std match) and Option B
  (CORAL centroid + covariance + norm) operate on **distribution moments**, not
  per-token targets, and are allowed.
- **No pre-extracted vision-feature cache** anywhere. Frames decoded + encoded on
  the fly every step; the encoder has no cross-frame attention, so it runs in
  frame chunks bit-exactly (§5 item 4).
- **Gate on the grounding probe** — centered cross-video cosine (`xcos_cen`) and
  `struct_ρ` vs the encoder — **not** raw `content_collapse`, which is dominated by
  an inherent SigLIP-NaViT common component (`cc_frac ≈ 0.94`, `content_collapse ≈
  0.87` on the raw encoder itself).
- **Eval is a per-task held-out slice carved out of the training pool** (§6
  "Held-out policy"), never a whole benchmark reserved wholesale and never an
  in-pool sample. Split by video, union across tasks, disjointness asserted by
  `dataset_util/build_phase3_blend.py`. This is why VDC can be training data.
  Still-image streams keep no slice; InternVid is never measured on; external
  benchmarks need a content-level overlap check (§6 "Benchmark contamination").
- **"Did X improve?" is answered answer-grounded:** CE on real annotated answers or
  benchmark accuracy, with a control checkpoint, an unseen slice and a bootstrap CI.
  Feature-space ratios / deletion sensitivity / attention maps only corroborate.
- **LLM frozen in Phases 1–2 and in the fold stages; `mm_projector` unfrozen.**
  Those checkpoints are evaluated as *compressor + the original frozen LLM* against a
  no-compression baseline.
- **No raw tokens next to compressed ones while training the compressor.** A per-unit
  raw-qbase escape hatch (`r_u`) was tried and dropped (§4b unit fold) — do not
  reintroduce it. The budget fold's raw bypass (a segment is raw OR folded) is not
  that. At inference, raw + compressed hybrids
  at an equal total budget are allowed.
- **Frame-order / temporal-blindness metrics are a diagnostic, not a gate.** Every
  fold checkpoint evaluated so far shows flat frame-order insensitivity regardless
  of time-embed choice or qbase-unfreeze — read as a property of CE-on-generic-
  captions training, not a compressor defect to engineer away.
