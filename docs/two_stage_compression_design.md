# Two-stage dynamic compression — design notes

**Status:** design agreed; §6.1–6.6 resolved. **Stage-2a is implemented** (2026-09-03):
`TwoStageCompressor` in `compressor.py` (`transformer_decoder_flat+mamba`), the
Stage-2 branch in `videollama3_arch.py::compress_visual_tokens_with_compressor`, and
`videollama3/train/stage2a_pretrain_compressor_fold.py` +
`shell/pretrain_stage2a_fold.sh`. Remaining for the first run: build the
length-filtered manifest (`anno_data/stage_2_training.json`, build order item 1) and
the collapse guardrail (4d). Stage-2b has two open items — the segmenter boundary
criterion (§6.1 sub-decision) and one integration point (model-side tokenised
`Time:` strings, §7). Stage-3/4 are design-only.

**Stage-2a — deltas from the spec below (deliberate, keep the first milestone small):**
- **Pure-PyTorch SSD, no `mamba_ssm`.** `segment_aggregator.py`'s reference
  `_ssd_chunk_scan` is used as-is; `mamba-ssm` stays an optional throughput dep
  (commented in `requirements.txt`). Revisit if the fold is a step-time bottleneck.
- **Per-unit loop, not one packed `seq_idx` scan.** `SegmentAggregator.forward` folds
  one unit per call; `TwoStageCompressor.compress_windows` loops over the `U ≤ 5`
  units. A packed varlen scan with unit-boundary state resets is the `mamba_ssm`
  upgrade, not needed at `U ≤ 5`.
- **`time_embed="index_sincos"` for 2a** (segment-index sinusoid — the aggregator's
  built-in, exercised in its `__main__`). `rel_gap_mlp` / `seconds_mlp` is the 2b
  form; wiring `segment_seconds` through `compress_windows` is deferred.
- **Segments derived in the arch, not supplied by the dataset.** The dataset emits
  only `U` contiguous `compression_parts`; `TwoStageCompressor` subdivides each into
  `ceil(T / frames_per_segment)` segments from `frames_per_segment` + the part's
  `(h, w)`. One new scalar, no new per-segment dataset field.
- **Per-unit fold-depth randomization is not on yet** — 2a starts with uniform
  `segs_per_unit`; the randomization is a follow-up flag on `Stage2UnitDataset`.
- **No chunked encoder forward yet** — the single `vision_encoder(...)` call is
  reused (proven at `max_frames` ≈ 200 by the Stage-1 recipe). Add chunking only if
  the encoder forward OOMs on the longer manifest.
- **Warm-start reads the qbase straight from an HF checkpoint dir**
  (`work_dirs/compressor_pretrain_video_norm`) — `TwoStageCompressor.load_stage1_pretrained`
  pulls `model.token_compressor.*` from the shards; no pre-extraction step.
**Scope:** the compression *strategy* for streaming + offline video. Stage-1
(`videollama3/model/compressor.py`) already exists and is pretrained
(`work_dirs/compressor_pretrain_video_norm`: `transformer_decoder_flat`,
`num_queries=64`, `match_encoder_scale` on, `distr_loss_weight=0.05`). Stage-2
(`videollama3/model/segment_aggregator.py`) is **specified in §4 / §7 but not yet in
the tree** — see build order item 2.

---

## 1. Goal

Compression should be **dynamic and non-uniform**, so the same mechanism serves
streaming and offline video, and so a spare token budget is spent on keeping
resolution rather than always compressing at a fixed ratio.

A dynamic scheme has to define three operations on the compressed representation:

1. `raw → compressed`
2. `compressed + raw → compressed`
3. `compressed + compressed → compressed`

Prior work mostly ignores case 3, or does it with training-free merging / pruning
(MovieChat, Flash-VStream, Video-XL, LongVU). Our novelty is the **LLM-CE-trained
lift + associative fold** (cases 1/2), the **bounded-`U` / unbounded-fold-depth
budget allocation** (§6.4), and the **streaming budget controller built on an exact
state-combine** (case 3, §4). Note that for the SSD state, case 3 is *mathematically
identical* to case 2 over the concatenation (§2) — the contribution there is that the
combine is exact and training-free, not that it carries information a plain fold
would not.

---

## 2. Unifying abstraction: an associative fold over a stream

The three cases are one algebraic structure — a monoid-style `reduce` over the
segment stream:

| case | role | operator |
|---|---|---|
| 1 | `lift`  : raw span ↦ element of the compressed semigroup | Stage-1 query compressor, then fold into a fresh state |
| 2 | `append`: `state ⊕ lift(span)` | Stage-1 lift + Mamba `update` |
| 3 | `merge` : `state ⊕ state` | **SSD-native state combine** (see §4), *not* training-free token merge |
| — | `readout`: `state ↦ M tokens` | M learned queries through the recurrence, run lazily |

The property that keeps this clean is **chunk-invariance / associativity**: the fold
result must not depend on how the stream was cut. The SSD recurrence has this for
the SSM state; a heuristic token merge does not.

**`append` and `merge` are equivalent for SSD.** By associativity,
`merge(fold(A), fold(B)) ≡ fold(A ++ B)` exactly on `ssm_state`. So `merge` carries
**no information advantage** over a plain deep `append` — it is a streaming /
numerical / parallelism tool, not a memory mechanism. The finite-state "flooding" on
long high-action video (§6.2) is *not* fixed by merging; hybrid memory (Stage-3) is
the answer. Consequences for the build:

- **Stage-2 uses `append` only** — one variable-depth SSD fold per unit (§6.5).
- **`merge` (case 3) is reserved for the Stage-2b streaming budget controller** —
  where two *already-materialised* unit states must be combined to hold `U` at its
  cap.
- With `d_conv` at its default there is a bounded `d_conv-1` (=3) token seam error on
  `merge` (§4), so `append` is strictly preferable wherever the whole unit is
  available (2a, and offline 2b).

---

## 3. The canonical object is the **SSM state**, not a token set

"State as canonical object" = the thing Stage-2 stores / streams / checkpoints /
merges is the recurrent SSM state, **not** a set of tokens.

| what you carry | case 3 becomes | verdict |
|---|---|---|
| the `readout()` M summary tokens | `M + M → M`, only doable by training-free merge/prune; Stage-2 never trained on merged summaries; non-associative; error accumulates over the merge tree | rejected |
| the **SSM state** (`(conv_state, ssm_state)` per layer) | combine two states with the SSD associative operator — exact, training-free, identical math to `_ssd_chunk_scan` step 3 (inter-chunk recurrence) | **adopted** |

Consequences:

- Query-based compression is **not** removed. It appears in two places: (a) the
  Stage-1 lift (bidirectional cross-attention within a segment), and (b) `readout`
  (M learned queries). What changes is only the *type of the thing passed around
  and re-merged* in Stage-2 — a state, not tokens.
- `state` and `readout(state)` are **never equivalent** — different spaces, different
  shapes. `state` is `(nheads, headdim, d_state)` per layer (an outer-product
  accumulation); `readout` output is `(M, d_output)` in token space. There is no
  `t=0` at which they coincide. The LLM can only eat `readout` output, so readout
  is mandatory before the LLM, always.

---

## 4. Case 3 in detail — SSD-native state combine

Mamba-2 / SSD state is a linear operator, so two chunk states combine with the
associative-scan operator `(A₁,b₁) ∘ (A₂,b₂) = (A₁·A₂, A₁·b₂ + b₁)`. This is already
implemented for the offline scan in `segment_aggregator.py::_ssd_chunk_scan` step 3
(to be written — build order item 2); case 3 is that same combine exposed as a public
op on `AggregatorState`.

Caveat — the depthwise causal conv: the full per-layer state is
`(conv_state, ssm_state)`. The combine is exact for `ssm_state`; `conv_state` is just
the last `d_conv-1` input vectors, so at a merge seam the conv window straddles the
join → a bounded approximation of `d_conv-1` (=3) tokens per seam.

**Resolved (§6.6): keep `d_conv` at its default.** The earlier proposal to set
`d_conv=1` (to make case 3 fully exact, at a capacity cost) is **dropped**. The conv
capacity is worth more than seam exactness because case 3 only runs in the Stage-2b
streaming budget controller, over coarse / old memory, where a 3-token seam
approximation is acceptable. Everything that has the whole unit available uses
`append` (§2, §6.5), which has no seam.

---

## 5. Training stages

Reconstruction / MSE / masked-feature objectives collapse the bottleneck — a
**settled finding that applies to Stage-2/3/4 as well**. Every stage is trained with
LLM cross-entropy only.

Constraints shared by all video stages:

- Compression path is `B == 1` (arch); the collator flattens
  (`use_batch_flattening`). Single CE forward, `use_dual_forward=False`.
- **No pre-extracted vision-feature cache** (disk budget). Frames are decoded and
  encoded on the fly every step — see §7 "On-the-fly encoding (no feature cache)".
- Data: `/root/datasets/videoxl/Finetuning/*`. Build a **length-filtered manifest**
  first (≥ ~100 frames / ≥ ~60 s). Measured durations of the caption-heavy sources:
  sharegpt4v ~10–52 s (med ~18), gpt4o_video ~11–35 s (med ~22), baaicaption
  ~7–62 s (med ~27), **vcg ~23–235 s (med ~132)**, **cinepile ~86–218 s (med ~154)**.
  The filter therefore keeps mostly **vcg + cinepile + the long tail of
  sharegpt4video**; median target ≈ 130–160 frames at 1 fps. Short clips barely
  exercise the fold. `eval_ablation/build_manifest.py` has a duration probe + clip
  picker to reuse.

### Stage-1 — done, not touched

`videollama3/train/compressor_pretrain_with_videollama3.py` stays as-is: whole span
→ one window → K tokens → `mm_projector` → frozen LLM → CE. "LLM-readable K tokens"
is load-bearing and a good Mamba input distribution. Do not modify it for Stage-2.
Stage-2 reuses this same qbase (`work_dirs/compressor_pretrain_video_norm`,
`transformer_decoder_flat`, `num_queries=64`) and its K-token query bank, but applies
it **per 1–8-frame segment** rather than to the whole video; the pretrained weights
warm-start `TwoStageCompressor.stage1` (§7, `load_stage1_pretrained`).

### Stage-2 — segment fold trainable, LLM frozen

**New script** (`videollama3/train/stage2_pretrain_compressor_fold.py` or similar).
It borrows from `compressor_pretrain_with_videollama3.py` only the frozen-
everything-but-the-fold setup, the single CE forward (`use_dual_forward=False`) and
`zero1`. The data path is new — multi-unit `compression_parts`, per-segment
boundaries, randomized per-unit fold depth — none of which that script has; it also
needs the multi-part collator offset logic from
`videollama3_chat_finetune_compressor.py`. Populates the empty
`anno_data/stage_2_training.json`.

**Vocabulary (Stage-2 onward).**

| sym | meaning | set by | ex. |
|---|---|---|---|
| N | segments the video is cut into | segmenter — uniform `frames_per_segment` in 2a, causal content-adaptive in 2b | ~38 (155 f @ 4 f/seg) |
| **K** | tokens Stage-1 (qbase) emits **per segment** — fixed | `num_queries` (pretrained) | 64 |
| **M** | tokens **one** Stage-2 readout emits **per unit** — **tied to K** | `stage2_n_summary_tokens` | 64 |
| **U** | readout-units the video is split into (= independent folds) — **capped** | 2a: dataset partition, ≤ `stage2_max_units` · 2b: streaming controller | ≤5 (typ. 2–4) |
| — | frames per segment | `--stage2_frames_per_segment`, clamp [1, 8] | 4 |
| — | **segments per unit (fold depth)** | **variable** — `--stage2_segs_per_unit` target, randomized per unit in training; grows with length at inference | ~6 (2a), unbounded |
| — | vision tokens the LLM sees for the video | `U·M` | ≤320 (typ. 128–256) |

```
T frames ─encoder→ T×(h·w) tokens
  └ segmenter (frames_per_segment f/seg, uniform in 2a) → N segments
      └ qbase, per segment → N×K tokens
          └ group into U ≤ stage2_max_units CONTIGUOUS units
            (segs-per-unit variable; randomized per unit in training; remainder → last unit)
              └ Stage-2 SSD fold, fresh state per unit → 1 SSM state / unit
                  └ readout: M learned queries through the state → M tokens / unit
  LLM input = U·M ≤ 320 tokens   (each unit gets its own "Time:{a}s-{b}s:" prefix)
```

- **Multi-unit is mandatory, in 2a too.** A fixed-size SSM state carries O(1)
  information; folding a whole long video into one M-token readout starves the LLM
  and wastes its token budget. But **`U` is capped** (≤ `stage2_max_units` = 5) and
  **`M` is fixed** (= K = 64), so `U·M ≤ 320` for any video length — the LLM's
  vision-token cost is bounded and predictable. **Video length is absorbed by the
  fold depth (`segs_per_unit`), not by more units.** At 2a the dataset partition
  guarantees the cap: `U = min(stage2_max_units, ceil(N / segs_per_unit_target))`;
  each unit is an **independent** fold over a **contiguous** slice of segments
  (fresh `init_state`). This is what "fold **adjacent** segments' states" means.
- **M and U are bounded; fold depth is not.** `M ≤ effective state rank` (§7). With
  `d_model=1152, expand=2, headdim=64 → nheads≈36, d_state=128` the state is
  `(36,64,128)` and the readout's usable rank is well below that — `M = 64` sits
  comfortably inside it. `M ≈ 1000` would be mostly duplicated rows. Spend more
  representational budget via a bigger state (`d_state`/`nheads`/`n_layers`) or
  (Stage-3) retained high-res segments — never by inflating `M` past K.
- **Before the projector (§6.3).** `SegmentAggregatorConfig.d_input = d_output =
  1152` (compressor hidden). Fold in compressor space; readout → the single existing
  frozen `mm_projector` call → LLM.
- **Stage-1 flavour.** `transformer_decoder_flat` only (K flat `num_queries`, a
  clean Mamba input). `compressor_type = "transformer_decoder_flat+mamba"`;
  `build_token_compressor` must **strip the `+mamba` suffix before** its existing
  `ct == "transformer_decoder_flat"` / `"transformer_decoder" in ct` checks
  (otherwise the substring match builds the grid variant — see §7).
- **Streaming budget controller** (2b only): fixed heuristic, **no gradient** (like
  `prune_kv_by_common_component`'s hard top-k; gradients still flow along whichever
  path each state took). At 2a `U ≤ 5` is guaranteed by the dataset partition, so
  there is no controller. At 2b, as units arrive on the stream, when a new unit
  would push `U` past the cap the controller `combine_states` (§4) the two adjacent
  oldest unit-states. Never training-free token merge.
- **Data mix: caption-heavy** (sharegpt4v, gpt4o_video, baaicaption, vcg, cinepile);
  short-answer QA barely pressures a frozen-LLM compressor — defer to Stage-3.
  Length-filtered manifest (≥ ~100 frames), bucketed by frame count so the fold
  depth stays in a stable range per grad-accum window (recurrence depth varying
  wildly within a window is unstable).

#### Stage-2a — unit boundaries fixed by the dataset, uniform segments

Goal: prove the multi-unit fold + readout does not collapse and CE trains, with the
**smallest** code change.

- **Units:** a **new partition function** (NOT `select_full_compression_parts`, which
  cuts fixed frame windows and discards a ≤3-frame remainder). It cuts the video
  into `U = min(stage2_max_units, ceil(N / segs_per_unit_target))` contiguous
  segment ranges, each range one `compression_part`. Remainder segments fold into
  the **last** unit. `segs_per_unit_target` from `--stage2_segs_per_unit`.
- **Per-unit fold depth is randomized in training.** Each unit's segment count
  `N_u` is drawn around `segs_per_unit_target` (e.g. `Uniform[4, 16]`, `Σ N_u = N`,
  `U ≤ stage2_max_units`) so the fold sees a range of recurrence depths and learns
  to extrapolate; it also doubles as augmentation (same video, different fold path
  per epoch). Bucket by `N` within a grad-accum window; grad-clip. The **LLM-side
  sequence length is invariant** to this (M queries regardless of `N_u`), so 2a
  stays easy to debug.
- **Segments:** produced **once**, globally, right after the encoder — not per unit.
  In 2a: fixed uniform `frames_per_segment` (default 4, ≤ 8) — no content-adaptive
  segmenter yet. The dataset supplies `segment_boundaries` (frame indices) and
  `frame_seconds`.
- **Compute — one vectorised path, no per-unit Python loop.** Inside
  `TwoStageCompressor.forward`:
  1. Stage-1: one varlen cross-attention over **all N segments** → `N×K`.
  2. SSD fold over the `N×K` sequence with **state reset at unit boundaries**
     (`mamba_ssm.Mamba2` with `seq_idx` / packed varlen) → `U` states.
  3. Readout: batched `M`-step scan over the `U` states → `U×M`.
- **Arch:** `output_hw_for → (1, M)`; the **existing multi-part machinery** in
  `prepare_inputs_labels_for_multimodal` (per-part `<|compression_start|>…
  <|compression_end|>`, per-part `compression_ts_info` → `Time:{a}s-{b}s:` via
  `build_range_ts_info`) handles the `U` blocks unchanged. **No
  `prepare_inputs_labels_for_multimodal` refactor.** The only new arch code is the
  Stage-2 branch in `compress_visual_tokens_with_compressor` (build a segment-level
  `cu_seqlens` alongside the existing part-level one, call
  `TwoStageCompressor.forward`, scatter `U·M` tokens via the existing
  `replace_mask`) plus the chunked encoder forward.
- **Recurrence depth at 2a is modest** — the filtered manifest tops out near ~235
  frames, so `N ≈ 30–60` and `N_u ≈ 6–15`. **Full backprop through the fold; no
  TBPTT** (that is a Stage-4 concern).
- **Time embedding:** keep it trainable, but 2a can use a **simple per-segment
  scalar** (cumulative frames within the unit, or segment duration in seconds)
  through `time_mlp`. The full `rel_gap_mlp` is the 2b form (§7).
- **Trainable:** the Stage-2 fold only — `Mamba2` block, `input_proj`/`input_norm`,
  `output_proj`, `summary_tokens`, `time_mlp`. **qbase frozen** (the known-good
  baseline to debug collapse against; warm-started via `load_stage1_pretrained`
  from `work_dirs/compressor_pretrain_video_norm`). `mm_projector` optionally
  unfrozen (tiny, low risk); else it joins in Stage-3. Single CE
  (`use_dual_forward=False`), `zero1`.

#### Stage-2b — in-model content-adaptive segmenter + budget controller

- **Segmenter:** in-model, **causal**, on encoder features — boundary when the
  consecutive-frame mean-feature cosine distance exceeds τ, forced at 8 frames.
  Non-uniformity comes from this segment density (§6.1(a)); the exact criterion is
  the open sub-decision (§6.1), pluggable, but every segment is clamped to [1, 8]
  frames whatever the criterion. An offline whole-video segmenter would not transfer
  to Stage-4. The per-unit fold-depth randomization from 2a carries over; now
  segment *lengths* are non-uniform on top of it.
- **Arch refactor (the big one):** `U`, per-unit token count and per-unit time range
  are now decided inside the model, so the dataset can't pre-build the parts.
  `encode_images` returns the unit structure; the compressed-block assembly in
  `prepare_inputs_labels_for_multimodal` is rewritten to consume it (§7). Land 2a
  with this path frozen first.
  - **Open integration point:** the model now owns the per-unit time ranges, but the
    compressed-block assembly needs them as **tokenised** `Time:{a}s-{b}s:` strings,
    and `prepare_inputs_labels_for_multimodal` has no tokenizer. Resolve by either
    passing the tokenizer into the arch call, or having the collator supply a small
    pre-tokenised lookup the model indexes. Decide before the refactor.
- **Streaming budget controller:** keeps `U ≤ stage2_max_units` by `combine_states`
  (§4) on adjacent oldest unit-states; fixed no-gradient heuristic; accepts the
  3-token conv seam (`d_conv` default).
- **qbase joint polish:** unfreeze at ~10× smaller LR than the fold (its consumer is
  now the fold, not the LLM) — not from step 0; two CE-only trainable modules from
  cold is under-constrained. Needs `create_optimizer` to split
  `token_compressor.stage1` / `.stage2` into separate LR groups (currently one group
  keyed on `token_compressor`).

**Gate to Stage-3**: collapse guardrail passes (`feature_distribution.py` on readout
output — `cos(real, shuffled)` low, effrank high) **and** caption metrics have
plateaued **and** the budget ablation (`U` at the cap vs. `U` unconstrained) shows
the constraint is nearly free.

### Stage-3 — + `mm_projector` + full-parameter LLM finetune + hybrid memory

- **Hybrid memory enters here (§6.2 — moved up from Stage-4).** On top of the
  always-on O(1) SSD fold, a small variable-size set of *retained* less-compressed
  segments contribute their K qbase tokens directly, interleaved in time order with
  a unit's M readout tokens (`n_u = M + Σ K_retained`). Because **M = K = 64** the
  interleaving is dimension-clean. Which segments are retained is a fixed
  no-gradient heuristic (e.g. lowest neighbour similarity / highest motion, within
  budget), like the merge controller. The Stage-2b arch refactor already returns a
  variable per-unit token count, so this drops in with no further shape change.
- **Split 3a / 3b:** 3a = hybrid path on, LLM still frozen (isolate it); 3b =
  unfreeze the LLM. Every trainable module + full LLM + the new retained path in one
  step is too many moving parts.
- Trainable: qbase, fold, `mm_projector`, **full LLM** (not LoRA — LoRA is the
  documented fallback if compute / stability forces it).
- **DeepSpeed**: move from `zero1.json` to **`zero2.json` (shard grads) or
  `zero3.json`** — 7B full FT + on-the-fly encoder forward + compressor + fold will
  not fit ZeRO-1 comfortably.
- **Staggered LR**: `llm_lr` well below the rest, e.g. `llm_lr 5e-6–2e-5` vs
  `compressor_lr 1e-4` (the trainer already has per-group LRs).
- **Anti-forgetting**: mix in a slice of VideoLLaMA3's original SFT data (or a
  general video-instruction set) as replay; keep Stage-3 short — it is a re-align,
  not a retrain. QA data (nextqa, ego4d, cinepile) enters here.
- **Do not reorder**: Stage-2 (frozen LLM) must converge first. Unfreezing the LLM
  over a half-baked, still-moving compressed representation lets the LLM race ahead
  and adapt to a moving target.
- **Measurement requirement (not optional)**: with the LLM trainable, end metrics no
  longer isolate "the compressor improved". Every Stage-3 checkpoint must **also** be
  evaluated as *compressor + the original frozen LLM*, reported against the Stage-2
  frozen-LLM baseline — otherwise the gains cannot be attributed to compression
  rather than the LLM memorising the task.

### Stage-4 — long-video finetune (only partly optional)

- **Hybrid memory is already in as of Stage-3 (§6.2).** Stage-4 only tunes its
  retained-set size policy under genuine long-video forgetting.
- **Unbounded folding for real**: this is where the variable `segs_per_unit` is
  actually pushed deep. Forward streamed segment-by-segment; backward is **TBPTT
  over a window of recent segments** (older segments detached). Mandatory here —
  with no feature cache the offline `[N·K ; M]` sequence cannot be materialised for
  a long video (§7).
- **Length curriculum**: grow max video length across Stage-4; do not start at
  hours. (Stage-2a only ever reached fold depth ~15.)
- **Data**: videoxl tops out near cinepile length; Stage-4 likely needs external
  long-video data (MovieChat-1K, LVBench, Ego4D-NLQ, VideoMME-long) and
  retrieval / needle-style eval to expose forgetting. May switch to the online data
  format (`videollama3_chat_finetune_online.py`).

---

## 6. Decisions

1. **Where does non-uniformity come from? — RESOLVED: (a) segment density, fixed K.**
   Fixed K per segment (`transformer_decoder_flat` learned queries), content-adaptive
   causal boundaries; dynamic content → shorter segments → more segments → more
   tokens for that span. `tokens_per_segment` (K) stays fixed, so `forward()`'s
   `S % K == 0` holds and no varlen packing is needed. Variable-K / Matryoshka
   queries (former option b) is dropped for now.
   - **2a default:** `frames_per_segment = 4`, uniform (still inside the [1, 8]
     clamp), so `N` is large enough that `segs_per_unit` stays sane without starving
     `U`.
   - **`segs_per_unit` (fold depth) is a *separate* axis from segment density** — it
     is variable and randomized in training, and is the lever that absorbs video
     length (§6.4). Segment density controls *where within a span* the tokens go;
     fold depth controls *how many spans* fold into one state.
   - **Still open (sub-decision): the exact boundary criterion.** Candidates to
     ablate — (i) consecutive-frame encoder mean-feature cosine distance > τ (the
     working default); (ii) uniform 8 (ablation floor); (iii) cumulative drift from
     a segment-start anchor > τ (won't over-cut slow pans); (iv) RGB shot boundary
     (PySceneDetect / TransNetV2 style, no encoder needed); (v) attention-entropy /
     K-budget triggered. All share the hard **[1, 8]-frame clamp**.

2. **Bounded state vs budget-adaptivity? — RESOLVED: fixed O(1) state for Stage-2,
   hybrid memory from Stage-3.** Stage-2 folds into a single fixed-size SSM state per
   unit, no retained set. The hybrid (O(1) SSD backbone + a small variable-size set
   of *retained* less-compressed segments, compressive-transformer / ∞-former style)
   enters at **Stage-3**, not Stage-4. Note that splitting a fold and `combine`-ing
   the sub-states is *exact* for SSD (§2), so **merge is not a memory mechanism** — a
   finite state still floods on a uniformly high-action long passage; hybrid memory
   is the answer, not merge. A good global budget controller under streaming is still
   a genuine open problem, deferred with Stage-4.

3. **Stage-2 before or after `mm_projector`? — RESOLVED: before.** `d_input` =
   compressor hidden (1152), Stage-2 folds in compressor space, readout → the single
   existing frozen `mm_projector` call → LLM. Smallest change to `encode_images`
   (projector stays one call at the end); CE through the frozen projector pulls the
   readout onto a projector-compatible manifold. `SegmentAggregatorConfig` must set
   `d_input = d_output = 1152`, not the 3584 default.

4. **`U` cap vs. fold-depth lever? — RESOLVED: cap `U`, vary fold depth.**
   `U ≤ stage2_max_units = 5`, `M = K = 64`, so `U·M ≤ 320` for any video. Video
   length is absorbed by `segs_per_unit` (unbounded SSD fold depth), never by more
   units. Rationale: a bounded, predictable LLM vision-token cost; the SSD fold is
   the O(1)-state mechanism that makes arbitrary depth viable; Stage-3 hybrid memory
   is the variable-size escape hatch for when a fixed state is not enough.

5. **`append` vs `merge` for the Stage-2 fold? — RESOLVED: `append`.** One
   variable-depth SSD fold per unit. `combine_states` (case 3) is reserved for the
   Stage-2b **streaming** budget controller (two already-materialised states forced
   together). For SSD the two are mathematically equivalent on `ssm_state`
   (associativity), and with `d_conv` at default `combine` carries a bounded 3-token
   seam error — so `append` is strictly preferable wherever the whole unit is in
   hand (2a, offline 2b).

6. **`d_conv`? — RESOLVED: keep the default (no `d_conv=1`).** The proposal to set
   `d_conv=1` for an exact `combine` is dropped. Conv capacity outweighs seam
   exactness because case 3 only runs over coarse / old memory in the 2b controller.

---

## 7. Implementation details / caveats

### `segment_aggregator.py` (to be created — build order item 2)

- **K / M config-driven** from the compressor config, **K = M = 64** (K from the
  pretrained qbase, M tied to it). `forward()`'s `S % K == 0` assert stays valid (K
  fixed).
- **`d_input = d_output = 1152`** (before projector, §6.3) — not the 3584 default.
- **`AggregatorState.combine(a, b)` — case 3, needed by the 2b streaming controller.**
  Add a per-layer `log_decay` accumulator to `AggregatorState` (`(B, nheads)`, 0 at
  `init_state`); the step returns its per-step `dt·A`, `update` accumulates it.
  `combine`: per layer `ssm = a.ssm · exp(b.log_decay)[…,None,None] + b.ssm`,
  `log_decay = a.log_decay + b.log_decay`, `conv_state = b.conv_state`, `n_seen`
  summed. **`d_conv` stays at its default** — the `d_conv-1` (=3) token seam per
  merge is accepted (§4, §6.6). Unit-test
  `combine(reduce(segs[:i]), reduce(segs[i:])) ≈ reduce(segs)` for several `i`,
  within the seam tolerance. *(Assumes SSD's scalar-per-head `A`; if the module ends
  up with a full diagonal `A`, the combine formula must be revised.)*
- **Time embedding.** Drop the unbounded `n_seen`-indexed / raw-seconds paths
  (`index_sincos` / `seconds_mlp` run off the training range for large N and fight
  randomized / non-uniform segmentation).
  - **2a:** a single per-segment scalar (cumulative frames in the unit, or segment
    duration in seconds) through `time_mlp`.
  - **2b:** `time_embed="rel_gap_mlp"` — per segment feed
    `(gap_seconds since previous segment, duration_seconds)` through `time_mlp`.
- **Fold parallelism — no per-unit loop.** The fold over the whole `N×K` sequence
  with state reset at unit boundaries is **one call** via `mamba_ssm.Mamba2` with
  `seq_idx` (packed variable-length sequences). The readout is a batched `M`-step
  scan over the `U` states. Nothing loops over units in Python.
- **`mamba_ssm` is a training dependency.** It provides the `seq_idx` varlen path
  the fold needs. Add `mamba-ssm` + `causal-conv1d` to `requirements.txt` (build
  against Torch 2.9.1 / CUDA 13.0; may need building from source). The pure-PyTorch
  `_ssd_chunk_scan` is the **numerical reference / unit-test oracle only** — it need
  not support varlen.
- **Offline `forward()` memory / TBPTT.** `forward()` builds `[N·K ; M]` as one
  sequence and runs the full SSD scan → train memory is O(total video length). TBPTT
  over a window of recent segments is mandatory at **Stage-4** (and any run whose max
  fold depth × K exceeds memory) — forward streamed, older segments detached.
  Stage-2a's length-filtered manifest keeps depth ≤ ~15 segments, so **full backprop
  through the fold is fine at 2a / 2b**.
- `readout()` runs M queries through the step from a **copy** of the state
  (non-destructive) — streaming continues after a readout. Cost per readout =
  M steps × n_layers, batched over U.
- **M ≤ effective state rank.** With `M = 64` and a `(≈36, 64, 128)` state this is
  comfortably satisfied.
- Streaming boundary detection must be **causal**; a segment is buffered before the
  bidirectional Stage-1 compresses it → added latency ≈ one segment length.

### On-the-fly encoding (no feature cache)

- No pre-extracted `.pt` feature cache (disk budget). The frozen SigLIP-NaViT
  encoder runs every step. Its output is also what the global segmenter reads.
- The encoder has **no cross-frame attention** —
  `videollama3/model/videollama3_encoder/modeling_videollama3_encoder.py` builds
  `cu_seqlens` per frame (`h*w` per frame), 2-D spatial RoPE only. Frame *t*'s output
  is independent of the other frames, so **encode segment-by-segment** (≤ 8 frames at
  a time) and never materialise all `T×HW` encoder outputs at once. Encoder-side
  memory is then bounded by one segment regardless of video length.
- Combined with streaming fold `update`, the vision + compressor + fold forward is
  O(1) in video length; only the LLM forward over the final compressed + text
  sequence is O(budget) = O(`U·M`) ≤ O(320).
- Cost moves from disk to CPU decode. Source videos are already extracted under
  `/root/datasets/videoxl/Finetuning/data`; the loader re-reads + ffmpeg-decodes them
  each epoch. Mitigate with high `dataloader_num_workers`, persistent workers,
  prefetch, fixed 1 fps, hard `max_frames`.
- `--separate_modality_batches` is not needed (all decoded pixels, one modality).

### `videollama3_arch.py` — the real integration point

- **`compressor = Stage-1 + Stage-2` wrapper** (`TwoStageCompressor` in
  `compressor.py`): `.stage1` (a `transformer_decoder_flat`, K = 64) + `.stage2`
  (`SegmentAggregator`), all params under `token_compressor.stage{1,2}.*` so the
  trainer's existing save (`keys_to_match` already has `token_compressor`) and
  `create_optimizer` grouping pick them up. `build_token_compressor` recognises a
  `"…+mamba"` suffix on `compressor_type` — **strip it first**, then run the existing
  `ct == "transformer_decoder_flat"` / `"transformer_decoder" in ct` branch
  selection (the substring check would otherwise pick the grid variant).
  `output_hw_for(h, w)` returns `(1, M)` = `(1, 64)`.
- **`TwoStageCompressor.load_stage1_pretrained(path)`** — warm-start helper. The
  full checkpoint keys are `model.token_compressor.layers.{0..7}.*`,
  `model.token_compressor.query`, `model.token_compressor.out_gamma`,
  `model.token_compressor.out_beta`; a separately-saved `compressor_pretrained.pt` is
  relative. Strip a leading `model.token_compressor.` / `token_compressor.` prefix,
  then `self.stage1.load_state_dict(sd, strict=False)` — `query`, `out_gamma`,
  `out_beta` (the `match_encoder_scale` affine) ride along; `stage2.*` stays at its
  fresh init (expected "missing").
- **Stage-2a — no `prepare_inputs_labels_for_multimodal` refactor.** The dataset
  emits `U` contiguous `compression_parts` (one per readout-unit) plus, per part,
  uniform `frames_per_segment` `segment_boundaries` and `frame_seconds`.
  `compress_visual_tokens_with_compressor` gets a Stage-2 branch: build a
  **segment-level `cu_seqlens`** (segment-count × K, for the fold) alongside the
  existing **raw-token `cu_seqlens`** (for the Stage-1 KV gather), run
  `TwoStageCompressor.forward(part_tokens, segment_cu_seqlens, unit_cu_seqlens,
  per_seg_grid_hws, segment_seconds)` → `(U·M, hidden)`, scatter into each part's
  first M slots via the existing `replace_mask`. per-segment `(h, w)` = the video
  grid broadcast. The existing multi-part block assembly handles the `U` blocks
  as-is.
- **Stage-2b — the refactor.** The in-model causal segmenter + streaming budget
  controller decide `U`, per-unit token count and per-unit time range from encoder
  features, so the dataset can no longer pre-build the parts. `encode_images`
  returns the unit structure (`[{tokens, start_sec, end_sec}], unit_lengths`); the
  compressed-block assembly loop in `prepare_inputs_labels_for_multimodal` is
  rewritten to consume that instead of `compression_parts` — cut the whole video's
  per-frame `Time X.0s:` + image tokens, re-insert `U` blocks. **Open point:** the
  model-decided time ranges must be tokenised into `Time:{a}s-{b}s:` strings, and
  this function has no tokenizer — pass it in, or have the collator supply a
  pre-tokenised lookup. Land 2a first with this path frozen.
- **Stage-3 hybrid.** Same returned structure, but a unit's token block is
  `M + Σ K_retained` — the retained segments' raw qbase tokens interleaved in time
  order with the M readout tokens. `M = K = 64`, and `unit_lengths` is already
  variable, so no further arch shape change.
- `_grid_hw_for_compression_parts` still supplies per-part `(h, w)` for Stage-1's
  cross-RoPE (used per segment via broadcast); unchanged.
- **Chunked encoder forward.** Replace the single `vision_encoder(pixel_values, …)`
  in `encode_images` with a loop over ≤ 8-frame groups under `torch.no_grad()`
  (encoder frozen, no cross-frame attention — see "On-the-fly encoding"); concat.
  Bounds encoder memory to one segment regardless of video length.
- **Length check.** The current pre-compression `seq_len > model_max_length` guard in
  the dataset counts the *uncompressed* `T×HW` tokens. Switch it to the
  post-compression estimate (`U·M` + text) so long clips aren't needlessly skipped;
  still assert `U·M ≤ 320`.

### Carry-overs from current work

- Option A `match_encoder_scale` / Option B `compressor_distr_loss_weight` (the
  compressor-output norm-inflation fix) still matter: Stage-1 output should be on a
  sane scale before it hits Stage-2's `input_proj` / `input_norm`, not only before
  `mm_projector`. Already on in `work_dirs/compressor_pretrain_video_norm`
  (`match_encoder_scale: true`, `distr_loss_weight: 0.05`); no new change, just keep
  it.
- Reuse `eval_ablation/feature_distribution.py` collapse metrics on **Stage-2
  readout** output: `cos(real, shuffled)` ≈ 1, effective rank, PCA participation
  ratio — the same signals that caught the reconstruction-pretrain collapse.

---

## 8. Build order

§6.1–6.6 are resolved (§6). Status:

1. **TODO — Length-filter** `/root/datasets/videoxl/Finetuning/*` (caption-heavy:
   sharegpt4v, gpt4o_video, baaicaption, vcg, cinepile) into a manifest (≥ ~100
   frames — vcg / cinepile / sharegpt4video long tail) and write
   `anno_data/stage_2_training.json`, bucketed by frame count so fold depth is
   stable per grad-accum window. Reuse `eval_ablation/build_manifest.py`'s duration
   probe + video index.
2. **DONE (user-supplied) — `videollama3/model/segment_aggregator.py`.** Pure-PyTorch
   `Mamba2Mixer` (scalar-decay SSD) + `_ssd_chunk_scan` + `SegmentAggregator`
   (`forward` = one fold; `init_state`/`update`/`readout`/`reduce` = streaming).
   `AggregatorState` has no `log_decay` / `combine` yet — that is a Stage-2b item.
   `mamba-ssm` left commented in `requirements.txt` (optional throughput).
3. **DONE — `TwoStageCompressor` in `compressor.py`** (`.stage1` flat qbase + `.stage2`
   `SegmentAggregator`), `build_token_compressor` `"…+mamba"` suffix handling,
   `output_hw_for → (1, M)`, `freeze_stage1`, `load_stage1_pretrained` (bare `.pt`
   **or** HF checkpoint dir), `compress_windows` (per-unit loop). `stage2_*` fields
   on `Videollama3TokenCompressorConfig`.
4. **Stage-2a — DONE (code); first run pending 1 + 4d:**
   a. **DONE — dataset** `Stage2UnitDataset` (in the new script): `select_stage2_units`
      → `U = min(stage2_max_units, ceil(N / (frames_per_segment·segs_per_unit)))`
      contiguous `compression_parts`, unit widened to fit the cap, remainder → last
      unit; `build_stage2_ts_info` per part (degrades to `(0, [])` without
      timestamps). Per-unit `N_u` randomization: **not on yet** (follow-up flag).
      Segments are derived in the arch from `frames_per_segment`, not a dataset field.
   b. **DONE — `videollama3_arch.py`:** Stage-2 branch in
      `compress_visual_tokens_with_compressor` —
      `compressor.compress_windows(gathered, compression_cu_seqlens, grid_hws)` when
      exposed, scatter via the existing `replace_mask`. No
      `prepare_inputs_labels_for_multimodal` refactor, no chunked encoder forward yet.
   c. **DONE — `videollama3/train/stage2a_pretrain_compressor_fold.py`** (thin
      monkeypatch over `compressor_pretrain_with_videollama3.py`): forces
      `compressor_type` → `…+mamba`, warm-starts + freezes stage-1, trains only the
      fold, single CE, `zero1`. Wrapper `shell/pretrain_stage2a_fold.sh`.
   d. **TODO — collapse guardrail** on readout output; sanity that CE drops and
      `U·M ≤ 320`.
5. **Stage-2b** — in-model causal segmenter (criterion per §6.1 sub-decision, [1, 8]
   clamp) + streaming `combine_states` budget controller (holds `U ≤
   stage2_max_units`); the `prepare_inputs_labels_for_multimodal` refactor to consume
   model-returned units (+ resolve the tokenised-`Time:` question, §7);
   `create_optimizer` split `stage1` / `stage2` LR groups; qbase joint polish at 10×
   lower LR. Gate to Stage-3 = guardrail + caption plateau + budget ablation.
6. **Stage-3** — retained-segment hybrid path (3a: LLM frozen), then full LLM FT
   (3b): `zero2` / `zero3`, staggered LR, replay data, frozen-LLM eval as a
   measurement requirement.
7. **Stage-4** — unbounded fold (deep `segs_per_unit`) + TBPTT window, length
   curriculum, retained-set size policy, external long-video data + retrieval /
   needle eval.
