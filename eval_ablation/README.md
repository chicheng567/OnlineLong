# eval_ablation — prune-vs-noprune compressor evaluation

Evaluation suite for the two token compressors trained by
`shell/pretrain_compressor_prune_ablation.sh`:

| tag       | output dir                                          | `--token_prune_ratio` |
|-----------|-----------------------------------------------------|-----------------------|
| `prune`   | `work_dirs/compressor_prune_ablation/prune0.30`     | 0.30                  |
| `noprune` | `work_dirs/compressor_prune_ablation/noprune`       | 0                     |

Both are `transformer_decoder_flat`, `num_queries=64`, 8 layers, trained
whole-video (`fixed_frames=8`, one 8-frame window per clip) on cached InternVid
features with the detail-caption prompt. `token_prune_ratio` only acts in
`.train()` mode, so at eval the two models differ **only in learned weights** —
this suite quantifies that difference two ways.

## What it does

1. **`feature_distribution.py`** — for each model, run `encoder → compressor →
   projector` on the manifest videos (frames cut into consecutive 8-frame groups,
   each group compressed on its own) and compare:
   - per model: L2-norm distribution, per-dim std / dead dims, excess kurtosis,
     PCA spectrum (participation ratio, effective rank, dims for 90/99 % var),
     mean pairwise cosine (token collapse), intra-window collapse, and the
     **compressed / raw-encoder norm ratio** (has the compressor drifted off the
     scale the frozen projector expects?);
   - across models, on the *same* windows: row cosine, relative L2, linear CKA,
     centroid shift, norm ratio.

2. **`caption_eval.py`** — each model captions the manifest videos with the
   training prompt, `--max_frames 64` (relaxed), 8-frame groups, **non-greedy**
   decoding (`do_sample=True`, `repetition_penalty=1.1`). Scores every caption
   against the dataset's gold caption (ROUGE-1/2/L, BLEU-4, unigram recall,
   length ratio) plus degeneration signals (distinct-2, duplicate-4-gram rate,
   max 4-gram repeat), and measures `prune` vs `noprune` agreement.

### Dynamic HW

Compression is always driven through the model's own
`_grid_hw_for_compression_parts` → `compressor.output_hw_for(h, w)` path: the
per-window `(h, w)` grid the encoder actually produced sizes the cross-attention
RoPE and the compressed placeholder block. Nothing hardcodes 16×16 / 256. Each
run logs the per-window input `(h,w)` and output grid (`describe_grid`,
`window_grid_hw` / `window_out_hw` in the JSON).

With the default `--force_image_size 448` (training geometry) every window lands
on `(16,16) → (1,64)`; the value is still *computed* from the encoder grid, not
assumed. Pass `--force_image_size 0` for native aspect-ratio resolution, where the
grids genuinely vary video-to-video (off-distribution — the compressors only ever
saw 16×16 — so use it only to stress the dynamic-HW path).

## Run

```bash
# one shot: waits for BOTH training runs to write their final model, then evaluates
bash eval_ablation/run_all.sh
#   env knobs: WAIT=0  DEVICE=cuda:7  MAX_FRAMES=64  WINDOW_SIZE=8
#              FORCE_IMAGE_SIZE=448  REP_PENALTY=1.1  OUT_DIR=work_dirs/ablation_eval
#              NOPRUNE_DIR=work_dirs/compressor_prune_ablation/noprune/checkpoint-9545
```

Or piecemeal (repo root, `PYTHONPATH=.`):

```bash
# (re)build the video manifest from ../datasets/videoxl  (12 clips, 45–150 s, with gold captions)
python eval_ablation/build_manifest.py --num 12

python eval_ablation/feature_distribution.py \
    --models prune=work_dirs/compressor_prune_ablation/prune0.30 \
             noprune=work_dirs/compressor_prune_ablation/noprune \
    --manifest eval_ablation/manifest.json --num_videos 12 \
    --out work_dirs/ablation_eval/feature_distribution --device cuda:0

python eval_ablation/caption_eval.py \
    --models prune=work_dirs/compressor_prune_ablation/prune0.30 \
             noprune=work_dirs/compressor_prune_ablation/noprune \
    --manifest eval_ablation/manifest.json \
    --out work_dirs/ablation_eval/caption_eval --device cuda:0
```

`--models` takes any number of `tag=path` pairs (cross-model stats need exactly 2).
`path` may be a final run dir or a `checkpoint-*` dir.

## Outputs (`work_dirs/ablation_eval/`)

```
feature_distribution/feature_distribution.json   full numbers + PCA spectra + per-video grids
feature_distribution/feature_distribution.md     comparison tables
caption_eval/captions.jsonl                      one row per (video, model): caption + metrics + meta
caption_eval/captions_side_by_side.md            reference vs each model, per video
caption_eval/summary.json                        per-model mean metrics + prune-vs-noprune agreement
caption_eval/caption_eval.md                     summary table
REPORT.md                                        feature + caption markdown, stitched (run_all.sh)
```

## Files

| file                     | role |
|--------------------------|------|
| `common.py`              | model / processor load, video → training-style model inputs, dynamic-HW grids, raw feature extraction |
| `metrics.py`             | dependency-free ROUGE / BLEU / distinct-n / repetition |
| `build_manifest.py`      | pick clips from `../datasets/videoxl` (+ gold captions) → `manifest.json` |
| `feature_distribution.py`| feature-distribution comparison |
| `caption_eval.py`        | caption-ability comparison |
| `run_all.sh`             | wait-for-training → both evals → `REPORT.md` |

## Notes / caveats

- The gold captions come from ShareGPT4V / VideoChatGPT, written to their own
  (varied) prompts; generation here uses the compressor's training prompt. Lexical
  metrics are for **relative** `prune` vs `noprune` comparison, not absolute
  quality.
- `caption_eval` seeds per video (`--seed + i`) so reruns reproduce; it is still
  sampled decoding, so small run-to-run drift is expected.
- Needs a free GPU with ~20 GB (7B model, bf16). `run_all.sh` waits for training
  to finish first; if you run earlier, point `--device` at an idle GPU.
- `_video_index.json` is a cached basename→path map of `../datasets/videoxl`;
  delete it to rescan.
