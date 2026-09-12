#!/usr/bin/env python3
"""Find the Phase-2 fold training bottleneck.

Two modes:

  --mode data   (no GPU / no torchrun)
      Build the REAL Phase-2 dataset + collator and measure how fast ONE rank's
      DataLoader can supply samples at a given worker/prefetch setting, plus the
      per-sample decode-time distribution (num_workers=0 pass) broken down by
      source dataset. Compare "samples/s supplied" against "samples/s the 8-GPU
      run consumes" (== 512 / observed_s_per_it).

  --mode train  (run under: torchrun --standalone --nproc_per_node=1 ...)
      Monkeypatch timers into encode_images / compress_visual_tokens_with_compressor
      / training_step and run ~25 real steps, then print where wall-time goes:
      data-wait vs compute, and inside compute: vision-encoder fwd vs compressor
      fwd vs LLM(fwd+bwd)+opt.

    python eval_ablation/profile_phase2.py --mode data --workers 16 --prefetch 8 --n 60
    torchrun --standalone --nproc_per_node=1 eval_ablation/profile_phase2.py --mode train --workers 8
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = "pretrained_models/videollama3_7b_local"
META = "anno_data/phase2_training.json"
MAX_FRAMES = 160
OBSERVED_S_PER_IT = 55.0            # steady-state s/it median (set from your run)
GLOBAL_BATCH = 512


# --------------------------------------------------------------------------- #
def _mk_dataset(workers_for_args: int):
    import transformers
    from videollama3.model.processor import Videollama3Processor, DEFAULT_CHAT_TEMPLATE
    from videollama3.train.compressor_pretrain_with_videollama3 import DataArguments
    from videollama3.train.phase2_pretrain_fold import (
        Phase2DataArguments, Phase2ModelArguments, Phase2FoldDataset,
    )
    from videollama3.train.data.global_compressor import make_global_compressor_data_module

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token
    proc = Videollama3Processor.from_pretrained(MODEL_DIR, trust_remote_code=False, fix_mistral_regex=True)
    proc.tokenizer = tok
    # base checkpoint's chat_template needs an `image_token` kwarg (jinja UndefinedError);
    # real training uses the hardcoded-`<image>` DEFAULT_CHAT_TEMPLATE. Match it.
    proc.chat_template = DEFAULT_CHAT_TEMPLATE
    proc.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    proc.image_processor.force_size = None            # dynamic HW, like training
    proc.image_processor.max_tokens = 16384
    proc.image_processor.min_tokens = 16

    da = Phase2DataArguments(
        data_path=[META], multi_dataset=True, fps=1, max_frames=MAX_FRAMES,
        video_merge_size=2, use_batch_flattening=True, stage2_max_units=5,
    )
    ma = Phase2ModelArguments(stage2_n_summary_tokens=64)
    dm = make_global_compressor_data_module(
        vlprocessor=proc, data_args=da, output_dir=None,
        dataset_cls=Phase2FoldDataset, model_args=ma,
    )
    return dm["train_dataset"], dm["data_collator"], proc


def mode_data(args):
    import torch
    from torch.utils.data import DataLoader

    ds, collator, _ = _mk_dataset(args.workers)
    n_total = len(ds)
    consume_rate = GLOBAL_BATCH / OBSERVED_S_PER_IT / 8.0     # samples/s/GPU the real run needs
    print(f"dataset: {n_total:,} samples across the 9-video meta")
    print(f"target supply rate: {consume_rate:.2f} samples/s/rank  "
          f"(= {GLOBAL_BATCH}/{OBSERVED_S_PER_IT:.0f}s / 8 ranks)\n")

    # --- (a) per-sample decode cost, single process, by source dataset ---
    print(f"[a] per-__getitem__ latency, num_workers=0, {args.n0} samples:")
    import random
    rng = random.Random(0)
    idxs = [rng.randrange(n_total) for _ in range(args.n0)]
    by_src = defaultdict(list)
    t_each = []
    for i in idxs:
        t0 = time.perf_counter()
        _ = ds[i]
        dt = time.perf_counter() - t0
        t_each.append(dt)
        src = "?"
        try:
            # ConcatDatasetWithLengths -> find which sub-dataset i landed in
            import bisect
            k = bisect.bisect_right(ds.cumulative_sizes, i)
            src = getattr(ds.datasets[k], "dataset_name", f"ds{k}")
        except Exception:
            pass
        by_src[src].append(dt)
    t_each.sort()
    print(f"    overall  n={len(t_each)}  min={t_each[0]*1e3:.0f}ms  "
          f"p50={t_each[len(t_each)//2]*1e3:.0f}ms  p90={t_each[int(len(t_each)*0.9)]*1e3:.0f}ms  "
          f"max={t_each[-1]*1e3:.0f}ms  mean={sum(t_each)/len(t_each)*1e3:.0f}ms")
    print(f"    -> 1 worker sustains ~{1.0/(sum(t_each)/len(t_each)):.2f} samples/s; "
          f"{args.workers} workers ~{args.workers/(sum(t_each)/len(t_each)):.1f} samples/s (ideal, no GIL/IO contention)")
    for src, v in sorted(by_src.items(), key=lambda kv: -sum(kv[1])/len(kv[1])):
        print(f"    {src:14s} n={len(v):3d}  mean={sum(v)/len(v)*1e3:5.0f}ms  max={max(v)*1e3:5.0f}ms")

    # --- (b) real DataLoader throughput at the configured worker/prefetch ---
    print(f"\n[b] DataLoader throughput  (batch_size=2, num_workers={args.workers}, "
          f"prefetch_factor={args.prefetch}, persistent):")
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=args.workers,
                    prefetch_factor=args.prefetch, persistent_workers=True,
                    collate_fn=collator)
    it = iter(dl)
    for _ in range(5):                                        # warm up workers
        next(it)
    gaps = []
    t_prev = time.perf_counter()
    for _ in range(args.n):
        next(it)
        now = time.perf_counter()
        gaps.append(now - t_prev)
        t_prev = now
    gaps.sort()
    bps = 1.0 / (sum(gaps) / len(gaps))
    print(f"    batches/s={bps:.2f}  ->  {bps*2:.2f} samples/s/rank   "
          f"(need {consume_rate:.2f})")
    print(f"    inter-batch gap  p50={gaps[len(gaps)//2]*1e3:.0f}ms  "
          f"p90={gaps[int(len(gaps)*0.9)]*1e3:.0f}ms  max={gaps[-1]*1e3:.0f}ms")
    verdict = ("DATA-BOUND: supply < consume — the GPUs starve waiting for frames."
               if bps * 2 < consume_rate * 1.3 else
               "data supply keeps up at this setting — bottleneck is elsewhere (compute).")
    print(f"\n    VERDICT: {verdict}")


# --------------------------------------------------------------------------- #
_T = defaultdict(list)


def mode_train(args):
    import torch
    import videollama3.train.compressor_pretrain_with_videollama3 as base
    import videollama3.train.phase2_pretrain_fold as s2
    from videollama3.model.videollama3_arch import Videollama3MetaForCausalLM
    from videollama3.train.videollama3_trainer import VideoLLaMA3Trainer

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # -- patch: split encode_images into vision-encoder vs compressor --
    _orig_comp = Videollama3MetaForCausalLM.compress_visual_tokens_with_compressor
    _orig_enc = Videollama3MetaForCausalLM.encode_images

    def comp_timed(self, *a, **k):
        _sync(); t0 = time.perf_counter()
        r = _orig_comp(self, *a, **k)
        _sync(); _T["compressor_fwd"].append(time.perf_counter() - t0)
        return r

    def enc_timed(self, *a, **k):
        _sync(); t0 = time.perf_counter()
        r = _orig_enc(self, *a, **k)
        _sync(); _T["encode_images_total"].append(time.perf_counter() - t0)
        return r

    Videollama3MetaForCausalLM.compress_visual_tokens_with_compressor = comp_timed
    Videollama3MetaForCausalLM.encode_images = enc_timed

    # -- patch: training_step -> data-wait gap vs compute duration --
    _orig_step = VideoLLaMA3Trainer.training_step
    state = {"prev_end": None}

    def step_timed(self, *a, **k):
        now = time.perf_counter()
        if state["prev_end"] is not None:
            _T["gap_between_steps"].append(now - state["prev_end"])
        _sync(); t0 = time.perf_counter()
        r = _orig_step(self, *a, **k)
        _sync()
        _T["step_compute"].append(time.perf_counter() - t0)
        state["prev_end"] = time.perf_counter()
        return r

    VideoLLaMA3Trainer.training_step = step_timed

    out = f"/tmp/claude-0/-root-OnlineLong/4fe5ab90-98e5-4ea3-96cb-5c41bce55e5c/scratchpad/prof_{os.getpid()}"
    sys.argv = [
        "prof", "--deepspeed", "shell/zero1.json",
        "--model_name_or_path", MODEL_DIR, "--vision_encoder", "DAMO-NLP-SG/SigLIP-NaViT",
        "--compressor_type", "transformer_decoder_flat", "--num_queries", "64",
        "--compressor_num_layers", "8", "--compressor_num_attention_heads", "8",
        "--match_encoder_scale", "True", "--compressor_distr_loss_weight", "0.05",
        "--stage1_pretrained", "pretrained_models/compressor_pretrain_video_norm",
        "--stage2_n_summary_tokens", "64", "--stage2_max_units", "5",
        "--multi_dataset", "True", "--data_path", META,
        "--fps", "1", "--max_frames", str(MAX_FRAMES), "--video_merge_size", "2",
        "--use_batch_flattening", "True", "--bf16", "True", "--tf32", "True",
        "--output_dir", out, "--overwrite_output_dir", "True",
        "--max_steps", str(args.steps), "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", str(args.grad_acc),
        "--compressor_lr", "5e-5", "--llm_lr", "0", "--vision_encoder_lr", "0",
        "--mm_projector_lr", "0", "--warmup_ratio", "0.0", "--lr_scheduler_type", "constant",
        "--logging_steps", "1", "--save_strategy", "no", "--gradient_checkpointing", "True",
        "--model_max_length", "32768", "--dataloader_num_workers", str(args.workers),
        "--dataloader_persistent_workers", "True", "--dataloader_prefetch_factor", "8",
        "--report_to", "none",
    ]
    try:
        base.train(
            attn_implementation="flash_attention_2",
            model_args_cls=s2.Phase2ModelArguments, data_args_cls=s2.Phase2DataArguments,
            dataset_cls=s2.Phase2FoldDataset,
            build_token_compressor_config=s2._build_stage2_token_compressor_config,
            configure_image_processor=s2._configure_stage2_image_processor,
            on_compressor_built=s2._warmstart_and_freeze_stage1,
        )
    except SystemExit:
        pass

    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return
    warm = 5
    def rep(key, drop=warm):
        v = _T[key][drop:]
        if not v:
            return f"  {key:22s}  (no samples)"
        return (f"  {key:22s}  n={len(v):2d}  mean={st.mean(v)*1e3:7.0f}ms  "
                f"median={st.median(v)*1e3:7.0f}ms  max={max(v)*1e3:7.0f}ms")
    print("\n" + "=" * 66 + "\n  STAGE-2 STEP PROFILE  (per micro-step, first "
          f"{warm} dropped as warmup)\n" + "=" * 66)
    for k in ("gap_between_steps", "step_compute", "encode_images_total",
              "compressor_fwd"):
        print(rep(k))
    g = _T["gap_between_steps"][warm:]
    c = _T["step_compute"][warm:]
    e = _T["encode_images_total"][warm:]
    cm = _T["compressor_fwd"][warm:]
    if g and c:
        tot = st.mean(g) + st.mean(c)
        print(f"\n  wall/microstep ~= {tot*1e3:.0f}ms   "
              f"data-wait {st.mean(g)/tot*100:.0f}%   compute {st.mean(c)/tot*100:.0f}%")
    if e and c:
        venc = st.mean(e) - (st.mean(cm) if cm else 0)
        rest = st.mean(c) - st.mean(e)
        print(f"  inside compute: vision-encoder ~{venc*1e3:.0f}ms   "
              f"compressor ~{st.mean(cm)*1e3 if cm else 0:.0f}ms   "
              f"LLM fwd+bwd+opt ~{rest*1e3:.0f}ms")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["data", "train"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--n", type=int, default=60, help="data mode: DataLoader batches to time")
    ap.add_argument("--n0", type=int, default=40, help="data mode: single-process samples to time")
    ap.add_argument("--steps", type=int, default=25, help="train mode: optimizer steps")
    ap.add_argument("--grad_acc", type=int, default=4, help="train mode: grad-accum")
    args = ap.parse_args()
    (mode_data if args.mode == "data" else mode_train)(args)


if __name__ == "__main__":
    main()
