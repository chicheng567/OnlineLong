#!/usr/bin/env python3
"""
Caption-ability probe for the prune-vs-noprune compressor ablation.

Runs each model on the manifest videos with the TRAINING grouping -- frames cut
into consecutive `--window_size` (default 8) groups, every group compressed on its
own through the token compressor, the compressed tokens (not the raw frames) fed to
the LLM -- and the training caption prompt.  Frame budget is relaxed to
`--max_frames` (default 64), so a typical clip becomes 8 windows of 8 frames.

Decoding is NOT greedy: `do_sample=True` with `--repetition_penalty` (default 1.1).
The per-window (h, w) grid produced by the encoder drives the compressor
(dynamic HW) and is logged per video.

Outputs (under --out):
  captions.jsonl              one row per (video, model): caption + metrics + meta
  captions_side_by_side.md    reference vs each model's caption, per video
  summary.json                per-model mean metrics, and model-A vs model-B agreement
  caption_eval.md             the summary as a table

Usage
-----
PYTHONPATH=. python eval_ablation/caption_eval.py \
    --models prune=work_dirs/compressor_prune_ablation/prune0.30 \
             noprune=work_dirs/compressor_prune_ablation/noprune \
    --manifest eval_ablation/manifest.json \
    --out work_dirs/ablation_eval/caption_eval --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_ablation.common import (  # noqa: E402
    DEFAULT_FORCE_IMAGE_SIZE, DEFAULT_MAX_FRAMES, DEFAULT_WINDOW_SIZE,
    TRAIN_CAPTION_PROMPT, describe_grid, free_model, load_model, load_processor,
    prepare_video_sample,
)
from eval_ablation.metrics import caption_metrics, mean_dict, pair_metrics  # noqa: E402


def parse_models(pairs: List[str]) -> List[tuple]:
    out = []
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--models entries must be tag=path, got {p!r}")
        tag, path = p.split("=", 1)
        out.append((tag, path))
    return out


def load_manifest(path: str) -> List[Dict]:
    data = json.load(open(path))
    items = []
    for e in data:
        vp = e["video"]
        if not os.path.exists(vp):
            print(f"[manifest] missing, skipping: {vp}")
            continue
        items.append({
            "video": vp,
            "prompt": e.get("prompt") or TRAIN_CAPTION_PROMPT,
            "reference": e.get("reference", ""),
            "source": e.get("source", ""),
            "duration": e.get("duration"),
        })
    if not items:
        raise SystemExit(f"No usable entries in {path}")
    return items


@torch.no_grad()
def generate_caption(model, proc, sample: Dict, gen_kwargs: Dict) -> str:
    out = model.generate(
        input_ids=sample["input_ids"],
        pixel_values=sample["pixel_values"],
        grid_sizes=sample["grid_sizes"],
        merge_sizes=sample["merge_sizes"],
        modals=sample["modals"],
        compression_parts=sample["compression_parts"],
        compression_ts_info=sample["compression_ts_info"],
        **gen_kwargs,
    )
    return proc.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def run_one_model(tag: str, model_path: str, items: List[Dict], args,
                  gen_kwargs: Dict) -> List[Dict]:
    proc = load_processor(model_path, force_image_size=args.force_image_size)
    model = load_model(model_path, device=args.device)
    rows = []
    for i, it in enumerate(items):
        rec = {"video": it["video"], "model": tag, "source": it["source"]}
        try:
            # deterministic per (video) so a rerun reproduces, but still sampled
            torch.manual_seed(args.seed + i)
            np.random.seed(args.seed + i)
            sample = prepare_video_sample(
                proc, it["video"], prompt=it["prompt"], fps=args.fps,
                max_frames=args.max_frames, window_size=args.window_size,
                device=args.device,
                out_hw_fn=model.get_token_compressor().output_hw_for,
            )
            caption = generate_caption(model, proc, sample, gen_kwargs)
            rec["caption"] = caption
            rec["grid"] = describe_grid(sample["meta"])
            rec["meta"] = {
                k: sample["meta"][k] for k in
                ("num_frames", "n_windows", "window_frame_spans", "window_grid_hw",
                 "compressed_tokens_total", "total_vision_tokens")
            }
            if it["reference"]:
                rec["metrics_vs_reference"] = caption_metrics(caption, it["reference"])
            print(f"  [{tag}] {os.path.basename(it['video'])}  {rec['grid']}  "
                  f"({len(caption.split())} words)")
        except Exception:
            rec["error"] = traceback.format_exc()
            print(f"  [{tag}] [SKIP] {it['video']}\n{rec['error']}")
        rows.append(rec)
    free_model(model)
    return rows


def build_summary(all_rows: Dict[str, List[Dict]], items: List[Dict]) -> Dict:
    tags = list(all_rows.keys())
    summary: Dict = {"per_model": {}, "cross_model": {}}
    for t in tags:
        mts = [r["metrics_vs_reference"] for r in all_rows[t]
               if "metrics_vs_reference" in r]
        summary["per_model"][t] = {
            "n_ok": sum(1 for r in all_rows[t] if "caption" in r),
            "n_with_reference": len(mts),
            "mean_metrics_vs_reference": mean_dict(mts),
        }
    if len(tags) == 2:
        a, b = tags
        by_vid = {}
        for r in all_rows[a]:
            by_vid.setdefault(r["video"], {})[a] = r
        for r in all_rows[b]:
            by_vid.setdefault(r["video"], {})[b] = r
        pairs = []
        for v, d in by_vid.items():
            if "caption" in d.get(a, {}) and "caption" in d.get(b, {}):
                pairs.append(pair_metrics(d[a]["caption"], d[b]["caption"]))
        summary["cross_model"] = {
            "pair": f"{a}_vs_{b}",
            "n_pairs": len(pairs),
            "mean_agreement": mean_dict(pairs),
        }
    return summary


def to_markdown(summary: Dict, gen_kwargs: Dict, cfg: Dict) -> str:
    tags = list(summary["per_model"].keys())
    L = ["# Compressor caption-ability ablation\n"]
    L.append(f"- frames/clip: {cfg['max_frames']} @ {cfg['fps']} fps | "
             f"window: {cfg['window_size']} frames/group | "
             f"force_image_size: {cfg['force_image_size']}")
    L.append(f"- decoding: {gen_kwargs} (not greedy)")
    L.append(f"- prompt ({'TRAINING prompt' if cfg.get('prompt_is_training_prompt') else 'NEW / non-training prompt'}): "
             f"`{cfg.get('prompt', '')}`")
    L.append("- dynamic HW: per-window (h,w) from the encoder grid drives the compressor "
             "(see per-video `window_grid_hw` in captions.jsonl)\n")
    metric_keys = ["rougeL_f", "rouge1_f", "rouge2_f", "bleu4", "unigram_recall",
                   "length_ratio", "cand_words", "distinct_2", "dup_4gram_rate",
                   "repeat_4gram_max"]
    L.append("## mean metrics vs reference caption\n")
    L.append("| metric | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    for k in metric_keys:
        cells = []
        for t in tags:
            v = summary["per_model"][t]["mean_metrics_vs_reference"].get(k)
            cells.append("n/a" if v is None else f"{v:.4g}")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L.append(f"\n_n videos scored per model: "
             + ", ".join(f"`{t}`={summary['per_model'][t]['n_with_reference']}" for t in tags)
             + "_")
    if summary.get("cross_model"):
        cm = summary["cross_model"]
        L.append(f"\n## {cm['pair']} agreement (n={cm['n_pairs']})\n")
        L.append("| metric | value |")
        L.append("|---|---|")
        for k, v in cm["mean_agreement"].items():
            L.append(f"| {k} | {v:.4g} |")
    L.append("\n_Side-by-side captions: `captions_side_by_side.md`. Raw rows: `captions.jsonl`._\n")
    return "\n".join(L)


def side_by_side(all_rows: Dict[str, List[Dict]], items: List[Dict]) -> str:
    tags = list(all_rows.keys())
    idx = {t: {r["video"]: r for r in all_rows[t]} for t in tags}
    L = ["# Captions side by side\n"]
    for it in items:
        v = it["video"]
        L.append(f"## {os.path.basename(v)}")
        meta = next((idx[t][v].get("grid") for t in tags
                     if v in idx[t] and idx[t][v].get("grid")), "")
        if meta:
            L.append(f"`{meta}`  \nsource: {it['source']}  duration: {it.get('duration')}\n")
        if it["reference"]:
            L.append(f"**reference**\n\n> {it['reference']}\n")
        for t in tags:
            r = idx[t].get(v, {})
            if "caption" in r:
                mv = r.get("metrics_vs_reference", {})
                tail = (f"  _(rougeL={mv.get('rougeL_f', 0):.3f}, "
                        f"bleu4={mv.get('bleu4', 0):.3f}, "
                        f"dup4={mv.get('dup_4gram_rate', 0):.3f})_" if mv else "")
                L.append(f"**{t}**{tail}\n\n> {r['caption']}\n")
            else:
                L.append(f"**{t}**: _failed_\n")
        L.append("\n---\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="tag=path pairs")
    ap.add_argument("--manifest", default="eval_ablation/manifest.json")
    ap.add_argument("--prompt", default=None,
                    help="override the caption instruction for every video "
                         "(default: manifest/training prompt)")
    ap.add_argument("--num_videos", type=int, default=0, help="0 = all in manifest")
    ap.add_argument("--out", default="work_dirs/ablation_eval/caption_eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    ap.add_argument("--force_image_size", type=int, default=DEFAULT_FORCE_IMAGE_SIZE)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    models = parse_models(args.models)
    items = load_manifest(args.manifest)
    if args.prompt:
        for it in items:
            it["prompt"] = args.prompt
    if args.num_videos:
        items = items[: args.num_videos]
    print(f"[caption_eval] prompt = {items[0]['prompt']!r}")

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "num_beams": 1,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.top_k and args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k

    print(f"[caption_eval] {len(models)} models x {len(items)} videos | {gen_kwargs}")

    all_rows: Dict[str, List[Dict]] = {}
    jsonl_path = os.path.join(args.out, "captions.jsonl")
    open(jsonl_path, "w").close()
    for tag, path in models:
        print(f"\n=== {tag} :: {path} ===")
        rows = run_one_model(tag, path, items, args, gen_kwargs)
        all_rows[tag] = rows
        with open(jsonl_path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = build_summary(all_rows, items)
    summary["generation_kwargs"] = gen_kwargs
    summary["config"] = {k: getattr(args, k) for k in
                         ("fps", "max_frames", "window_size", "force_image_size",
                          "manifest", "seed")}
    summary["config"]["prompt"] = items[0]["prompt"]
    summary["config"]["prompt_is_training_prompt"] = (items[0]["prompt"] == TRAIN_CAPTION_PROMPT)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "caption_eval.md"), "w") as f:
        f.write(to_markdown(summary, gen_kwargs, summary["config"]))
    with open(os.path.join(args.out, "captions_side_by_side.md"), "w", encoding="utf-8") as f:
        f.write(side_by_side(all_rows, items))
    print(f"\n[caption_eval] wrote {args.out}/{{captions.jsonl,summary.json,"
          f"caption_eval.md,captions_side_by_side.md}}")


if __name__ == "__main__":
    main()
