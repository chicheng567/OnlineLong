#!/usr/bin/env python3
"""
Measure and compare the COMPRESSED-FEATURE distribution of two token compressors
(the prune-vs-noprune ablation from shell/pretrain_compressor_prune_ablation.sh).

For every video it runs, per model:  frozen SigLIP-NaViT  ->  token compressor  ->
mm_projector, and collects
  * the compressor output tokens          (pre-projector, dim = mm_hidden_size)
  * their projection into the LLM space   (post-projector, dim = hidden_size)
  * a random sample of the UNCOMPRESSED encoder tokens + their projection, as the
    reference manifold the projector was trained on.

Frames are cut into consecutive `--window_size` (default 8) groups -- the training
grouping -- and each group is compressed independently. The per-window (h, w) grid
actually produced by the encoder drives the compressor (dynamic HW); it is logged.

Reported per model: L2-norm distribution, per-dim std / dead dims, excess kurtosis,
PCA spectrum (participation ratio, effective rank, dims for 90/99 % variance),
mean pairwise cosine (token collapse), intra-window collapse, and the
compressed/raw norm ratio (has the compressor drifted off the projector's input
scale?).  Reported across models on the SAME windows: row cosine, relative L2,
linear CKA, centroid shift, norm ratio.

Usage
-----
PYTHONPATH=. python eval_ablation/feature_distribution.py \
    --models prune=work_dirs/compressor_prune_ablation/prune0.30 \
             noprune=work_dirs/compressor_prune_ablation/noprune \
    --manifest eval_ablation/manifest.json --num_videos 12 \
    --out work_dirs/ablation_eval/feature_distribution --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
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
    describe_grid, extract_features, free_model, load_model, load_processor,
    prepare_video_sample,
)

try:
    from scipy.stats import kurtosis as _scipy_kurtosis
    def _excess_kurtosis(x):  # (n, d) -> (d,)
        return _scipy_kurtosis(x, axis=0, fisher=True, bias=False)
except Exception:  # pragma: no cover
    def _excess_kurtosis(x):
        m = x.mean(0); s = x.std(0) + 1e-12
        return ((x - m) ** 4).mean(0) / s ** 4 - 3.0


# --------------------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------------------
def _pca_spectrum(x: np.ndarray, max_rows: int = 8192) -> Dict:
    if x.shape[0] > max_rows:
        x = x[np.random.choice(x.shape[0], max_rows, replace=False)]
    xc = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(xc, compute_uv=False)
    lam = (s ** 2) / max(x.shape[0] - 1, 1)
    tot = float(lam.sum()) + 1e-12
    p = lam / tot
    nz = p[p > 0]
    eff_rank = float(np.exp(-(nz * np.log(nz)).sum()))
    part_ratio = float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))
    csum = np.cumsum(p)
    return {
        "participation_ratio": part_ratio,
        "effective_rank": eff_rank,
        "evr_top1": float(p[0]),
        "evr_top10": float(p[:10].sum()),
        "n_dims_90pct": int(np.searchsorted(csum, 0.90) + 1),
        "n_dims_99pct": int(np.searchsorted(csum, 0.99) + 1),
        "spectrum_head": [float(v) for v in p[:24]],
    }


def _mean_pairwise_cos(x: np.ndarray, max_rows: int = 2048) -> float:
    if x.shape[0] > max_rows:
        x = x[np.random.choice(x.shape[0], max_rows, replace=False)]
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    g = xn @ xn.T
    n = g.shape[0]
    off = (g.sum() - np.trace(g)) / (n * (n - 1))
    return float(off)


def describe(x: np.ndarray, tag: str) -> Dict:
    norms = np.linalg.norm(x, axis=1)
    gstd = float(x.std()) + 1e-12
    per_dim_std = x.std(0)
    kurt = _excess_kurtosis(x)
    d = {
        "n_rows": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "l2_norm": {
            "mean": float(norms.mean()), "std": float(norms.std()),
            "p5": float(np.percentile(norms, 5)),
            "p50": float(np.percentile(norms, 50)),
            "p95": float(np.percentile(norms, 95)),
        },
        "elem_mean": float(x.mean()),
        "elem_std": float(x.std()),
        "per_dim_std_mean": float(per_dim_std.mean()),
        "dead_dim_frac": float((per_dim_std < 1e-3 * gstd).mean()),
        "excess_kurtosis_mean": float(np.nanmean(kurt)),
        "excess_kurtosis_p95": float(np.nanpercentile(kurt, 95)),
        "mean_pairwise_cos": _mean_pairwise_cos(x),
        "pca": _pca_spectrum(x),
    }
    return d


def intra_window_collapse(comp: np.ndarray, n_per_window: np.ndarray) -> Dict:
    """comp rows are ordered window-by-window. For each window: mean pairwise cosine
    of its tokens and its participation ratio. High cosine / low PR == the window's
    output tokens have collapsed together."""
    cos, pr = [], []
    off = 0
    for k in n_per_window:
        k = int(k)
        w = comp[off:off + k]
        off += k
        if w.shape[0] < 3:
            continue
        cos.append(_mean_pairwise_cos(w, max_rows=k))
        wc = w - w.mean(0, keepdims=True)
        s = np.linalg.svd(wc, compute_uv=False)
        lam = s ** 2
        pr.append(float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12)))
    return {
        "n_windows": len(cos),
        "mean_within_window_cos": float(np.mean(cos)) if cos else None,
        "mean_within_window_participation_ratio": float(np.mean(pr)) if pr else None,
    }


def linear_cka(x: np.ndarray, y: np.ndarray, max_rows: int = 8192) -> float:
    if x.shape[0] > max_rows:
        idx = np.random.choice(x.shape[0], max_rows, replace=False)
        x, y = x[idx], y[idx]
    x = x - x.mean(0, keepdims=True)
    y = y - y.mean(0, keepdims=True)
    xty = np.linalg.norm(x.T @ y, "fro") ** 2
    xtx = np.linalg.norm(x.T @ x, "fro") ** 2
    yty = np.linalg.norm(y.T @ y, "fro") ** 2
    return float(xty / (np.sqrt(xtx * yty) + 1e-12))


def cross_model(a: np.ndarray, b: np.ndarray) -> Dict:
    """a, b are row-aligned (same videos, same windows, same token order)."""
    assert a.shape == b.shape, (a.shape, b.shape)
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    row_cos = (an * bn).sum(1)
    rel_l2 = np.linalg.norm(a - b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)
    norm_ratio = np.linalg.norm(b, axis=1) / (np.linalg.norm(a, axis=1) + 1e-12)
    ca, cb = a.mean(0), b.mean(0)
    return {
        "row_cosine_mean": float(row_cos.mean()),
        "row_cosine_p5": float(np.percentile(row_cos, 5)),
        "row_cosine_p50": float(np.percentile(row_cos, 50)),
        "row_rel_l2_mean": float(rel_l2.mean()),
        "row_norm_ratio_b_over_a_mean": float(norm_ratio.mean()),
        "linear_cka": linear_cka(a, b),
        "centroid_shift_rel": float(np.linalg.norm(ca - cb) / (np.linalg.norm(ca) + 1e-12)),
    }


def _spark(vals: List[float]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(blocks[min(7, int((v - lo) / rng * 7))] for v in vals)


# --------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------
def parse_models(pairs: List[str]) -> List[tuple]:
    out = []
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--models entries must be tag=path, got {p!r}")
        tag, path = p.split("=", 1)
        out.append((tag, path))
    return out


def resolve_videos(args) -> List[str]:
    vids: List[str] = []
    if args.manifest:
        data = json.load(open(args.manifest))
        vids += [e["video"] for e in data]
    if args.video_dir:
        vids += sorted(glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True))
    vids += list(args.videos or [])
    vids = [v for v in vids if os.path.exists(v)]
    if args.num_videos:
        vids = vids[: args.num_videos]
    if not vids:
        raise SystemExit("No videos resolved (use --manifest / --video_dir / --videos).")
    return vids


def run_one_model(model_path: str, videos: List[str], args) -> Dict:
    # Re-seed per model so the frozen encoder's random raw-token sample is drawn
    # identically -> `raw` rows are aligned across models (row_cosine ~ 1.0 is the
    # harness sanity check), and any `comp` difference is purely the compressor.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    proc = load_processor(model_path, force_image_size=args.force_image_size)
    model = load_model(model_path, device=args.device)
    per_video, buf = [], {"comp": [], "comp_proj": [], "raw": [], "raw_proj": [], "npw": []}
    for vp in videos:
        try:
            sample = prepare_video_sample(
                proc, vp, fps=args.fps, max_frames=args.max_frames,
                window_size=args.window_size, device=args.device,
                out_hw_fn=model.get_token_compressor().output_hw_for,
            )
            feats = extract_features(model, sample)
            per_video.append({"video": vp, "grid": describe_grid(sample["meta"]),
                              "meta": sample["meta"]})
            for k, dst in (("comp", "comp"), ("comp_proj", "comp_proj"),
                           ("raw", "raw"), ("raw_proj", "raw_proj")):
                buf[dst].append(feats[k])
            buf["npw"].append(feats["n_per_window"])
            print(f"  [{os.path.basename(vp)}] {per_video[-1]['grid']}  "
                  f"comp{feats['comp'].shape} proj{feats['comp_proj'].shape}")
        except Exception:
            print(f"  [SKIP] {vp}\n{traceback.format_exc()}")
    free_model(model)

    comp = np.concatenate(buf["comp"], 0)
    comp_proj = np.concatenate(buf["comp_proj"], 0)
    raw = np.concatenate(buf["raw"], 0)
    raw_proj = np.concatenate(buf["raw_proj"], 0)
    npw = np.concatenate(buf["npw"], 0)

    result = {
        "model_path": model_path,
        "n_videos_ok": len(per_video),
        "compressor_type": "transformer_decoder_flat",
        "compressed": describe(comp, "compressed"),
        "compressed_projected": describe(comp_proj, "compressed_projected"),
        "raw_encoder": describe(raw, "raw_encoder"),
        "raw_encoder_projected": describe(raw_proj, "raw_encoder_projected"),
        "intra_window_collapse": intra_window_collapse(comp, npw),
        "norm_ratio_comp_over_raw_preproj": float(
            np.median(np.linalg.norm(comp, axis=1)) /
            (np.median(np.linalg.norm(raw, axis=1)) + 1e-12)),
        "norm_ratio_comp_over_raw_postproj": float(
            np.median(np.linalg.norm(comp_proj, axis=1)) /
            (np.median(np.linalg.norm(raw_proj, axis=1)) + 1e-12)),
        "per_video": [{"video": p["video"], "grid": p["grid"]} for p in per_video],
    }
    # keep the row-aligned matrices around for the cross-model pass
    return result, {"comp": comp, "comp_proj": comp_proj, "raw": raw}


def to_markdown(report: Dict) -> str:
    tags = list(report["models"].keys())
    L = []
    L.append("# Compressor feature-distribution ablation\n")
    model_list = ", ".join("`%s` = %s" % (t, report["models"][t]["model_path"]) for t in tags)
    L.append("- models: " + model_list)
    L.append(f"- videos: {report['config']['n_videos']} | "
             f"frames/clip: {report['config']['max_frames']} | "
             f"window: {report['config']['window_size']} | "
             f"force_image_size: {report['config']['force_image_size']}")
    L.append(f"- dynamic HW: per-window (h,w) taken from the encoder grid and passed "
             f"to the compressor for every window (see per-video grids in the JSON)\n")

    def row(label, fn):
        return "| " + label + " | " + " | ".join(f"{fn(report['models'][t]):.4g}" for t in tags) + " |"

    for space, key in [("compressor output (pre-projector)", "compressed"),
                       ("projected into LLM space", "compressed_projected"),
                       ("uncompressed encoder tokens", "raw_encoder")]:
        L.append(f"\n## {space}\n")
        L.append("| metric | " + " | ".join(f"`{t}`" for t in tags) + " |")
        L.append("|---|" + "---|" * len(tags))
        L.append(row("L2 norm (median)", lambda m: m[key]["l2_norm"]["p50"]))
        L.append(row("L2 norm (p5..p95 spread)",
                     lambda m: m[key]["l2_norm"]["p95"] - m[key]["l2_norm"]["p5"]))
        L.append(row("per-dim std (mean)", lambda m: m[key]["per_dim_std_mean"]))
        L.append(row("dead-dim fraction", lambda m: m[key]["dead_dim_frac"]))
        L.append(row("excess kurtosis (mean)", lambda m: m[key]["excess_kurtosis_mean"]))
        L.append(row("mean pairwise cosine", lambda m: m[key]["mean_pairwise_cos"]))
        L.append(row("participation ratio", lambda m: m[key]["pca"]["participation_ratio"]))
        L.append(row("effective rank", lambda m: m[key]["pca"]["effective_rank"]))
        L.append(row("dims for 90% var", lambda m: m[key]["pca"]["n_dims_90pct"]))
        L.append(row("dims for 99% var", lambda m: m[key]["pca"]["n_dims_99pct"]))
        for t in tags:
            L.append(f"  - `{t}` spectrum head: `{_spark(report['models'][t][key]['pca']['spectrum_head'])}`")

    L.append("\n## collapse / drift\n")
    L.append("| metric | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    L.append(row("within-window cosine (mean)",
                 lambda m: m["intra_window_collapse"]["mean_within_window_cos"]))
    L.append(row("within-window participation ratio",
                 lambda m: m["intra_window_collapse"]["mean_within_window_participation_ratio"]))
    L.append(row("norm ratio compressed/raw (pre-proj)",
                 lambda m: m["norm_ratio_comp_over_raw_preproj"]))
    L.append(row("norm ratio compressed/raw (post-proj)",
                 lambda m: m["norm_ratio_comp_over_raw_postproj"]))

    if "cross_model" in report:
        L.append("\n## cross-model, same windows\n")
        for space, cm in report["cross_model"].items():
            L.append(f"**{space}**\n")
            L.append("| metric | value |")
            L.append("|---|---|")
            for k, v in cm.items():
                L.append(f"| {k} | {v:.4g} |")
            L.append("")
    L.append("\n_Full numbers, PCA spectra and per-video grids: `feature_distribution.json`._\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="tag=path pairs, e.g. prune=work_dirs/.../prune0.30")
    ap.add_argument("--manifest", default=None, help="caption manifest json (uses its 'video' field)")
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--num_videos", type=int, default=12)
    ap.add_argument("--out", default="work_dirs/ablation_eval/feature_distribution")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    ap.add_argument("--force_image_size", type=int, default=DEFAULT_FORCE_IMAGE_SIZE,
                    help="0 = native aspect-ratio resolution (off-distribution, varies (h,w))")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    models = parse_models(args.models)
    videos = resolve_videos(args)
    print(f"[feature_distribution] {len(models)} models x {len(videos)} videos")

    report = {"config": vars(args) | {"n_videos": len(videos)}, "models": {}}
    mats = {}
    for tag, path in models:
        print(f"\n=== {tag} :: {path} ===")
        res, mat = run_one_model(path, videos, args)
        report["models"][tag] = res
        mats[tag] = mat

    if len(models) == 2:
        (ta, _), (tb, _) = models
        A, B = mats[ta], mats[tb]
        report["cross_model"] = {}
        for space in ("comp", "comp_proj", "raw"):
            if A[space].shape == B[space].shape:
                report["cross_model"][space] = cross_model(A[space], B[space])
        report["cross_model_note"] = (
            f"`comp`/`comp_proj` rows are aligned window-for-window between {ta} and "
            f"{tb}; `raw` rows are a random encoder-token sample (aligned only if the "
            f"same frames were drawn)."
        )

    with open(os.path.join(args.out, "feature_distribution.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "feature_distribution.md"), "w") as f:
        f.write(to_markdown(report))
    print(f"\n[feature_distribution] wrote {args.out}/feature_distribution.{{json,md}}")


if __name__ == "__main__":
    main()
