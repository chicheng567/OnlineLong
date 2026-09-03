#!/usr/bin/env python3
"""Focused effective-rank / spectrum analysis of the COMPRESSED tokens for the
image_mix vs noprune compressors.

For every manifest video: frozen SigLIP-NaViT -> token compressor -> mm_projector,
same 8-frame-window training grouping and dynamic HW as eval_ablation/common.
Then, per model, on three token sets --
  comp       compressor output, pre-projector      (dim = mm_hidden_size, 1152)
  comp_proj  what the LLM actually receives         (dim = hidden_size, 3584)
  raw        uncompressed encoder tokens (sample)   (reference ceiling)
report several rank measures on centered features:
  erank_entropy   exp(-sum p_i log p_i),  p_i = lambda_i / sum(lambda)   (Roy-Vetterli)
  participation   (sum lambda)^2 / sum(lambda^2)
  stable_rank     sum(lambda) / lambda_max   ( = ||X||_F^2 / sigma_max^2 )
  dims_9x_pct     #components for 90/95/99 % cumulative variance
GLOBAL (all windows of all videos pooled), PER-VIDEO (mean +/- std over videos),
and PER-WINDOW (mean over every 64-token window).  Also dumps the top-64
normalised eigenvalue spectrum of `comp` so the decay is inspectable, and a
per-video paired sign test image_mix vs noprune on erank_entropy.

Usage:
PYTHONPATH=. python eval_ablation/effrank.py \
  --models noprune=work_dirs/compressor_prune_ablation/noprune \
           image_mix=work_dirs/compressor_pretrain_image_mix \
  --manifest eval_ablation/manifest_imagemix.json --num_videos 0 \
  --out work_dirs/image_mix_eval/effrank --device cuda:0
"""
from __future__ import annotations
import argparse, json, os, sys, traceback
from typing import Dict, List
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_ablation.common import (  # noqa: E402
    DEFAULT_FORCE_IMAGE_SIZE, DEFAULT_MAX_FRAMES, DEFAULT_WINDOW_SIZE,
    extract_features, free_model, load_model, load_processor, prepare_video_sample,
)


def _eigs(x: np.ndarray, center: bool = True, max_rows: int = 20000) -> np.ndarray:
    if x.shape[0] > max_rows:
        x = x[np.random.choice(x.shape[0], max_rows, replace=False)]
    if center:
        x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    lam = (s ** 2) / max(x.shape[0] - 1, 1)
    return lam[lam > 0]


def rank_measures(x: np.ndarray, center: bool = True) -> Dict:
    lam = _eigs(x, center=center)
    if lam.size == 0:
        return {}
    tot = float(lam.sum())
    p = lam / tot
    erank = float(np.exp(-(p * np.log(p)).sum()))
    part = float((lam.sum() ** 2) / np.square(lam).sum())
    stable = float(lam.sum() / lam.max())
    csum = np.cumsum(p)
    return {
        "n_rows": int(x.shape[0]),
        "ambient_dim": int(x.shape[1]),
        "erank_entropy": erank,
        "participation": part,
        "stable_rank": stable,
        "dims_90pct": int(np.searchsorted(csum, 0.90) + 1),
        "dims_95pct": int(np.searchsorted(csum, 0.95) + 1),
        "dims_99pct": int(np.searchsorted(csum, 0.99) + 1),
        "evr_top1": float(p[0]),
        "evr_top8": float(p[:8].sum()),
    }


def per_group_erank(x: np.ndarray, sizes: np.ndarray, center: bool = True) -> Dict:
    """erank_entropy / participation for each contiguous group of `sizes` rows."""
    er, pr = [], []
    off = 0
    for k in sizes:
        k = int(k)
        w = x[off:off + k]; off += k
        if w.shape[0] < 3:
            continue
        lam = _eigs(w, center=center)
        if lam.size == 0:
            continue
        p = lam / lam.sum()
        er.append(float(np.exp(-(p * np.log(p)).sum())))
        pr.append(float((lam.sum() ** 2) / np.square(lam).sum()))
    return {
        "n_groups": len(er),
        "erank_entropy_mean": float(np.mean(er)) if er else None,
        "erank_entropy_std": float(np.std(er)) if er else None,
        "participation_mean": float(np.mean(pr)) if pr else None,
    }


def spectrum(x: np.ndarray, k: int = 64, center: bool = True) -> List[float]:
    lam = _eigs(x, center=center)
    p = lam / lam.sum()
    return [float(v) for v in p[:k]]


def run_model(path: str, videos: List[str], args) -> Dict:
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    proc = load_processor(path, force_image_size=args.force_image_size)
    model = load_model(path, device=args.device)
    comp_all, proj_all, raw_all, npw_all = [], [], [], []
    per_video = []
    for vp in videos:
        try:
            sample = prepare_video_sample(
                proc, vp, fps=args.fps, max_frames=args.max_frames,
                window_size=args.window_size, device=args.device,
                out_hw_fn=model.get_token_compressor().output_hw_for,
            )
            f = extract_features(model, sample)
            comp_all.append(f["comp"]); proj_all.append(f["comp_proj"])
            raw_all.append(f["raw"]); npw_all.append(f["n_per_window"])
            per_video.append({
                "video": os.path.basename(vp),
                "n_comp_tokens": int(f["comp"].shape[0]),
                "n_windows": int(len(f["n_per_window"])),
                "comp": rank_measures(f["comp"]),
                "comp_proj": rank_measures(f["comp_proj"]),
            })
            print(f"  [{os.path.basename(vp)}] comp{f['comp'].shape} "
                  f"erank(comp)={per_video[-1]['comp']['erank_entropy']:.2f} "
                  f"erank(proj)={per_video[-1]['comp_proj']['erank_entropy']:.2f}")
        except Exception:
            print(f"  [SKIP] {vp}\n{traceback.format_exc()}")
    free_model(model)

    comp = np.concatenate(comp_all, 0)
    proj = np.concatenate(proj_all, 0)
    raw = np.concatenate(raw_all, 0)
    npw = np.concatenate(npw_all, 0)

    pv_er_comp = np.array([p["comp"]["erank_entropy"] for p in per_video])
    pv_er_proj = np.array([p["comp_proj"]["erank_entropy"] for p in per_video])
    return {
        "model_path": path,
        "n_videos": len(per_video),
        "global": {
            "comp":      rank_measures(comp),
            "comp_proj": rank_measures(proj),
            "raw":       rank_measures(raw),
            "comp_uncentered": rank_measures(comp, center=False),
        },
        "per_video_summary": {
            "erank_comp_mean": float(pv_er_comp.mean()),
            "erank_comp_std": float(pv_er_comp.std()),
            "erank_comp_min": float(pv_er_comp.min()),
            "erank_comp_max": float(pv_er_comp.max()),
            "erank_comp_proj_mean": float(pv_er_proj.mean()),
            "erank_comp_proj_std": float(pv_er_proj.std()),
        },
        "per_window": {
            "comp":      per_group_erank(comp, npw),
            "comp_proj": per_group_erank(proj, npw),
        },
        "spectrum_comp_top64": spectrum(comp, 64),
        "spectrum_comp_proj_top64": spectrum(proj, 64),
        "per_video": per_video,
        "_pv_erank_comp": pv_er_comp.tolist(),
        "_pv_erank_proj": pv_er_proj.tolist(),
        "_videos": [p["video"] for p in per_video],
    }


def md_table(report: Dict) -> str:
    tags = list(report["models"].keys())
    L = ["# Effective rank of compressed tokens -- image_mix vs noprune\n"]
    cfg = report["config"]
    L.append(f"- videos: {cfg['n_videos']} | frames/clip {cfg['max_frames']} | "
             f"window {cfg['window_size']} | force_image_size {cfg['force_image_size']}")
    L.append("- measures on CENTERED features. erank_entropy = exp(spectral entropy); "
             "participation = (Σλ)²/Σλ²; stable_rank = Σλ/λ_max.\n")

    def rows(space, label):
        L.append(f"\n## {label}\n")
        L.append("| measure | " + " | ".join(f"`{t}`" for t in tags) + " |")
        L.append("|---|" + "---|" * len(tags))
        for key, disp in [
            ("erank_entropy", "effective rank (entropy)"),
            ("participation", "participation ratio"),
            ("stable_rank", "stable rank"),
            ("dims_90pct", "dims for 90% var"),
            ("dims_95pct", "dims for 95% var"),
            ("dims_99pct", "dims for 99% var"),
            ("evr_top1", "top-1 eigenvalue share"),
            ("evr_top8", "top-8 eigenvalue share"),
            ("ambient_dim", "ambient dim"),
            ("n_rows", "n tokens"),
        ]:
            cells = []
            for t in tags:
                v = report["models"][t]["global"][space].get(key)
                cells.append("n/a" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
            L.append(f"| {disp} | " + " | ".join(cells) + " |")

    rows("comp", "compressor output (pre-projector, global pool)")
    rows("comp_proj", "projected into LLM space (global pool)")
    rows("raw", "uncompressed encoder tokens (reference ceiling)")

    L.append("\n## per-video effective rank (entropy), mean ± std over videos\n")
    L.append("| space | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    for space, disp in [("erank_comp", "comp (pre-proj)"), ("erank_comp_proj", "comp_proj (LLM space)")]:
        cells = []
        for t in tags:
            s = report["models"][t]["per_video_summary"]
            cells.append(f"{s[space+'_mean']:.3g} ± {s[space+'_std']:.2g}")
        L.append(f"| {disp} | " + " | ".join(cells) + " |")

    L.append("\n## per-window effective rank (entropy), mean over all 64-token windows\n")
    L.append("| space | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    for space, disp in [("comp", "comp (pre-proj)"), ("comp_proj", "comp_proj (LLM space)")]:
        cells = []
        for t in tags:
            w = report["models"][t]["per_window"][space]
            cells.append(f"{w['erank_entropy_mean']:.3g} ± {w['erank_entropy_std']:.2g}  (n={w['n_groups']})")
        L.append(f"| {disp} | " + " | ".join(cells) + " |")

    if "paired" in report:
        p = report["paired"]
        L.append("\n## per-video paired comparison (image_mix − noprune), erank_entropy\n")
        for space in ("comp", "comp_proj"):
            d = p[space]
            L.append(f"- **{space}**: mean Δ = {d['mean_delta']:+.3f}, "
                     f"image_mix higher on {d['n_pos']}/{d['n']} videos "
                     f"(median Δ {d['median_delta']:+.3f})")
    L.append("\n_Full spectra & per-video rows: effrank.json_\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default="eval_ablation/manifest_imagemix.json")
    ap.add_argument("--num_videos", type=int, default=0)
    ap.add_argument("--out", default="work_dirs/image_mix_eval/effrank")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    ap.add_argument("--force_image_size", type=int, default=DEFAULT_FORCE_IMAGE_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    models = [p.split("=", 1) for p in args.models]
    data = json.load(open(args.manifest))
    videos = [e["video"] for e in data if os.path.exists(e["video"])]
    if args.num_videos:
        videos = videos[: args.num_videos]
    print(f"[effrank] {len(models)} models x {len(videos)} videos")

    report = {"config": vars(args) | {"n_videos": len(videos)}, "models": {}}
    for tag, path in models:
        print(f"\n=== {tag} :: {path} ===")
        report["models"][tag] = run_model(path, videos, args)

    if len(models) == 2:
        (ta, _), (tb, _) = models
        A, B = report["models"][ta], report["models"][tb]
        va, vb = A["_videos"], B["_videos"]
        common = [v for v in va if v in vb]
        ia = {v: i for i, v in enumerate(va)}; ib = {v: i for i, v in enumerate(vb)}
        report["paired"] = {}
        for space, keya in (("comp", "_pv_erank_comp"), ("comp_proj", "_pv_erank_proj")):
            da = np.array([A[keya][ia[v]] for v in common])
            db = np.array([B[keya][ib[v]] for v in common])
            delta = db - da  # image_mix (b) minus noprune (a) if models given in that order
            report["paired"][space] = {
                "n": len(common),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "n_pos": int((delta > 0).sum()),
                "order": f"{tb} - {ta}",
            }

    with open(os.path.join(args.out, "effrank.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "effrank.md"), "w") as f:
        f.write(md_table(report))
    print(f"\n[effrank] wrote {args.out}/effrank.{{json,md}}")


if __name__ == "__main__":
    main()
