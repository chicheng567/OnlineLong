#!/usr/bin/env python3
"""
Is the compressor's low effective rank a GENERIC bottleneck effect, or did the
caption-only training objective SHAPE which subspace survives?

Two checks, per model, on the compressed windows of the manifest videos.
The compressor never sees text, so nothing here depends on the eval prompt --
the question is only about what the *training* loss left in the weights.

Check 1 -- subspace geometry
  S_comp = top-k principal subspace of the pooled compressor output (pre-projector,
  1152-dim SigLIP space -- same space as the raw encoder tokens).
  S_raw  = top-k principal subspace of the raw encoder tokens.
  * mean cos^2 of the principal angles between S_comp and S_raw  (1 = identical,
    ~k/1152 = random)
  * raw variance captured by S_comp, vs by S_raw's own top-k (ceiling), vs by a
    random k-subspace (floor)
  * per-PCA-band: does S_comp's alignment fall off smoothly with the raw variance
    ranking (=> generic PCA-like compression) or drop off a cliff after the first
    few "gist" dims (=> selective / task-shaped)?
  * S_comp(prune) vs S_comp(noprune) overlap

Check 2 -- what is linearly recoverable from `comp`, token-holdout
  Each window's raw tokens are split checkerboard into halves A and B (both cover
  every frame and the whole frame). The compressor saw ALL tokens.
  For target statistics of half B -- computed in the raw-PCA basis:
     gist      = mean(B) . V_raw[:, 0:16]            (coarse semantics, caption-y)
     fine      = mean(B) . V_raw[:, 64:256]          (fine feature detail)
     spatial   = per-quadrant mean(B) . V_raw[:, 0:16], mean-centred  (where things are)
     temporal  = (late frames - early frames) mean(B) . V_raw[:, 0:16]  (change)
  we fit ridge (5-fold CV R^2) from
     comp  : mean+std of the 64 compressed tokens of that window
     rawA  : mean+std of half A            (the "no compression" control)
  retention(target) = R2(comp -> B) / R2(rawA -> B).
  Caption-shaped  => retention(gist) >> retention(fine) ~ retention(spatial) ~ retention(temporal).
  Generic low-rank => retention roughly flat, or a smooth decay with PCA index and
  spatial/temporal no worse than fine.

Usage
-----
PYTHONPATH=. python eval_ablation/subspace_probe.py \
    --models prune=work_dirs/compressor_prune_ablation/prune0.30 \
             noprune=work_dirs/compressor_prune_ablation/noprune \
    --manifest eval_ablation/manifest_probe.json \
    --out work_dirs/ablation_eval/subspace_probe --device cuda:0
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
    free_model, load_model, load_processor, prepare_video_sample,
)
from eval_ablation.common import _grid_hw_for_compression_parts  # noqa: E402

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


# --------------------------------------------------------------------------------------
# per-window extraction: comp tokens + raw half-A/half-B summaries
# --------------------------------------------------------------------------------------
def _checkerboard_masks(h: int, w: int) -> np.ndarray:
    r = np.arange(h)[:, None]
    c = np.arange(w)[None, :]
    return ((r + c) % 2 == 0).reshape(-1)          # (h*w,) True = half A


@torch.no_grad()
def extract_windows(model, sample: Dict) -> List[Dict]:
    m = model.get_model()
    pv, gs, ms = sample["pixel_values"], sample["grid_sizes"], sample["merge_sizes"]
    parts = sample["compression_parts"]
    mm = m.get_vision_encoder()(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)   # (N, C)
    grid_hws = _grid_hw_for_compression_parts(parts, gs, ms)
    comp_all = model.compress_visual_tokens_with_compressor(mm.clone(), parts, grid_hws)  # (sum n_out, C)

    compressor = model.get_token_compressor()
    n_out = [int(np.prod(compressor.output_hw_for(int(h), int(w)))) for (h, w) in grid_hws]

    mm = mm.float().cpu().numpy()
    comp_all = comp_all.float().cpu().numpy()
    C = mm.shape[1]
    tpf = sample["meta"]["tokens_per_frame"]

    rows, off = [], 0
    for (s, e), (h, w), k in zip(parts, grid_hws, n_out):
        raw_win = mm[s:e]                                   # (T*h*w, C)
        comp_win = comp_all[off:off + k]; off += k          # (k, C)
        T = raw_win.shape[0] // (h * w)
        raw_thwc = raw_win.reshape(T, h * w, C)
        A = _checkerboard_masks(h, w)                       # (h*w,)
        Braw = raw_thwc[:, ~A, :]                           # (T, nB, C)
        Araw = raw_thwc[:, A, :]                            # (T, nA, C)

        # quadrant means of B  (2x2 over the h x w grid)
        qidx = np.zeros(h * w, dtype=int)
        rr = np.arange(h)[:, None] * np.ones(w, dtype=int)[None, :]
        cc = np.ones(h, dtype=int)[:, None] * np.arange(w)[None, :]
        qidx = ((rr >= h / 2).astype(int) * 2 + (cc >= w / 2).astype(int)).reshape(-1)
        qB = qidx[~A]
        quad_means = np.stack([raw_thwc[:, ~A, :][:, qB == q, :].mean(axis=(0, 1))
                               for q in range(4)], axis=0)   # (4, C)

        nlate = T // 2
        rows.append({
            "B_mean": Braw.mean(axis=(0, 1)),
            "B_fine_src": Braw.mean(axis=(0, 1)),            # same vec, projected on a different band
            "B_quad": quad_means,
            "B_early": Braw[:T - nlate].mean(axis=(0, 1)),
            "B_late": Braw[T - nlate:].mean(axis=(0, 1)),
            "A_mean": Araw.mean(axis=(0, 1)),
            "A_std": Araw.std(axis=(0, 1)),
            "comp_mean": comp_win.mean(axis=0),
            "comp_std": comp_win.std(axis=0),
            "comp_rows": comp_win,                            # (k, C) for the subspace pool
            "raw_rows": raw_win[np.random.choice(raw_win.shape[0],
                                min(400, raw_win.shape[0]), replace=False)],
        })
    return rows


# --------------------------------------------------------------------------------------
# check 1
# --------------------------------------------------------------------------------------
def _basis(x: np.ndarray, k: int) -> np.ndarray:
    xc = x - x.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return vt[:k].T                                           # (C, k) orthonormal columns


def _var_captured(x: np.ndarray, V: np.ndarray) -> float:
    xc = x - x.mean(0, keepdims=True)
    tot = float((xc ** 2).sum())
    proj = xc @ V
    return float((proj ** 2).sum() / (tot + 1e-12))


def check1(comp_pool: np.ndarray, raw_pool: np.ndarray, k: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    Vc = _basis(comp_pool, k)
    Vr = _basis(raw_pool, k)
    Vrand = np.linalg.qr(rng.standard_normal((comp_pool.shape[1], k)))[0]

    M = Vr.T @ Vc                                             # (k, k)
    ang_cos = np.linalg.svd(M, compute_uv=False)              # cos of principal angles
    out = {
        "k": k,
        "principal_angle_cos2_mean": float((ang_cos ** 2).mean()),
        "principal_angle_cos_sorted": [float(v) for v in ang_cos],
        "raw_var_in_S_comp": _var_captured(raw_pool, Vc),
        "raw_var_in_S_raw_topk_CEILING": _var_captured(raw_pool, Vr),
        "raw_var_in_random_k_FLOOR": _var_captured(raw_pool, Vrand),
        "comp_var_in_S_raw": _var_captured(comp_pool, Vr),
        "comp_var_in_S_comp_self": _var_captured(comp_pool, Vc),
    }
    # per-band alignment: overlap of S_comp with raw PCA dirs [b0:b1]
    xc = raw_pool - raw_pool.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    bands = [(0, 4), (4, 16), (16, 48), (48, 128), (128, 384)]
    out["S_comp_overlap_with_raw_band"] = {
        f"{b0}-{b1}": float(((vt[b0:b1] @ Vc) ** 2).sum() / (b1 - b0))
        for b0, b1 in bands
    }
    return out, Vc


def subspace_overlap(Va: np.ndarray, Vb: np.ndarray) -> Dict:
    ang = np.linalg.svd(Va.T @ Vb, compute_uv=False)
    return {"cos2_mean": float((ang ** 2).mean()),
            "cos_sorted": [float(v) for v in ang]}


# --------------------------------------------------------------------------------------
# check 2
# --------------------------------------------------------------------------------------
def _cv_r2(X: np.ndarray, Y: np.ndarray, seed: int) -> float:
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    if Y.ndim == 1:
        Y = Y[:, None]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y)
    for tr, te in kf.split(X):
        model = RidgeCV(alphas=(1.0, 10.0, 100.0, 1000.0, 1e4))
        model.fit(X[tr], Y[tr])
        preds[te] = model.predict(X[te])
    # variance-weighted mean R^2 across target dims (avoids dead dims dominating)
    r2s, wts = [], []
    for j in range(Y.shape[1]):
        v = Y[:, j].var()
        if v < 1e-12:
            continue
        r2s.append(r2_score(Y[:, j], preds[:, j]))
        wts.append(v)
    return float(np.average(r2s, weights=wts)) if r2s else float("nan")


def check2(rows: List[Dict], Vraw: np.ndarray, seed: int) -> Dict:
    B_mean = np.stack([r["B_mean"] for r in rows])
    B_quad = np.stack([r["B_quad"] for r in rows])                  # (n,4,C)
    B_delta = np.stack([r["B_late"] - r["B_early"] for r in rows])
    Xcomp = np.concatenate([np.stack([r["comp_mean"] for r in rows]),
                            np.stack([r["comp_std"] for r in rows])], axis=1)
    XrawA = np.concatenate([np.stack([r["A_mean"] for r in rows]),
                            np.stack([r["A_std"] for r in rows])], axis=1)

    g16 = Vraw[:, 0:16]
    tgt = {
        "gist_pca0_16":  B_mean @ g16,
        "fine_pca64_256": B_mean @ Vraw[:, 64:256],
        "spatial_contrast": (B_quad @ g16 -
                             (B_quad @ g16).mean(axis=1, keepdims=True)
                             ).reshape(len(rows), -1),
        "temporal_delta": B_delta @ g16,
        # a smooth-decay reference: gist vs successive PCA bands
        "band_pca16_32": B_mean @ Vraw[:, 16:32],
        "band_pca32_64": B_mean @ Vraw[:, 32:64],
    }
    res = {}
    for name, Y in tgt.items():
        r_comp = _cv_r2(Xcomp, Y, seed)
        r_rawA = _cv_r2(XrawA, Y, seed)
        res[name] = {
            "R2_comp": r_comp,
            "R2_rawA_control": r_rawA,
            "retention": float(r_comp / r_rawA) if r_rawA > 0.02 else None,
            "target_dim": int(Y.shape[1] if Y.ndim > 1 else 1),
        }
    return res


# --------------------------------------------------------------------------------------
def parse_models(pairs):
    out = []
    for p in pairs:
        tag, path = p.split("=", 1)
        out.append((tag, path))
    return out


def run_model(model_path, videos, args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    proc = load_processor(model_path, force_image_size=args.force_image_size)
    model = load_model(model_path, device=args.device)
    rows = []
    for vp in videos:
        try:
            s = prepare_video_sample(proc, vp, fps=args.fps, max_frames=args.max_frames,
                                     window_size=args.window_size, device=args.device,
                                     out_hw_fn=model.get_token_compressor().output_hw_for)
            rows.extend(extract_windows(model, s))
            print(f"  ok {os.path.basename(vp)}  (+{s['meta']['n_windows']} windows, "
                  f"total {len(rows)})")
        except Exception:
            print(f"  SKIP {vp}\n{traceback.format_exc().splitlines()[-1]}")
    free_model(model)

    comp_pool = np.concatenate([r["comp_rows"] for r in rows])
    raw_pool = np.concatenate([r["raw_rows"] for r in rows])
    if raw_pool.shape[0] > 40000:
        raw_pool = raw_pool[np.random.choice(raw_pool.shape[0], 40000, replace=False)]
    Vraw_full = _basis(raw_pool, 384)

    c1, Vc = check1(comp_pool, raw_pool, args.k, args.seed)
    c2 = check2(rows, Vraw_full, args.seed)
    return {"n_windows": len(rows), "check1_subspace": c1, "check2_probe": c2}, Vc


def to_md(report: Dict) -> str:
    tags = list(report["models"].keys())
    L = ["# Compressor subspace / probe -- is the low rank task-shaped?\n"]
    L.append(f"- videos: {report['config']['n_videos']}  windows/model: "
             + ", ".join(f"`{t}`={report['models'][t]['n_windows']}" for t in tags))
    L.append(f"- k (subspace dim): {report['config']['k']} | "
             f"force_image_size: {report['config']['force_image_size']}\n")

    L.append("## Check 1 -- S_comp vs S_raw geometry\n")
    L.append("| metric | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    def r1(lbl, key):
        return "| " + lbl + " | " + " | ".join(
            f"{report['models'][t]['check1_subspace'][key]:.4g}" for t in tags) + " |"
    L.append(r1("principal-angle mean cos^2  (1=identical, ~k/1152=random)", "principal_angle_cos2_mean"))
    L.append(r1("raw var in S_comp", "raw_var_in_S_comp"))
    L.append(r1("raw var in S_raw top-k  (CEILING)", "raw_var_in_S_raw_topk_CEILING"))
    L.append(r1("raw var in random k-subspace  (FLOOR)", "raw_var_in_random_k_FLOOR"))
    L.append(r1("comp var lying in S_raw", "comp_var_in_S_raw"))
    L.append("\n**S_comp overlap with raw PCA bands** (smooth fall-off = generic PCA; cliff after 0-4 = selective)\n")
    bandkeys = list(report["models"][tags[0]]["check1_subspace"]["S_comp_overlap_with_raw_band"].keys())
    L.append("| raw PCA band | " + " | ".join(f"`{t}`" for t in tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    for bk in bandkeys:
        L.append(f"| {bk} | " + " | ".join(
            f"{report['models'][t]['check1_subspace']['S_comp_overlap_with_raw_band'][bk]:.4g}"
            for t in tags) + " |")
    if "cross_model_S_comp" in report:
        cm = report["cross_model_S_comp"]
        L.append(f"\n**S_comp(prune) vs S_comp(noprune)**: mean cos^2 = {cm['cos2_mean']:.4g}\n")

    L.append("\n## Check 2 -- linear recoverability from `comp` (token-holdout)\n")
    L.append("retention = R2(comp -> half B) / R2(raw half A -> half B). "
             "gist should retain; if fine/spatial/temporal retain just as well it's generic.\n")
    for t in tags:
        L.append(f"\n### `{t}`\n")
        L.append("| target | R2 comp->B | R2 rawA->B (control) | retention |")
        L.append("|---|---|---|---|")
        for name, d in report["models"][t]["check2_probe"].items():
            ret = "n/a" if d["retention"] is None else f"{d['retention']:.3f}"
            L.append(f"| {name} (dim {d['target_dim']}) | {d['R2_comp']:.3f} | "
                     f"{d['R2_rawA_control']:.3f} | {ret} |")
    L.append("\n_Full numbers: `subspace_probe.json`._\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--manifest", default="eval_ablation/manifest_probe.json")
    ap.add_argument("--num_videos", type=int, default=0)
    ap.add_argument("--out", default="work_dirs/ablation_eval/subspace_probe")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=DEFAULT_MAX_FRAMES)
    ap.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    ap.add_argument("--force_image_size", type=int, default=DEFAULT_FORCE_IMAGE_SIZE)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    models = parse_models(args.models)
    data = json.load(open(args.manifest))
    videos = [e["video"] for e in data if os.path.exists(e["video"])]
    if args.num_videos:
        videos = videos[: args.num_videos]
    print(f"[subspace_probe] {len(models)} models x {len(videos)} videos, k={args.k}")

    report = {"config": vars(args) | {"n_videos": len(videos)}, "models": {}}
    Vc = {}
    for tag, path in models:
        print(f"\n=== {tag} :: {path} ===")
        report["models"][tag], Vc[tag] = run_model(path, videos, args)

    if len(models) == 2:
        (ta, _), (tb, _) = models
        report["cross_model_S_comp"] = subspace_overlap(Vc[ta], Vc[tb])

    with open(os.path.join(args.out, "subspace_probe.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.out, "subspace_probe.md"), "w") as f:
        f.write(to_md(report))
    print(f"\n[subspace_probe] wrote {args.out}/subspace_probe.{{json,md}}")
    print("\n" + to_md(report))


if __name__ == "__main__":
    main()
