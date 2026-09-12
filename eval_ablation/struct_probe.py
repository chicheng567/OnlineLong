#!/usr/bin/env python3
"""Structural / collapse gate for a compressed representation (design doc §2.3, §4).

Answers the question the raw `content_collapse` cannot: *does the compressor pass
the encoder's video-to-video geometry through, or has it scrambled the
discriminative residual while keeping the norm?* This is the Phase-1 -> Phase-2
gate, and the same probe re-run on the fold readout is the Phase-2 -> Phase-3 gate.

FEATURE level only (frozen encoder -> compressor, pre-`mm_projector`, no generate):
one mean-pooled vector per video per representation, then

  * xcos_raw   mean off-diagonal cosine of the L2-normed pooled vectors.
               ~0.87-0.91 on the raw encoder itself (an inherent SigLIP-NaViT
               common component) -> NOT a collapse signal on its own.
  * xcos_cen   same, after subtracting the across-video mean vector. Should sit
               near the encoder's (~ -0.08 for a dozen clips). A compressor value
               far above the encoder's = centered structure lost.
  * cc_frac    ||mean_v p_v|| / mean_v ||p_v||  -- shared-component energy.
               ~0.94 on the raw encoder; a compressor value pinned to ~1.0 means
               every pooled vector points the same way (RMSNorm-style collapse).
  * struct_rho Spearman of this rep's video-to-video cosine ranking against the
               encoder's. qbase passes it through at ~0.999; a value below ~0.9
               is a red flag (the superseded fold scored 0.39).
  * token_collapse       mean pairwise cosine among ONE video's output tokens.
  * temporal_blindness   row-cosine  C(video)  vs  C(frames reversed).

Works on the Phase-1 `transformer_decoder_flat` + `--adaptive_segmentation`
checkpoint: it drives one whole-video compression part straight through
`compress_visual_tokens_with_compressor`, so the model-side fixed-count adaptive
segmenter runs exactly as in training and no `input_ids` placeholder bookkeeping
is needed.

    PYTHONPATH=. python eval_ablation/struct_probe.py \
        --models phase1=work_dirs/phase1_qbase_internvid/checkpoint-500 \
        --manifest eval_ablation/manifest_probe.json \
        --num_videos 12 --max_frames 180 \
        --out work_dirs/phase1_qbase_internvid/struct_probe --device cuda:0
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

from eval_ablation.common import free_model, load_model, load_processor  # noqa: E402
from videollama3.mm_utils import load_video  # noqa: E402
from videollama3.model.videollama3_arch import _grid_hw_for_compression_parts  # noqa: E402

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".gif")


# --------------------------------------------------------------------------- math
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _selfcos(x: np.ndarray, cap: int = 512) -> float:
    x = x[:cap]
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    g = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    return float(np.mean(g[iu])) if len(iu[0]) else float("nan")


def _rowcos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return float(np.mean(num / den))


def _pool_stats(vecs: List[np.ndarray]) -> Dict[str, float]:
    """Cross-video geometry of a list of per-video pooled vectors."""
    if len(vecs) < 2:
        return dict(xcos_raw=float("nan"), xcos_cen=float("nan"), cc_frac=float("nan"))
    P = np.stack(vecs).astype(float)
    norms = np.linalg.norm(P, axis=-1)
    cc_frac = float(np.linalg.norm(P.mean(0)) / (norms.mean() + 1e-8))
    iu = np.triu_indices(len(P), k=1)
    Pn = P / (norms[:, None] + 1e-8)
    xcos_raw = float((Pn @ Pn.T)[iu].mean())
    Pc = P - P.mean(0, keepdims=True)
    Pc = Pc / (np.linalg.norm(Pc, axis=-1, keepdims=True) + 1e-8)
    xcos_cen = float((Pc @ Pc.T)[iu].mean())
    return dict(xcos_raw=xcos_raw, xcos_cen=xcos_cen, cc_frac=cc_frac)


def _vv_cos(vecs: List[np.ndarray]) -> np.ndarray:
    """Off-diagonal video-to-video cosine of L2-normed pooled vectors."""
    P = np.stack(vecs).astype(float)
    P = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-8)
    iu = np.triu_indices(len(P), k=1)
    return (P @ P.T)[iu]


# ------------------------------------------------------------------- model plumbing
def _reverse_frames(pv: torch.Tensor, num_frames: int) -> torch.Tensor:
    """pixel_values is patch rows concatenated frame-by-frame; flip the frame order."""
    if num_frames <= 1 or pv.shape[0] % num_frames != 0:
        return pv
    tpf = pv.shape[0] // num_frames
    return pv.view(num_frames, tpf, *pv.shape[1:]).flip(0).reshape_as(pv)


@torch.no_grad()
def _encode_and_compress(model, pv, gs, ms):
    """(compressor output tokens, raw encoder tokens) for one whole-video window."""
    m = model.get_model()
    mm = m.get_vision_encoder()(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)
    parts = [[0, int(mm.shape[0])]]
    grid_hws = _grid_hw_for_compression_parts(parts, gs, ms)
    c, _ = model.compress_visual_tokens_with_compressor(mm.clone(), parts, grid_hws)
    return c.detach().float().cpu().numpy(), mm.detach().float().cpu().numpy()


def run_model(tag: str, path: str, videos: List[str], device: str,
              fps: int, max_frames: int, merge_size: int) -> Dict:
    proc = load_processor(path, force_image_size=0, native_max_tokens=16384)  # dynamic HW, training budget
    model = load_model(path, device=device)
    comp = model.get_token_compressor()
    adaptive = bool(getattr(comp, "adaptive_segmentation", False))

    per: List[Dict] = []
    cmp_pooled: List[np.ndarray] = []
    enc_pooled: List[np.ndarray] = []
    for vp in videos:
        try:
            frames, ts = load_video(vp, fps=fps, max_frames=max_frames)
            T = len(frames)
            if T < 8:
                print(f"  [{tag}] [skip <8f] {os.path.basename(vp)}")
                continue
            conv = [{"role": "user", "content": [
                {"type": "video", "timestamps": [float(t) for t in ts], "num_frames": T},
                {"type": "text", "text": "x"}]}]
            inp = proc(images=[frames], text=conv, merge_size=merge_size, return_tensors="pt")
            pv = inp["pixel_values"].to(device, torch.bfloat16)
            gs = inp["grid_sizes"].to(device)
            ms = inp["merge_sizes"].to(device)

            c_real, mm_real = _encode_and_compress(model, pv, gs, ms)
            c_rev, _ = _encode_and_compress(model, _reverse_frames(pv, T), gs, ms)

            cmp_pooled.append(c_real.mean(0))
            enc_pooled.append(mm_real.mean(0))
            n_seg = c_real.shape[0] // int(getattr(comp, "num_queries", c_real.shape[0]) or c_real.shape[0])
            per.append({
                "video": os.path.basename(vp),
                "n_frames": T,
                "n_out_tokens": int(c_real.shape[0]),
                "n_segments": int(n_seg) if adaptive else None,
                "token_collapse": _selfcos(c_real),
                "temporal_blindness": _rowcos(c_real, c_rev),
                "feat_reldiff": float(np.linalg.norm(c_real - c_rev) / (np.linalg.norm(c_real) + 1e-8)),
            })
            print(f"  [{tag}] {per[-1]['video']:36s} T={T:3d} "
                  f"seg={per[-1]['n_segments']} out={per[-1]['n_out_tokens']:5d} "
                  f"tok_collapse={per[-1]['token_collapse']:.3f} "
                  f"temporal_blind={per[-1]['temporal_blindness']:.3f}")
        except Exception:
            print(f"  [{tag}] [skip] {vp}\n{traceback.format_exc()}")
    free_model(model)

    cmp_s = _pool_stats(cmp_pooled)
    enc_s = _pool_stats(enc_pooled)
    struct_rho = float("nan")
    if len(cmp_pooled) >= 3 and len(cmp_pooled) == len(enc_pooled):
        struct_rho = _spearman(_vv_cos(cmp_pooled), _vv_cos(enc_pooled))

    def _m(k):
        v = [p[k] for p in per if isinstance(p.get(k), float) and not np.isnan(p[k])]
        return float(np.mean(v)) if v else float("nan")

    agg = {
        "n_videos": len(per),
        "adaptive_segmentation": adaptive,
        "struct_rho": struct_rho,
        "xcos_raw": cmp_s["xcos_raw"],
        "xcos_cen": cmp_s["xcos_cen"],
        "cc_frac": cmp_s["cc_frac"],
        "xcos_raw_enc": enc_s["xcos_raw"],
        "xcos_cen_enc": enc_s["xcos_cen"],
        "cc_frac_enc": enc_s["cc_frac"],
        "token_collapse": _m("token_collapse"),
        "temporal_blindness": _m("temporal_blindness"),
        "feat_reldiff": _m("feat_reldiff"),
    }
    return {"aggregate": agg, "per_video": per}


# ------------------------------------------------------------------------- verdict
def verdict(a: Dict) -> List[str]:
    out: List[str] = []
    sr = a["struct_rho"]
    xc, xce = a["xcos_cen"], a["xcos_cen_enc"]
    cf, cfe = a["cc_frac"], a["cc_frac_enc"]
    tc, tb = a["token_collapse"], a["temporal_blindness"]

    if not np.isnan(sr):
        out.append(f"- struct_rho={sr:.3f} vs encoder  "
                   + ("PASS - encoder video-to-video geometry preserved" if sr >= 0.95
                      else "BORDERLINE - geometry partly scrambled" if sr >= 0.90
                      else "FAIL - discriminative geometry scrambled"))
    if not np.isnan(xc) and not np.isnan(xce):
        d = abs(xc - xce)
        out.append(f"- xcos_cen={xc:+.3f} (encoder {xce:+.3f}, delta {d:.3f})  "
                   + ("PASS - tracks the encoder" if d <= 0.05
                      else "BORDERLINE - drifting from encoder" if d <= 0.10
                      else "FAIL - centered cross-video structure lost"))
    if not np.isnan(cf) and not np.isnan(cfe):
        out.append(f"- cc_frac={cf:.3f} (encoder {cfe:.3f})  "
                   + ("FAIL - pinned to one shared direction" if cf >= 0.995
                      else "BORDERLINE" if cf >= 0.99 or abs(cf - cfe) > 0.04
                      else "PASS - near the encoder's shared-component energy"))
    if not np.isnan(tc):
        out.append(f"- token_collapse={tc:.3f}  "
                   + ("FAIL - output tokens are ~one vector" if tc >= 0.98
                      else "BORDERLINE" if tc >= 0.90
                      else "PASS - output tokens are diverse"))
    if not np.isnan(tb):
        out.append(f"- temporal_blindness={tb:.3f}  "
                   + ("FAIL - output ignores frame order" if tb >= 0.98
                      else "BORDERLINE - weak temporal sensitivity" if tb >= 0.90
                      else "PASS - frame order changes the output"))
    return out


def _gate_pass(a: Dict) -> bool:
    sr, xc, xce = a["struct_rho"], a["xcos_cen"], a["xcos_cen_enc"]
    cf = a["cc_frac"]
    ok = True
    if not np.isnan(sr):
        ok &= sr >= 0.95
    if not np.isnan(xc) and not np.isnan(xce):
        ok &= abs(xc - xce) <= 0.05
    if not np.isnan(cf):
        ok &= cf < 0.99
    return bool(ok)


# ---------------------------------------------------------------------------- main
def _collect_videos(args) -> List[str]:
    if args.manifest:
        items = json.load(open(args.manifest))
        vids = [it["video"] if isinstance(it, dict) else it for it in items]
    elif args.video_dir:
        vids = sorted(
            os.path.join(args.video_dir, f)
            for f in os.listdir(args.video_dir)
            if f.lower().endswith(VIDEO_EXTS)
        )
    else:
        raise SystemExit("pass --manifest <json list> or --video_dir <dir>")
    vids = [v for v in vids if os.path.exists(v)]
    return vids[: args.num_videos]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="tag=path (repeatable)")
    ap.add_argument("--manifest", default=None, help="json list of {'video': path} or [path, ...]")
    ap.add_argument("--video_dir", default=None, help="directory of video files (alt to --manifest)")
    ap.add_argument("--num_videos", type=int, default=12)
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--max_frames", type=int, default=180)
    ap.add_argument("--merge_size", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="work_dirs/struct_probe")
    args = ap.parse_args()

    videos = _collect_videos(args)
    if len(videos) < 3:
        raise SystemExit(f"need >= 3 readable videos, found {len(videos)}")
    os.makedirs(args.out, exist_ok=True)
    print(f"[struct_probe] {len(videos)} videos, fps={args.fps} max_frames={args.max_frames}")

    results = {}
    for pair in args.models:
        tag, path = pair.split("=", 1)
        print(f"=== struct probe: {tag} ({path}) ===")
        results[tag] = run_model(tag, path, videos, args.device,
                                 args.fps, args.max_frames, args.merge_size)

    with open(os.path.join(args.out, "struct.json"), "w") as f:
        json.dump(results, f, indent=1)

    lines = [
        "## Structural / collapse gate (`struct_probe.py`)\n",
        "_`xcos_cen` should sit near the encoder's; `cc_frac` near the encoder's "
        "(not ~1.0); `struct_rho` >= 0.95. Raw `xcos_raw` is ~0.9 on the encoder "
        "itself and is not a gate._\n",
        "| model | n | struct_rho | xcos_cen (enc) | cc_frac (enc) | xcos_raw (enc) | "
        "token_collapse | temporal_blind | gate |",
        "|---|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for tag, r in results.items():
        a = r["aggregate"]
        lines.append(
            f"| `{tag}` | {a['n_videos']} | {a['struct_rho']:.3f} | "
            f"{a['xcos_cen']:+.3f} ({a['xcos_cen_enc']:+.3f}) | "
            f"{a['cc_frac']:.3f} ({a['cc_frac_enc']:.3f}) | "
            f"{a['xcos_raw']:.3f} ({a['xcos_raw_enc']:.3f}) | "
            f"{a['token_collapse']:.3f} | {a['temporal_blindness']:.3f} | "
            f"{'PASS' if _gate_pass(a) else 'FAIL'} |"
        )
    for tag, r in results.items():
        lines.append(f"\n**`{tag}`**  ({'gate PASS' if _gate_pass(r['aggregate']) else 'gate FAIL'})")
        lines += verdict(r["aggregate"])
    md = os.path.join(args.out, "struct.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[struct_probe] wrote {md}")


if __name__ == "__main__":
    main()
