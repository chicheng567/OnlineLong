#!/usr/bin/env python3
"""Is the compressed representation actually a function of the video, or has the
frozen LLM just learned to autocomplete templated captions?

A flat / very-low training CE (0.2-0.3) on caption data is ambiguous: the frozen
7B can predict most caption tokens from language priors alone, so the compressor
can score well while carrying almost no visual information. This probe measures
whether it does, per model, on a handful of videos:

  FEATURE level (encoder -> compressor, pre-LLM)
    * temporal_blindness   mean row-cosine  C(video) vs C(frames reversed).
                           ~1.0  -> output ignores frame order.
    * struct_rho           Spearman of this rep's video-to-video cosine ranking
                           against the encoder's. >=0.95 -> geometry preserved;
                           <0.9 -> discriminative residual scrambled. GATE.
    * xcos_cen             content_collapse after removing the across-video mean
                           vector. Should sit near the encoder's (~ -0.08). GATE.
    * cc_frac              ||mean_v p_v|| / mean_v ||p_v|| -- shared-component
                           energy. ~0.94 on the encoder; ~1.0 -> collinear. GATE.
    * content_collapse     mean cosine between DIFFERENT videos' mean-pooled C
                           (== xcos_raw). ~0.9 on the raw encoder itself, so it is
                           NOT a gate on its own (design doc §2.3).
    * token_collapse       mean pairwise cosine among one video's output tokens.
                           ~1.0  -> the M readout tokens are all the same vector.

  CAPTION level (full generate)
    * caption_order_sim    rougeL( caption(video), caption(frames reversed) ).
                           ~1.0  -> the caption does not depend on frame order.
    * specificity_gap      rougeL(cap_i, ref_i) - mean_j!=i rougeL(cap_i, ref_j).
                           ~0    -> captions are not video-specific (generic prose).

Verdict thresholds are conservative; borderline numbers are reported, not hidden.

    PYTHONPATH=. python eval_ablation/grounding_probe.py \
        --models qbase=work_dirs/phase1_qbase_internvid \
                 fold=work_dirs/phase2_fold_internvid \
        --manifest eval_ablation/manifest_probe.json \
        --out work_dirs/phase2_probe/grounding --device cuda:0
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
    free_model, load_model, load_processor, prepare_video_sample,
)
from eval_ablation.metrics import rouge_l, tokenize  # noqa: E402
from eval_ablation.struct_probe import _pool_stats, _spearman, _vv_cos  # noqa: E402
from videollama3.model.videollama3_arch import _grid_hw_for_compression_parts  # noqa: E402

GEN = dict(do_sample=False, num_beams=1, max_new_tokens=200, repetition_penalty=1.1)


def _rowcos(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return float(np.mean(num / den))


def _selfcos(x: np.ndarray, cap: int = 512) -> float:
    x = x[:cap]
    x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    g = x @ x.T
    iu = np.triu_indices(len(x), k=1)
    return float(np.mean(g[iu])) if len(iu[0]) else float("nan")


def _reverse_frames(pv: torch.Tensor, num_frames: int) -> torch.Tensor:
    """pixel_values is patch rows concatenated frame-by-frame; flip the frame order."""
    if pv.shape[0] % num_frames != 0:
        return pv
    tpf = pv.shape[0] // num_frames
    return pv.view(num_frames, tpf, *pv.shape[1:]).flip(0).reshape_as(pv)


@torch.no_grad()
def _compress(model, pv, gs, ms, parts):
    """(compressor output tokens, raw-encoder pooled vector) for the sample."""
    m = model.get_model()
    mm = m.get_vision_encoder()(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)
    grid_hws = _grid_hw_for_compression_parts(parts, gs, ms)
    c, _ = model.compress_visual_tokens_with_compressor(mm.clone(), parts, grid_hws)
    enc_pooled = mm.detach().float().mean(0).cpu().numpy()
    return c.detach().float().cpu().numpy(), enc_pooled


@torch.no_grad()
def _caption(model, proc, sample, pv) -> str:
    out = model.generate(
        input_ids=sample["input_ids"], pixel_values=pv,
        grid_sizes=sample["grid_sizes"], merge_sizes=sample["merge_sizes"],
        modals=sample["modals"], compression_parts=sample["compression_parts"],
        compression_ts_info=sample["compression_ts_info"], **GEN,
    )
    return proc.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def run_model(tag: str, path: str, items: List[Dict], device: str, max_frames: int,
              window_size: int, whole_video: bool = False) -> Dict:
    proc = load_processor(path, force_image_size=0)          # dynamic HW
    model = load_model(path, device=device)
    per: List[Dict] = []
    pooled: List[np.ndarray] = []
    enc_pooled: List[np.ndarray] = []
    caps_real: List[str] = []
    refs: List[str] = []
    for i, it in enumerate(items):
        try:
            s = prepare_video_sample(
                proc, it["video"], prompt=it.get("prompt") or "Describe this video in detail.",
                fps=1, max_frames=max_frames, window_size=window_size, device=device,
                out_hw_fn=model.get_token_compressor().output_hw_for,
                whole_video=whole_video,
            )
            nf = s["meta"]["num_frames"]
            pv = s["pixel_values"]
            pv_rev = _reverse_frames(pv, nf)
            c_real, enc_real = _compress(model, pv, s["grid_sizes"], s["merge_sizes"], s["compression_parts"])
            c_rev, _ = _compress(model, pv_rev, s["grid_sizes"], s["merge_sizes"], s["compression_parts"])
            cap_real = _caption(model, proc, s, pv)
            cap_rev = _caption(model, proc, s, pv_rev)
            per.append({
                "video": os.path.basename(it["video"]),
                "n_frames": nf,
                "temporal_blindness": _rowcos(c_real, c_rev),
                "feat_reldiff": float(np.linalg.norm(c_real - c_rev) / (np.linalg.norm(c_real) + 1e-8)),
                "token_collapse": _selfcos(c_real),
                "caption_order_sim": rouge_l(tokenize(cap_real), tokenize(cap_rev))["f"],
                "cap_real": cap_real, "cap_rev": cap_rev,
            })
            pooled.append(c_real.mean(0))
            enc_pooled.append(enc_real)
            caps_real.append(cap_real)
            refs.append(it.get("reference") or "")
            print(f"  [{tag}] {per[-1]['video']}: temporal_blind={per[-1]['temporal_blindness']:.3f} "
                  f"token_collapse={per[-1]['token_collapse']:.3f} "
                  f"cap_order_sim={per[-1]['caption_order_sim']:.3f}")
        except Exception:
            print(f"  [{tag}] [skip] {it['video']}\n{traceback.format_exc()}")
    free_model(model)

    # cross-video geometry: raw off-diagonal cosine (== content_collapse), the
    # common-component-removed version (xcos_cen), the shared-component energy
    # fraction (cc_frac), and the Spearman of this rep's video-to-video cosine
    # ranking against the encoder's (struct_rho). Gate on xcos_cen / cc_frac /
    # struct_rho, NOT on raw content_collapse (design doc §2.3).
    cmp_s = _pool_stats(pooled)
    enc_s = _pool_stats(enc_pooled)
    content_collapse = cmp_s["xcos_raw"]
    struct_rho = float("nan")
    if len(pooled) >= 3 and len(pooled) == len(enc_pooled):
        struct_rho = _spearman(_vv_cos(pooled), _vv_cos(enc_pooled))

    # caption specificity: own-ref rougeL vs other-refs rougeL
    spec_gap = float("nan")
    if len(caps_real) >= 2 and any(refs):
        gaps = []
        for i, cap in enumerate(caps_real):
            if not refs[i]:
                continue
            own = rouge_l(tokenize(cap), tokenize(refs[i]))["f"]
            others = [rouge_l(tokenize(cap), tokenize(refs[j]))["f"]
                      for j in range(len(refs)) if j != i and refs[j]]
            if others:
                gaps.append(own - float(np.mean(others)))
        if gaps:
            spec_gap = float(np.mean(gaps))

    def _m(k):
        v = [p[k] for p in per if isinstance(p.get(k), float) and not np.isnan(p[k])]
        return float(np.mean(v)) if v else float("nan")

    agg = {
        "n_videos": len(per),
        "temporal_blindness": _m("temporal_blindness"),
        "feat_reldiff": _m("feat_reldiff"),
        "token_collapse": _m("token_collapse"),
        "caption_order_sim": _m("caption_order_sim"),
        "content_collapse": content_collapse,
        "xcos_cen": cmp_s["xcos_cen"],
        "cc_frac": cmp_s["cc_frac"],
        "struct_rho": struct_rho,
        "xcos_cen_enc": enc_s["xcos_cen"],
        "cc_frac_enc": enc_s["cc_frac"],
        "specificity_gap": spec_gap,
    }
    return {"aggregate": agg, "per_video": per}


def verdict(a: Dict) -> List[str]:
    out = []
    tb, cc, tc = a["temporal_blindness"], a["content_collapse"], a["token_collapse"]
    cos_, sg = a["caption_order_sim"], a["specificity_gap"]
    sr = a.get("struct_rho", float("nan"))
    xc, xce = a.get("xcos_cen", float("nan")), a.get("xcos_cen_enc", float("nan"))
    cf, cfe = a.get("cc_frac", float("nan")), a.get("cc_frac_enc", float("nan"))
    if not np.isnan(tb):
        out.append(f"- temporal_blindness={tb:.3f}  "
                   + ("⚠ output nearly frame-order-invariant" if tb > 0.98
                      else "△ weak temporal sensitivity" if tb > 0.90
                      else "✓ frame order changes the output"))
    if not np.isnan(sr):
        out.append(f"- struct_rho={sr:.3f} vs encoder  "
                   + ("✓ encoder video-to-video geometry preserved" if sr >= 0.95
                      else "△ geometry partly scrambled" if sr >= 0.90
                      else "⚠ discriminative geometry scrambled (gate fail)"))
    if not np.isnan(xc) and not np.isnan(xce):
        out.append(f"- xcos_cen={xc:+.3f} (encoder {xce:+.3f})  "
                   + ("✓ tracks the encoder" if abs(xc - xce) <= 0.05
                      else "△ drifting from encoder" if abs(xc - xce) <= 0.10
                      else "⚠ centered cross-video structure lost"))
    if not np.isnan(cf) and not np.isnan(cfe):
        out.append(f"- cc_frac={cf:.3f} (encoder {cfe:.3f})  "
                   + ("⚠ pinned to one shared direction" if cf >= 0.995
                      else "△" if cf >= 0.99 or abs(cf - cfe) > 0.04
                      else "✓ near the encoder's shared-component energy"))
    if not np.isnan(cc):
        out.append(f"- content_collapse={cc:.3f} (raw xcos; ~0.9 on the encoder itself — not a gate)  "
                   + ("⚠ very high" if cc > 0.98
                      else "△" if cc > 0.95
                      else "✓"))
    if not np.isnan(tc):
        out.append(f"- token_collapse={tc:.3f}  "
                   + ("⚠ the M readout tokens are ~one vector" if tc > 0.98
                      else "✓ readout tokens are diverse" if tc < 0.9 else "△"))
    if not np.isnan(cos_):
        out.append(f"- caption_order_sim={cos_:.3f}  "
                   + ("⚠ caption ignores frame order" if cos_ > 0.85
                      else "✓ caption changes when frames are reversed"))
    if not np.isnan(sg):
        out.append(f"- specificity_gap={sg:+.3f}  "
                   + ("⚠ captions not video-specific (generic prose)" if sg < 0.03
                      else "✓ captions match their own video better than others"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="tag=path")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="work_dirs/phase2_probe/grounding")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_videos", type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=160)
    ap.add_argument("--window_size", type=int, default=24)
    ap.add_argument("--whole_video", action="store_true",
                    help="one compression part covering the whole clip (Plan-X "
                         "training geometry) instead of consecutive window_size groups")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    items = json.load(open(args.manifest))[: args.num_videos]
    results = {}
    for pair in args.models:
        tag, path = pair.split("=", 1)
        print(f"=== grounding probe: {tag} ({path}) ===")
        results[tag] = run_model(tag, path, items, args.device, args.max_frames,
                                 args.window_size, whole_video=args.whole_video)

    with open(os.path.join(args.out, "grounding.json"), "w") as f:
        json.dump(results, f, indent=1)

    lines = ["## 4. Visual grounding probe\n",
             "_Does the compressed representation depend on the video, or is the "
             "low CE just the frozen LLM autocompleting templated captions?_\n",
             "| model | struct_rho | xcos_cen (enc) | cc_frac (enc) | temporal_blindness | "
             "content_collapse | token_collapse | caption_order_sim | specificity_gap |",
             "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for tag, r in results.items():
        a = r["aggregate"]
        lines.append(f"| `{tag}` | {a['struct_rho']:.3f} | "
                     f"{a['xcos_cen']:+.3f} ({a['xcos_cen_enc']:+.3f}) | "
                     f"{a['cc_frac']:.3f} ({a['cc_frac_enc']:.3f}) | "
                     f"{a['temporal_blindness']:.3f} | {a['content_collapse']:.3f} | "
                     f"{a['token_collapse']:.3f} | {a['caption_order_sim']:.3f} | {a['specificity_gap']:+.3f} |")
    lines.append("\n_gate: struct_rho >= 0.95, xcos_cen within ~0.05 of the encoder's, "
                 "cc_frac near the encoder's (not ~1.0). raw content_collapse is ~0.9 on "
                 "the encoder itself — not a gate. lower temporal_blindness / token_collapse / "
                 "caption_order_sim is better; higher specificity_gap is better._\n")
    for tag, r in results.items():
        lines.append(f"\n**`{tag}`**")
        lines += verdict(r["aggregate"])
    md = os.path.join(args.out, "grounding.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[grounding_probe] wrote {md}")


if __name__ == "__main__":
    main()
