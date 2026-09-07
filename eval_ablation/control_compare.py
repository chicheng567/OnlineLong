#!/usr/bin/env python3
"""Training-geometry control comparison for the Stage-2 runs.

Re-runs the grounding + caption checks at the SAME geometry the compressors were
trained under (dynamic HW, per-video token budget 16384, fps=1, max 160 frames,
each model partitioned the way it was trained) and adds two controls:

  * ``base``  -- raw VideoLLaMA3, NO compression (upper bound on caption quality).
  * ``qbase`` -- the Stage-1 qbase alone (single whole-video window -> 64 tokens),
                 the known-good single-stage compressor.
  * ``stage2a`` / ``stage2b`` -- our runs (select_stage2_units -> <=5x64 tokens).

Per model: greedy caption vs the reference (rougeL / rouge1 / bleu4), a
video-specificity gap (own-ref rougeL minus mean other-ref rougeL) and -- for the
compressor models -- the grounding metrics from grounding_probe.py
(temporal_blindness, token_collapse, content_collapse, caption_order_sim).

    PYTHONPATH=. python eval_ablation/control_compare.py \
        --manifest work_dirs/stage2_analysis/manifest.json \
        --base pretrained_models/videollama3_7b_local \
        --qbase pretrained_models/compressor_pretrain_video_norm \
        --stage2a work_dirs/stage2a_fold_videoxl \
        --stage2b work_dirs/stage2_unfreeze_full \
        --out work_dirs/stage2_analysis/control --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from videollama3.constants import DEFAULT_IMAGE_TOKEN  # noqa: E402
from videollama3.model import Videollama3Qwen2ForCausalLM  # noqa: E402
from videollama3.model.processor import Videollama3Processor, DEFAULT_CHAT_TEMPLATE  # noqa: E402
from videollama3.model.videollama3_arch import _grid_hw_for_compression_parts  # noqa: E402
from videollama3.mm_utils import load_video  # noqa: E402
from eval_ablation.common import _build_ts_info  # noqa: E402
from eval_ablation.metrics import rouge_l, rouge_n, bleu, tokenize  # noqa: E402
from videollama3.train.data.compressor import select_full_compression_parts  # noqa: E402
from videollama3.train.stage2a_pretrain_compressor_fold import select_stage2_units  # noqa: E402

GEN = dict(do_sample=False, num_beams=1, max_new_tokens=220, repetition_penalty=1.1)
PROMPT = ("Describe this video in detail: the main subjects, what happens in order, "
          "the setting, and any camera movement. One factual paragraph.")
MAX_FRAMES = 160
NATIVE_MAX_TOKENS = 16384          # == training per-video budget
FRAMES_PER_SEGMENT, SEGS_PER_UNIT, MAX_UNITS, M = 4, 6, 5, 64


def _load(path: str, device: str):
    try:
        m = Videollama3Qwen2ForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    except TypeError:
        m = Videollama3Qwen2ForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    m.config.use_cache = True
    m.to(device).eval()
    return m


def _proc(path: str) -> Videollama3Processor:
    p = Videollama3Processor.from_pretrained(path, trust_remote_code=False, fix_mistral_regex=True)
    if p.tokenizer.pad_token is None and p.tokenizer.unk_token is not None:
        p.tokenizer.pad_token = p.tokenizer.unk_token
    # The base checkpoint ships a chat template that needs an `image_token` kwarg
    # (jinja UndefinedError otherwise); the trained checkpoints hardcode `<image>`.
    # Force the hardcoded one everywhere so all 4 models see identical prompts.
    p.chat_template = DEFAULT_CHAT_TEMPLATE
    p.tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE
    p.image_processor.force_size = None                 # dynamic HW
    p.image_processor.max_tokens = NATIVE_MAX_TOKENS
    return p


def _pack(proc, video_path, device):
    frames, ts = load_video(video_path, fps=1, max_frames=MAX_FRAMES)
    nf = len(frames)
    ts = [float(t) for t in ts]
    conv = [{"role": "user", "content": [
        {"type": "video", "timestamps": ts, "num_frames": nf},
        {"type": "text", "text": PROMPT}]}]
    inp = proc(images=[frames], text=conv, merge_size=2, return_tensors="pt")
    img_id = proc.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    tvt = int((inp["input_ids"] == img_id).sum())
    if nf == 0 or tvt % nf != 0:
        raise RuntimeError(f"{video_path}: {tvt} vis tokens / {nf} frames")
    return inp, nf, ts, tvt


def _parts_for(kind: str, nf: int, tvt: int):
    if kind == "base":
        return None
    if kind == "qbase":
        return [[0, tvt]]                                # one whole-video window
    p = select_stage2_units(nf, tvt, FRAMES_PER_SEGMENT, SEGS_PER_UNIT, MAX_UNITS, min_part_tokens=M)
    return p or [[0, tvt]]


def _reverse_frames(pv: torch.Tensor, nf: int) -> torch.Tensor:
    if pv.shape[0] % nf != 0:
        return pv
    tpf = pv.shape[0] // nf
    return pv.view(nf, tpf, *pv.shape[1:]).flip(0).reshape_as(pv)


@torch.no_grad()
def _gen(model, proc, inp, pv, parts, ts_info, device):
    kw = dict(input_ids=inp["input_ids"].to(device),
              pixel_values=pv.to(device=device, dtype=torch.bfloat16),
              grid_sizes=inp["grid_sizes"].to(device),
              merge_sizes=inp["merge_sizes"].to(device),
              modals=["video"], **GEN)
    if parts is not None:
        kw["compression_parts"] = parts
        kw["compression_ts_info"] = ts_info
    out = model.generate(**kw)
    return proc.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


@torch.no_grad()
def _compress_feat(model, inp, pv, parts, device):
    m = model.get_model()
    mm = m.get_vision_encoder()(pixel_values=pv.to(device=device, dtype=torch.bfloat16),
                                grid_sizes=inp["grid_sizes"].to(device),
                                merge_sizes=inp["merge_sizes"].to(device))
    ghw = _grid_hw_for_compression_parts(parts, inp["grid_sizes"].to(device),
                                         inp["merge_sizes"].to(device))
    c = model.compress_visual_tokens_with_compressor(mm.clone(), parts, ghw)
    return c.detach().float().cpu().numpy()


def _rowcos(a, b):
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    return float(np.mean((a * b).sum(-1) /
                 (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)))


def _selfcos(x, cap=512):
    x = x[:cap]; x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    g = x @ x.T; iu = np.triu_indices(len(x), k=1)
    return float(np.mean(g[iu])) if len(iu[0]) else float("nan")


def _cap_metrics(cand: str, ref: str) -> Dict[str, float]:
    c, r = tokenize(cand), tokenize(ref)
    return {"rougeL": rouge_l(c, r)["f"], "rouge1": rouge_n(c, r, 1)["f"],
            "bleu4": bleu(c, r, 4), "words": float(len(c))}


def run_model(kind: str, path: str, items: List[Dict], device: str) -> Dict:
    print(f"\n=== {kind}  ({path}) ===")
    proc = _proc(path)
    model = _load(path, device)
    is_comp = kind in ("qbase", "stage2a", "stage2b")
    per, caps, refs, pooled = [], [], [], []
    for it in items:
        row: Dict = {"video": os.path.basename(it["video"])}
        try:
            inp, nf, ts, tvt = _pack(proc, it["video"], device)
            parts = _parts_for(kind, nf, tvt)
            ts_info = _build_ts_info(ts, parts, nf, tvt, proc.tokenizer) if parts else None
            pv = inp["pixel_values"]
            cap = _gen(model, proc, inp, pv, parts, ts_info, device)
            row.update(_cap_metrics(cap, it.get("reference") or ""))
            row["n_frames"] = nf
            row["n_tokens_to_llm"] = (int(M * len(parts)) if (parts and kind != "qbase")
                                      else (M if kind == "qbase" else tvt))
            row["caption"] = cap
            caps.append(cap); refs.append(it.get("reference") or "")
            if is_comp:
                pv_rev = _reverse_frames(pv, nf)
                cap_rev = _gen(model, proc, inp, pv_rev, parts, ts_info, device)
                row["caption_order_sim"] = rouge_l(tokenize(cap), tokenize(cap_rev))["f"]
                c_real = _compress_feat(model, inp, pv, parts, device)
                c_rev = _compress_feat(model, inp, pv_rev, parts, device)
                row["temporal_blindness"] = _rowcos(c_real, c_rev)
                row["token_collapse"] = _selfcos(c_real)
                pooled.append(c_real.mean(0))
            print(f"  {row['video']}: rougeL={row['rougeL']:.3f} "
                  + (f"tblind={row.get('temporal_blindness', float('nan')):.3f} "
                     f"order_sim={row.get('caption_order_sim', float('nan')):.3f}" if is_comp else "")
                  + f" ({int(row['words'])}w, {row['n_tokens_to_llm']} vis tok)")
        except Exception:
            row["error"] = traceback.format_exc().splitlines()[-1]
            print(f"  {row['video']}: SKIP {row['error']}")
        per.append(row)
    try:
        model.to("cpu"); del model; torch.cuda.empty_cache()
    except Exception:
        pass

    ok = [p for p in per if "error" not in p]
    def _m(k):
        v = [p[k] for p in ok if isinstance(p.get(k), float) and not np.isnan(p[k])]
        return float(np.mean(v)) if v else float("nan")
    agg = {"n": len(ok), "rougeL": _m("rougeL"), "rouge1": _m("rouge1"),
           "bleu4": _m("bleu4"), "words": _m("words"),
           "n_tokens_to_llm": float(np.mean([p["n_tokens_to_llm"] for p in ok])) if ok else float("nan")}
    if is_comp:
        agg.update(temporal_blindness=_m("temporal_blindness"),
                   token_collapse=_m("token_collapse"),
                   caption_order_sim=_m("caption_order_sim"))
        if len(pooled) >= 2:
            P = np.stack(pooled); P = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-8)
            G = P @ P.T; iu = np.triu_indices(len(P), k=1)
            agg["content_collapse"] = float(np.mean(G[iu]))
    # specificity gap
    gaps = []
    for i, cap in enumerate(caps):
        if not refs[i]:
            continue
        own = rouge_l(tokenize(cap), tokenize(refs[i]))["f"]
        oth = [rouge_l(tokenize(cap), tokenize(refs[j]))["f"] for j in range(len(refs)) if j != i and refs[j]]
        if oth:
            gaps.append(own - float(np.mean(oth)))
    agg["specificity_gap"] = float(np.mean(gaps)) if gaps else float("nan")
    return {"aggregate": agg, "per_video": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", default="pretrained_models/videollama3_7b_local")
    ap.add_argument("--qbase", default="pretrained_models/compressor_pretrain_video_norm")
    ap.add_argument("--stage2a", default="work_dirs/stage2a_fold_videoxl")
    ap.add_argument("--stage2b", default="work_dirs/stage2_unfreeze_full")
    ap.add_argument("--out", default="work_dirs/stage2_analysis/control")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_videos", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    items = json.load(open(args.manifest))[: args.num_videos]

    models = [("base", args.base), ("qbase", args.qbase),
              ("stage2a", args.stage2a), ("stage2b", args.stage2b)]
    res = {}
    for kind, path in models:
        if not os.path.exists(path):
            print(f"[skip] {kind}: {path} missing"); continue
        try:
            res[kind] = run_model(kind, path, items, args.device)
        except Exception:
            res[kind] = {"aggregate": {}, "per_video": [], "fatal": traceback.format_exc()}
            print(f"[{kind}] FATAL\n{res[kind]['fatal']}")

    json.dump(res, open(os.path.join(args.out, "control.json"), "w"), indent=1)

    cols = ["n_tokens_to_llm", "rougeL", "rouge1", "bleu4", "words",
            "specificity_gap", "temporal_blindness", "caption_order_sim",
            "token_collapse", "content_collapse"]
    L = ["# Training-geometry control comparison\n",
         f"geometry: dynamic HW, per-video budget {NATIVE_MAX_TOKENS}, fps=1, max_frames={MAX_FRAMES}, greedy\n",
         "| model | " + " | ".join(cols) + " |",
         "|---|" + "---|" * len(cols)]
    for kind, _ in models:
        if kind not in res:
            continue
        a = res[kind]["aggregate"]
        L.append(f"| `{kind}` | " + " | ".join(
            (f"{a[c]:.3f}" if isinstance(a.get(c), float) and not np.isnan(a[c])
             else (str(int(a[c])) if c == "n_tokens_to_llm" and isinstance(a.get(c), float) else "-"))
            for c in cols) + " |")
    L += ["",
          "_rougeL/rouge1/bleu4/specificity_gap: higher = better. "
          "temporal_blindness/caption_order_sim/token_collapse/content_collapse: lower = better._",
          "",
          "Read: if `base`/`qbase` caption these videos well but `stage2a`/`stage2b` do not, "
          "the Stage-2 training regressed the representation (not an eval-geometry artefact)."]
    # side by side
    L.append("\n## captions (first 4 videos)\n")
    for i, it in enumerate(items[:4]):
        L.append(f"### {os.path.basename(it['video'])}")
        L.append(f"> **ref:** {(it.get('reference') or '')[:400]}\n")
        for kind, _ in models:
            if kind not in res or i >= len(res[kind]["per_video"]):
                continue
            pv = res[kind]["per_video"][i]
            L.append(f"**{kind}**: {pv.get('caption', pv.get('error', '-'))[:400]}\n")
    md = os.path.join(args.out, "control.md")
    open(md, "w").write("\n".join(L) + "\n")
    print(f"\n[control_compare] wrote {md}")


if __name__ == "__main__":
    main()
