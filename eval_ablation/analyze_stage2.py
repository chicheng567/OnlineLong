#!/usr/bin/env python3
"""Post-training analysis for the two Stage-2 runs (2a frozen-qbase fold, then the
joint-polish run that unfreezes the qbase on the full videoxl video set).

Produces ``<out>/REPORT.md`` from three parts, each degrading gracefully:

  1. LOSS CURVES  -- parsed from every run's ``trainer_state.json``: start / end /
     min CE, slope over the last 25 %, grad-norm range, LR schedule sanity. Always
     runs (pure JSON).
  2. COLLAPSE GUARDRAIL  -- ``feature_distribution.py`` on the Stage-2 readout of
     each run: mean pairwise cosine, effective rank, PCA participation ratio,
     ``cos(real, shuffled)``. The design-doc gate (low shuffle-cosine, high
     effrank == not collapsed). Dynamic HW (``--force_image_size 0``).
  3. CAPTION ABILITY  -- ``caption_eval.py`` rougeL / bleu4 vs the reference
     caption, per run. Shows the compressed representation still supports the LLM.

Parts 2/3 are subprocess calls wrapped so a failure lands in the report instead of
aborting it.

    PYTHONPATH=. python eval_ablation/analyze_stage2.py \
        --runs stage2a=work_dirs/stage2a_fold_videoxl \
               stage2b=work_dirs/stage2_unfreeze_full \
        --out work_dirs/stage2_analysis --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
from typing import Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
# 1. loss curves
# --------------------------------------------------------------------------- #
def _load_history(run_dir: str) -> List[Dict]:
    cands = [os.path.join(run_dir, "trainer_state.json")]
    if os.path.isdir(run_dir):
        cands += [
            os.path.join(run_dir, d, "trainer_state.json")
            for d in sorted(os.listdir(run_dir), reverse=True)
            if d.startswith("checkpoint-")
        ]
    for cand in cands:
        if os.path.isfile(cand):
            with open(cand) as f:
                return json.load(f).get("log_history", [])
    return []


def _series(history: List[Dict], key: str):
    xs, ys = [], []
    for e in history:
        if key in e and "step" in e:
            xs.append(e["step"]); ys.append(e[key])
    return xs, ys


def loss_section(runs: Dict[str, str]) -> str:
    out = ["## 1. Loss curves\n"]
    plot_runs = {}
    for tag, d in runs.items():
        hist = _load_history(d)
        if not hist:
            out.append(f"### `{tag}`  ({d})\n\n_no trainer_state.json / log_history found._\n")
            continue
        sx, sy = _series(hist, "loss")
        if not sy:
            for k in ("loss_partial", "train_loss"):
                sx, sy = _series(hist, k)
                if sy:
                    break
        gx, gy = _series(hist, "grad_norm")
        lx, ly = _series(hist, "learning_rate")
        if not sy:
            out.append(f"### `{tag}`\n\n_history has no loss field. keys seen: "
                       f"{sorted({k for e in hist for k in e})}_\n")
            continue
        plot_runs[tag] = (sx, sy)
        n = len(sy)
        tail = sy[max(0, int(n * 0.75)):]
        head = sy[:max(1, int(n * 0.1))]
        slope = (sum(tail) / len(tail)) - (sum(head) / len(head))
        out.append(textwrap.dedent(f"""\
            ### `{tag}`  ({d})

            | metric | value |
            |---|---|
            | logged steps | {n} (step {sx[0]}..{sx[-1]}) |
            | start CE (first 10%) | {sum(head)/len(head):.4f} |
            | end CE (last 25%) | {sum(tail)/len(tail):.4f} |
            | min CE | {min(sy):.4f} @ step {sx[sy.index(min(sy))]} |
            | net change (end - start) | {slope:+.4f} {'✓ decreasing' if slope < -1e-3 else '⚠ flat/rising'} |
            | grad_norm range | {min(gy):.2f} .. {max(gy):.2f} | {'' if not gy else ''}
            | final LR / peak LR | {ly[-1]:.2e} / {max(ly):.2e} |
            """))
    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for tag, (sx, sy) in plot_runs.items():
            # light smoothing
            w = max(1, len(sy) // 60)
            sm = [sum(sy[max(0, i-w):i+1]) / len(sy[max(0, i-w):i+1]) for i in range(len(sy))]
            ax.plot(sx, sm, label=tag, lw=1.5)
        ax.set_xlabel("optimizer step"); ax.set_ylabel("CE loss"); ax.legend(); ax.grid(alpha=.3)
        ax.set_title("Stage-2 training loss")
        p = os.path.join(OUT, "loss_curves.png")
        fig.tight_layout(); fig.savefig(p, dpi=110)
        out.append(f"\n![loss curves](loss_curves.png)\n")
    except Exception as e:  # noqa: BLE001
        out.append(f"\n_(loss plot skipped: {e})_\n")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# small held-out manifest from the training annotations
# --------------------------------------------------------------------------- #
def build_manifest(path: str, n: int = 12) -> Optional[str]:
    srcs = [
        ("sharegpt4v", "/root/datasets/videoxl/Finetuning/sharegpt4v.json"),
        ("gpt4o_video", "/root/datasets/videoxl/Finetuning/gpt4o_video.json"),
        ("baaicaption", "/root/datasets/videoxl/Finetuning/baaicaption.json"),
        ("vcg_20k", "/root/datasets/videoxl/Finetuning/vcg_20k.json"),
    ]
    root = "/root/datasets/videoxl/Finetuning"
    rng = random.Random(1234)
    rows: List[Dict] = []
    per = max(1, n // len(srcs))
    for tag, jp in srcs:
        if not os.path.isfile(jp):
            continue
        data = json.load(open(jp))
        picks = data[-2000:] if len(data) > 2000 else data          # tail == least likely early in an epoch
        rng.shuffle(picks)
        got = 0
        for s in picks:
            v = s.get("video")
            v = v[0] if isinstance(v, list) else v
            vp = os.path.join(root, v) if v else None
            if not vp or not os.path.exists(vp):
                continue
            conv = s.get("conversations", [])
            hu = next((c["value"] for c in conv if c.get("from") in ("human", "user")), "")
            gp = next((c["value"] for c in conv if c.get("from") in ("gpt", "assistant")), "")
            rows.append({
                "video": vp,
                "prompt": hu.replace("<image>", "").replace("<video>", "").strip(),
                "reference": gp.strip(),
                "source": tag,
            })
            got += 1
            if got >= per:
                break
    if not rows:
        return None
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    return path


# --------------------------------------------------------------------------- #
# 2 / 3. subprocess evals
# --------------------------------------------------------------------------- #
def run_capture(cmd: List[str], log_path: str) -> tuple[int, str]:
    env = dict(os.environ, PYTHONPATH=".")
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    tail = ""
    try:
        with open(log_path) as lf:
            tail = "".join(lf.readlines()[-40:])
    except Exception:  # noqa: BLE001
        pass
    return p.returncode, tail


def feature_section(runs: Dict[str, str], manifest: Optional[str], device: str) -> str:
    if not manifest:
        return "## 2. Collapse guardrail\n\n_skipped: could not build a manifest._\n"
    model_args = [f"{t}={d}" for t, d in runs.items()]
    md = os.path.join(OUT, "feature_distribution", "feature_distribution.md")
    rc, tail = run_capture(
        [sys.executable, "eval_ablation/feature_distribution.py",
         "--models", *model_args, "--manifest", manifest,
         "--num_videos", "10", "--out", os.path.join(OUT, "feature_distribution"),
         "--device", device, "--max_frames", "160", "--window_size", "24",
         "--force_image_size", "0"],
        os.path.join(OUT, "feature_distribution.log"))
    body = ""
    if os.path.isfile(md):
        body = open(md).read()
    return (f"## 2. Collapse guardrail  (feature_distribution.py, exit {rc})\n\n"
            + (body if body else f"```\n{tail}\n```\n"))


def grounding_section(runs: Dict[str, str], manifest: Optional[str], device: str) -> str:
    if not manifest:
        return "## 4. Visual grounding probe\n\n_skipped: could not build a manifest._\n"
    model_args = [f"{t}={d}" for t, d in runs.items()]
    md = os.path.join(OUT, "grounding", "grounding.md")
    rc, tail = run_capture(
        [sys.executable, "eval_ablation/grounding_probe.py",
         "--models", *model_args, "--manifest", manifest,
         "--out", os.path.join(OUT, "grounding"), "--device", device,
         "--num_videos", "8", "--max_frames", "160", "--window_size", "24"],
        os.path.join(OUT, "grounding.log"))
    body = open(md).read() if os.path.isfile(md) else ""
    return (f"## 4. Visual grounding probe  (grounding_probe.py, exit {rc})\n\n"
            + (body if body else f"```\n{tail}\n```\n"))


def caption_section(runs: Dict[str, str], manifest: Optional[str], device: str) -> str:
    if not manifest:
        return "## 3. Caption ability\n\n_skipped: could not build a manifest._\n"
    model_args = [f"{t}={d}" for t, d in runs.items()]
    md = os.path.join(OUT, "caption_eval", "caption_eval.md")
    rc, tail = run_capture(
        [sys.executable, "eval_ablation/caption_eval.py",
         "--models", *model_args, "--manifest", manifest,
         "--out", os.path.join(OUT, "caption_eval"), "--device", device,
         "--max_frames", "160", "--window_size", "24",
         "--force_image_size", "0", "--repetition_penalty", "1.1"],
        os.path.join(OUT, "caption_eval.log"))
    body = open(md).read() if os.path.isfile(md) else ""
    return (f"## 3. Caption ability  (caption_eval.py, exit {rc})\n\n"
            + (body if body else f"```\n{tail}\n```\n"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="tag=run_dir pairs")
    ap.add_argument("--out", default="work_dirs/stage2_analysis")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip_evals", action="store_true",
                    help="only the loss-curve section (no model loading)")
    args = ap.parse_args()

    global OUT
    OUT = os.path.join(REPO, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(OUT, exist_ok=True)

    runs: Dict[str, str] = {}
    for pair in args.runs:
        assert "=" in pair, f"--runs entries must be tag=dir, got {pair!r}"
        t, d = pair.split("=", 1)
        runs[t] = d if os.path.isabs(d) else os.path.join(REPO, d)

    parts = ["# Stage-2 training analysis\n",
             f"runs: " + ", ".join(f"`{t}` = `{d}`" for t, d in runs.items()) + "\n"]
    parts.append(loss_section(runs))

    if not args.skip_evals:
        manifest = build_manifest(os.path.join(OUT, "manifest.json"), n=12)
        parts.append(feature_section(runs, manifest, args.device))
        parts.append(grounding_section(runs, manifest, args.device))
        parts.append(caption_section(runs, manifest, args.device))
    else:
        parts.append("_(model evals skipped: --skip_evals)_\n")

    report = os.path.join(OUT, "REPORT.md")
    with open(report, "w") as f:
        f.write("\n\n---\n\n".join(parts) + "\n")
    print(f"[analyze_stage2] wrote {report}")


if __name__ == "__main__":
    main()
