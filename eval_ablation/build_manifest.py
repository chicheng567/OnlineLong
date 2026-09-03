#!/usr/bin/env python3
"""
Pick a handful of real videos from ../datasets/videoxl for the caption eval and
write eval_ablation/manifest.json:  [{video, prompt, reference, source, duration}].

The dataset's own gold caption is kept as `reference` (for lexical-overlap
metrics); `prompt` is forced to the compressor's training instruction so
generation stays on distribution. Only videos whose duration falls in
[--min_dur, --max_dur] are kept, so `--max_frames 64` @ 1 fps is meaningful.

Run once (or with --num / --min_dur to regenerate):
    python eval_ablation/build_manifest.py --num 12
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_ablation.common import TRAIN_CAPTION_PROMPT  # noqa: E402

VIDEOXL = "/root/datasets/videoxl/Finetuning"
EXTRACTED = os.path.join(VIDEOXL, "data/_extracted")
INDEX_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_video_index.json")

# annotation files, tried round-robin so the manifest is content-diverse
SOURCES_CAPTION = ["sharegpt4v.json", "vcg_20k.json"]                 # have gold captions
SOURCES_DIVERSE = ["sharegpt4v.json", "vcg_20k.json", "ego_4d.json",
                   "anomaly_det.json", "cinepine_30k.json"]           # refs may be QA, fine for the probe


def build_index() -> dict:
    if os.path.exists(INDEX_CACHE):
        return json.load(open(INDEX_CACHE))
    print(f"[index] walking {EXTRACTED} ...")
    idx = {}
    for base, _, files in os.walk(EXTRACTED):
        for f in files:
            if f.lower().endswith((".mp4", ".mkv", ".webm", ".mov")):
                idx.setdefault(f, os.path.join(base, f))
    json.dump(idx, open(INDEX_CACHE, "w"))
    print(f"[index] {len(idx)} videos, cached to {INDEX_CACHE}")
    return idx


def duration(path: str):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


def gold_caption(conv) -> str:
    for c in conv:
        if c.get("from") in ("gpt", "assistant"):
            return c["value"].strip()
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=12)
    ap.add_argument("--min_dur", type=float, default=45.0)
    ap.add_argument("--max_dur", type=float, default=150.0)
    ap.add_argument("--min_ref_words", type=int, default=60)
    ap.add_argument("--diverse", action="store_true",
                    help="round-robin over 5 datasets (probe set); refs may be QA/empty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "manifest.json"))
    args = ap.parse_args()

    random.seed(args.seed)
    idx = build_index()
    picked, seen = [], set()

    sources = SOURCES_DIVERSE if args.diverse else SOURCES_CAPTION
    pools = []
    for anno_file in sources:
        p = os.path.join(VIDEOXL, anno_file)
        if not os.path.exists(p):
            print(f"[skip] {p} not found")
            continue
        data = json.load(open(p))
        random.shuffle(data)
        pools.append((anno_file, iter(data)))

    # round-robin: one accepted video per source per lap until --num reached
    active = True
    while active and len(picked) < args.num:
        active = False
        for anno_file, it in pools:
            if len(picked) >= args.num:
                break
            for e in it:                       # advance this source to its next acceptable video
                active = True
                v = e.get("video")
                if not v:
                    continue
                b = os.path.basename(v)
                real = idx.get(b)
                if not real or real in seen:
                    continue
                ref = gold_caption(e.get("conversations", []))
                if len(ref.split()) < args.min_ref_words:
                    continue
                d = duration(real)
                if d is None or not (args.min_dur <= d <= args.max_dur):
                    continue
                seen.add(real)
                picked.append({
                    "video": real,
                    "prompt": TRAIN_CAPTION_PROMPT,
                    "reference": ref,
                    "source": anno_file.replace(".json", ""),
                    "duration": round(d, 1),
                })
                print(f"  + {b}  dur={d:.1f}s  ref_words={len(ref.split())}  ({anno_file})")
                break

    if not picked:
        raise SystemExit("Nothing picked -- loosen --min_dur/--max_dur/--min_ref_words.")
    json.dump(picked, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\n[manifest] {len(picked)} videos -> {args.out}")


if __name__ == "__main__":
    main()
