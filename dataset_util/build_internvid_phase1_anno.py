#!/usr/bin/env python3
"""Phase-1 (qbase, <180 s) InternVid annotation + meta.

Filters ``anno_online/internvid_qwen3vl.json`` to clips whose ffprobed duration is
< --max_seconds (default 180) and whose .mp4 is on disk, then writes:
  * <anno_out>       LLaVA-shape annotation (subset, same schema as the input)
  * <meta_out>       anno_data-style meta: { name: {annotation, data_root, ...} }

Usage:
  python dataset_util/build_internvid_phase1_anno.py \
    --anno anno_online/internvid_qwen3vl.json \
    --durations anno_data/internVid_durations.json \
    --data_root /share/dataset/internVid \
    --max_seconds 180 \
    --anno_out anno_online/internvid_qwen3vl_lt180.json \
    --meta_out anno_data/internvid_qwen3vl_lt180.json
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anno", default="anno_online/internvid_qwen3vl.json")
    p.add_argument("--durations", default="anno_data/internVid_durations.json")
    p.add_argument("--data_root", default="/share/dataset/internVid")
    p.add_argument("--max_seconds", type=float, default=180.0)
    p.add_argument("--min_seconds", type=float, default=0.0)
    p.add_argument("--anno_out", default="anno_online/internvid_qwen3vl_lt180.json")
    p.add_argument("--meta_out", default="anno_data/internvid_qwen3vl_lt180.json")
    p.add_argument("--name", default=None, help="meta dataset name (default: anno_out stem)")
    p.add_argument("--limit", type=int, default=0, help="cap kept entries (debug)")
    p.add_argument("--require_on_disk", action="store_true", default=True)
    p.add_argument("--no_require_on_disk", dest="require_on_disk", action="store_false")
    args = p.parse_args()

    dur = {}
    for d in json.load(open(args.durations)):
        if d.get("ok") and d.get("duration_sec") is not None:
            dur[d["video_id"]] = float(d["duration_sec"])
        # also key by filename for robustness
        if d.get("video"):
            dur.setdefault(os.path.splitext(d["video"])[0], dur.get(d.get("video_id"), None))

    anno = json.load(open(args.anno))
    kept, no_dur, out_of_range, missing = [], 0, 0, 0
    for e in anno:
        vid = os.path.splitext(os.path.basename(e["video"]))[0]
        s = dur.get(vid)
        if s is None:
            no_dur += 1
            continue
        if not (args.min_seconds <= s < args.max_seconds):
            out_of_range += 1
            continue
        if args.require_on_disk and not os.path.exists(os.path.join(args.data_root, e["video"])):
            missing += 1
            continue
        kept.append(e)
        if args.limit and len(kept) >= args.limit:
            break

    os.makedirs(os.path.dirname(args.anno_out) or ".", exist_ok=True)
    json.dump(kept, open(args.anno_out, "w"))
    name = args.name or os.path.splitext(os.path.basename(args.anno_out))[0]
    meta = {name: {
        "annotation": os.path.abspath(args.anno_out),
        "data_root": args.data_root,
        "online_mode": False,
        "prefix_captioning": False,
    }}
    os.makedirs(os.path.dirname(args.meta_out) or ".", exist_ok=True)
    json.dump(meta, open(args.meta_out, "w"), indent=1)

    print(f"input entries      : {len(anno):,}")
    print(f"kept (<{args.max_seconds:g}s, on disk): {len(kept):,}")
    print(f"  dropped no duration : {no_dur:,}")
    print(f"  dropped out of range: {out_of_range:,}")
    print(f"  dropped missing file: {missing:,}")
    print(f"anno_out : {args.anno_out}")
    print(f"meta_out : {args.meta_out}  (name={name})")


if __name__ == "__main__":
    main()
