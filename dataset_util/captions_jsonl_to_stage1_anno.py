#!/usr/bin/env python3
"""
Turn a re-caption JSONL (``recaption_videos.py`` / ``recaption_videos_vllm.py``
output, one ``{"video", "prompt", "caption", ...}`` record per line) into a
Stage-1 compressor-pretrain annotation.

Stage-1 (``videollama3/train/compressor_pretrain_with_videollama3.py``) reads
LLaVA-shape entries through ``LazySupervisedDataset._convert_normal``:

    {
      "video": "<id>.mp4",
      "prompt_name": "...",              # carried through, ignored by training
      "recaption_model": "...",          # carried through, ignored by training
      "conversations": [
        {"from": "human", "value": "<video>\\n<prompt>"},
        {"from": "gpt",   "value": "<caption>"}
      ]
    }

This is byte-for-byte the same shape ``recaption_videos_vllm.py --annotation_out``
emits for a ``--video_dir`` run; this script exists so the annotation can be
rebuilt from the JSONL alone (after filtering, or when only the per-rank shards
survived), and so the ``anno_data/`` meta registry that ``--multi_dataset True``
needs is produced in the same step.

The compression path takes one whole-video window per sample, so multi-turn
records are out of scope: every line becomes exactly one human/gpt pair.

Example
-------
    python dataset_util/captions_jsonl_to_stage1_anno.py \
        --captions recaption/qwen3vl/captions.jsonl \
        --anno_out anno_online/internvid_qwen3vl.json \
        --meta_out anno_data/internvid_qwen3vl.json \
        --data_root /share/dataset/internVid

Point the Phase-1 / Phase-2 training at the meta file:

    ... --multi_dataset True --data_path anno_data/internvid_qwen3vl.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append("./")

from videollama3.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--captions", required=True,
                   help="Input re-caption JSONL (recaption_videos*.py --output_file).")
    p.add_argument("--anno_out", required=True,
                   help="Output annotation JSON (anno_online/ style).")
    p.add_argument("--meta_out", default=None,
                   help="Also write an anno_data-style meta registry pointing at --anno_out. "
                        "Needs --data_root.")
    p.add_argument("--data_root", default=None,
                   help="Directory the `video` paths are relative to. Written into --meta_out and, "
                        "with --verify_videos, used to drop entries whose file is missing.")
    p.add_argument("--dataset_name", default=None,
                   help="Key for the single dataset in --meta_out (default: --anno_out stem).")
    p.add_argument("--modal", choices=["video", "image"], default="video",
                   help="Media type of the samples; picks the <video> / <image> tag (default: video).")
    p.add_argument("--online_mode", action="store_true",
                   help="Set online_mode=true in --meta_out (default false: the _convert_normal path).")
    p.add_argument("--prefix_captioning", action="store_true",
                   help="Set prefix_captioning=true in --meta_out (default false).")
    p.add_argument("--verify_videos", action="store_true",
                   help="Drop entries whose <data_root>/<video> does not exist (needs --data_root).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --anno_out if it already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    captions_path = Path(args.captions)
    anno_path = Path(args.anno_out)
    if not captions_path.exists():
        sys.exit(f"--captions {captions_path} does not exist.")
    if anno_path.exists() and not args.force:
        sys.exit(f"--anno_out {anno_path} exists; pass --force to overwrite.")
    if args.verify_videos and not args.data_root:
        sys.exit("--verify_videos needs --data_root.")
    if args.meta_out and not args.data_root:
        sys.exit("--meta_out needs --data_root.")

    modal_token = DEFAULT_VIDEO_TOKEN if args.modal == "video" else DEFAULT_IMAGE_TOKEN

    n_total = 0
    n_blank = 0
    n_dup = 0
    n_missing = 0
    by_video: Dict[str, Dict] = {}  # last record for a given video path wins

    with captions_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            rec = json.loads(line)

            video = rec.get("video") or rec.get("video_id")
            prompt = (rec.get("prompt") or "").strip()
            caption = (rec.get("caption") or "").strip()
            if not video or not prompt or not caption:
                n_blank += 1
                continue

            if video in by_video:
                n_dup += 1
            entry: Dict = {"video": video}
            if rec.get("prompt_name"):
                entry["prompt_name"] = rec["prompt_name"]
            if rec.get("model"):
                entry["recaption_model"] = rec["model"]
            # Per-frame times, if the JSONL carried real ones. `_convert_normal`
            # ignores this field; it is kept only so a later pass can filter on it.
            if rec.get("timestamps") is not None:
                entry["frame_timestamps"] = rec["timestamps"]
                if rec.get("num_frames") is not None:
                    entry["num_frames"] = rec["num_frames"]
                if rec.get("timestamps_synthetic"):
                    entry["synthetic_timestamps"] = True
            entry["conversations"] = [
                {"from": "human", "value": f"{modal_token}\n{prompt}"},
                {"from": "gpt", "value": caption},
            ]
            by_video[video] = entry

    entries: List[Dict] = list(by_video.values())

    if args.verify_videos:
        kept = []
        for e in entries:
            if os.path.exists(os.path.join(args.data_root, e["video"])):
                kept.append(e)
            else:
                n_missing += 1
        entries = kept

    anno_path.parent.mkdir(parents=True, exist_ok=True)
    with anno_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"Read {n_total} JSONL record(s) from {captions_path}")
    if n_blank:
        print(f"  - skipped {n_blank} with an empty video / prompt / caption")
    if n_dup:
        print(f"  - {n_dup} duplicate video path(s) collapsed (last record wins)")
    if n_missing:
        print(f"  - dropped {n_missing} whose file is missing under {args.data_root}")
    print(f"Wrote {anno_path} with {len(entries)} sample(s)")

    if args.meta_out:
        name = args.dataset_name or anno_path.stem
        meta = {
            name: {
                "annotation": str(anno_path),
                "data_root": args.data_root,
                "online_mode": bool(args.online_mode),
                "prefix_captioning": bool(args.prefix_captioning),
            }
        }
        meta_path = Path(args.meta_out)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"Wrote {meta_path} (dataset '{name}', data_root {args.data_root})")


if __name__ == "__main__":
    main()
