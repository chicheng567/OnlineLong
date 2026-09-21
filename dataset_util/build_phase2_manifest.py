#!/usr/bin/env python3
"""Build the Plan-X **Phase 2** data manifest (docs/two_stage_compression_design.md
§4 Phase 2 / §6) from the finished InternVid Qwen3-VL re-captions.

Phase 2 mixes three streams (the ``--multi_dataset`` meta format has no per-entry
weight, so the mix ratio == the annotation-file sizes):

  1. **mid bucket** -- every clip with ``180 <= duration_sec < 420`` (~273k). The
     fold's main training signal.
  2. **fold replay** -- a random slice of the Phase-1 ``< 180 s`` pool
     (``--replay_frac`` of it, ~79k at 0.30). Short clips give the *shallow*-fold
     signal (``N ~ 8-16`` -> ``U = 1-2``) and keep short-video retention.
  3. **pure-qbase replay** -- a small disjoint slice of the same ``< 180 s`` pool
     (``--qbase_frac`` of streams 1+2, ~5-8 %), emitted with ``"qbase_only": true``
     so the model routes it straight through stage-1 (``N*K`` qbase tokens, no unit
     split, no fold) -- keeps the unfrozen ``mm_projector`` / qbase anchored to the
     raw-qbase manifold.

Writes the three annotation JSONs (LLaVA list, byte-for-byte a subset of
``--anno``) and the meta registry ``--meta_out`` that
``phase2_pretrain_fold.py --multi_dataset True --data_path <meta_out>`` reads.

Durations come from the ffprobe scan (``anno_data/internVid_durations.json`` -- a
list of ``{"video_id", "duration_sec", "est_frames_1fps", "ok"}`` records).

Example
-------
    python dataset_util/build_phase2_manifest.py \
        --anno       anno_online/internvid_qwen3vl.json \
        --durations  anno_data/internVid_durations.json \
        --data_root  /share/dataset/internVid \
        --meta_out   anno_data/phase2_internvid.json

Deterministic given ``--seed``. Re-run with ``--force`` to overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

sys.path.append("./")


def _stem(v) -> str:
    if isinstance(v, list):
        v = v[0] if v else ""
    return os.path.splitext(os.path.basename(str(v)))[0]


def load_durations(path: str) -> Dict[str, float]:
    raw = json.load(open(path))
    items = raw.items() if isinstance(raw, dict) else (
        ((r.get("video_id") or r.get("video")), r) for r in raw
    )
    out: Dict[str, float] = {}
    for k, v in items:
        if k is None:
            continue
        if isinstance(v, dict):
            if v.get("ok") is False:
                continue
            d = v.get("duration_sec")
            if d is None and v.get("est_frames_1fps"):
                d = float(v["est_frames_1fps"])          # fps=1 -> frames ~= seconds
        else:
            d = v
        if d is not None:
            out[_stem(k)] = float(d)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--anno", default="anno_online/internvid_qwen3vl.json",
                   help="Full re-caption annotation (LLaVA list) to slice from.")
    p.add_argument("--durations", default="anno_data/internVid_durations.json",
                   help="ffprobe duration scan (list of records OR {clip: sec} map).")
    p.add_argument("--out_dir", default="anno_online",
                   help="Directory for the three sliced annotation JSONs.")
    p.add_argument("--meta_out", default="anno_data/phase2_internvid.json",
                   help="Meta registry written for --multi_dataset True.")
    p.add_argument("--data_root", default="/share/dataset/internVid",
                   help="Directory the `video` paths are relative to (into --meta_out).")
    p.add_argument("--prefix", default="internvid_qwen3vl",
                   help="Basename prefix for the sliced annotation files.")
    p.add_argument("--mid_lo", type=float, default=180.0)
    p.add_argument("--mid_hi", type=float, default=420.0)
    p.add_argument("--mid_cap", type=int, default=0,
                   help="Subsample the mid bucket to this many clips (0 = keep all).")
    p.add_argument("--replay_frac", type=float, default=0.30,
                   help="Fold-replay count = this * |< mid_lo pool|.")
    p.add_argument("--replay_cap", type=int, default=0)
    p.add_argument("--qbase_frac", type=float, default=0.07,
                   help="Pure-qbase-replay count = this * (mid + fold-replay). Drawn "
                        "disjoint from the fold-replay slice.")
    p.add_argument("--qbase_cap", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260910)
    p.add_argument("--verify_videos", action="store_true",
                   help="Drop entries whose <data_root>/<video> is missing.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output files.")
    return p.parse_args()


def _dump(path: Path, obj, force: bool) -> None:
    if path.exists() and not force:
        sys.exit(f"refusing to overwrite {path} (pass --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def main() -> None:
    a = parse_args()
    rng = random.Random(a.seed)

    anno: List[dict] = json.load(open(a.anno))
    dur = load_durations(a.durations)
    print(f"[phase2-manifest] anno={len(anno):,}  durations={len(dur):,}")

    if a.verify_videos:
        n0 = len(anno)
        anno = [e for e in anno if os.path.exists(os.path.join(a.data_root, _stem(e["video"]) + ".mp4"))
                or os.path.exists(os.path.join(a.data_root, str(e["video"] if not isinstance(e["video"], list) else e["video"][0])))]
        print(f"[phase2-manifest] verify_videos: dropped {n0 - len(anno):,} missing")

    short, mid, long, unk = [], [], [], 0
    for e in anno:
        d = dur.get(_stem(e["video"]))
        if d is None:
            unk += 1
            continue
        if d < a.mid_lo:
            short.append(e)
        elif d < a.mid_hi:
            mid.append(e)
        elif d <= 1200.0:
            long.append(e)
    print(f"[phase2-manifest] buckets: <{a.mid_lo:g}s={len(short):,}  "
          f"{a.mid_lo:g}-{a.mid_hi:g}s={len(mid):,}  {a.mid_hi:g}-1200s={len(long):,}  "
          f"no_duration={unk:,}  (Phase 3 uses the 420-1200 bucket)")

    # --- mid bucket ---
    rng.shuffle(mid)
    if a.mid_cap and a.mid_cap < len(mid):
        mid = mid[: a.mid_cap]

    # --- fold replay: replay_frac of the short pool ---
    rng.shuffle(short)
    n_replay = int(round(a.replay_frac * len(short)))
    if a.replay_cap:
        n_replay = min(n_replay, a.replay_cap)
    n_replay = min(n_replay, len(short))
    replay = short[:n_replay]

    # --- pure-qbase replay: qbase_frac of (mid + replay), drawn disjoint from replay ---
    rest_short = short[n_replay:]
    n_qbase = int(round(a.qbase_frac * (len(mid) + len(replay))))
    if a.qbase_cap:
        n_qbase = min(n_qbase, a.qbase_cap)
    n_qbase = min(n_qbase, len(rest_short))
    qbase = rest_short[:n_qbase]

    out_dir = Path(a.out_dir)
    f_mid = out_dir / f"{a.prefix}_mid_{int(a.mid_lo)}_{int(a.mid_hi)}.json"
    f_replay = out_dir / f"{a.prefix}_lt{int(a.mid_lo)}_replay.json"
    f_qbase = out_dir / f"{a.prefix}_qbase_replay.json"
    _dump(f_mid, mid, a.force)
    _dump(f_replay, replay, a.force)
    _dump(f_qbase, qbase, a.force)

    total = len(mid) + len(replay) + len(qbase)
    def pct(n):
        return f"{100.0 * n / total:.1f}%" if total else "-"
    print(f"[phase2-manifest] wrote:")
    print(f"  {f_mid}            {len(mid):>8,}  ({pct(len(mid))})")
    print(f"  {f_replay}    {len(replay):>8,}  ({pct(len(replay))})")
    print(f"  {f_qbase}       {len(qbase):>8,}  ({pct(len(qbase))})  [qbase_only]")
    print(f"  total steps / epoch (per-sample) ~= {total:,}")

    root = os.path.abspath(a.data_root)
    # Registry keys track --mid_lo/--mid_hi (they used to be hardcoded to the
    # 180/420 defaults, so a Phase-3 420-1200 build produced a meta whose key names
    # contradicted the files they pointed at).
    k_mid = f"internvid_mid_{int(a.mid_lo)}_{int(a.mid_hi)}"
    k_replay = f"internvid_replay_lt{int(a.mid_lo)}"
    k_qbase = f"internvid_qbase_replay_lt{int(a.mid_lo)}"
    meta = {
        k_mid: {
            "annotation": str(f_mid.resolve()),
            "data_root": root, "online_mode": False, "prefix_captioning": False,
        },
        k_replay: {
            "annotation": str(f_replay.resolve()),
            "data_root": root, "online_mode": False, "prefix_captioning": False,
        },
        k_qbase: {
            "annotation": str(f_qbase.resolve()),
            "data_root": root, "online_mode": False, "prefix_captioning": False,
            "qbase_only": True,
        },
    }
    _dump(Path(a.meta_out), meta, a.force)
    print(f"[phase2-manifest] meta -> {a.meta_out}  (3 datasets; {k_qbase} has qbase_only=true)")
    print(f"[phase2-manifest] point training at it:\n"
          f"    ... --multi_dataset True --data_path {a.meta_out} \\\n"
          f"        --durations_json {a.durations} --group_by_compression_depth True")


if __name__ == "__main__":
    main()
