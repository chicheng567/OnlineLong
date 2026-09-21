#!/usr/bin/env python3
"""
Build the Phase-3 ``unified`` blend: LLaVA-shape annotations + the
``--multi_dataset`` meta registry (design doc §4 Phase 3 / §6, §5 item 14).

Why this blend exists: the Qwen3-VL re-caption pipeline samples a FIXED frame
count per clip regardless of duration, so a 420-1200 s InternVid clip's caption
carries no more temporal detail than a 60 s one. Training Phase 3 on InternVid
captions alone chases a ceiling the *data* imposes. These sources give CE actual
pressure to preserve detail:

  * **activitynet** (19,994 clips, ``annotations/activitynet/train.json`` +
    ``val_1.json``) -- real human multi-event annotation, ``timestamps`` +
    ``sentences``. This is the only source here that is not already LLaVA-shape;
    it is converted into one human/gpt pair whose answer is the event list with
    its real ``{a}s-{b}s:`` ranges, so the ``Time:{a}s-{b}s:`` the compressor
    emits per unit is trained against genuine localization rather than a
    frame-capped VLM guess.
  * **llava-video nextqa / perceptiontest / activitynetqa** -- precise-event VQA
    ("what did the man in blue do at the end of the video?"), which penalizes
    averaging away order/recency in a way a generic caption never does. Already
    LLaVA-shape; this script only path-verifies and registers them.
  * **vcd ``VDC_1k``** -- 5-turn multi-granularity detailed captioning (camera /
    short / background / main-object / detailed) per clip, far denser supervision
    than a single InternVid sentence. Built from the ORIGINAL
    ``vcd/VDC_1k.jsonl``, not ``unified/annotations/vcd/VDC_1k.json``: the
    unified copy re-ids every clip to a UUID whose ``vcd/videos/<uuid>.mp4`` path
    does not exist on disk, while the jsonl's ``video_id`` IS the on-disk file
    stem for the pixabay / pexels / panda / mixkit sources, and the ego4d ids
    resolve once ``videos/videos_3.tar.gz`` is extracted.

**Eval policy (project rule, see docs §6 "Held-out policy").** Evaluation is
never a whole external benchmark reserved wholesale, and never an in-pool
sample: this script carves a small **per-task** held-out slice out of every
sub-dataset it registers (``--heldout_frac``, seeded by ``--seed``), splitting
**by video** so no clip can appear in both. The training meta (``--meta_out``)
and the eval meta (``--eval_meta_out``) are disjoint by construction, and
``--manifest_out`` writes the eval slice as an ``eval_ablation``-style video
manifest. This is what makes VDC usable as training data: its own held-out slice
is what it is evaluated on. ``--merge``-d registries are split the same way
unless ``--no_split_merged``.

Example
-------
    python dataset_util/build_phase3_blend.py \\
        --unified_root ../datasets/unified \\
        --out_dir anno_online --meta_out anno_data/phase3_blend.json \\
        --eval_meta_out anno_data/phase3_heldout.json \\
        --merge anno_data/phase3_internvid.json

Then:

    ... --multi_dataset True --data_path anno_data/phase3_blend.json

``--probe_durations`` additionally ffprobes every registered clip into a
``{video_id: est_frames_1fps}`` map so ``--group_by_compression_depth`` /
``--durations_json`` works on the blend the same way it does on InternVid.
Deterministic given ``--seed``; re-run with ``--force`` to overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

sys.path.append("./")

# llava-video families kept (design doc §6): precise-event VQA only. The
# *_academic_v0_1 / *_youtube_v0_1 families are generic captioning -- the same
# signal InternVid already supplies in bulk -- so they are off by default.
LLAVA_FAMILIES = ("nextqa", "perceptiontest", "activitynetqa")
LLAVA_DURATION_PREFIXES = ("0_30_s", "30_60_s", "1_2_m", "2_3_m", "3_5_m")

ACTIVITYNET_PROMPT = (
    "<video>\nDescribe what happens in this video as a list of events, each with the "
    "time range it covers."
)

# VDC's five caption granularities, in the order the AuroraCap release documents
# them; prompt wording matches unified/annotations/vcd/VDC_1k.json so a model
# trained on either copy sees the same instructions.
VDC_TURNS = (
    ("camera_caption",
     "Describe the camera work in this video: shot types, angles, movements, and transitions."),
    ("short_caption",
     "Summarize this video in one detailed sentence, capturing the key action and overall mood."),
    ("background_caption",
     "Describe the background of this video in detail: objects, location, weather, time, "
     "and any dynamic elements."),
    ("main_object_caption",
     "Describe the main subject's actions, attributes, interactions, and movements throughout "
     "this video."),
    ("detailed_caption",
     "Give a detailed, comprehensive description of this video covering everything above."),
)


# ---------------------------------------------------------------- activitynet
def _index_videos(root: str, exts=(".mp4", ".mkv", ".webm")) -> Dict[str, str]:
    """``{stem: path relative to root}`` for every video under ``root``. One walk
    of the extracted tree (~20k files); activitynet ids are unique across the
    v1-2 / v1-3 splits, so a flat stem index is enough."""
    out: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                out.setdefault(stem, os.path.relpath(os.path.join(dirpath, fn), root))
    return out


def build_activitynet(unified_root: str, splits: List[str], video_index: Dict[str, str],
                      max_events: int = 0) -> Tuple[List[dict], Dict[str, int]]:
    """Dense-caption dict -> LLaVA-shape list. ``{vid: {duration, timestamps,
    sentences}}`` becomes one human/gpt pair whose answer lists every event with
    its real seconds range."""
    rows: List[dict] = []
    stats = {"entries": 0, "missing_video": 0, "no_events": 0}
    for split in splits:
        path = os.path.join(unified_root, "annotations", "activitynet", split)
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        data = json.load(open(path))
        for vid, rec in data.items():
            stats["entries"] += 1
            rel = video_index.get(vid)
            if rel is None:
                stats["missing_video"] += 1
                continue
            ts = rec.get("timestamps") or []
            sent = rec.get("sentences") or []
            pairs = [(t, s.strip()) for t, s in zip(ts, sent) if s and s.strip()]
            if not pairs:
                stats["no_events"] += 1
                continue
            pairs.sort(key=lambda p: (float(p[0][0]), float(p[0][1])))
            if max_events > 0:
                pairs = pairs[:max_events]
            lines = [f"{float(a):.1f}s-{float(b):.1f}s: {s}" for (a, b), s in pairs]
            rows.append({
                "id": vid,
                "video": rel,
                "data_source": f"activitynet/{os.path.splitext(split)[0]}",
                "duration": rec.get("duration"),
                "conversations": [
                    {"from": "human", "value": ACTIVITYNET_PROMPT},
                    {"from": "gpt", "value": "\n".join(lines)},
                ],
            })
    return rows, stats


# --------------------------------------------------------------- llava-video
def collect_llava_subsets(unified_root: str, families: Tuple[str, ...]) -> List[str]:
    root = os.path.join(unified_root, "annotations", "llava-video")
    if not os.path.isdir(root):
        return []
    keep = []
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        if any(name.endswith(f"_{fam}") for fam in families):
            keep.append(name)
    return keep


def load_llava_subset(unified_root: str, subset: str, data_root: str,
                      verify: int, rng: random.Random) -> Tuple[List[dict], Dict[str, int]]:
    """Already LLaVA-shape; ``.eval.json`` siblings are held-out splits and are
    skipped. Verifies ``verify`` sampled video paths (``-1`` = all)."""
    anno_dir = os.path.join(unified_root, "annotations", "llava-video", subset)
    rows: List[dict] = []
    for fn in sorted(os.listdir(anno_dir)):
        if not fn.endswith(".json") or fn.endswith(".eval.json"):
            continue
        data = json.load(open(os.path.join(anno_dir, fn)))
        for r in data:
            if r.get("video"):
                rows.append(r)
    stats = {"entries": len(rows), "checked": 0, "missing_video": 0}
    idx = list(range(len(rows)))
    if verify >= 0:
        idx = rng.sample(idx, min(verify, len(idx)))
    for i in idx:
        stats["checked"] += 1
        if not os.path.exists(os.path.join(data_root, rows[i]["video"])):
            stats["missing_video"] += 1
    return rows, stats


# ---------------------------------------------------------------------- vcd
def build_vcd(jsonl: str, video_index: Dict[str, str]) -> Tuple[List[dict], Dict[str, int]]:
    """``VDC_1k.jsonl`` -> one LLaVA-shape entry per clip with the five caption
    granularities as five human/gpt turns. ``video_id`` is the on-disk stem (the
    unified JSON's UUID re-id is not, see the module docstring)."""
    rows: List[dict] = []
    stats = {"entries": 0, "missing_video": 0, "no_caption": 0}
    missing_by_source: Dict[str, int] = {}
    with open(jsonl) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stats["entries"] += 1
            vid = r.get("video_id") or os.path.splitext(str(r.get("video_name", "")))[0]
            rel = video_index.get(vid)
            if rel is None:
                stats["missing_video"] += 1
                src = str(r.get("video_source", "?"))
                missing_by_source[src] = missing_by_source.get(src, 0) + 1
                continue
            convs = []
            for field, prompt in VDC_TURNS:
                text = (r.get(field) or "").strip()
                if not text:
                    continue
                q = f"<video>\n{prompt}" if not convs else prompt
                convs += [{"from": "human", "value": q}, {"from": "gpt", "value": text}]
            if not convs:
                stats["no_caption"] += 1
                continue
            rows.append({
                "id": vid,
                "video": rel,
                "data_source": f"vcd/{r.get('video_source', 'unknown')}",
                "conversations": convs,
            })
    stats["missing_by_source"] = missing_by_source           # type: ignore[assignment]
    return rows, stats


# ------------------------------------------------------ per-task held-out split
def row_stem(row: dict) -> str:
    """The row's video identity, comparable ACROSS datasets: the file stem. The
    same clip is referenced by different relative paths in different sub-datasets
    (activitynet's own mp4 vs. llava-video's `*_activitynetqa` copy of it), so a
    path-level comparison would miss the overlap."""
    v = row.get("video")
    v = v[0] if isinstance(v, list) and v else v
    return os.path.splitext(os.path.basename(str(v)))[0]


def pick_heldout_videos(rows: List[dict], frac: float, lo: int, hi: int,
                        rng: random.Random, cap_frac: float = 0.2) -> set:
    """Choose this task's held-out VIDEOS (docs §6 "Held-out policy"). By video,
    not by row: llava-video keeps several QA rows per clip and VDC five caption
    turns, so a row-level split would leak the same footage into both sides.
    Deterministic given ``rng``."""
    if frac <= 0 or not rows:
        return set()
    stems = sorted({row_stem(r) for r in rows})
    n_hold = int(round(frac * len(stems)))
    if lo > 0:
        n_hold = max(n_hold, lo)
    if hi > 0:
        n_hold = min(n_hold, hi)
    # The `lo` floor must not eat a small task: cap the slice at a fraction of the
    # task's own videos (llava_2_3_m_nextqa has 31 clips -- a flat floor of 20 would
    # leave it with almost no training rows).
    n_hold = min(n_hold, max(1, int(cap_frac * len(stems))))
    n_hold = max(0, min(n_hold, len(stems) - 1))             # never hold out everything
    return set(rng.sample(stems, n_hold)) if n_hold else set()


def partition_by_stems(rows: List[dict], held: set) -> Tuple[List[dict], List[dict]]:
    """Split one task's rows against the GLOBAL held-out video set. Applying the
    union (not just this task's own picks) is what stops the same footage being
    trained on under task A while it is evaluated under task B -- activitynet's
    clips reappear in llava-video's `*_activitynetqa` subsets."""
    train = [r for r in rows if row_stem(r) not in held]
    evalr = [r for r in rows if row_stem(r) in held]
    return train, evalr


# ------------------------------------------------------------------ durations
def _probe_one(args: Tuple[str, str]) -> Tuple[str, Optional[float]]:
    vid, path = args
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        )
        return vid, float(out.stdout.strip())
    except Exception:
        return vid, None


def probe_durations(pairs: List[Tuple[str, str]], workers: int) -> Dict[str, dict]:
    done: Dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (vid, dur) in enumerate(ex.map(_probe_one, pairs, chunksize=32), 1):
            if dur and dur > 0:
                done[vid] = {"duration_sec": dur, "est_frames_1fps": int(round(dur)), "ok": True}
            else:
                done[vid] = {"ok": False}
            if n % 5000 == 0:
                print(f"    ffprobe {n}/{len(pairs)}")
    return done


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unified_root", default="../datasets/unified")
    p.add_argument("--out_dir", default="anno_online")
    p.add_argument("--meta_out", default="anno_data/phase3_blend.json")
    p.add_argument("--merge", nargs="*", default=[],
                   help="Existing meta registries to merge into --meta_out (e.g. the "
                        "InternVid long bucket / replay streams).")
    p.add_argument("--activitynet_splits", nargs="*", default=["train.json", "val_1.json"])
    p.add_argument("--activitynet_max_events", type=int, default=0,
                   help="Truncate each clip's event list (0 = keep all).")
    p.add_argument("--llava_families", nargs="*", default=list(LLAVA_FAMILIES))
    p.add_argument("--no_activitynet", action="store_true")
    p.add_argument("--no_vcd", action="store_true",
                   help="Skip vcd VDC_1k (included by default; its own held-out slice is "
                        "what it is evaluated on -- see the module docstring).")
    p.add_argument("--vcd_root", default="../datasets/vcd")
    p.add_argument("--vcd_jsonl", default="",
                   help="Default: <vcd_root>/VDC_1k.jsonl (the ORIGINAL release, whose "
                        "video_id matches the on-disk file stem).")
    # -- per-task held-out policy (docs §6) ---------------------------------
    p.add_argument("--heldout_frac", type=float, default=0.02,
                   help="Fraction of each task's VIDEOS held out for eval (0 disables).")
    p.add_argument("--heldout_min", type=int, default=20,
                   help="Floor on the per-task held-out video count (small tasks).")
    p.add_argument("--heldout_max", type=int, default=300,
                   help="Cap on the per-task held-out video count (0 = no cap).")
    p.add_argument("--heldout_cap_frac", type=float, default=0.2,
                   help="Hard ceiling on the held-out share of ONE task's videos, so the "
                        "--heldout_min floor cannot swallow a small task.")
    p.add_argument("--eval_meta_out", default="anno_data/phase3_heldout.json",
                   help="Meta registry for the held-out slices (disjoint from --meta_out).")
    p.add_argument("--manifest_out", default="eval_ablation/manifest_phase3_heldout.json",
                   help="eval_ablation-style video manifest of the held-out slice.")
    p.add_argument("--no_split_merged", action="store_true",
                   help="Do NOT carve a held-out slice out of --merge-d registries "
                        "(they are split by default, same rule).")
    p.add_argument("--verify", type=int, default=200,
                   help="Sampled video-path existence checks per sub-dataset (-1 = all).")
    p.add_argument("--probe_durations", action="store_true",
                   help="ffprobe every registered clip -> --durations_out (for "
                        "--group_by_compression_depth).")
    p.add_argument("--durations_out", default="anno_data/phase3_blend_durations.json")
    p.add_argument("--probe_workers", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    unified = os.path.abspath(args.unified_root)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in (args.meta_out, args.eval_meta_out, args.manifest_out):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # tasks: (name, rows, data_root, extra meta keys)
    tasks: List[Tuple[str, Optional[List[dict]], str, dict]] = []

    # -- activitynet ---------------------------------------------------------
    if not args.no_activitynet:
        an_root = os.path.join(unified, "source", "activitynet", "extracted")
        print(f"[activitynet] indexing videos under {an_root} ...")
        index = _index_videos(an_root)
        print(f"[activitynet] {len(index)} video files indexed")
        rows, stats = build_activitynet(unified, args.activitynet_splits, index,
                                        args.activitynet_max_events)
        print(f"[activitynet] {len(rows)} clips  (entries={stats['entries']}, "
              f"missing_video={stats['missing_video']}, no_events={stats['no_events']})")
        if rows:
            tasks.append(("activitynet_events", rows, an_root, {}))

    # -- llava-video precise-event VQA --------------------------------------
    for subset in collect_llava_subsets(unified, tuple(args.llava_families)):
        data_root = os.path.join(unified, "source", "llava-video", subset)
        if not os.path.isdir(data_root):
            print(f"[llava-video] {subset}: no media dir at {data_root}, skipped")
            continue
        rows, stats = load_llava_subset(unified, subset, data_root, args.verify, rng)
        if not rows:
            continue
        miss = stats["missing_video"]
        flag = "" if miss == 0 else f"  !! {miss}/{stats['checked']} sampled paths MISSING"
        print(f"[llava-video] {subset}: {len(rows)} entries{flag}")
        tasks.append((f"llava_{subset}", rows, data_root, {}))

    # -- vcd VDC_1k ----------------------------------------------------------
    if not args.no_vcd:
        vcd_root = os.path.abspath(args.vcd_root)
        jsonl = args.vcd_jsonl or os.path.join(vcd_root, "VDC_1k.jsonl")
        if not os.path.exists(jsonl):
            print(f"[vcd] {jsonl} not found; skipped")
        else:
            vindex = _index_videos(vcd_root)
            rows, stats = build_vcd(jsonl, vindex)
            miss = stats["missing_video"]
            note = ""
            if miss:
                by_src = stats.get("missing_by_source") or {}
                note = (f"  ({miss} unresolved: {by_src}; ego4d ids live in "
                        f"videos/videos_3.tar.gz -- extract it to get them)")
            print(f"[vcd] {len(rows)}/{stats['entries']} clips resolved{note}")
            if rows:
                tasks.append(("vcd_vdc1k", rows, vcd_root, {}))

    # -- merged registries join the task list (split by the same rule) ------
    for m in args.merge:
        for name, cfg in json.load(open(m)).items():
            extra = {k: v for k, v in cfg.items() if k not in ("annotation", "data_root")}
            if args.no_split_merged or args.heldout_frac <= 0:
                tasks.append((name, None, cfg["data_root"], {**extra, "_as_is": cfg}))
            else:
                tasks.append((name, json.load(open(cfg["annotation"])), cfg["data_root"], extra))

    # -- per-task held-out split + write (docs §6 "Held-out policy") ---------
    # Pass 1: each task picks its OWN small slice. Pass 2: every task is split
    # against the UNION, so a clip held out for one task is never trained on
    # under another (activitynet's videos reappear in llava `*_activitynetqa`).
    held: set = set()
    for name, rows, _root, _extra in tasks:
        if rows is None:
            continue
        held |= pick_heldout_videos(rows, args.heldout_frac, args.heldout_min,
                                    args.heldout_max, random.Random(f"{args.seed}:{name}"),
                                    args.heldout_cap_frac)
    print(f"\n[split] per-task held-out slice (frac={args.heldout_frac}, min={args.heldout_min}, "
          f"max={args.heldout_max}, seed={args.seed}; by VIDEO, union across tasks) "
          f"-> {len(held)} videos held out")

    meta: Dict[str, dict] = {}
    eval_meta: Dict[str, dict] = {}
    manifest: List[dict] = []
    written: List[Tuple[str, str, str]] = []

    def _write(path: str, rows: List[dict]) -> None:
        if os.path.exists(path) and not args.force:
            print(f"    {path} exists (use --force to overwrite); reusing")
            return
        json.dump(rows, open(path, "w"), ensure_ascii=False)

    def _entry(anno_path: str, data_root: str, extra: dict) -> dict:
        e = {"annotation": os.path.abspath(anno_path), "data_root": data_root,
             "online_mode": False, "prefix_captioning": False}
        e.update({k: v for k, v in extra.items() if not k.startswith("_")})
        return e

    for name, rows, data_root, extra in tasks:
        if rows is None:                                  # --no_split_merged passthrough
            meta[name] = extra["_as_is"]
            print(f"  {name:32s} merged as-is (not split)")
            continue
        train_rows, eval_rows = partition_by_stems(rows, held)
        tr_out = os.path.join(args.out_dir, f"phase3_{name}.json")
        _write(tr_out, train_rows)
        meta[name] = _entry(tr_out, data_root, extra)
        written.append((name, tr_out, data_root))
        n_vid_e = 0
        if eval_rows:
            ev_out = os.path.join(args.out_dir, f"phase3_{name}_heldout.json")
            _write(ev_out, eval_rows)
            eval_meta[f"{name}_heldout"] = _entry(ev_out, data_root, extra)
            uniq = sorted({os.path.join(data_root, str(
                r["video"][0] if isinstance(r.get("video"), list) and r["video"] else r.get("video")
            )) for r in eval_rows})
            n_vid_e = len(uniq)
            manifest += [{"video": v, "task": name} for v in uniq]
        print(f"  {name:32s} train {len(train_rows):7d} rows | heldout {len(eval_rows):5d} rows "
              f"/ {n_vid_e:4d} videos")

    # -- registries ----------------------------------------------------------
    json.dump(meta, open(args.meta_out, "w"), indent=1)
    print(f"\n[meta] {len(meta)} TRAIN entries -> {args.meta_out}")
    if eval_meta and args.eval_meta_out:
        json.dump(eval_meta, open(args.eval_meta_out, "w"), indent=1)
        print(f"[meta] {len(eval_meta)} HELD-OUT entries -> {args.eval_meta_out}")
    if manifest and args.manifest_out:
        json.dump(manifest, open(args.manifest_out, "w"), indent=1)
        print(f"[meta] {len(manifest)} held-out videos -> {args.manifest_out}")

    # -- disjointness check (the whole point of the policy) ------------------
    def _vids(reg: Dict[str, dict]) -> set:
        # by STEM, not by path: the same clip lives under different relative paths
        # in different sub-datasets, and that is exactly the leak worth catching.
        out = set()
        for cfg in reg.values():
            for r in json.load(open(cfg["annotation"])):
                out.add(row_stem(r))
        return out

    if eval_meta:
        overlap = _vids(meta) & _vids(eval_meta)
        print(f"[check] train x heldout video overlap: {len(overlap)}"
              + ("" if not overlap else f"  !! {sorted(overlap)[:3]}"))
        if overlap:
            print("[check] FAILED -- the held-out slice is not disjoint; do not train on this.")
            return 1

    # -- optional duration scan ---------------------------------------------
    if args.probe_durations:
        pairs: List[Tuple[str, str]] = []
        seen = set()
        for name, anno_path, data_root in written:
            for r in json.load(open(anno_path)):
                v = r.get("video")
                v = v[0] if isinstance(v, list) and v else v
                if not v:
                    continue
                stem = os.path.splitext(os.path.basename(str(v)))[0]
                if stem in seen:
                    continue
                seen.add(stem)
                pairs.append((stem, os.path.join(data_root, str(v))))
        print(f"[durations] ffprobing {len(pairs)} clips with {args.probe_workers} workers ...")
        dur = probe_durations(pairs, args.probe_workers)
        json.dump(dur, open(args.durations_out, "w"))
        ok = sum(1 for d in dur.values() if d.get("ok"))
        print(f"[durations] {ok}/{len(dur)} ok -> {args.durations_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
