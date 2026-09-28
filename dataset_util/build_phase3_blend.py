#!/usr/bin/env python3
"""
Build the Phase-3 blend: ``/root/datasets`` (unified + vcd) is the PRIMARY
source, InternVid is capped to a minority share (design doc §4 Phase 3 / §6).

Why this shape (rebuilt 2026-09-21, supersedes the InternVid-dominated build):
the Qwen3-VL re-caption pipeline pins EVERY clip to 64 frames regardless of
duration (``shell/recaption_vllm.sh`` ``NUM_FRAMES=64``), so a 678 s InternVid
clip's caption is written from ~0.05 fps -- the lowest sampling density of any
annotation on disk -- yet the previous build let it carry 98.7 % of the CE token
mass. ``/root/datasets`` holds 278k clips with far denser supervision that were
either partly used (activitynet, 4 % of llava-video) or not used at all
(videoxl / videoxl_pro, 70k clips). This builder registers all of it and demotes
InternVid to ``--merge_share``.

**InternVid cannot go to zero.** It is the only source of 420-1200 s length on
this machine: outside it the whole of ``/root/datasets`` has ~170 clips >= 420 s
(measured). So it stays as the long-bucket supplier, just not as the bulk of the
gradient. Raise ``--merge_share`` if a run needs deeper folds.

Sources
-------
* **activitynet** (`unified/annotations/activitynet/`) -- human multi-event
  `timestamps` + `sentences`, converted to one event-list answer with real
  `{a}s-{b}s:` ranges. The only source that trains the `Time:{a}s-{b}s:` string
  the compressor emits per unit against genuine localization.
* **llava-video, ALL subsets** (`unified/annotations/llava-video/`) -- the
  previous build kept only `nextqa`/`perceptiontest`/`activitynetqa`
  (57k rows, mean answer 10-18 chars: letter-only MCQ) and dropped the
  `*_academic_v0_1` / `*_youtube_v0_1` families, which is where all the dense
  supervision lives: 176,954 GPT-4o recursive captions (mean 2,019-3,010 chars)
  and 952,396 open-ended QA rows (mean ~410 chars). All families are now in.
* **videoxl / videoxl_pro** (`unified/annotations/videoxl*/`) -- previously
  0 % used, media 100 % on disk via `source/videoxl_pro/media/`:
  ShareGPT4Video dense captions (37,769), `time_sft_opens`' ActivityNet slice
  (29,896 rows of "the given query happens in a - b seconds" temporal
  grounding), BAAI captions, CinePile MCQ + VICO event-ordering, VideoChatGPT
  full-video descriptions, ShareGPT-4o captions, Ego4D, anomaly detection.
  Image-only files (`bunny_union`, `ocrvqa_19k`, `Pretraining/pretrain.json`)
  and `finevideo_43k` (0 % of its media on disk) are skipped.
* **vcd `VDC_1k`** -- 5-turn multi-granularity captioning, built from the
  ORIGINAL `vcd/VDC_1k.jsonl` (the `unified` copy re-ids to UUIDs that do not
  exist on disk).
* **InternVid**, via ``--merge`` -- subsampled to ``--merge_share`` of the final
  blend, each stream keeping its relative proportion.

Turn merging (``--max_turns``)
------------------------------
llava-video keeps 1 caption + ~5.5 open-ended + ~1 MCQ rows per clip in separate
files; a row-per-QA registry would decode and vision-encode the same clip ~7x
per epoch. Rows sharing a video are merged into ONE multi-turn sample (caption
first, then QA), so a clip is decoded once and the whole supervision lands in a
single CE forward. ``<video>`` is emitted on the first human turn only.

Video identity (``--heldout_*``)
--------------------------------
The held-out split is by VIDEO, unioned across tasks. Identity is the file
**stem** when the stem is distinctive, and the **realpath** when it is not:
`ShareGPT-4o/pvideo/video_10001.mp4` and
`perception_test/videos/video_10001.mp4` are different videos with the same
stem (1,511 such collisions), while `vqa/Activity_Videos/v_09MaNbzc2TA.mp4` and
activitynet's own copy are the same video behind two paths. Stem-only would
merge the first pair; realpath-only would split the second.

``--protect_from`` keeps clips that Phase 1/2 already trained on out of the
held-out slice (the previous build put 192/300 and 209/300 of the InternVid
replay held-out clips inside the Phase-2 training pool).

Example
-------
    python dataset_util/build_phase3_blend.py \\
        --datasets_root /root/datasets \\
        --merge anno_data/phase3_internvid.json --merge_share 0.35 \\
        --protect_from anno_data/phase2_internvid.json anno_data/internvid_qwen3vl_lt180.json \\
        --probe_durations --force

Then:  ... --multi_dataset True --data_path anno_data/phase3_blend.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

sys.path.append("./")

VIDEO_TAG = "<video>"

# A stem that carries no identity of its own -- `video_10001`, `0001`, `clip-7`.
# These repeat across corpora as DIFFERENT videos, so they fall back to realpath.
# The length floor matters: NextQA ids ARE bare 10-11 digit numbers, and the same
# clip is shipped as two separate physical copies (`0_30_s_nextqa/NextQA/NExTVideo/
# 1202/3264772244.mp4` and `0_30_s_academic_v0_1/academic_source/NextQA/1202/
# 3264772244.mp4`, identical bytes, different inodes), so realpath cannot merge
# them and the stem has to.
GENERIC_STEM = re.compile(r"^(?:video|vid|clip|sample|v|img|frame)[-_]?\d+$", re.I)
MIN_DISTINCTIVE_STEM = 8

ACTIVITYNET_PROMPT = (
    "Describe what happens in this video as a list of events, each with the "
    "time range it covers."
)

# VDC's five caption granularities, in the order the AuroraCap release documents them.
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

# videoxl_pro finetune files, grouped by the media pool they share so clips that
# carry two kinds of supervision (CinePile MCQ + VICO event ordering) become one
# multi-turn sample instead of two decodes. Files are listed caption/context
# first -- turn merging keeps the FIRST pair when it has to drop some.
# Skipped on purpose: bunny_union / ocrvqa_19k (images), finevideo_43k (0 % of
# its media is on disk -- see annotations/videoxl_pro/MEDIA_NOTE.md).
VXLP_TASKS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("vxlp_sharegpt4video", ("sharegpt4v_38k.json",)),
    ("vxlp_baaicaption",    ("baaicaption_10k.json",)),
    ("vxlp_vcg",            ("vcg_union.json",)),
    ("vxlp_gpt4o_video",    ("gpt4o_video_2k.json",)),
    ("vxlp_time_grounding", ("time_sft_opens.json",)),
    ("vxlp_cinepile",       ("cinepine_union.json", "VICO.json")),
    ("vxlp_ego4d",          ("ego_4d_0.7k.json",)),
)
VXL_TASKS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("vxl_anomaly_det", ("anomaly_det.json",)),
)

# llava-video file kinds, in turn-merge priority order: the dense caption is the
# turn that must survive a --max_turns trim, the letter-only MCQ is the first to go.
LLAVA_KIND_ORDER = ("cap", "oe", "mc", "other")

# Not duration-bucketed datasets: `llava_hound` rows are not LLaVA-shape, and the
# two `gpt4o_*_prompt` dirs hold the release's prompt templates, not annotations.
LLAVA_SKIP_DIRS = ("llava_hound", "gpt4o_caption_prompt", "gpt4o_qa_prompt")


# --------------------------------------------------------------- video identity
def video_identity(data_root: str, video_rel: str) -> str:
    """Identity of a clip, comparable ACROSS sub-datasets. See the module
    docstring: distinctive stem, else realpath."""
    stem = os.path.splitext(os.path.basename(str(video_rel)))[0]
    if len(stem) >= MIN_DISTINCTIVE_STEM and not GENERIC_STEM.match(stem):
        return f"stem:{stem}"
    return f"path:{os.path.realpath(os.path.join(data_root, str(video_rel)))}"


def row_video(row: dict) -> Optional[str]:
    v = row.get("video")
    v = v[0] if isinstance(v, list) and v else v
    return str(v) if v else None


# ---------------------------------------------------------------- turn merging
def _qa_pairs(conversations: List[dict]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    i = 0
    while i + 1 < len(conversations):
        h, g = conversations[i], conversations[i + 1]
        if h.get("from") in ("human", "system") and g.get("from") == "gpt":
            pairs.append((str(h.get("value") or ""), str(g.get("value") or "")))
            i += 2
        else:
            i += 1
    return pairs


def merge_rows_by_video(rows: List[dict], task: str, max_turns: int,
                        seed: int) -> List[dict]:
    """Rows sharing a video -> one multi-turn sample. Input order decides turn
    order, so callers load the caption file first. The first pair is always kept."""
    by_vid: "OrderedDict[str, List[dict]]" = OrderedDict()
    for r in rows:
        v = row_video(r)
        if v:
            by_vid.setdefault(v, []).append(r)

    out: List[dict] = []
    for vid, group in by_vid.items():
        pairs: List[Tuple[str, str]] = []
        for r in group:
            pairs += _qa_pairs(r.get("conversations") or [])
        pairs = [(h, g) for h, g in pairs if g.strip()]
        if not pairs:
            continue
        if max_turns > 0 and len(pairs) > max_turns:
            rng = random.Random(f"{seed}:{task}:{vid}")
            keep = sorted(rng.sample(range(1, len(pairs)), max_turns - 1))
            pairs = [pairs[0]] + [pairs[i] for i in keep]
        convs: List[dict] = []
        for j, (h, g) in enumerate(pairs):
            h = h.replace("<image>", "").replace(VIDEO_TAG, "").strip()
            if j == 0:
                h = f"{VIDEO_TAG}\n{h}" if h else VIDEO_TAG
            convs += [{"from": "human", "value": h}, {"from": "gpt", "value": g}]
        out.append({
            "id": f"{task}/{os.path.splitext(os.path.basename(vid))[0]}",
            "video": vid,
            "data_source": task,
            "conversations": convs,
        })
    return out


# ------------------------------------------------------------------ media check
def keep_resolvable(rows: List[dict], data_root: str) -> Tuple[List[dict], int]:
    """Drop rows whose media is not on disk. Checked once per unique video."""
    seen: Dict[str, bool] = {}
    out = []
    for r in rows:
        v = row_video(r)
        if not v:
            continue
        ok = seen.get(v)
        if ok is None:
            ok = os.path.exists(os.path.join(data_root, v))
            seen[v] = ok
        if ok:
            out.append(r)
    return out, sum(1 for ok in seen.values() if not ok)


# ---------------------------------------------------------------- activitynet
def _index_videos(root: str, exts=(".mp4", ".mkv", ".webm")) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            if ext.lower() in exts:
                out.setdefault(stem, os.path.relpath(os.path.join(dirpath, fn), root))
    return out


def build_activitynet(unified: str, splits: List[str], video_index: Dict[str, str],
                      max_events: int = 0) -> Tuple[List[dict], Dict[str, int]]:
    rows: List[dict] = []
    stats = {"entries": 0, "missing_video": 0, "no_events": 0}
    for split in splits:
        path = os.path.join(unified, "annotations", "activitynet", split)
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        for vid, rec in json.load(open(path)).items():
            stats["entries"] += 1
            rel = video_index.get(vid)
            if rel is None:
                stats["missing_video"] += 1
                continue
            ts, sent = rec.get("timestamps") or [], rec.get("sentences") or []
            pairs = [(t, s.strip()) for t, s in zip(ts, sent) if s and s.strip()]
            if not pairs:
                stats["no_events"] += 1
                continue
            pairs.sort(key=lambda p: (float(p[0][0]), float(p[0][1])))
            if max_events > 0:
                pairs = pairs[:max_events]
            lines = [f"{float(x):.1f}s-{float(y):.1f}s: {s}" for (x, y), s in pairs]
            rows.append({
                "id": f"activitynet_events/{vid}",
                "video": rel,
                "data_source": "activitynet_events",
                "duration": rec.get("duration"),
                "conversations": [
                    {"from": "human", "value": f"{VIDEO_TAG}\n{ACTIVITYNET_PROMPT}"},
                    {"from": "gpt", "value": "\n".join(lines)},
                ],
            })
    return rows, stats


# ----------------------------------------------------------------------- vcd
def build_vcd(jsonl: str, video_index: Dict[str, str]) -> Tuple[List[dict], Dict[str, int]]:
    rows: List[dict] = []
    stats = {"entries": 0, "missing_video": 0, "no_caption": 0}
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
                continue
            convs: List[dict] = []
            for field, prompt in VDC_TURNS:
                text = (r.get(field) or "").strip()
                if not text:
                    continue
                q = f"{VIDEO_TAG}\n{prompt}" if not convs else prompt
                convs += [{"from": "human", "value": q}, {"from": "gpt", "value": text}]
            if not convs:
                stats["no_caption"] += 1
                continue
            rows.append({"id": f"vcd_vdc1k/{vid}", "video": rel,
                         "data_source": f"vcd/{r.get('video_source', 'unknown')}",
                         "conversations": convs})
    return rows, stats


# --------------------------------------------------------------- llava-video
def llava_kind(filename: str) -> str:
    if "_cap_" in filename:
        return "cap"
    if "_oe_" in filename:
        return "oe"
    if "_mc_" in filename:
        return "mc"
    return "other"


def load_llava_subset(unified: str, subset: str) -> List[dict]:
    """All of a subset's annotation files, caption first. ``.eval.json`` siblings
    are the release's own held-out splits and are never training data."""
    anno_dir = os.path.join(unified, "annotations", "llava-video", subset)
    files = [fn for fn in sorted(os.listdir(anno_dir))
             if fn.endswith(".json") and not fn.endswith(".eval.json")]
    files.sort(key=lambda fn: (LLAVA_KIND_ORDER.index(llava_kind(fn)), fn))
    rows: List[dict] = []
    for fn in files:
        data = json.load(open(os.path.join(anno_dir, fn)))
        if not isinstance(data, list):
            continue
        rows += [r for r in data if isinstance(r, dict) and r.get("video")]
    return rows


def collect_llava_subsets(unified: str, families: Optional[Tuple[str, ...]]) -> List[str]:
    root = os.path.join(unified, "annotations", "llava-video")
    if not os.path.isdir(root):
        return []
    keep = []
    for name in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, name)) or name in LLAVA_SKIP_DIRS:
            continue
        if families and not any(name.endswith(f"_{fam}") for fam in families):
            continue
        keep.append(name)
    return keep


def load_plain(paths: List[str]) -> List[dict]:
    rows: List[dict] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"  [skip] {p} not found")
            continue
        data = json.load(open(p))
        if isinstance(data, list):
            rows += [r for r in data if isinstance(r, dict) and r.get("video")]
    return rows


# ------------------------------------------------------ per-task held-out split
def pick_heldout(ids: List[str], frac: float, lo: int, hi: int, cap_frac: float,
                 rng: random.Random, protected: set) -> set:
    if frac <= 0 or not ids:
        return set()
    uniq = set(ids)
    pool = sorted(uniq - protected)
    if not pool:
        return set()
    n_hold = int(round(frac * len(uniq)))
    if lo > 0:
        n_hold = max(n_hold, lo)
    if hi > 0:
        n_hold = min(n_hold, hi)
    n_hold = min(n_hold, max(1, int(cap_frac * len(uniq))))
    n_hold = max(0, min(n_hold, len(pool) - 1))
    return set(rng.sample(pool, n_hold)) if n_hold > 0 else set()


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


def load_duration_seed(paths: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        raw = json.load(open(p))
        items = raw.items() if isinstance(raw, dict) else (
            ((r.get("video_id") or r.get("video")), r) for r in raw)
        for k, v in items:
            if not k:
                continue
            if isinstance(v, dict):
                if v.get("ok") is False:
                    continue
                d = v.get("duration_sec")
            else:
                d = v
            if d and float(d) > 0:
                out[str(k)] = {"duration_sec": float(d),
                               "est_frames_1fps": int(round(float(d))), "ok": True}
        print(f"  [durations] seeded {len(out)} after {p}")
    return out


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets_root", default="/root/datasets")
    p.add_argument("--unified_root", default="", help="Default: <datasets_root>/unified")
    p.add_argument("--vcd_root", default="", help="Default: <datasets_root>/vcd")
    p.add_argument("--out_dir", default="anno_online/phase3",
                   help="Annotations are written to <out_dir>/train/<task>.json and "
                        "<out_dir>/heldout/<task>.json -- one file per registry entry, "
                        "which is what --multi_dataset and the per-task held-out policy "
                        "require. See anno_online/README.md.")
    p.add_argument("--meta_out", default="anno_data/phase3_blend.json")
    p.add_argument("--eval_meta_out", default="anno_data/phase3_heldout.json")
    p.add_argument("--manifest_out", default="eval_ablation/manifest_phase3_heldout.json")
    # -- sources ------------------------------------------------------------
    p.add_argument("--activitynet_splits", nargs="*", default=["train.json", "val_1.json"])
    p.add_argument("--activitynet_max_events", type=int, default=0)
    p.add_argument("--llava_families", nargs="*", default=[],
                   help="Restrict llava-video to these subset suffixes (default: ALL "
                        "subsets -- the academic/youtube families are where the dense "
                        "captions and open-ended QA live).")
    p.add_argument("--no_activitynet", action="store_true")
    p.add_argument("--no_llava", action="store_true")
    p.add_argument("--no_videoxl", action="store_true")
    p.add_argument("--no_vcd", action="store_true")
    p.add_argument("--vcd_jsonl", default="", help="Default: <vcd_root>/VDC_1k.jsonl")
    p.add_argument("--max_turns", type=int, default=6,
                   help="Cap on QA pairs per merged sample (0 = keep all). The first pair "
                        "(the dense caption, for llava-video) always survives.")
    p.add_argument("--no_merge_turns", action="store_true",
                   help="Keep one sample per annotation row (a clip is then decoded once "
                        "per QA -- ~7x for llava-video).")
    # -- InternVid (secondary) ----------------------------------------------
    p.add_argument("--merge", nargs="*", default=[],
                   help="Meta registries merged in as SECONDARY streams (InternVid).")
    p.add_argument("--merge_share", type=float, default=0.35,
                   help="Target share of the final train rows taken by --merge-d "
                        "registries. Each stream keeps its relative proportion. "
                        "<=0 or >=1 disables the cap.")
    # -- held-out policy (docs §6) ------------------------------------------
    p.add_argument("--heldout_frac", type=float, default=0.02)
    p.add_argument("--heldout_min", type=int, default=20)
    p.add_argument("--heldout_max", type=int, default=300)
    p.add_argument("--heldout_cap_frac", type=float, default=0.2)
    p.add_argument("--protect_from", nargs="*", default=[],
                   help="Meta registries whose clips must NOT be chosen as held-out "
                        "(they were already trained on -- Phase 1/2 InternVid pools).")
    # -- durations ----------------------------------------------------------
    p.add_argument("--probe_durations", action="store_true")
    p.add_argument("--durations_seed", nargs="*",
                   default=["anno_data/internVid_durations.json"],
                   help="Existing {stem: duration} maps reused instead of re-probing.")
    p.add_argument("--durations_out", default="anno_data/phase3_durations.json")
    p.add_argument("--probe_workers", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                   help="Print the composition, write nothing.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    unified = os.path.abspath(a.unified_root or os.path.join(a.datasets_root, "unified"))
    vcd_root = os.path.abspath(a.vcd_root or os.path.join(a.datasets_root, "vcd"))
    merge_turns = not a.no_merge_turns

    train_dir = os.path.join(a.out_dir, "train")
    eval_dir = os.path.join(a.out_dir, "heldout")
    if not a.dry_run:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        for path in (a.meta_out, a.eval_meta_out, a.manifest_out, a.durations_out):
            if path:
                os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)

    # tasks: [name, rows, data_root, extra, is_merged]
    tasks: List[List] = []

    def add(name: str, rows: List[dict], data_root: str, extra: dict = None,
            premerged: bool = False) -> None:
        rows, n_missing = keep_resolvable(rows, data_root)
        if not rows:
            print(f"  {name:34s} 0 resolvable rows, skipped")
            return
        n_raw = len(rows)
        if merge_turns and not premerged:
            rows = merge_rows_by_video(rows, name, a.max_turns, a.seed)
        vids = len({row_video(r) for r in rows})
        turns = sum(len(r.get("conversations", [])) // 2 for r in rows)
        note = f"  (dropped {n_missing} videos not on disk)" if n_missing else ""
        print(f"  {name:34s} rows {n_raw:>8d} -> {len(rows):>8d} samples | "
              f"{vids:>7d} videos | {turns:>8d} turns{note}")
        tasks.append([name, rows, data_root, dict(extra or {}), False])

    # -- activitynet ---------------------------------------------------------
    if not a.no_activitynet:
        an_root = os.path.join(unified, "source", "activitynet", "extracted")
        print(f"[activitynet] indexing {an_root} ...")
        rows, st = build_activitynet(unified, a.activitynet_splits,
                                     _index_videos(an_root), a.activitynet_max_events)
        print(f"[activitynet] {len(rows)} clips (entries={st['entries']}, "
              f"missing_video={st['missing_video']}, no_events={st['no_events']})")
        add("activitynet_events", rows, an_root, premerged=True)

    # -- llava-video (ALL subsets by default) -------------------------------
    if not a.no_llava:
        print("[llava-video]")
        for subset in collect_llava_subsets(unified, tuple(a.llava_families) or None):
            data_root = os.path.join(unified, "source", "llava-video", subset)
            if not os.path.isdir(data_root):
                print(f"  {subset:34s} no media dir, skipped")
                continue
            add(f"llava_{subset}", load_llava_subset(unified, subset), data_root)

    # -- videoxl / videoxl_pro ----------------------------------------------
    if not a.no_videoxl:
        print("[videoxl_pro]")
        vxlp_root = os.path.join(unified, "source", "videoxl_pro", "media")
        vxlp_anno = os.path.join(unified, "annotations", "videoxl_pro", "finetune")
        for name, files in VXLP_TASKS:
            add(name, load_plain([os.path.join(vxlp_anno, f) for f in files]), vxlp_root)
        print("[videoxl]")
        vxl_root = os.path.join(unified, "source", "videoxl", "Finetuning")
        vxl_anno = os.path.join(unified, "annotations", "videoxl", "Finetuning")
        for name, files in VXL_TASKS:
            add(name, load_plain([os.path.join(vxl_anno, f) for f in files]), vxl_root)

    # -- vcd VDC_1k ----------------------------------------------------------
    if not a.no_vcd:
        jsonl = a.vcd_jsonl or os.path.join(vcd_root, "VDC_1k.jsonl")
        if not os.path.exists(jsonl):
            print(f"[vcd] {jsonl} not found; skipped")
        else:
            rows, st = build_vcd(jsonl, _index_videos(vcd_root))
            print(f"[vcd] {len(rows)}/{st['entries']} clips resolved "
                  f"(missing={st['missing_video']})")
            add("vcd_vdc1k", rows, vcd_root, premerged=True)   # already multi-turn

    primary_rows = sum(len(t[1]) for t in tasks)

    # -- InternVid, capped to --merge_share ---------------------------------
    merged: List[List] = []
    for m in a.merge:
        for name, cfg in json.load(open(m)).items():
            extra = {k: v for k, v in cfg.items() if k not in ("annotation", "data_root")}
            merged.append([name, json.load(open(cfg["annotation"])), cfg["data_root"],
                           extra, True])
    if merged:
        total_merged = sum(len(t[1]) for t in merged)
        budget = total_merged
        if 0 < a.merge_share < 1 and primary_rows:
            budget = int(round(a.merge_share / (1.0 - a.merge_share) * primary_rows))
        print(f"\n[merge] {len(merged)} secondary streams, {total_merged:,} rows "
              f"-> budget {min(budget, total_merged):,} "
              f"(share {a.merge_share:g} of {primary_rows:,} primary rows)")
        for t in merged:
            n = min(len(t[1]), int(round(budget * len(t[1]) / max(1, total_merged))))
            if n < len(t[1]):
                rng = random.Random(f"{a.seed}:merge:{t[0]}")
                idx = sorted(rng.sample(range(len(t[1])), n))
                t[1] = [t[1][i] for i in idx]
            print(f"  {t[0]:34s} -> {len(t[1]):>8d} samples")
            tasks.append(t)

    # -- identities ----------------------------------------------------------
    print("\n[identity] resolving video identities ...")
    ident: Dict[int, List[str]] = {}
    for ti, (name, rows, data_root, _extra, _mg) in enumerate(tasks):
        cache: Dict[str, str] = {}
        ids = []
        for r in rows:
            v = row_video(r)
            k = cache.get(v)
            if k is None:
                k = video_identity(data_root, v)
                cache[v] = k
            ids.append(k)
        ident[ti] = ids

    protected: set = set()
    for p in a.protect_from:
        for _n, cfg in json.load(open(p)).items():
            root = cfg["data_root"]
            for r in json.load(open(cfg["annotation"])):
                v = row_video(r)
                if v:
                    protected.add(video_identity(root, v))
    if protected:
        print(f"[protect] {len(protected)} clips already trained on in "
              f"{len(a.protect_from)} registries -> never held out")

    # -- per-task held-out picks, applied as a UNION -------------------------
    held: set = set()
    for ti, (name, _rows, _root, _extra, _mg) in enumerate(tasks):
        held |= pick_heldout(ident[ti], a.heldout_frac, a.heldout_min, a.heldout_max,
                             a.heldout_cap_frac,
                             random.Random(f"{a.seed}:{name}"), protected)
    print(f"[split] held out {len(held)} videos (frac={a.heldout_frac}, "
          f"min={a.heldout_min}, max={a.heldout_max}, seed={a.seed}; union across tasks)")

    # -- write ---------------------------------------------------------------
    meta: Dict[str, dict] = {}
    eval_meta: Dict[str, dict] = {}
    manifest: List[dict] = []
    written: List[Tuple[str, str, str]] = []

    def _entry(anno_path: str, data_root: str, extra: dict) -> dict:
        e = {"annotation": os.path.abspath(anno_path), "data_root": data_root,
             "online_mode": False, "prefix_captioning": False}
        e.update({k: v for k, v in extra.items() if not k.startswith("_")})
        return e

    def _write(path: str, rows: List[dict]) -> None:
        if a.dry_run:
            return
        if os.path.exists(path) and not a.force:
            print(f"    {path} exists (use --force); reusing")
            return
        json.dump(rows, open(path, "w"), ensure_ascii=False)

    print("\n[write]")
    n_train = n_eval = 0
    for ti, (name, rows, data_root, extra, _mg) in enumerate(tasks):
        ids = ident[ti]
        train_rows = [r for r, k in zip(rows, ids) if k not in held]
        eval_rows = [r for r, k in zip(rows, ids) if k in held]
        n_train += len(train_rows)
        n_eval += len(eval_rows)
        tr_out = os.path.join(train_dir, f"{name}.json")
        _write(tr_out, train_rows)
        meta[name] = _entry(tr_out, data_root, extra)
        written.append((name, tr_out, data_root))
        n_vid_e = 0
        if eval_rows:
            ev_out = os.path.join(eval_dir, f"{name}.json")
            _write(ev_out, eval_rows)
            eval_meta[f"{name}_heldout"] = _entry(ev_out, data_root, extra)
            uniq = sorted({os.path.join(data_root, row_video(r)) for r in eval_rows})
            n_vid_e = len(uniq)
            manifest += [{"video": v, "task": name} for v in uniq]
        print(f"  {name:34s} train {len(train_rows):>8d} | heldout {len(eval_rows):>5d} "
              f"/ {n_vid_e:>4d} videos")

    if not a.dry_run:
        json.dump(meta, open(a.meta_out, "w"), indent=1)
        print(f"\n[meta] {len(meta)} TRAIN entries / {n_train:,} rows -> {a.meta_out}")
        if eval_meta and a.eval_meta_out:
            json.dump(eval_meta, open(a.eval_meta_out, "w"), indent=1)
            print(f"[meta] {len(eval_meta)} HELD-OUT entries / {n_eval:,} rows "
                  f"-> {a.eval_meta_out}")
        if manifest and a.manifest_out:
            json.dump(manifest, open(a.manifest_out, "w"), indent=1)
            print(f"[meta] {len(manifest)} held-out videos -> {a.manifest_out}")
    else:
        print(f"\n[dry-run] {len(meta)} train entries / {n_train:,} rows; "
              f"{len(eval_meta)} held-out entries / {n_eval:,} rows")

    # -- disjointness check --------------------------------------------------
    train_ids, eval_ids = set(), set()
    for ti in range(len(tasks)):
        for k in ident[ti]:
            (eval_ids if k in held else train_ids).add(k)
    overlap = train_ids & eval_ids
    print(f"[check] train x heldout video overlap: {len(overlap)}")
    if overlap:
        print("[check] FAILED -- held-out slice is not disjoint; do not train on this.")
        return 1
    if protected:
        bad = held & protected
        print(f"[check] held-out clips seen in --protect_from pools: {len(bad)}")
        if bad:
            return 1

    # -- durations -----------------------------------------------------------
    if a.probe_durations and not a.dry_run:
        dur = load_duration_seed(a.durations_seed)
        pairs: List[Tuple[str, str]] = []
        seen = set()
        for name, anno_path, data_root in written:
            for r in json.load(open(anno_path)):
                v = row_video(r)
                if not v:
                    continue
                stem = os.path.splitext(os.path.basename(v))[0]
                if stem in seen or stem in dur:
                    continue
                seen.add(stem)
                pairs.append((stem, os.path.join(data_root, v)))
        print(f"[durations] {len(dur)} reused, ffprobing {len(pairs)} new clips "
              f"with {a.probe_workers} workers ...")
        if pairs:
            with ProcessPoolExecutor(max_workers=a.probe_workers) as ex:
                for n, (vid, d) in enumerate(ex.map(_probe_one, pairs, chunksize=32), 1):
                    dur[vid] = ({"duration_sec": d, "est_frames_1fps": int(round(d)),
                                 "ok": True} if d and d > 0 else {"ok": False})
                    if n % 20000 == 0:
                        print(f"    ffprobe {n}/{len(pairs)}")
        json.dump(dur, open(a.durations_out, "w"))
        ok = sum(1 for d in dur.values() if d.get("ok"))
        print(f"[durations] {ok}/{len(dur)} ok -> {a.durations_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
