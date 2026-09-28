#!/usr/bin/env python3
"""Does an external video benchmark overlap the compressor's TRAINING pool?

Video-MME, LongVideoBench and most long-video benchmarks are built from YouTube,
and so are large parts of this training pool:

    InternVid            <YouTubeID>.mp4              (Phase 1/2/3)
    llava-video youtube  ytb_<YouTubeID>.mp4
    activitynet          v_<YouTubeID>.mp4

so a benchmark clip can already have been trained on. This walks every registry the
model has seen, normalises each clip to a YouTube id where one is recoverable (and
to `build_phase3_blend.video_identity` otherwise), and intersects with the
benchmark's own ids.

    python dataset_util/check_benchmark_contamination.py \\
      --benchmark videomme --benchmark_file anno_data/benchmarks/videomme_test.parquet

Exit code is 0 always; read the report. `--emit_clean` writes the benchmark rows
whose video is NOT in the training pool, which is the set that may be scored.
"""
import argparse, json, os, re, sys, collections

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataset_util.build_phase3_blend import video_identity  # noqa: E402

# A YouTube v1 id: 11 chars of [A-Za-z0-9_-].
YT = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Prefixes the training corpora bolt onto a bare YouTube id.
YT_PREFIXES = ("ytb_", "v_", "yt_")


def youtube_id(video_rel: str):
    """Recover the YouTube id from a training clip's filename, or None."""
    stem = os.path.splitext(os.path.basename(str(video_rel)))[0]
    for p in YT_PREFIXES:
        if stem.startswith(p) and YT.match(stem[len(p):]):
            return stem[len(p):]
    return stem if YT.match(stem) else None


def iter_registry(meta_path):
    """Yield (source_name, data_root, video_rel) over a meta JSON of annotations."""
    meta = json.load(open(meta_path))
    for name, cfg in meta.items():
        ann = cfg["annotation"] if isinstance(cfg, dict) else cfg
        root = cfg.get("data_root", "") if isinstance(cfg, dict) else ""
        try:
            rows = json.load(open(ann))
        except Exception as e:
            print(f"  [warn] {name}: cannot read {ann} ({e})", file=sys.stderr)
            continue
        for r in rows:
            v = r.get("video")
            if isinstance(v, list):
                v = v[0] if v else None
            if v:
                yield name, root, v


def load_videomme(path):
    import pandas as pd
    df = pd.read_parquet(path)
    return [{"video_id": str(r.videoID), "qid": str(r.question_id),
             "duration": str(r.duration)} for r in df.itertuples()]


def load_longvideobench(path):
    """`lvb_val.json` from the gated repo, or the ungated
    `longvideobench/LongVideoBench-Meta` parquet (same rows, no videos)."""
    if path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        return [{"video_id": str(r.video_id), "qid": str(r.id),
                 "duration": f"{int(r.duration_group)}s"} for r in df.itertuples()]
    rows = json.load(open(path))
    return [{"video_id": str(r.get("video_id") or
                           os.path.splitext(str(r.get("video_path", "")))[0]),
             "qid": str(r.get("id", "")),
             "duration": f"{int(r.get('duration_group', 0))}s"} for r in rows]


def load_mlvu(path):
    """MLVU generation subtasks: pass the directory holding MLVU/json/*.json, or a
    single subtask json. `video` is a bare filename whose stem is the clip id."""
    files = []
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if f.endswith(".json")]
    else:
        files = [path]
    out = []
    for f in files:
        task = os.path.splitext(os.path.basename(f))[0]
        for i, r in enumerate(json.load(open(f))):
            out.append({"video_id": os.path.splitext(str(r["video"]))[0],
                        "qid": f"{task}:{i}",
                        "duration": f"{int(float(r.get('duration', 0)))}s"})
    return out


LOADERS = {"videomme": load_videomme, "longvideobench": load_longvideobench,
           "mlvu": load_mlvu}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, choices=sorted(LOADERS))
    ap.add_argument("--benchmark_file", required=True)
    ap.add_argument("--registries", nargs="+", default=[
        "anno_data/phase3_blend.json",      # Phase 3 train
        "anno_data/phase3_internvid.json",  # Phase 3 InternVid buckets
        "anno_data/phase2_internvid.json",  # Phase 2
        "anno_data/internvid_qwen3vl_lt180.json",  # Phase 1
    ])
    ap.add_argument("--extra_video_dirs", nargs="*", default=["/share/dataset/internVid"],
                    help="raw pools the model may have seen that no registry enumerates; "
                         "every file's stem is treated as a trained clip id.")
    ap.add_argument("--emit_clean", default=None,
                    help="write the non-overlapping benchmark rows here (JSON).")
    args = ap.parse_args()

    trained_yt = {}          # youtube id -> set(source)
    trained_ident = set()    # non-YouTube identities
    for reg in args.registries:
        p = reg if os.path.isabs(reg) else os.path.join(REPO, reg)
        if not os.path.exists(p):
            print(f"[skip] registry not found: {reg}")
            continue
        n = 0
        for name, root, rel in iter_registry(p):
            n += 1
            y = youtube_id(rel)
            if y:
                trained_yt.setdefault(y, set()).add(name)
            else:
                trained_ident.add(video_identity(root, rel))
        print(f"[registry] {reg}: {n} rows")
    for d in args.extra_video_dirs or []:
        if not os.path.isdir(d):
            print(f"[skip] dir not found: {d}")
            continue
        n = 0
        for fn in os.listdir(d):
            stem = os.path.splitext(fn)[0]
            if YT.match(stem):
                trained_yt.setdefault(stem, set()).add(os.path.basename(d))
                n += 1
        print(f"[raw pool] {d}: {n} YouTube-id files")
    print(f"[pool] {len(trained_yt)} trained YouTube ids, {len(trained_ident)} other identities\n")

    rows = LOADERS[args.benchmark](args.benchmark_file)
    vids = {r["video_id"] for r in rows}
    hit = {v for v in vids if v in trained_yt}
    print(f"[{args.benchmark}] {len(rows)} questions over {len(vids)} videos")
    print(f"[{args.benchmark}] CONTAMINATED videos: {len(hit)} / {len(vids)} "
          f"({100*len(hit)/max(len(vids),1):.1f}%)")
    q_hit = [r for r in rows if r["video_id"] in hit]
    print(f"[{args.benchmark}] affected questions: {len(q_hit)} / {len(rows)} "
          f"({100*len(q_hit)/max(len(rows),1):.1f}%)\n")

    if hit:
        src = collections.Counter()
        for v in hit:
            for s in trained_yt[v]:
                src[s] += 1
        print("  overlap by training source (a clip can appear in several):")
        for s, c in src.most_common():
            print(f"    {c:6d}  {s}")
        print()
    by = collections.Counter(r["duration"] for r in rows)
    byh = collections.Counter(r["duration"] for r in q_hit)
    print("  by benchmark duration split:")
    for k in sorted(by):
        print(f"    {k:10s} {byh[k]:5d} / {by[k]:5d} questions contaminated "
              f"({100*byh[k]/by[k]:.1f}%)")

    if args.emit_clean:
        clean = [r for r in rows if r["video_id"] not in hit]
        with open(args.emit_clean, "w") as f:
            json.dump(clean, f, indent=1)
        print(f"\n[clean] {len(clean)} questions -> {args.emit_clean}")


if __name__ == "__main__":
    main()
