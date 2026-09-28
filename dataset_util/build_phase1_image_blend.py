#!/usr/bin/env python3
"""Phase-1 IMAGE stream: registry + per-task held-out slices for the qbase pretrain.

Why images in Phase 1
---------------------
The qbase's whole training signal has so far been InternVid captions, and
``shell/recaption_vllm.sh`` pins every clip to ``NUM_FRAMES=64`` regardless of
duration -- a 678 s clip is captioned at ~0.05 fps. There is therefore no CE
gradient anywhere in Phase 1 that rewards preserving detail finer than that
sampling density, which is the "**the qbase was pretrained on very sparse
input**" root cause in ``docs/two_stage_compression_design.md`` §2.2.

A still image is the degenerate case of the whole-video compression path and the
densest possible supervision for it:

    T = 1 frame  ->  adaptive_segment_count(1) = 1 segment  ->  K = 64 tokens

With dynamic HW the frozen encoder hands that one frame up to ``--vision_max_tokens``
tokens, so the qbase squeezes ~1k tokens into 64 against a caption/VQA answer that
actually names small things (OCR strings, attributes, counts, spatial relations).
That is the detail pressure InternVid structurally cannot supply.

No ``Time`` marker
------------------
``_rewrite_image_block_as_single_frame_video`` (train/data/global_compressor.py)
turns the image content block into a one-frame video block and drops its
timestamp, so the chat template renders a bare ``<image>`` with no ``Time X.0s:``
prefix, and the datasets flag the part with ``compression_is_image=[True]`` so
``prepare_inputs_labels_for_multimodal`` emits no ``Time:{a}s-{b}s:`` range for
it either (single- and two-stage paths both).

Sources (verified on disk 2026-09-26)
-------------------------------------
============================  ==========  =========  ====================================
file                          rows        resolves   media root
============================  ==========  =========  ====================================
videoxl Pretraining/          2,000,000   100.0 %    videoxl/Pretraining/_media/
  pretrain.json                                        pretrain_images/images
videoxl_pro bunny_union.json    624,174    86.5 %    videoxl_pro/media
videoxl_pro ocrvqa_19k.json      19,258    99.6 %    videoxl_pro/media
============================  ==========  =========  ====================================

``bunny_union`` is really four corpora (visual_genome 316k / coco_2017 206k /
ocrvqa 80k / open_images 22k rows) and is registered as four tasks, so each gets
its own held-out slice and its own share of the mix. Its per-root resolution
differs a lot (visual_genome 93.6 %, coco 89.0 %, ocrvqa 66.3 %, open_images
57.9 %), which is why every row is existence-checked rather than trusted.

``pretrain.json`` is alt-text (mean answer ~90 chars, "Piece of dark jeans fabric
Royalty Free Stock Photography"): volume, not detail. It is capped by
``--vxl_pretrain_max`` so it cannot swamp the dense instruction data.

Held-out policy: NONE by default (deliberate, 2026-09-26)
---------------------------------------------------------
The image stream keeps **no** held-out slice -- every resolvable row goes into
training (``--heldout_frac 0``, the default). This is a considered exception to
the CLAUDE.md per-task held-out rule, which exists so that the *measurements*
reported for the compressor are never taken on training data. Nothing measures
this stream: the images are here purely as detail pressure on the qbase at a
budget it can actually spend (a still image is one segment, so K = 64 tokens
carry the whole frame), and every reported number comes from the video held-out
slices that ``build_phase3_blend.py`` carves. Reserving image rows would only
shrink that pressure.

``--heldout_frac > 0`` re-enables the carve if a later recipe does want to
measure this stream. It then splits by **image**, not by row (bunny ships ~2.1
rows per image), and holds out the **union across tasks**: identity is the media
**realpath**, so bunny's ``ocrvqa/763667781.jpg`` and ocrvqa_19k's
``ocrvqa/763667781.jpg`` -- the same physical file through two symlinks -- are
held out together. Image stems here are generic (``0010278167``), so the
stem-vs-realpath rule of the video builder does not apply: realpath always.

Usage
-----
    # image stream only
    python dataset_util/build_phase1_image_blend.py

    # blended with the existing Phase-1 InternVid video stream (what Phase 1 runs)
    python dataset_util/build_phase1_image_blend.py \\
      --merge anno_data/internvid_qwen3vl_lt180.json \\
      --image_share 0.4 \\
      --blend_out anno_data/phase1_blend.json
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TAG = "<image>"

# (task, annotation, media_root_key, optional image-root filter)
VXL = "videoxl/Pretraining"
VXLP = "videoxl_pro/finetune"
BUNNY_ROOTS = ("visual_genome", "coco_2017", "ocrvqa", "open_images")


def media_roots(datasets_root: str) -> Dict[str, str]:
    unified = os.path.join(datasets_root, "unified")
    return {
        "vxl_pretrain": os.path.join(
            unified, "source/videoxl/Pretraining/_media/pretrain_images/images"),
        "vxl_pro": os.path.join(unified, "source/videoxl_pro/media"),
    }


def annotations(datasets_root: str) -> Dict[str, str]:
    unified = os.path.join(datasets_root, "unified")
    return {
        "vxl_pretrain": os.path.join(unified, "annotations", VXL, "pretrain.json"),
        "bunny_union": os.path.join(unified, "annotations", VXLP, "bunny_union.json"),
        "ocrvqa_19k": os.path.join(unified, "annotations", VXLP, "ocrvqa_19k.json"),
    }


# ------------------------------------------------------------------- utilities
def row_image(row: dict) -> Optional[str]:
    v = row.get("image")
    v = v[0] if isinstance(v, list) and v else v
    return str(v) if v else None


def require_image(row: dict) -> str:
    """``row_image`` for rows that are known to have one (everything downstream of
    ``keep_resolvable`` / ``merge_rows_by_image``, both of which drop the rest)."""
    v = row_image(row)
    assert v, f"row without an image survived filtering: {row.get('id')}"
    return v


def image_identity(data_root: str, image_rel: str) -> str:
    """Identity comparable ACROSS tasks. Image stems in these corpora are generic
    digit strings and the SAME physical file is reachable under two different
    relative paths (bunny's ``ocrvqa/x.jpg`` and ocrvqa_19k's ``ocrvqa/x.jpg``
    are two symlinks to one file), so realpath is the only correct key."""
    return os.path.realpath(os.path.join(data_root, str(image_rel)))


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


def merge_rows_by_image(rows: List[dict], task: str, max_turns: int,
                        seed: int) -> List[dict]:
    """Rows sharing an image -> one multi-turn sample, so the frozen encoder runs
    once per image per epoch instead of ~2.1x (bunny_union: 624,174 rows over
    297,131 images). The LONGEST answer leads -- the dense-caption turn is the one
    that has to survive the ``--max_turns`` cut, the same reason
    build_phase3_blend.py loads its caption file first."""
    by_img: "OrderedDict[str, List[dict]]" = OrderedDict()
    for r in rows:
        v = row_image(r)
        if v:
            by_img.setdefault(v, []).append(r)

    out: List[dict] = []
    for img, group in by_img.items():
        pairs: List[Tuple[str, str]] = []
        for r in group:
            pairs += _qa_pairs(r.get("conversations") or [])
        pairs = [(h, g) for h, g in pairs if g.strip()]
        if not pairs:
            continue
        lead = max(range(len(pairs)), key=lambda j: len(pairs[j][1]))
        pairs = [pairs[lead]] + pairs[:lead] + pairs[lead + 1:]
        if max_turns > 0 and len(pairs) > max_turns:
            rng = random.Random(f"{seed}:{task}:{img}")
            keep = sorted(rng.sample(range(1, len(pairs)), max_turns - 1))
            pairs = [pairs[0]] + [pairs[i] for i in keep]
        convs: List[dict] = []
        for j, (h, g) in enumerate(pairs):
            h = h.replace(IMAGE_TAG, "").replace("<video>", "").strip()
            if j == 0:
                h = f"{IMAGE_TAG}\n{h}" if h else IMAGE_TAG
            convs += [{"from": "human", "value": h}, {"from": "gpt", "value": g}]
        out.append({
            "id": f"{task}/{os.path.splitext(os.path.basename(img))[0]}",
            "image": img,
            "data_source": task,
            "conversations": convs,
        })
    return out


def keep_resolvable(rows: List[dict], data_root: str) -> Tuple[List[dict], int]:
    """Drop rows whose media is not on disk. Checked once per unique image."""
    seen: Dict[str, bool] = {}
    out = []
    for r in rows:
        v = row_image(r)
        if not v:
            continue
        ok = seen.get(v)
        if ok is None:
            ok = os.path.exists(os.path.join(data_root, v))
            seen[v] = ok
        if ok:
            out.append(r)
    return out, sum(1 for ok in seen.values() if not ok)


def native_tokens(path: str, factor: int = 28, merge: int = 2) -> Optional[int]:
    """Merged vision tokens the frozen encoder produces for this file at its NATIVE
    resolution -- ``(round(h/factor)*factor/patch/merge) * (same for w)``, the
    ``simple_batched_resize`` arithmetic with neither clamp applied. PIL reads only
    the header, so this is a stat, not a decode."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 -- unreadable / truncated / not an image
        return None
    patch = factor // merge
    return (round(h / factor) * factor // patch // merge) * \
           (round(w / factor) * factor // patch // merge)


def keep_big_enough(rows: List[dict], data_root: str, min_tokens: int,
                    workers: int) -> Tuple[List[dict], int]:
    """Drop images too small to fill one qbase segment.

    A still image is ONE segment, so the compressor emits ``N*K = 64`` rows for it,
    and ``compress_visual_tokens_with_compressor`` reserves those rows INSIDE the
    part they replace -- an image whose native grid holds fewer than ``K`` vision
    tokens overflows into the next part and the scatter dies on a shape mismatch
    (the assert in videollama3_arch.py names it).

    ``--vision_min_tokens K`` would also satisfy the contract, by upscaling, but
    upscaling a 84x168 thumbnail invents no detail while still costing a full
    64-token segment and a caption's worth of CE -- exactly the sparse-input
    signal images were added to fix. Drop them instead; the processor floor stays
    as the belt-and-braces guard.
    """
    if min_tokens <= 0:
        return rows, 0
    from concurrent.futures import ThreadPoolExecutor
    uniq = sorted({require_image(r) for r in rows})
    with ThreadPoolExecutor(max_workers=workers) as ex:  # header reads: I/O bound
        toks = list(ex.map(lambda v: native_tokens(os.path.join(data_root, v)), uniq))
    ok = {v: (t is not None and t >= min_tokens) for v, t in zip(uniq, toks)}
    return [r for r in rows if ok[require_image(r)]], sum(1 for v in ok.values() if not v)


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


# --------------------------------------------------------------------- loading
def load_sources(args) -> "OrderedDict[str, Tuple[List[dict], str]]":
    """task -> (rows, data_root). bunny_union is split by its image root."""
    roots = media_roots(args.datasets_root)
    annos = annotations(args.datasets_root)
    tasks: "OrderedDict[str, Tuple[List[dict], str]]" = OrderedDict()

    if not args.no_bunny:
        path = annos["bunny_union"]
        rows = json.load(open(path))
        by_root: Dict[str, List[dict]] = {}
        for r in rows:
            v = row_image(r)
            if v:
                by_root.setdefault(v.split("/")[0], []).append(r)
        for sub in BUNNY_ROOTS:
            if sub in by_root:
                tasks[f"img_bunny_{sub}"] = (by_root[sub], roots["vxl_pro"])
        extra = sorted(set(by_root) - set(BUNNY_ROOTS))
        if extra:
            print(f"[phase1-img] bunny_union: unregistered image roots {extra} "
                  f"({sum(len(by_root[k]) for k in extra):,} rows) -- skipped")

    if not args.no_ocrvqa:
        tasks["img_ocrvqa"] = (json.load(open(annos["ocrvqa_19k"])), roots["vxl_pro"])

    if not args.no_vxl_pretrain:
        rows = json.load(open(annos["vxl_pretrain"]))
        if args.vxl_pretrain_max > 0 and len(rows) > args.vxl_pretrain_max:
            rng = random.Random(args.seed)
            rows = rng.sample(rows, args.vxl_pretrain_max)
        tasks["img_vxl_pretrain"] = (rows, roots["vxl_pretrain"])

    return tasks


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets_root", default="/root/datasets")
    p.add_argument("--out_dir", default="anno_online/phase1_images",
                   help="per-task annotation files land in <out_dir>/{train,heldout}/")
    p.add_argument("--meta_out", default="anno_data/phase1_image_blend.json")
    p.add_argument("--eval_meta_out", default="anno_data/phase1_image_heldout.json")
    p.add_argument("--manifest_out",
                   default="eval_ablation/manifest_phase1_image_heldout.json")

    p.add_argument("--no_bunny", action="store_true")
    p.add_argument("--no_ocrvqa", action="store_true")
    p.add_argument("--no_vxl_pretrain", action="store_true")
    p.add_argument("--vxl_pretrain_max", type=int, default=60_000,
                   help="cap on the 2M alt-text set (volume, not detail; mean answer "
                        "~90 chars and a 88-token median native grid, vs 216-999 for "
                        "the bunny corpora); 0 = all")
    p.add_argument("--min_native_tokens", type=int, default=64,
                   help="drop images whose NATIVE grid holds fewer than this many "
                        "vision tokens. Must be >= K (=64): a still image is one "
                        "segment, so the compressor emits K rows into a part that "
                        "only holds the image's own tokens. 0 = no filter (then the "
                        "run needs --vision_min_tokens >= K to upscale instead).")
    p.add_argument("--probe_workers", type=int, default=32)

    p.add_argument("--max_turns", type=int, default=6,
                   help="rows sharing an image merge into one multi-turn sample, "
                        "capped here; the longest answer always leads. 0 = no cap")
    p.add_argument("--no_merge_turns", action="store_true")

    p.add_argument("--merge", nargs="*", default=[],
                   help="existing --multi_dataset meta JSON(s) (the Phase-1 InternVid "
                        "video stream) to fold in, written to --blend_out")
    p.add_argument("--blend_out", default="anno_data/phase1_blend.json")
    p.add_argument("--image_share", type=float, default=0.4,
                   help="target fraction of the blended registry's rows that are "
                        "images. The meta format has no per-entry weight, so the mix "
                        "IS the row counts: the image tasks are subsampled (keeping "
                        "their relative proportions) to hit this share. Too high and "
                        "the qbase becomes a single-frame expert and loses the 1-8 "
                        "frame segment behaviour it is actually deployed with.")

    p.add_argument("--heldout_frac", type=float, default=0.0,
                   help="0 (the default) = NO held-out slice; every row trains. The "
                        "image stream is detail pressure for the qbase, not something "
                        "any recipe measures -- see the module docstring. >0 re-enables "
                        "the per-task carve (by image realpath, union across tasks) and "
                        "writes --eval_meta_out / --manifest_out.")
    p.add_argument("--heldout_min", type=int, default=20)
    p.add_argument("--heldout_max", type=int, default=300)
    p.add_argument("--heldout_cap_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true",
                   help="report composition and write nothing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    os.chdir(REPO)

    for path in annotations(args.datasets_root).values():
        if not os.path.exists(path):
            print(f"[phase1-img] MISSING annotation: {path}", file=sys.stderr)
    for path in media_roots(args.datasets_root).values():
        if not os.path.isdir(path):
            print(f"[phase1-img] MISSING media root: {path}", file=sys.stderr)
            return 1

    tasks = load_sources(args)
    if not tasks:
        print("[phase1-img] no sources selected", file=sys.stderr)
        return 1

    # -- resolve + merge turns -------------------------------------------------
    prepared: "OrderedDict[str, Tuple[List[dict], str]]" = OrderedDict()
    for task, (rows, root) in tasks.items():
        n_raw = len(rows)
        rows, n_missing = keep_resolvable(rows, root)
        rows, n_small = keep_big_enough(rows, root, args.min_native_tokens,
                                        args.probe_workers)
        if not args.no_merge_turns:
            rows = merge_rows_by_image(rows, task, args.max_turns, args.seed)
        n_img = len({require_image(r) for r in rows})
        print(f"[phase1-img] {task:28s} {n_raw:9,} rows -> {len(rows):8,} samples "
              f"({n_img:,} images, {n_missing:,} unresolvable, "
              f"{n_small:,} under {args.min_native_tokens} tokens)")
        if rows:
            prepared[task] = (rows, root)

    # -- held-out: split by image, UNION across tasks ---------------------------
    # A clip held out for one task is excluded from EVERY task's training: bunny's
    # ocrvqa rows and ocrvqa_19k point at the same physical files.
    heldout_ids: set = set()
    per_task_ids: Dict[str, List[str]] = {}
    for task, (rows, root) in prepared.items():
        ids = [image_identity(root, require_image(r)) for r in rows]
        per_task_ids[task] = ids
        heldout_ids |= pick_heldout(ids, args.heldout_frac, args.heldout_min,
                                    args.heldout_max, args.heldout_cap_frac,
                                    rng, protected=set())

    train_tasks: "OrderedDict[str, Tuple[List[dict], str]]" = OrderedDict()
    eval_tasks: "OrderedDict[str, Tuple[List[dict], str]]" = OrderedDict()
    for task, (rows, root) in prepared.items():
        ids = per_task_ids[task]
        tr = [r for r, i in zip(rows, ids) if i not in heldout_ids]
        ev = [r for r, i in zip(rows, ids) if i in heldout_ids]
        if tr:
            train_tasks[task] = (tr, root)
        if ev:
            eval_tasks[task] = (ev, root)

    # -- image share against the merged video stream ---------------------------
    merged: "OrderedDict[str, dict]" = OrderedDict()
    n_video_rows = 0
    for meta_path in args.merge:
        for name, cfg in json.loads(open(meta_path).read()).items():
            merged[name] = cfg
            try:
                n_video_rows += len(json.load(open(cfg["annotation"])))
            except Exception as exc:  # noqa: BLE001
                print(f"[phase1-img] cannot count {cfg['annotation']}: {exc}",
                      file=sys.stderr)

    n_image_rows = sum(len(r) for r, _ in train_tasks.values())
    if merged and 0.0 < args.image_share < 1.0 and n_video_rows > 0:
        target = int(round(args.image_share / (1.0 - args.image_share) * n_video_rows))
        if target < n_image_rows:
            keep = target / n_image_rows
            for task, (rows, root) in list(train_tasks.items()):
                n_keep = max(1, int(round(len(rows) * keep)))
                sub_rng = random.Random(f"{args.seed}:share:{task}")
                train_tasks[task] = (sub_rng.sample(rows, n_keep), root)
            n_image_rows = sum(len(r) for r, _ in train_tasks.values())
        else:
            print(f"[phase1-img] image_share {args.image_share} wants {target:,} image "
                  f"rows but only {n_image_rows:,} exist -- using all of them "
                  f"(actual share {n_image_rows / (n_image_rows + n_video_rows):.3f})")

    # -- disjointness assert ----------------------------------------------------
    train_ids = {image_identity(root, require_image(r))
                 for rows, root in train_tasks.values() for r in rows}
    eval_ids = {image_identity(root, require_image(r))
                for rows, root in eval_tasks.values() for r in rows}
    overlap = train_ids & eval_ids
    assert not overlap, (
        f"train x held-out overlap on {len(overlap)} images, e.g. "
        f"{sorted(overlap)[:3]} -- the per-task held-out policy is broken"
    )

    print(f"[phase1-img] train {len(train_tasks)} tasks / {n_image_rows:,} samples / "
          f"{len(train_ids):,} images")
    if eval_tasks:
        print(f"[phase1-img] held-out {len(eval_tasks)} tasks / "
              f"{sum(len(r) for r, _ in eval_tasks.values()):,} samples / "
              f"{len(eval_ids):,} images; train x held-out overlap 0")
    else:
        print("[phase1-img] held-out: none (--heldout_frac 0) -- every row trains")
    if merged:
        share = n_image_rows / max(1, n_image_rows + n_video_rows)
        print(f"[phase1-img] blend: {n_image_rows:,} image + {n_video_rows:,} video rows "
              f"-> image share {share:.3f}")

    if args.dry_run:
        print("[phase1-img] --dry_run: nothing written")
        return 0

    # -- write ------------------------------------------------------------------
    def dump(split: str, group) -> Dict[str, dict]:
        d = os.path.join(args.out_dir, split)
        os.makedirs(d, exist_ok=True)
        meta = {}
        for task, (rows, root) in group.items():
            path = os.path.join(d, f"{task}.json")
            with open(path, "w") as f:
                json.dump(rows, f)
            meta[task] = {
                "annotation": os.path.join(REPO, path),
                "data_root": root,
                "online_mode": False,
                "prefix_captioning": False,
            }
        return meta

    train_meta = dump("train", train_tasks)
    to_write = [(args.meta_out, train_meta)]
    if eval_tasks:
        to_write.append((args.eval_meta_out, dump("heldout", eval_tasks)))
    for path, obj in to_write:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=1)
        print(f"[phase1-img] wrote {path} ({len(obj)} tasks)")

    # No held-out slice (the default) -> write no eval registry and no manifest, and
    # clear any left behind by an earlier run, so nothing downstream can pick up a
    # stale slice that is now training data.
    if eval_tasks:
        manifest = [{"task": task, "image": os.path.join(root, require_image(r)),
                     "id": r.get("id")}
                    for task, (rows, root) in eval_tasks.items() for r in rows]
        os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"[phase1-img] wrote {args.manifest_out} ({len(manifest)} entries)")
    else:
        stale = [args.eval_meta_out, args.manifest_out]
        stale += [os.path.join(args.out_dir, "heldout", f)
                  for f in (os.listdir(os.path.join(args.out_dir, "heldout"))
                            if os.path.isdir(os.path.join(args.out_dir, "heldout")) else [])]
        for path in stale:
            if os.path.exists(path):
                os.remove(path)
                print(f"[phase1-img] removed stale {path}")
        hd = os.path.join(args.out_dir, "heldout")
        if os.path.isdir(hd) and not os.listdir(hd):
            os.rmdir(hd)
        print("[phase1-img] no held-out slice: every image row is training data")

    if merged:
        blend = OrderedDict(train_meta)
        blend.update(merged)
        os.makedirs(os.path.dirname(args.blend_out) or ".", exist_ok=True)
        with open(args.blend_out, "w") as f:
            json.dump(blend, f, indent=1)
        print(f"[phase1-img] wrote {args.blend_out} ({len(blend)} tasks: "
              f"{len(train_meta)} image + {len(merged)} video)")

    counts = Counter({t: len(r) for t, (r, _) in train_tasks.items()})
    print("[phase1-img] train composition:")
    for task, n in counts.most_common():
        print(f"    {task:28s} {n:8,}  {100 * n / max(1, n_image_rows):5.1f}% of images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
