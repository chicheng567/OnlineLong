#!/usr/bin/env python3
"""
Re-annotate a video dataset with detailed captions from a **vLLM-served Qwen3-VL**
model, letting vLLM sample + preprocess frames on its own (backend "A").

This is the vLLM sibling of ``recaption_videos.py``. It keeps that script's whole
data / prompt / output half unchanged -- same ``--meta_path`` / ``--video_dir``
inputs, same prompt pool, same resume-safe per-rank JSONL, same
``--annotation_out`` / ``--meta_out`` writers -- and swaps the frozen-VideoLLaMA3
"vision_encoder -> mm_projector -> scatter -> Qwen2.generate(inputs_embeds)"
pipeline for a single ``vllm.LLM`` running ``Qwen/Qwen3-VL-8B-Instruct``.

What changes vs recaption_videos.py
-----------------------------------
* No manual vision tower / projector / embedding scatter, no MLP seq-chunking, no
  token-budget batch packing, no ``--llm_impl`` stock/vendored split. vLLM owns
  the KV cache, continuous batching and the vision preprocessing.
* Frames are **not decoded here**. Only the file path is handed to vLLM; frame
  sampling is Qwen3-VL's own fps grid, tuned by ``--fps`` / ``--max-pixels`` /
  ``--max-frames`` which flow through as ``mm_processor_kwargs``. Qwen3-VL also
  interleaves its own per-frame timestamp markers from that grid.
* Per-frame timestamps in the JSONL are therefore a **synthetic uniform grid**
  re-derived from ``--fps`` (frame order is real, the seconds are approximate),
  tagged ``"timestamps_synthetic": true``. Pass ``--timestamps none`` to omit
  them. This path is for plain captioning, not second-referencing annotation.

Environment
-----------
Needs ``vllm`` (a build with Qwen3-VL support) and ``qwen-vl-utils`` importable.
vLLM pins its own Torch build, so install it in a **separate venv** from the
training env (``requirements.txt``). Local video files are read by vLLM's media
loader only under ``--allowed_local_media_path`` (default ``/``).

Multi-GPU
---------
Data-parallel, the same ``rank::world_size`` shard split recaption_videos.py
uses: launch with ``torchrun`` and every rank builds its **own**
``LLM(tensor_parallel_size=1)`` pinned to one card (``CUDA_VISIBLE_DEVICES`` is
set per local rank before vLLM starts; only a gloo group is created, for the
end-of-run barrier). For a model too big for one card, drop ``torchrun`` and use
``--tensor_parallel_size N`` instead -- the two cannot be combined here.

Usage
-----
8 GPUs, data-parallel (one Qwen3-VL per card), re-annotate a registry in place:
    torchrun --nproc_per_node=8 dataset_util/recaption_videos_vllm.py \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --meta_path anno_data/finetune_online.json \
        --output_file recaption/qwen3vl/online.jsonl \
        --meta_out recaption/qwen3vl/meta_recaption.json

Single GPU, bare folder of clips, one prompt drawn per sample:
    python dataset_util/recaption_videos_vllm.py \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --video_dir /share/dataset/internVid \
        --output_file recaption/internvid/captions.jsonl \
        --annotation_out anno_online/internvid_recap.json

2-GPU tensor parallel for a bigger model (no torchrun):
    python dataset_util/recaption_videos_vllm.py --tensor_parallel_size 2 \
        --model Qwen/Qwen3-VL-30B-A3B-Instruct \
        --meta_path anno_online/my_videos.json --data_root /root/datasets/... \
        --output_file recaption/detail_captions.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist
from tqdm import tqdm

sys.path.append("./")

from videollama3.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Describe this video in detail. Cover the main subjects and their appearance, "
    "the actions and events in the order they happen, the scene and background, "
    "any notable camera movement, and any visible text. "
    "Write one coherent, factual paragraph and do not speculate about what is not shown."
)

# Keys this script adds to a meta entry for its own bookkeeping; stripped again
# before an entry is written to --annotation_out.
_INTERNAL_KEYS = ("_data_root", "_dataset", "_prompt", "_prompt_name", "_source")

_PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

# The prompt pool. With neither --prompt nor --prompt_file given, every sample
# draws one of these, so a dataset captioned in a single run carries several
# caption styles instead of one voice repeated N times. See
# dataset_util/prompts/README.md. Override the membership with --prompt_pool.
DEFAULT_PROMPT_POOL = (
    "baseline_default",
    "scene_shift",
    "motion_scene",
)

# Video file extensions for the --annotate_unannotated / --video_dir disk walks.
_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".flv")


# ---------------------------------------------------------------------------
# Prompt pool  (verbatim from recaption_videos.py)
# ---------------------------------------------------------------------------

def _load_prompt_pool(names: List[str]) -> List[Tuple[str, str]]:
    """[(name, text), ...] for each pool member.

    A bare name resolves against dataset_util/prompts/<name>.txt (next to this
    file, so it works from any cwd); anything with a suffix or a separator is
    read as a path.
    """
    pool: List[Tuple[str, str]] = []
    for name in names:
        name = name.strip()
        if not name:
            continue
        path = Path(name)
        if not path.suffix and os.sep not in name:
            path = _PROMPT_DIR / f"{name}.txt"
        try:
            text = path.read_text().strip()
        except OSError as exc:  # noqa: BLE001
            logger.warning("Prompt %s is unreadable (%s); dropped from the pool.", path, exc)
            continue
        if text:
            pool.append((path.stem, text))
    if not pool:
        logger.warning("No prompt in the pool could be read; falling back to the built-in prompt.")
        pool = [("builtin_default", DEFAULT_PROMPT)]
    return pool


def _pick_prompt(pool: List[Tuple[str, str]], video_id: str, seed: int) -> Tuple[str, str]:
    """This sample's prompt, drawn per sample rather than per run.

    Keyed on a hash of the video id instead of a running RNG so the assignment is
    a property of the sample: a resume, a different GPU count and the
    longest-first sort all leave every video on the prompt it had before. Change
    --seed to reshuffle the whole dataset.
    """
    digest = hashlib.blake2b(f"{seed}:{video_id}".encode(), digest_size=8).digest()
    return pool[int.from_bytes(digest, "big") % len(pool)]


# ---------------------------------------------------------------------------
# Distributed  (gloo only -- torch collectives are never used here)
# ---------------------------------------------------------------------------

_GLOO_PG = None


def _pin_gpu_for_local_rank() -> None:
    """torchrun data-parallel: give each rank exactly one visible GPU BEFORE vLLM
    (or any CUDA call) runs, so a tensor_parallel_size=1 engine lands on this
    rank's own card. A no-op outside torchrun.
    """
    if "LOCAL_RANK" not in os.environ:
        return
    lr = int(os.environ["LOCAL_RANK"])
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = [x for x in vis.split(",") if x != ""] if vis else None
    os.environ["CUDA_VISIBLE_DEVICES"] = ids[lr] if ids and lr < len(ids) else str(lr)


def _init_dist() -> Tuple[int, int, int]:
    global _GLOO_PG
    if "LOCAL_RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="gloo", timeout=timedelta(days=1))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        _GLOO_PG = dist.new_group(backend="gloo", timeout=timedelta(days=1))
    else:
        rank = local_rank = 0
        world_size = 1
    return rank, local_rank, world_size


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(group=_GLOO_PG)


# ---------------------------------------------------------------------------
# Sample collection  (verbatim from recaption_videos.py)
# ---------------------------------------------------------------------------

def _entry_media(entry: Dict) -> Tuple[Optional[str], List[str]]:
    """(video field, image files) of a meta entry, both normalised to str."""
    video = entry.get("video")
    if isinstance(video, (list, tuple)):
        video = video[0] if video else None
    images = entry.get("image")
    if images is None:
        images = []
    elif not isinstance(images, (list, tuple)):
        images = [images]
    return (str(video) if video is not None else None), [str(i) for i in images]


def _resolve_path(entry: Dict, rel: str, data_root: Optional[str]) -> str:
    if os.path.isabs(rel):
        return rel
    root = entry.get("_data_root") or data_root or ""
    return os.path.join(root, rel) if root else rel


def _sample_prompt(entry: Dict, default_prompt: str, prompt_source: str) -> str:
    """The user turn this sample is captioned with.

    --prompt_source annotation reads the entry's own user text and drops every
    assistant/gpt/system turn, so the model never sees the GT it is supposed to
    be replacing; entries without usable user text fall back to --prompt.
    """
    if prompt_source != "annotation":
        return default_prompt
    texts: List[str] = []
    for turn in entry.get("conversations") or []:
        role = turn.get("role") or turn.get("from", "")
        if role not in ("human", "user"):
            continue
        text = str(turn.get("value", turn.get("content", "")))
        text = text.replace(DEFAULT_VIDEO_TOKEN, "").replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if text:
            texts.append(text)
    return "\n".join(texts) if texts else default_prompt


def _scan_unannotated(
    entries: List[Dict], registry: Dict, data_root: Optional[str]
) -> List[Dict]:
    """Synthetic entries for videos on disk that the annotation never mentions.
    They carry no conversations, so they always caption with --prompt.
    """
    known = set()
    for entry in entries:
        video, images = _entry_media(entry)
        for media in ([video] if video else []) + images:
            known.add(Path(media).stem)

    roots: List[Tuple[str, Optional[str]]] = []
    if registry:
        for ds_name, ds_cfg in registry.items():
            root = ds_cfg.get("data_root") or data_root
            if root:
                roots.append((root, ds_name))
    elif data_root:
        roots.append((data_root, None))

    extra: List[Dict] = []
    for root, ds_name in roots:
        if not os.path.isdir(root):
            logger.warning("--annotate_unannotated: %s is not a directory.", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()  # deterministic walk: the shard split depends on the order
            for filename in sorted(filenames):
                if os.path.splitext(filename)[1].lower() not in _VIDEO_EXTS:
                    continue
                stem = Path(filename).stem
                if stem in known:
                    continue
                known.add(stem)
                entry = {
                    "video": os.path.relpath(os.path.join(dirpath, filename), root),
                    "_data_root": root,
                    "_unannotated": True,
                }
                if ds_name:
                    entry["_dataset"] = ds_name
                extra.append(entry)
    if extra:
        logger.info(
            "--annotate_unannotated: %d video(s) on disk are not referenced by the "
            "annotation; captioning them too.", len(extra),
        )
    return extra


def _scan_video_dir(video_dir: str) -> List[Dict]:
    """One synthetic entry per video file found (recursively) under `video_dir`."""
    root = os.path.abspath(video_dir)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"--video_dir {video_dir} is not a directory.")
    entries: List[Dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()  # deterministic walk: the shard split depends on the order
        for filename in sorted(filenames):
            if os.path.splitext(filename)[1].lower() not in _VIDEO_EXTS:
                continue
            entries.append({
                "video": os.path.relpath(os.path.join(dirpath, filename), root),
                "_data_root": root,
            })
    return entries


def _load_samples(
    meta_path: Optional[str],
    data_root: Optional[str],
    video_dir: Optional[str] = None,
    annotate_unannotated: bool = False,
    prompt: str = "",
    prompt_source: str = "fixed",
    prompt_pool: Optional[List[Tuple[str, str]]] = None,
    seed: int = 0,
) -> Tuple[List[Dict], Dict]:
    """Build the work list, plus the anno_data registry it came from (empty for a
    plain list or --video_dir) so --meta_out can be written in that shape.
    """
    samples: List[Dict] = []
    registry: Dict = {}

    if video_dir is not None:
        entries = _scan_video_dir(video_dir)
        logger.info("--video_dir: %d video file(s) under %s", len(entries), video_dir)
    else:
        assert meta_path is not None, "need --meta_path or --video_dir"
        with open(meta_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            registry = raw
            entries = []
            for ds_name, ds_cfg in raw.items():
                with open(ds_cfg["annotation"]) as fa:
                    ann = json.load(fa)
                for entry in ann:
                    entry = dict(entry)
                    entry.setdefault("_data_root", ds_cfg.get("data_root", data_root or ""))
                    entry["_dataset"] = ds_name
                    entries.append(entry)
        else:
            entries = [dict(e) for e in raw]

    if annotate_unannotated:
        entries = entries + _scan_unannotated(entries, registry, data_root)

    seen = set()
    n_no_input = 0
    for entry in entries:
        video_field, image_files = _entry_media(entry)
        if video_field is None and not image_files:
            continue
        key = video_field if video_field is not None else image_files[0]
        stem = Path(key).stem
        if stem in seen:  # one caption per video, even if the meta has many turns
            continue
        seen.add(stem)

        if image_files and video_field is None:
            source = "image"
        else:
            if video_dir is None:
                path = _resolve_path(entry, video_field, data_root)
                if not os.path.exists(path):
                    n_no_input += 1
                    logger.warning("No readable video for %s -- skipped.", key)
                    continue
            source = "video"

        if prompt_pool is not None:
            prompt_name, sample_default = _pick_prompt(prompt_pool, stem, seed)
        else:
            prompt_name, sample_default = "fixed", prompt
        entry["_prompt"] = _sample_prompt(entry, sample_default, prompt_source)
        entry["_prompt_name"] = prompt_name if entry["_prompt"] == sample_default else "annotation"
        entry["_source"] = source
        samples.append({
            "video": key,
            "video_id": stem,
            "source": source,
            "prompt": entry["_prompt"],
            "prompt_name": entry["_prompt_name"],
            "images": image_files,
            "raw": entry,
        })

    if n_no_input:
        logger.warning("%d sample(s) had no usable input and were dropped.", n_no_input)
    return samples, registry


def _load_done_ids(output_file: Path) -> set:
    """Collect already-captioned video_ids from every rank's shard file."""
    done = set()
    for p in output_file.parent.glob(f"{output_file.stem}.rank*{output_file.suffix}"):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["video_id"])
                except Exception:
                    continue
    return done


# ---------------------------------------------------------------------------
# Clip probe (fps sampling) -- shard balancing + synthetic timestamp grid
# ---------------------------------------------------------------------------

def _probe_clip(
    sample: Dict, fps: float, max_frames: int, duration_key: str, data_root: Optional[str],
    num_frames: int = 0, need_duration: bool = True,
) -> Tuple[int, Optional[float]]:
    """(approx frames vLLM feeds Qwen3-VL, clip duration in seconds or None).

    The frame count length-balances the rank shards; the duration lets
    --timestamps uniform spread the synthetic grid over the *real* runtime (for
    a clip long enough to hit --max_frames, Qwen3-VL samples across the whole
    video, not the first max_frames/fps seconds). Prefers a duration field; only
    opens a decord header when the annotation has none -- and only when
    `need_duration` (skipped entirely for --num_frames + --timestamps none).
    """
    raw = sample["raw"]
    if sample.get("source") == "image":
        return max(1, len(sample.get("images") or [])), None
    fixed = (num_frames - (num_frames % 2)) if num_frames and num_frames > 0 else 0
    if fixed and not need_duration:
        return fixed, None  # every clip contributes the same count; no probe needed
    dur = raw.get(duration_key)
    if not isinstance(dur, (int, float)):
        dur = raw.get("video_duration")
    if not isinstance(dur, (int, float)) or dur <= 0:
        try:
            from decord import VideoReader

            vr = VideoReader(
                _resolve_path(raw, str(sample["video"]), data_root), num_threads=1
            )
            n, r = len(vr), float(vr.get_avg_fps())
            dur = n / r if r > 0 else 0.0
        except Exception as exc:  # noqa: BLE001
            logger.debug("duration probe failed on %s: %s", sample["video_id"], exc)
            dur = 0.0
    duration = float(dur) if dur and dur > 0 else None
    if fixed:
        return fixed, duration
    if duration is None:
        return (max_frames if max_frames and max_frames > 0 else 32), None
    n = max(4, int(round(duration * fps)))
    if max_frames and max_frames > 0:
        n = min(n, max_frames)
    return n, duration


# ---------------------------------------------------------------------------
# vLLM request assembly
# ---------------------------------------------------------------------------

def _parse_kv_int(spec: str) -> Dict[str, int]:
    """"image=16,video=1" -> {"image": 16, "video": 1}."""
    out: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = int(v)
    return out


def _build_mm_processor_kwargs(args) -> Dict:
    """Video-preprocessing knobs handed to Qwen3-VL's HF processor (backend A).

    `fps` (+ `cap_pixels_per_frame`, the transformers-5.16-era switch to the
    qwen-vl-utils per-frame pixel cap) are passed by default; the pixel / frame
    ceilings are opt-in because their exact key names drift across transformers
    versions. --mm_processor_kwargs_json is the escape hatch and is merged last
    (it wins). vLLM drops keys the installed processor does not accept with a
    warning rather than an error.
    """
    kw: Dict = {}
    # Qwen3-VL's temporal_patch_size is 2: an ODD sampled-frame count trips a
    # FATAL `len(timestamps) == len(tokens_per_frame)` assert in vLLM 0.28's
    # video processor -- it kills the rank's EngineCore, and torchrun then tears
    # the whole run down. The literal `num_frames` kwarg is also broken here
    # ("Failed to apply Qwen3VLProcessor ... do_sample_frames=True"). What works:
    # fps sampling with min_frames == max_frames, which pins every clip to an
    # exact EVEN count through the codepath vLLM actually handles.
    fixed = (args.num_frames - args.num_frames % 2) if args.num_frames and args.num_frames > 0 else 0
    if fixed:
        kw["fps"] = args.fps if (args.fps and args.fps > 0) else 2.0
        kw["min_frames"] = fixed
        kw["max_frames"] = fixed
    elif args.fps and args.fps > 0:
        kw["fps"] = args.fps
        if args.max_frames and args.max_frames > 0:
            kw["max_frames"] = args.max_frames - (args.max_frames % 2)
        if args.min_frames and args.min_frames > 0:
            kw["min_frames"] = args.min_frames + (args.min_frames % 2)
    if args.cap_pixels_per_frame:
        kw["cap_pixels_per_frame"] = True
    if args.max_pixels and args.max_pixels > 0:
        kw["max_pixels"] = args.max_pixels
    if args.min_pixels and args.min_pixels > 0:
        kw["min_pixels"] = args.min_pixels
    if args.mm_processor_kwargs_json:
        kw.update(json.loads(args.mm_processor_kwargs_json))
    return kw


def _media_url(entry: Dict, rel: str, data_root: Optional[str]) -> str:
    """A URL vLLM's chat parser accepts. Local paths become file:// URIs (vLLM
    reads them only under --allowed_local_media_path); http(s)/data URIs pass
    through untouched.
    """
    if rel.startswith(("http://", "https://", "file://", "data:")):
        return rel
    return "file://" + os.path.abspath(_resolve_path(entry, rel, data_root))


def _build_messages(sample: Dict, args, media_root: Optional[str]) -> List[Dict]:
    """One OpenAI-style chat conversation for `llm.chat`. The media is a
    file:// URL; vLLM's media loader reads it and Qwen3-VL's chat template does
    the frame sampling + timestamp interleaving.
    """
    source = sample.get("source", "video")
    content: List[Dict] = []
    if source == "image":
        files = sample.get("images") or []
        if not files:
            raise ValueError("image sample with no image files")
        for f in files:
            content.append({
                "type": "image_url",
                "image_url": {"url": _media_url(sample["raw"], f, media_root)},
            })
    else:
        content.append({
            "type": "video_url",
            "video_url": {"url": _media_url(sample["raw"], str(sample["video"]), media_root)},
        })
    content.append({"type": "text", "text": sample["prompt"]})

    msgs: List[Dict] = []
    if args.system_prompt:
        msgs.append({"role": "system", "content": args.system_prompt})
    msgs.append({"role": "user", "content": content})
    return msgs


def _run_chat(llm, sampling_params, convs: List[List[Dict]], rank: int) -> List[Optional[str]]:
    """llm.chat over a chunk. On any failure, retry the chunk one conversation at
    a time so a single unreadable video cannot sink the whole chunk. Returns a
    list aligned to `convs`; a slot is None when that sample failed.
    """
    try:
        outs = llm.chat(convs, sampling_params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
    except Exception as exc:  # noqa: BLE001
        logger.warning("rank %d: chunk chat failed (%s); retrying per-sample.", rank, exc)

    res: List[Optional[str]] = []
    for conv in convs:
        try:
            out = llm.chat([conv], sampling_params, use_tqdm=False)
            res.append(out[0].outputs[0].text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("rank %d: sample chat failed: %s", rank, exc)
            res.append(None)
    return res


def _make_record(sample: Dict, text: str, args) -> Dict:
    nf = sample.get("_approx_frames")
    dur = sample.get("_duration")
    if nf is None and args.timestamps == "uniform":
        nf, dur = _probe_clip(
            sample, args.fps or 2.0, args.max_frames, args.duration_key,
            args.data_root or args.video_root, num_frames=args.num_frames, need_duration=True,
        )
    timestamps: Optional[List[float]] = None
    ts_synthetic = False
    if args.timestamps == "uniform" and nf and sample.get("source") != "image":
        n = int(nf)
        if dur and dur > 0:
            # Qwen3-VL spreads the sampled frames across the whole runtime, so
            # frame i sits in segment [i*dur/n, (i+1)*dur/n) -- take its midpoint.
            step = float(dur) / n
        else:
            # No duration: assume the frames really are 1/fps apart (true only
            # when the clip was short enough not to hit --max_frames).
            step = 1.0 / (args.fps or 2.0)
        timestamps = [round((i + 0.5) * step, 2) for i in range(n)]
        ts_synthetic = True
    return {
        "video_id": sample["video_id"],
        "video": sample["video"],
        "num_frames": int(nf) if nf else None,
        "video_duration": round(dur, 2) if dur else None,
        "sampling_fps": args.fps or None,
        "timestamps": timestamps,
        # True => the times above were re-derived from --fps, not decoded. Filter
        # on this before using these captions for anything that cites seconds.
        "timestamps_synthetic": ts_synthetic,
        "prompt_name": sample["prompt_name"],
        "prompt": sample["prompt"],
        "caption": text.strip(),
        "model": args.model,
    }


# ---------------------------------------------------------------------------
# Registry output  (verbatim from recaption_videos.py)
# ---------------------------------------------------------------------------

def _write_meta_registry(
    meta_out: Path,
    anno: List[Tuple[Optional[str], Dict]],
    registry: Dict,
) -> None:
    """Write an anno_data-style registry over this run's annotations.

    One {dataset}_recaption.json per source dataset next to --meta_out, each
    registry entry keeping the original data_root / repeat_time / ... and pointing
    `annotation` at the new file. Drop the result straight into --data_path.
    """
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict]] = {}
    for dataset_name, entry in anno:
        grouped.setdefault(dataset_name or "recaption", []).append(entry)

    meta: Dict[str, Dict] = {}
    for dataset_name, entries in grouped.items():
        anno_path = meta_out.parent / f"{dataset_name}_recaption.json"
        with open(anno_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        cfg = dict(registry.get(dataset_name, {}))
        cfg["annotation"] = str(anno_path)
        meta[dataset_name] = cfg
        logger.info("Wrote %s with %d samples", anno_path, len(entries))

    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %s with %d dataset(s)", meta_out, len(meta))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # ---- data / sample selection (same surface as recaption_videos.py) ----
    parser.add_argument("--meta_path", default=None,
                        help="A plain sample list or an anno_data-style dict-of-datasets registry. "
                             "Exactly one of --meta_path / --video_dir is required.")
    parser.add_argument("--video_dir", default=None,
                        help="Caption every video file found (recursively) under this directory, "
                             "with no annotation. Mutually exclusive with --meta_path.")
    parser.add_argument("--data_root", default=None, help="Fallback root for relative video paths.")
    parser.add_argument("--video_root", default=None,
                        help="Root for relative video paths. Defaults to --data_root.")
    parser.add_argument("--output_file", default="recaption/detail_captions.jsonl",
                        help="JSONL results; each rank streams to {stem}.rank{r}{suffix}, "
                             "rank 0 merges into this path at the end.")
    parser.add_argument("--annotation_out", default=None,
                        help="Also write a conversation-format JSON (anno_online style) here.")
    parser.add_argument("--meta_out", default=None,
                        help="Also write an anno_data-style registry pointing at the annotations "
                             "produced by this run. Needs --meta_path to have been a registry.")
    parser.add_argument("--annotate_unannotated", action="store_true",
                        help="Also caption video files found under the data roots that the "
                             "annotation never mentions. They always use --prompt.")
    parser.add_argument("--duration_key", default="duration",
                        help="Meta field holding video duration in seconds.")

    # ---- prompts (same as recaption_videos.py) ----
    parser.add_argument("--prompt", default=None,
                        help="Caption every sample with this one prompt. Giving neither --prompt "
                             "nor --prompt_file turns on the prompt pool: each sample draws its "
                             "own prompt from --prompt_pool.")
    parser.add_argument("--prompt_file", default=None,
                        help="Read the single prompt from this file instead (also disables the pool).")
    parser.add_argument("--prompt_pool", default=",".join(DEFAULT_PROMPT_POOL),
                        help="Comma-separated pool used when no single prompt is given. A bare name "
                             "resolves to dataset_util/prompts/<name>.txt. Default: "
                             + ", ".join(DEFAULT_PROMPT_POOL) + ".")
    parser.add_argument("--prompt_source", choices=["fixed", "annotation"], default="fixed",
                        help="'fixed' captions every sample with --prompt/--prompt_file. "
                             "'annotation' uses each sample's own user turn instead (falling back "
                             "to --prompt when it has none); assistant/system turns are never shown.")
    parser.add_argument("--system_prompt", default=None,
                        help="Optional system turn prepended to every conversation. Left unset, "
                             "Qwen3-VL's own default system prompt from the chat template is used.")

    # ---- vLLM engine ----
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HF id or local path of the Qwen3-VL checkpoint vLLM serves.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Shards one model across N GPUs. Cannot be combined with a torchrun "
                             "(WORLD_SIZE>1) data-parallel launch; use one or the other.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=32768,
                        help="Context length ceiling. A long clip at a high --fps / --max-pixels "
                             "can exceed this -- lower one of them or raise this if vLLM rejects "
                             "prompts.")
    parser.add_argument("--max_num_seqs", type=int, default=16,
                        help="vLLM's concurrent-sequence cap. Lower to ~4-8 on a smaller card.")
    parser.add_argument("--limit_mm_per_prompt", default="image=16,video=1",
                        help="Per-prompt multimodal-item ceiling passed to vLLM, 'k=v,k=v'.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Skip CUDA-graph capture. Slower decode, but sidesteps capture-time "
                             "OOM on tight VRAM.")
    parser.add_argument("--allowed_local_media_path", default="/",
                        help="Filesystem prefix vLLM's media loader may read video/image files "
                             "from. Default '/' (this is an offline job over your own files).")
    parser.add_argument("--video_loader_backend", default=None,
                        help="Sets VLLM_VIDEO_LOADER_BACKEND (e.g. decord, opencv, pyav). Left "
                             "unset, vLLM's default is used.")

    # ---- frame sampling (backend A -> mm_processor_kwargs) ----
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Pin EVERY video to exactly this many frames, uniformly sampled "
                             "(0 = off, use --fps + --max_frames). Coerced to even; implemented as "
                             "fps + min_frames==max_frames. This is the safe choice for a "
                             "heterogeneous --video_dir: plain fps sampling lets a clip land on an "
                             "ODD frame count, which trips a FATAL assert in vLLM 0.28's Qwen3-VL "
                             "processor and kills the rank. A clip with fewer real frames than "
                             "this is skipped (that is ~1-2% of internVid, the <30 s clips).")
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frames per second vLLM samples from each video (Qwen3-VL builds its "
                             "timestamp markers from this grid). 0 leaves the processor default. "
                             "Ignored when --num_frames is set.")
    parser.add_argument("--max_pixels", type=int, default=0,
                        help="Per-frame pixel ceiling (0 = processor default). Opt-in: only sent "
                             "to the processor when > 0.")
    parser.add_argument("--min_pixels", type=int, default=0, help="Per-frame pixel floor (0 = off).")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Hard cap on sampled frames per video (0 = processor default). Also "
                             "caps the synthetic timestamp grid.")
    parser.add_argument("--min_frames", type=int, default=0, help="Floor on sampled frames (0 = off).")
    parser.add_argument("--no_cap_pixels_per_frame", dest="cap_pixels_per_frame",
                        action="store_false",
                        help="Don't send cap_pixels_per_frame=True. Default on: it makes Qwen3-VL's "
                             "video processor apply the qwen-vl-utils per-frame pixel cap, so long "
                             "clips don't balloon the token count. Harmless on transformers builds "
                             "that lack the kwarg (vLLM drops it with a warning).")
    parser.set_defaults(cap_pixels_per_frame=True)
    parser.add_argument("--mm_processor_kwargs_json", default=None,
                        help="JSON object merged (last, wins) into the video mm_processor_kwargs, "
                             "for keys this script does not expose.")
    parser.add_argument("--timestamps", choices=["none", "uniform"], default="uniform",
                        help="'uniform' writes a SYNTHETIC per-frame time grid re-derived from "
                             "--fps into the JSONL / --annotation_out (frame order real, seconds "
                             "approximate), tagged synthetic. 'none' omits per-frame times.")

    # ---- sampling params ----
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true",
                        help="Sample instead of greedy. Off => temperature 0.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="1.1 only flips near-ties, breaking the greedy repetition loop a "
                             "shaky video falls into; 1.0 disables it.")

    # ---- run control ----
    parser.add_argument("--chunk_size", type=int, default=64,
                        help="Samples handed to one llm.chat() call. vLLM batches internally; this "
                             "only bounds peak preprocessing memory and the resume checkpoint "
                             "granularity (the shard file is flushed after each chunk).")
    parser.add_argument("--no_sort_by_length", action="store_true",
                        help="Process in meta order instead of longest-first. Skips the per-sample "
                             "duration probe entirely -- pass it on a bare --video_dir where every "
                             "clip lacks a duration field and the probe would decord-open every "
                             "file on every rank.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Ignore existing results instead of resuming.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _pin_gpu_for_local_rank()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rank, local_rank, world_size = _init_dist()
    if rank != 0:
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    if args.video_loader_backend:
        os.environ["VLLM_VIDEO_LOADER_BACKEND"] = args.video_loader_backend
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if world_size > 1 and args.tensor_parallel_size > 1:
        parser.error("torchrun data-parallel (WORLD_SIZE>1) and --tensor_parallel_size>1 cannot "
                     "be combined here; pick one.")
    if bool(args.meta_path) == bool(args.video_dir):
        parser.error("pass exactly one of --meta_path / --video_dir.")
    if args.meta_path is not None and not Path(args.meta_path).exists():
        parser.error(f"--meta_path {args.meta_path} does not exist.")
    if args.video_dir is not None and not Path(args.video_dir).is_dir():
        parser.error(f"--video_dir {args.video_dir} is not a directory.")

    prompt = args.prompt or ""
    prompt_pool: Optional[List[Tuple[str, str]]] = None
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()
    elif args.prompt is None:
        prompt_pool = _load_prompt_pool(args.prompt_pool.split(","))
        if rank == 0:
            logger.info(
                "No single prompt given -- drawing one per sample from a pool of %d: %s",
                len(prompt_pool), ", ".join(n for n, _ in prompt_pool),
            )

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    rank_file = output_file.parent / f"{output_file.stem}.rank{rank}{output_file.suffix}"
    if args.overwrite and rank_file.exists():
        rank_file.unlink()

    samples, registry = _load_samples(
        args.meta_path, args.data_root,
        video_dir=args.video_dir,
        annotate_unannotated=args.annotate_unannotated,
        prompt=prompt,
        prompt_source=args.prompt_source,
        prompt_pool=prompt_pool,
        seed=args.seed,
    )
    if args.meta_out and not registry and rank == 0:
        logger.warning(
            "--meta_out was passed but --meta_path is not an anno_data-style registry; the "
            "generated meta will hold a single 'recaption' dataset.",
        )
    if args.limit > 0:
        samples = samples[: args.limit]
    done = set() if args.overwrite else _load_done_ids(output_file)
    todo = [s for s in samples if s["video_id"] not in done]

    media_root = args.data_root or args.video_root
    fixed_frames = args.num_frames and args.num_frames > 0
    if not args.no_sort_by_length and not fixed_frames:
        # Sort the whole list *before* sharding: round-robin over a sorted list
        # hands every rank a near-identical length distribution for free. The
        # probe also caches the duration for _make_record's timestamp grid.
        # Pointless with --num_frames (every clip contributes the same count).
        need_dur = args.timestamps == "uniform"
        for s in todo:
            s["_approx_frames"], s["_duration"] = _probe_clip(
                s, args.fps or 2.0, args.max_frames, args.duration_key, media_root,
                num_frames=args.num_frames, need_duration=need_dur,
            )
        todo.sort(key=lambda s: -(s.get("_approx_frames") or 0))
    shard = todo[rank::world_size]

    if rank == 0:
        logger.info(
            "Samples: %d total | %d already captioned | %d to do | %d on this rank",
            len(samples), len(samples) - len(todo), len(todo), len(shard),
        )

    # Only rank 0 reads `samples` again (at the end, for --annotation_out); the
    # dicts in `shard` are the same objects, so dropping the lists is enough.
    if rank != 0:
        samples = None
    del todo, done

    # ---- caption this rank's shard (skip the engine entirely if it's empty --
    #      a resume of a finished run, or more ranks than remaining work) ----
    if shard:
        from vllm import LLM, SamplingParams

        limit_mm = _parse_kv_int(args.limit_mm_per_prompt)
        mm_kwargs = _build_mm_processor_kwargs(args)
        if rank == 0:
            logger.info(
                "vLLM: model=%s | tp=%d | dp(world)=%d | mm_processor_kwargs=%s | limit_mm=%s",
                args.model, args.tensor_parallel_size, world_size, mm_kwargs, limit_mm,
            )
            if not fixed_frames:
                logger.warning(
                    "Using --fps sampling: a clip whose round(duration*fps) is ODD trips a FATAL "
                    "assert in vLLM 0.28's Qwen3-VL video processor and takes down this rank's "
                    "engine. Pass --num_frames <even> for a heterogeneous --video_dir.",
                )

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            limit_mm_per_prompt=limit_mm,
            mm_processor_kwargs=mm_kwargs or None,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            allowed_local_media_path=args.allowed_local_media_path,
            seed=args.seed,
            disable_log_stats=True,
        )

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature if args.do_sample else 0.0,
            top_p=args.top_p if args.do_sample else 1.0,
            top_k=args.top_k if args.do_sample else -1,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
        )

        n_fail = 0
        fout = open(rank_file, "a", encoding="utf-8")
        pbar = tqdm(total=len(shard), desc=f"[rank {rank}] captioning", disable=(rank != 0))
        for start in range(0, len(shard), args.chunk_size):
            chunk = shard[start:start + args.chunk_size]
            convs: List[List[Dict]] = []
            kept: List[Dict] = []
            for s in chunk:
                try:
                    convs.append(_build_messages(s, args, media_root))
                    kept.append(s)
                except Exception as exc:  # noqa: BLE001
                    n_fail += 1
                    logger.warning("Skipping %s: %s", s["video_id"], exc)

            if convs:
                texts = _run_chat(llm, sampling_params, convs, rank)
                for s, text in zip(kept, texts):
                    if text is None:
                        n_fail += 1
                        continue
                    fout.write(json.dumps(_make_record(s, text, args), ensure_ascii=False) + "\n")
                fout.flush()
            pbar.update(len(chunk))

        pbar.close()
        fout.close()
        if n_fail:
            logger.warning("rank %d: %d samples skipped.", rank, n_fail)
    elif rank == 0:
        logger.info("Nothing to caption on any rank at this shard split; merging existing shards.")

    _barrier()

    if rank == 0:
        # Glob rather than range(world_size): a run resumed under a different GPU
        # count still has to pick up the shards written by the earlier run.
        by_video: Dict[str, Dict] = {}
        for p in sorted(output_file.parent.glob(f"{output_file.stem}.rank*{output_file.suffix}")):
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    by_video[rec["video_id"]] = rec  # last write wins
        records = list(by_video.values())
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote %s with %d captions", output_file, len(records))

        mix: Dict[str, int] = {}
        for rec in records:
            key = rec.get("prompt_name") or "fixed"
            mix[key] = mix.get(key, 0) + 1
        if len(mix) > 1 or prompt_pool is not None:
            logger.info(
                "Prompt mix: %s",
                ", ".join(f"{k}={v} ({100 * v / len(records):.0f}%)"
                          for k, v in sorted(mix.items(), key=lambda kv: -kv[1])),
            )
        if args.timestamps == "uniform":
            logger.warning(
                "Per-frame timestamps in the output are a SYNTHETIC uniform grid at %.4g FPS "
                "(frame order is real; the seconds are re-derived from --fps, not decoded). "
                "Tagged \"timestamps_synthetic\": true -- do not use them for temporal grounding "
                "or dense captioning.", args.fps or 2.0,
            )

        if args.annotation_out or args.meta_out:
            by_id = {s["video_id"]: s for s in samples}
            anno = []
            for rec in records:
                source_sample = by_id.get(rec["video_id"], {})
                base = dict(source_sample.get("raw", {}))
                dataset_name = base.get("_dataset")
                for key in _INTERNAL_KEYS:
                    base.pop(key, None)
                base["video"] = rec["video"]
                if rec.get("timestamps") is not None:
                    base["frame_timestamps"] = rec["timestamps"]
                    base["num_frames"] = rec["num_frames"]
                    if rec.get("timestamps_synthetic"):
                        base["synthetic_timestamps"] = True
                if rec.get("video_duration") is not None:
                    base["video_duration"] = rec["video_duration"]
                if rec.get("prompt_name"):
                    base["prompt_name"] = rec["prompt_name"]
                if rec.get("model"):
                    base["recaption_model"] = rec["model"]
                modal_token = DEFAULT_IMAGE_TOKEN if source_sample.get("source") == "image" else DEFAULT_VIDEO_TOKEN
                base["conversations"] = [
                    {"from": "human", "value": f"{modal_token}\n{rec['prompt']}"},
                    {"from": "gpt", "value": rec["caption"]},
                ]
                anno.append((dataset_name, base))

            if args.annotation_out:
                anno_path = Path(args.annotation_out)
                anno_path.parent.mkdir(parents=True, exist_ok=True)
                with open(anno_path, "w", encoding="utf-8") as f:
                    json.dump([entry for _, entry in anno], f, ensure_ascii=False, indent=2)
                logger.info("Wrote %s with %d samples", anno_path, len(anno))

            if args.meta_out:
                _write_meta_registry(Path(args.meta_out), anno, registry)

    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
