#!/usr/bin/env python3
"""
Video Repackaging Advisor

This script scans annotation-configured datasets and flags videos that are likely
to require repackaging or re-encoding. It cross-validates each video with ffprobe
metadata and a light Decord decode probe to surface common failure modes such as:
    * Missing pixel format / codec parameters
    * Decord decode failures (e.g., `mmco: unref short failure`)
    * Videos that report zero decodable frames

Usage examples:
    python dataset_util/check_video_repackage.py
    python dataset_util/check_video_repackage.py --config anno_data/finetune_online.json --datasets anet_dvc_train
    python dataset_util/check_video_repackage.py --output work_dirs/video_repackage_report.json --verbose
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any

try:
    from decord import VideoReader, cpu
    from decord import DECORDError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    VideoReader = None  # type: ignore
    DECORDError = Exception  # type: ignore

COMMON_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_annotation_file(anno_path: str) -> List[Dict[str, Any]]:
    try:
        with open(anno_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and 'annotations' in data:
                return data['annotations']
    except Exception as exc:
        print(f"Error loading annotation file {anno_path}: {exc}")
    return []


def extract_video_files(annotation_data: List[Dict[str, Any]]) -> List[str]:
    video_files: List[str] = []
    for item in annotation_data:
        if 'video' in item and isinstance(item['video'], str):
            video_files.append(item['video'])
        elif 'image' in item and isinstance(item['image'], str):
            lower_path = item['image'].lower()
            if any(lower_path.endswith(ext) for ext in COMMON_EXTENSIONS):
                video_files.append(item['image'])
    return video_files


def resolve_video_path(video_file: str, data_root: str) -> Path:
    base_path = Path(data_root)
    candidate = base_path / video_file
    if candidate.exists():
        return candidate
    if '.' not in candidate.name:
        for ext in COMMON_EXTENSIONS:
            alt = candidate.with_suffix(ext)
            if alt.exists():
                return alt
    return candidate


def run_ffprobe(video_path: Path, analyzeduration: str = "64M", probesize: str = "64M") -> Dict[str, Any]:
    if not shutil.which("ffprobe"):
        return {"ok": False, "error": "ffprobe_not_found"}

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-hide_banner",
        "-of",
        "json",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,codec_type,pix_fmt,nb_frames,duration",
        "-analyzeduration",
        analyzeduration,
        "-probesize",
        probesize,
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "ffprobe_not_found"}

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": proc.stderr.strip() or proc.stdout.strip(),
        }

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"ffprobe_json_error: {exc}"}

    streams = payload.get("streams", [])
    if not streams:
        return {"ok": False, "error": "ffprobe_no_video_stream"}

    stream = streams[0]
    pix_fmt = stream.get("pix_fmt")
    codec_name = stream.get("codec_name")
    metadata = {
        "pix_fmt": pix_fmt,
        "codec_name": codec_name,
        "nb_frames": stream.get("nb_frames"),
        "duration": stream.get("duration"),
    }

    issues: List[str] = []
    if not pix_fmt or pix_fmt in ("unknown", "none", "N/A", "unspecified"):
        issues.append("missing_pixel_format")
    if codec_name in (None, "unknown", "none"):
        issues.append("missing_codec_name")

    return {"ok": len(issues) == 0, "issues": issues, "metadata": metadata}


def decode_with_decord(video_path: Path) -> Dict[str, Any]:
    if VideoReader is None:
        return {"skipped": True, "reason": "decord_not_installed"}

    try:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    except Exception as exc:  # pylint: disable=broad-except
        return {"ok": False, "error": f"open_error: {exc}"}

    vlen = len(vr)
    if vlen == 0:
        return {"ok": False, "error": "no_decodable_frames"}

    sample_indices = sorted({0, max(vlen - 1, 0), vlen // 2})
    try:
        vr.get_batch(sample_indices)
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = str(exc)
        issue = f"decode_error: {error_msg}"
        if "mmco: unref short failure" in error_msg:
            issue = "decode_error_mmco_unref_short_failure"
        return {"ok": False, "error": issue}

    try:
        vr.get_batch([vlen - 1])
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = str(exc)
        issue = f"tail_decode_error: {error_msg}"
        if "mmco: unref short failure" in error_msg:
            issue = "tail_decode_error_mmco_unref_short_failure"
        return {"ok": False, "error": issue}

    return {"ok": True, "frames": vlen}


def analyze_video(video_path: Path) -> Dict[str, Any]:
    if not video_path.exists():
        return {
            "status": "missing",
            "reasons": ["file_not_found"],
            "metadata": {},
        }

    reasons: List[str] = []
    metadata: Dict[str, Any] = {}

    ffprobe_result = run_ffprobe(video_path)
    if not ffprobe_result.get("ok", False):
        reasons.append(ffprobe_result.get("error", "ffprobe_failure"))
    else:
        metadata.update(ffprobe_result.get("metadata", {}))
        reasons.extend(ffprobe_result.get("issues", []))

    decord_result = decode_with_decord(video_path)
    if not decord_result.get("ok", False):
        if not decord_result.get("skipped", False):
            reasons.append(decord_result.get("error", "decord_failure"))
    else:
        metadata["decoded_frames"] = decord_result.get("frames")

    status = "healthy" if not reasons else "needs_repackage"

    return {
        "status": status,
        "reasons": reasons,
        "metadata": metadata,
    }


def analyze_dataset(dataset_name: str, dataset_config: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    annotation_path = dataset_config["annotation"]
    data_root = dataset_config["data_root"]

    results: List[Dict[str, Any]] = []

    if not os.path.exists(annotation_path):
        return {
            "dataset": dataset_name,
            "status": "annotation_missing",
            "videos_checked": 0,
            "videos_flagged": 0,
            "details": [],
        }

    if not os.path.exists(data_root):
        return {
            "dataset": dataset_name,
            "status": "data_root_missing",
            "videos_checked": 0,
            "videos_flagged": 0,
            "details": [],
        }

    annotations = load_annotation_file(annotation_path)
    video_files = extract_video_files(annotations)

    if verbose:
        print(f"\nDataset: {dataset_name}")
        print(f"  Annotation: {annotation_path}")
        print(f"  Data root:  {data_root}")
        print(f"  Samples detected: {len(video_files)}")

    flagged = 0
    for video_file in video_files:
        video_path = resolve_video_path(video_file, data_root)
        analysis = analyze_video(video_path)
        if analysis["status"] != "healthy":
            flagged += 1
        if verbose:
            reasons = analysis["reasons"] or ["OK"]
            print(f"  - {video_file}: {analysis['status']} | {', '.join(reasons)}")

        results.append({
            "video": video_file,
            "resolved_path": str(video_path),
            **analysis,
        })

    overall_status = "complete" if flagged == 0 else "needs_attention"

    return {
        "dataset": dataset_name,
        "status": overall_status,
        "videos_checked": len(video_files),
        "videos_flagged": flagged,
        "details": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect videos that likely need repackaging or re-encoding.")
    parser.add_argument(
        "--config",
        default="anno_data/finetune_online.json",
        help="Path to dataset configuration JSON.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Restrict the check to specific dataset names defined in the config.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to store a JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-video diagnostics.",
    )
    parser.add_argument(
        "--ignore_missing",
        action="store_true",
        help="Skip videos that are missing on disk instead of flagging them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_dataset_config(config_path)
    selected_datasets = args.datasets or list(config.keys())

    summary: List[Dict[str, Any]] = []
    total_flagged = 0
    total_checked = 0

    for dataset_name in selected_datasets:
        dataset_cfg = config.get(dataset_name)
        if not dataset_cfg:
            print(f"Dataset '{dataset_name}' not found in config, skipping.")
            continue

        result = analyze_dataset(dataset_name, dataset_cfg, verbose=args.verbose)
        summary.append(result)

        total_checked += result["videos_checked"]
        total_flagged += result["videos_flagged"]

        flagged = result["videos_flagged"]
        status_icon = "✅" if flagged == 0 else "⚠️"
        print(f"\n{status_icon} {dataset_name}: checked {result['videos_checked']} videos, flagged {flagged}.")

        if args.ignore_missing:
            for detail in result["details"]:
                if detail["status"] == "missing":
                    total_flagged -= 1

    print("\nSummary:")
    print(f"  Datasets processed: {len(summary)}")
    print(f"  Videos checked:    {total_checked}")
    print(f"  Videos flagged:    {max(total_flagged, 0)}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report written to: {output_path}")


if __name__ == "__main__":
    main()
