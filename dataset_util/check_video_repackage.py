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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from decord import VideoReader, cpu
    from decord import DECORDError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    VideoReader = None  # type: ignore
    DECORDError = Exception  # type: ignore

COMMON_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']


@dataclass
class RepackageOptions:
    output_root: Path
    codec: str
    pix_fmt: str
    crf: int
    preset: str
    audio_codec: str
    suffix: str
    replace_original: bool


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


def get_video_identifier(item: Dict[str, Any]) -> Optional[str]:
    value = item.get('video')
    if isinstance(value, str):
        return value
    image = item.get('image')
    if isinstance(image, str):
        lower_path = image.lower()
        if any(lower_path.endswith(ext) for ext in COMMON_EXTENSIONS):
            return image
    return None


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


def repackage_video(
    video_path: Path,
    dataset_name: str,
    relative_video: str,
    options: RepackageOptions,
) -> Dict[str, Any]:
    if not shutil.which("ffmpeg"):
        return {"performed": False, "error": "ffmpeg_not_found"}

    if not video_path.exists():
        return {"performed": False, "error": "source_missing"}

    rel_path = Path(relative_video)
    rel_parent = rel_path.parent
    base_stem = rel_path.stem or video_path.stem or video_path.name
    # Keep dataset separation under the output root to avoid collisions.
    dataset_root = options.output_root / dataset_name / rel_parent
    dataset_root.mkdir(parents=True, exist_ok=True)

    target_name = f"{base_stem}{options.suffix}.mp4"
    target_path = dataset_root / target_name

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-c:v",
        options.codec,
        "-pix_fmt",
        options.pix_fmt,
        "-preset",
        options.preset,
        "-crf",
        str(options.crf),
        "-movflags",
        "+faststart",
        "-c:a",
        options.audio_codec,
        str(target_path),
    ]

    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if proc.returncode != 0:
        return {
            "performed": False,
            "error": proc.stderr.strip() or proc.stdout.strip() or "ffmpeg_failed",
            "command": cmd,
        }

    backup_path: Optional[Path] = None
    replaced = False
    final_path = target_path
    if options.replace_original:
        backup_path = video_path.with_name(video_path.name + ".backup")
        if not backup_path.exists():
            shutil.copy2(video_path, backup_path)
        shutil.copy2(target_path, video_path)
        final_path = video_path
        replaced = True

    return {
        "performed": True,
        "output_path": str(target_path),
        "final_path": str(final_path),
        "replaced_original": replaced,
        "backup_path": str(backup_path) if backup_path else None,
        "command": cmd,
        "ffmpeg_stdout": proc.stdout.strip(),
        "ffmpeg_stderr": proc.stderr.strip(),
    }


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


def analyze_dataset(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    verbose: bool = False,
    repackage_options: Optional[RepackageOptions] = None,
) -> Dict[str, Any]:
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
    repackaged = 0
    for video_file in video_files:
        video_path = resolve_video_path(video_file, data_root)
        analysis = analyze_video(video_path)
        repair_info: Optional[Dict[str, Any]] = None
        if analysis["status"] != "healthy":
            flagged += 1

        if (
            repackage_options
            and analysis["status"] not in ("healthy", "missing")
        ):
            repair_info = repackage_video(video_path, dataset_name, video_file, repackage_options)
            if repair_info.get("performed"):
                repackaged += 1
                target_path = video_path if repackage_options.replace_original else Path(repair_info["output_path"])
                recheck = analyze_video(target_path)
                repair_info["post_check"] = recheck
                if repackage_options.replace_original and recheck["status"] == "healthy":
                    analysis = recheck
                    flagged -= 1
                    flagged = max(flagged, 0)

        if verbose:
            reasons = analysis["reasons"] or ["OK"]
            print(f"  - {video_file}: {analysis['status']} | {', '.join(reasons)}")

        entry = {
            "video": video_file,
            "resolved_path": str(video_path),
            **analysis,
        }
        if repair_info:
            entry["repair"] = repair_info
        results.append(entry)

    overall_status = "complete" if flagged == 0 else "needs_attention"

    return {
        "dataset": dataset_name,
        "status": overall_status,
        "videos_checked": len(video_files),
        "videos_flagged": flagged,
        "videos_repackaged": repackaged,
        "details": results,
    }


def collect_problematic_videos(details: List[Dict[str, Any]]) -> Set[str]:
    problematic: Set[str] = set()
    for detail in details:
        video = detail.get("video")
        status = detail.get("status")
        if not video or status == "healthy":
            continue
        problematic.add(video)
    return problematic


def build_repackage_report(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reduce the verbose dataset summary to only the entries that actually need
    repackaging so the JSON report stays focused on actionable failures.
    """
    report: List[Dict[str, Any]] = []
    for dataset_result in summary:
        dataset_name = dataset_result.get("dataset")
        flagged_details: List[Dict[str, Any]] = []
        for detail in dataset_result.get("details", []):
            if detail.get("status") != "needs_repackage":
                continue
            minimal_detail = {
                "video": detail.get("video"),
                "resolved_path": detail.get("resolved_path"),
                "status": detail.get("status"),
                "reasons": detail.get("reasons"),
                "metadata": detail.get("metadata"),
            }
            if "repair" in detail:
                minimal_detail["repair"] = detail["repair"]
            flagged_details.append(minimal_detail)
        if flagged_details:
            report.append(
                {
                    "dataset": dataset_name,
                    "videos_flagged": len(flagged_details),
                    "details": flagged_details,
                }
            )
    return report


def remove_entries_from_annotation(annotation_path: str, bad_videos: Set[str]) -> Dict[str, Any]:
    try:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        return {"removed": 0, "error": f"load_error: {exc}"}

    if isinstance(data, list):
        entries = data
        container_type = "list"
    elif isinstance(data, dict) and isinstance(data.get("annotations"), list):
        entries = data["annotations"]
        container_type = "dict"
    else:
        return {"removed": 0, "error": "unsupported_annotation_format"}

    filtered: List[Dict[str, Any]] = []
    removed = 0
    for item in entries:
        identifier = get_video_identifier(item)
        if identifier and identifier in bad_videos:
            removed += 1
            continue
        filtered.append(item)

    if removed == 0:
        return {"removed": 0, "error": None}

    backup_path = annotation_path + ".backup"
    with open(backup_path, "w", encoding="utf-8") as backup_file:
        json.dump(data, backup_file, indent=2, ensure_ascii=False)

    if container_type == "list":
        new_payload: Any = filtered
    else:
        data["annotations"] = filtered
        new_payload = data

    with open(annotation_path, "w", encoding="utf-8") as dest:
        json.dump(new_payload, dest, indent=2, ensure_ascii=False)

    return {
        "removed": removed,
        "remaining": len(filtered),
        "backup_path": backup_path,
    }


def fix_problematic_entries(config: Dict[str, Any], summary: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 60)
    print("Fixing annotations with --fix-missing")
    print("=" * 60)

    total_removed = 0
    for dataset_result in summary:
        dataset_name = dataset_result.get("dataset")
        details = dataset_result.get("details", [])
        bad_videos = collect_problematic_videos(details)
        if not bad_videos:
            continue

        dataset_cfg = config.get(dataset_name, {})
        annotation_path = dataset_cfg.get("annotation")
        if not annotation_path or not os.path.exists(annotation_path):
            print(f"Skipping {dataset_name}: annotation file missing")
            continue

        result = remove_entries_from_annotation(annotation_path, bad_videos)
        removed = result.get("removed", 0)
        total_removed += removed
        if removed == 0:
            print(f"{dataset_name}: no entries removed (already clean)")
            continue

        backup_path = result.get("backup_path")
        remaining = result.get("remaining")
        print(
            f"{dataset_name}: removed {removed} entries (remaining {remaining}). "
            f"Backup -> {backup_path}"
        )

    if total_removed == 0:
        print("No problematic entries were removed. All datasets already clean.")
    else:
        print(f"Total entries removed: {total_removed}")


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
    parser.add_argument(
        "--auto-repackage",
        action="store_true",
        help="Automatically attempt to re-encode flagged videos with ffmpeg.",
    )
    parser.add_argument(
        "--repackage-dir",
        default="work_dirs/repackaged_videos",
        help="Directory where repaired videos (and logs) are stored.",
    )
    parser.add_argument(
        "--repackage-codec",
        default="libx264",
        help="Video codec to use when auto-repackaging.",
    )
    parser.add_argument(
        "--repackage-pix-fmt",
        default="yuv420p",
        help="Pixel format to enforce during re-encoding.",
    )
    parser.add_argument(
        "--repackage-crf",
        type=int,
        default=18,
        help="CRF quality value for ffmpeg when auto-repackaging.",
    )
    parser.add_argument(
        "--repackage-preset",
        default="medium",
        help="ffmpeg preset to use for the video encoder.",
    )
    parser.add_argument(
        "--repackage-audio-codec",
        default="aac",
        help="Audio codec to use for the repaired file (use 'copy' to keep original).",
    )
    parser.add_argument(
        "--repackage-suffix",
        default="_fixed",
        help="Suffix appended to repaired filenames before the extension.",
    )
    parser.add_argument(
        "--replace-original",
        action="store_true",
        help="After a successful repair, copy the new file over the original (backup is kept).",
    )
    parser.add_argument(
        "--fix-missing",
        action="store_true",
        help="Remove annotation entries referencing videos that remain missing or damaged.",
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
    total_repackaged = 0

    repackage_options: Optional[RepackageOptions] = None
    if args.auto_repackage:
        output_root = Path(args.repackage_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        repackage_options = RepackageOptions(
            output_root=output_root,
            codec=args.repackage_codec,
            pix_fmt=args.repackage_pix_fmt,
            crf=args.repackage_crf,
            preset=args.repackage_preset,
            audio_codec=args.repackage_audio_codec,
            suffix=args.repackage_suffix,
            replace_original=args.replace_original,
        )

    for dataset_name in selected_datasets:
        dataset_cfg = config.get(dataset_name)
        if not dataset_cfg:
            print(f"Dataset '{dataset_name}' not found in config, skipping.")
            continue

        result = analyze_dataset(
            dataset_name,
            dataset_cfg,
            verbose=args.verbose,
            repackage_options=repackage_options,
        )
        summary.append(result)

        total_checked += result["videos_checked"]
        total_flagged += result["videos_flagged"]
        total_repackaged += result.get("videos_repackaged", 0)

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
    if args.auto_repackage:
        print(f"  Videos repaired:   {total_repackaged}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_summary = build_repackage_report(summary)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered_summary, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report written to: {output_path} (flagged videos only)")

    if args.fix_missing:
        fix_problematic_entries(config, summary)


if __name__ == "__main__":
    main()
