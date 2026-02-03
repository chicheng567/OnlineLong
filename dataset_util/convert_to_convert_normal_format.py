#!/usr/bin/env python3
"""
Convert annotation JSON files to a format compatible with `_convert_normal`.

Example:
    python dataset_util/convert_to_convert_normal_format.py \
        --input anno_online/dense_video_captioning/anet.json \
        --output anno_online/dense_video_captioning/anet.convert_normal.json
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_SINGLE_TURN_QUESTION = (
    "Summarize all the events in this video. For each event, include the duration. "
    "Format your response as: <start time> - <end time> (duration: <x> seconds), "
    "<description>"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and convert a dataset annotation file for _convert_normal."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input annotation JSON path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path.",
    )
    return parser.parse_args()


def normalize_video_field(item: Dict[str, Any]) -> int:
    """
    Ensure `video` follows `_convert_normal` requirements:
    - string -> [string]
    - [string] stays unchanged

    Returns 1 if a conversion was applied, else 0.
    """
    if "video" not in item or item["video"] is None:
        return 0

    video_value = item["video"]
    if isinstance(video_value, str):
        item["video"] = [video_value]
        return 1

    if isinstance(video_value, list) and len(video_value) == 1 and isinstance(video_value[0], str):
        return 0

    raise ValueError(
        f"Unsupported `video` format for sample: {video_value}. "
        "Expected string or single-item string list."
    )


def normalize_conversations(item: Dict[str, Any]) -> int:
    """
    Normalize conversation turns:
    - `from: gpt` -> `from: assistant` (optional cleanup)

    Returns number of modified turns.
    """
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return 0

    changed = 0
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from")
        if role == "gpt":
            turn["from"] = "assistant"
            changed += 1
    return changed


def collapse_to_single_turn(item: Dict[str, Any], question: str) -> int:
    """
    Replace multi-turn conversations with:
    - one human question
    - one assistant answer built by concatenating all assistant turns
    """
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return 0

    answers = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("from")
        if role in {"assistant", "gpt"}:
            text = turn.get("value", "")
            if isinstance(text, str) and text.strip():
                answers.append(text.strip())

    merged_answer = "\n".join(answers).strip()
    item["conversations"] = [
        {"from": "human", "value": f"<video>\n{question}"},
        {"from": "assistant", "value": merged_answer},
    ]
    return 1


def convert_data(data: Any) -> Dict[str, int]:
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of samples.")

    stats = {
        "total_samples": len(data),
        "video_wrapped_to_list": 0,
        "role_renamed_gpt_to_assistant": 0,
        "single_turn_collapsed": 0,
    }

    for item in data:
        if not isinstance(item, dict):
            continue
        stats["video_wrapped_to_list"] += normalize_video_field(item)
        stats["role_renamed_gpt_to_assistant"] += normalize_conversations(item)
        stats["single_turn_collapsed"] += collapse_to_single_turn(item, DEFAULT_SINGLE_TURN_QUESTION)

    return stats


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    converted = copy.deepcopy(raw)
    stats = convert_data(converted)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"Saved converted file to: {output_path}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
