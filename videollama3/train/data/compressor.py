"""Windowed-compression SFT dataset + collator, and the shared compression-window
selection helpers.

Extracted from ``videollama3_chat_finetune_compressor.py`` so both it and
``compressor_pretrain_with_videollama3.py`` (and ``videollama3/inference/captioning.py``,
``test_compressor_inference.py``) can import this machinery without importing from a
training entrypoint.
"""
import bisect
import json
import os
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.utils.data
import transformers

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.train.data import common
from videollama3.train.data.common import logger, rank0_print
from videollama3.train.data.supervised import ConcatDatasetWithLengths, LazySupervisedDataset

__all__ = [
    "select_full_compression_parts",
    "select_compression_parts",
    "count_video_frames_in_messages",
    "CompressorLazySupervisedDataset",
    "DataCollatorWithCompressor",
    "SubsetWithLengths",
    "make_compressor_data_module",
]

def _select_non_overlapping_windows(
    num_frames: int,
    window_size: int,
    target_frames: int,
    rng: random.Random,
) -> List[int]:
    if num_frames < window_size:
        return []
    target_windows = max(1, round(target_frames / window_size))
    candidates = list(range(0, num_frames - window_size + 1))
    rng.shuffle(candidates)
    selected = []
    for start in candidates:
        overlap = False
        for prev in selected:
            if not (start + window_size <= prev or prev + window_size <= start):
                overlap = True
                break
        if overlap:
            continue
        selected.append(start)
        if len(selected) >= target_windows:
            break
    return selected


def select_full_compression_parts(
    total_frames: int,
    total_vision_tokens: int,
    window_size: int,
) -> List[List[int]]:
    """Return compression_parts that cover every frame in the video.

    Frames are divided into consecutive windows of `window_size`.
    If total_frames % window_size > 3, the leftover frames form one extra part.
    If the remainder is <= 3 it is discarded.
    """
    if total_frames <= 0 or window_size <= 0 or total_frames < window_size:
        return []
    assert total_vision_tokens % total_frames == 0, (
        f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
    )
    tokens_per_frame = total_vision_tokens // total_frames
    n_full = total_frames // window_size
    remainder = total_frames % window_size
    selected_idx = []
    for i in range(n_full):
        s = i * window_size * tokens_per_frame
        e = (i + 1) * window_size * tokens_per_frame
        selected_idx.append([s, e])
    if remainder > 3:
        s = n_full * window_size * tokens_per_frame
        selected_idx.append([s, total_vision_tokens])
    return selected_idx


def select_compression_parts(
    total_frames: int,
    total_vision_tokens: int,
    ratio: float,
    window_size: int,
    rng: random.Random,
) -> List[List[int]]:
    if total_frames <= 0 or ratio <= 0 or window_size <= 0 or total_frames < window_size:
        return []
    assert total_vision_tokens % total_frames == 0, f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
    tokens_per_frame = total_vision_tokens // total_frames
    target_frames = max(window_size, int(round(total_frames * ratio)))
    starts = _select_non_overlapping_windows(total_frames, window_size, target_frames, rng)
    starts.sort()
    selected_idx = []
    for start in starts:
        assert start + window_size <= total_frames, f"Selected window [{start}, {start + window_size}) exceeds total frames {total_frames}."
        selected_idx.append([start * tokens_per_frame, (start + window_size) * tokens_per_frame])
    
    return selected_idx


def count_video_frames_in_messages(messages: List[Dict]) -> int:
    total_frames = 0
    for message in messages:
        if message.get("role") != "user":
            continue
        for content in message.get("content", []):
            if isinstance(content, dict) and content.get("type") == "video":
                total_frames += int(content.get("num_frames", 0))
    return total_frames


class CompressorLazySupervisedDataset(LazySupervisedDataset):
    def __init__(self, *args, compression_ratio: float = 0.3, compression_window_size: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.compression_ratio = compression_ratio
        self.compression_window_size = compression_window_size
        assert compression_window_size - 2 > 1, "Compression window size cannot be less than 3."

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw_sample = self.list_data_dict[i]
        try:
            sample = raw_sample
            if self.online_mode:
                modal, images, messages, merge_size = self._convert_online_video(sample)
            else:
                if "stream" in sample and sample["stream"]:
                    raise NotImplementedError("Online stream data is not supported in compressor training yet.")
                    modal, images, messages, merge_size = self._convert_stream(sample)
                else:
                    modal, images, messages, merge_size = self._convert_normal(sample)
            assert modal == "video", "Compressor training currently only supports video data."

            # Extract per-frame timestamps from messages before passing to processor,
            # so we can later compute range timestamps for each compression window.
            frame_timestamps = None
            for msg in messages:
                if msg.get("role") == "user":
                    for content_item in msg.get("content", []):
                        if isinstance(content_item, dict) and content_item.get("type") == "video":
                            frame_timestamps = content_item.get("timestamps", None)
                            break
                if frame_timestamps is not None:
                    break

            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=self.return_label,
                return_tensors="pt",
            )
            data_dict["modals"] = [modal] * len(images)
            if modal == "video":
                image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
                # NOTE: data_dict["pixel_values"].shape[0] will be 4 times than total_vision token because of 4x4 merging.
                total_vision_tokens = int((data_dict["input_ids"] == image_token_id).sum().item())
                total_frames = len(images[0])
                if total_frames < self.compression_window_size:
                    # Too few frames to form even one compression window; retry with a different sample
                    # to ensure every batch has valid compression_parts on every rank. Mixed
                    # compression/no-compression paths between ranks cause NCCL AllReduce desync
                    # when DeepSpeed overlap_comm fires during backward.
                    backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                    logger.warning(
                        "Sample %s has only %d frames (< window_size %d); retrying with sample %s.",
                        i, total_frames, self.compression_window_size, backup_idx,
                    )
                    return self.__getitem__(backup_idx)
                compression_part = select_compression_parts(
                    total_frames=total_frames,
                    total_vision_tokens=total_vision_tokens,
                    ratio=self.compression_ratio,
                    window_size=self.compression_window_size,
                    rng=random,
                )
                compression_part_full = select_full_compression_parts(
                    total_frames=total_frames,
                    total_vision_tokens=total_vision_tokens,
                    window_size=self.compression_window_size,
                )
            else:
                compression_part = []
                compression_part_full = []
            data_dict["compression_parts"] = compression_part
            data_dict["compression_parts_full"] = compression_part_full

            # Build per-part timestamp replacement info: (old_ts_token_count, new_ts_token_ids).
            # The chat template emits "Time X.0s:" before each frame's image tokens.
            # We record how long that old string is (to know how far back to cut) and
            # what the range string tokenizes to (so prepare_inputs_labels can splice it in).
            tokenizer = self.vlprocessor.tokenizer

            def _build_ts_info(parts):
                ts_info: List[tuple] = []
                if parts and frame_timestamps is not None and total_vision_tokens > 0:
                    tpf = total_vision_tokens // total_frames
                    for s, e in parts:
                        frame_s = s // tpf
                        frame_e = e // tpf
                        ts_start = float(frame_timestamps[frame_s])
                        ts_end = float(frame_timestamps[min(frame_e, len(frame_timestamps)) - 1])
                        old_ts_str = f"Time {ts_start:.1f}s:"
                        new_ts_str = f"Time:{ts_start:.1f}s-{ts_end:.1f}s:"
                        old_ts_ids = tokenizer.encode(old_ts_str, add_special_tokens=False)
                        new_ts_ids = tokenizer.encode(new_ts_str, add_special_tokens=False)
                        ts_info.append((len(old_ts_ids), new_ts_ids))
                else:
                    ts_info = [(0, []) for _ in parts]
                return ts_info

            data_dict["compression_ts_info"] = _build_ts_info(compression_part)
            data_dict["compression_ts_info_full"] = _build_ts_info(compression_part_full)

        except Exception:
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            logger.exception("Failed to process sample %s. Fallback index: %s.", i, backup_idx)
            return self.__getitem__(backup_idx)
        return data_dict


@dataclass
class DataCollatorWithCompressor:
    vlprocessor: transformers.ProcessorMixin

    def __call__(self, instances: Sequence[Dict], separator_id=-100) -> Dict[str, torch.Tensor]:
        # input_ids: List[torch.Tensor], labels: List[torch.Tensor]
        input_ids, labels, compression_parts = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "compression_parts"))
        new_input_ids = []
        new_labels = []
        position_ids = []
        new_compression_parts: List[List[int]] = []
        new_compression_ts_info: List[tuple] = []
        new_compression_parts_full: List[List[int]] = []
        new_compression_ts_info_full: List[tuple] = []
        # Two-stage fold side inputs (one entry per compression_part == per video;
        # offset-independent, so just concatenate in instance order).
        new_compression_retained: List[List[int]] = []
        new_compression_seed: List[int] = []
        new_compression_frame_sec: List[List[int]] = []
        new_compression_qbase_only: List[bool] = []
        accumulated_length = 0
        image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        for sample_idx in range(0, len(input_ids)):
            if input_ids[sample_idx].shape[0] > self.vlprocessor.tokenizer.model_max_length:
                warnings.warn(
                    f"Sample {sample_idx} length {input_ids[sample_idx].shape[0]} exceeds model max length "
                    f"{self.vlprocessor.tokenizer.model_max_length}. It will be truncated."
                )
            capped_ids = input_ids[sample_idx][: self.vlprocessor.tokenizer.model_max_length]
            capped_labels = labels[sample_idx][: self.vlprocessor.tokenizer.model_max_length]
            capped_labels[0] = separator_id
            new_input_ids.append(capped_ids)
            new_labels.append(capped_labels)
            position_ids.append(torch.arange(len(capped_ids), dtype=torch.long))
            new_compression_parts.extend([[startend[0] + accumulated_length, startend[1] + accumulated_length] for startend in compression_parts[sample_idx]])
            # Carry through per-part timestamp replacement info (offset-independent: it's token IDs, not positions).
            if "compression_ts_info" in instances[sample_idx]:
                new_compression_ts_info.extend(instances[sample_idx]["compression_ts_info"])
            else:
                new_compression_ts_info.extend([(0, []) for _ in compression_parts[sample_idx]])
            new_compression_retained.extend(instances[sample_idx].get("compression_retained", []))
            new_compression_seed.extend(instances[sample_idx].get("compression_seed", []))
            new_compression_frame_sec.extend(instances[sample_idx].get("compression_frame_sec", []))
            new_compression_qbase_only.extend(
                instances[sample_idx].get(
                    "compression_qbase_only", [False] * len(compression_parts[sample_idx])
                )
            )
            # Full compression parts (same offset logic).
            parts_full = instances[sample_idx].get("compression_parts_full", [])
            new_compression_parts_full.extend([[s + accumulated_length, e + accumulated_length] for s, e in parts_full])
            if "compression_ts_info_full" in instances[sample_idx]:
                new_compression_ts_info_full.extend(instances[sample_idx]["compression_ts_info_full"])
            else:
                new_compression_ts_info_full.extend([(0, []) for _ in parts_full])
            image_token_count = int((capped_ids == image_token_id).sum().item())
            accumulated_length += image_token_count

        flat_input_ids = torch.cat(new_input_ids)
        flat_labels = torch.cat(new_labels)
        flat_position_ids = torch.cat(position_ids)

        batch = dict(
            input_ids=flat_input_ids.unsqueeze(0),
            labels=flat_labels.unsqueeze(0),
            position_ids=flat_position_ids.unsqueeze(0),
        )
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances])
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances])
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances])
        batch["modals"] = sum([x["modals"] for x in instances], [])
        batch["compression_parts"] = new_compression_parts
        batch["compression_ts_info"] = new_compression_ts_info
        batch["compression_parts_full"] = new_compression_parts_full
        batch["compression_ts_info_full"] = new_compression_ts_info_full
        if new_compression_retained:
            batch["compression_retained"] = new_compression_retained
        if new_compression_seed:
            batch["compression_seed"] = new_compression_seed
        if new_compression_frame_sec:
            batch["compression_frame_sec"] = new_compression_frame_sec
        if any(new_compression_qbase_only):
            batch["compression_qbase_only"] = new_compression_qbase_only

        return batch


class SubsetWithLengths(torch.utils.data.Subset):
    """Subset that preserves `lengths`, `modality_lengths` and `compression_depths`
    for grouped sampling."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        parent_lengths = dataset.lengths
        parent_modality = dataset.modality_lengths
        self._lengths = [parent_lengths[i] for i in indices]
        self._modality_lengths = [parent_modality[i] for i in indices]
        # Keep the depth-class grouped sampler (--group_by_compression_depth) working
        # through a validation split; the parent list is in global-index order.
        parent_depths = getattr(dataset, "compression_depths", None)
        self._compression_depths = (
            [parent_depths[i] for i in indices] if parent_depths is not None else None
        )

    @property
    def lengths(self):
        return self._lengths

    @property
    def modality_lengths(self):
        return self._modality_lengths

    @property
    def compression_depths(self):
        return self._compression_depths


def _collect_val_video_paths(dataset, val_indices: List[int]) -> List[str]:
    """Return the full video paths for the given global indices in dataset.

    Handles both a single CompressorLazySupervisedDataset and a
    ConcatDatasetWithLengths that wraps multiple sub-datasets.
    """
    paths: List[str] = []
    for idx in val_indices:
        if isinstance(dataset, torch.utils.data.ConcatDataset):
            ds_idx = bisect.bisect_right(dataset.cumulative_sizes, idx)
            local_idx = idx if ds_idx == 0 else idx - dataset.cumulative_sizes[ds_idx - 1]
            sub_ds = dataset.datasets[ds_idx]
        else:
            sub_ds = dataset
            local_idx = idx

        sample = sub_ds.list_data_dict[local_idx]
        video = sample.get("video")
        if video is None:
            continue
        if isinstance(video, list):
            video = video[0]

        if sub_ds.online_mode:
            root = sub_ds.dataset_root or ""
        else:
            root = getattr(sub_ds.data_args, "data_folder", None) or ""

        full_path = os.path.join(root, video) if root else video
        paths.append(full_path)
    return paths


def make_compressor_data_module(vlprocessor: transformers.ProcessorMixin, data_args, output_dir: Optional[str] = None) -> Dict:
    if data_args.multi_dataset:
        rank0_print("Use meta file to control datasets loading. Data path will use as meta path")
        ds_collection = dict()
        meta_path = data_args.data_path[0]
        ds_collection.update(json.loads(open(meta_path).read()))
        collected_datasets = []
        for dataset_name, dataset_cfg in ds_collection.items():
            collected_datasets.append(
                CompressorLazySupervisedDataset(
                    vlprocessor=vlprocessor,
                    data_path=[dataset_cfg["annotation"]],
                    data_args=data_args,
                    dataset_name=dataset_name,
                    dataset_root=dataset_cfg["data_root"],
                    online_mode=dataset_cfg["online_mode"],
                    prefix_captioning=dataset_cfg.get("prefix_captioning", False),
                    compression_ratio=data_args.compression_ratio,
                    compression_window_size=data_args.compression_window_size,
                )
            )
        train_dataset = ConcatDatasetWithLengths(collected_datasets)
    else:
        train_dataset = CompressorLazySupervisedDataset(
            vlprocessor=vlprocessor,
            data_path=data_args.data_path,
            data_args=data_args,
            compression_ratio=data_args.compression_ratio,
            compression_window_size=data_args.compression_window_size,
        )
    if data_args.validation_split_rate > 0:
        n_total = len(train_dataset)
        n_val = max(1, int(round(n_total * data_args.validation_split_rate)))
        n_train = n_total - n_val
        indices = list(range(n_total))
        random.shuffle(indices)
        val_indices = indices[n_train:]
        original_dataset = train_dataset
        eval_dataset = SubsetWithLengths(train_dataset, val_indices)
        train_dataset = SubsetWithLengths(train_dataset, indices[:n_train])
        if output_dir is not None and common.local_rank in (0, -1):
            val_paths = _collect_val_video_paths(original_dataset, val_indices)
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "val_video_paths.txt")
            with open(out_path, "w") as f:
                for p in val_paths:
                    f.write(p + "\n")
            rank0_print(f"[INFO] Val dataset video paths ({len(val_paths)}) saved to {out_path}")
    else:
        eval_dataset = None
    data_collator = DataCollatorWithCompressor(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
