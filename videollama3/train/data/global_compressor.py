"""Whole-video ("global") compression dataset for the Phase-1 qbase CE pretrain, plus
the frame/message helpers it shares with the Phase-2 fold script.

Extracted from ``compressor_pretrain_with_videollama3.py`` so
``phase2_pretrain_fold.py`` can subclass ``GlobalCompressorLazySupervisedDataset``
and reuse the helpers without importing (or monkeypatching) a training entrypoint module.

Frames are always decoded and encoded on the fly (there is no pre-extracted
vision-feature cache -- see ``docs/two_stage_compression_design.md`` §"On-the-fly
encoding").
"""
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.train.data import common
from videollama3.train.data.common import cast_pixel_values_, logger, rank0_print
from videollama3.train.data.compressor import (
    DataCollatorWithCompressor,
    SubsetWithLengths,
    _collect_val_video_paths,
)
from videollama3.train.data.supervised import ConcatDatasetWithLengths, LazySupervisedDataset

__all__ = [
    "get_video_content",
    "resample_indices",
    "resample_video_frames",
    "build_range_ts_info",
    "GlobalCompressorLazySupervisedDataset",
    "make_global_compressor_data_module",
]


def get_video_content(messages: List[Dict]) -> Dict:
    """Return the sample's single video content dict (the block the chat template
    renders as "Time X.0s:<image>,..."). Multi-block samples are rejected."""
    contents = [
        content
        for message in messages if message.get("role") == "user"
        for content in message.get("content", [])
        if isinstance(content, dict) and content.get("type") == "video"
    ]
    assert len(contents) == 1, (
        f"Whole-video compression expects exactly one video block per sample, got {len(contents)} "
        f"(multi-turn online data is not supported here)."
    )
    return contents[0]


def _rewrite_image_block_as_single_frame_video(messages: List[Dict]) -> None:
    """In-place: turn ``_convert_normal``'s ``{"type": "image"}`` content block into a
    one-frame ``{"type": "video"}`` block, so a still image flows through the
    whole-video compression path unchanged.

    ``get_video_content`` then finds its single video block, ``content["num_frames"]``
    is 1, the one compression part covers the frame's ``HW`` tokens, and -- because no
    timestamps are attached -- ``build_range_ts_info`` returns ``[(0, [])]`` and the
    chat template renders a single bare ``<image>`` token with no ``"Time X.0s:"``
    prefix. The image is still fed to ``vlprocessor`` as raw pixels (``merge_size`` is
    switched to ``video_merge_size`` by the caller so one 448px frame yields the same
    ``HW`` tokens a decoded video frame does).
    """
    for message in messages:
        if message.get("role") != "user":
            continue
        for content in message.get("content", []):
            if isinstance(content, dict) and content.get("type") == "image":
                content.pop("timestamp", None)
                content["type"] = "video"
                content["num_frames"] = 1


def resample_indices(num_frames: int, target: int) -> List[int]:
    """Frame indices that turn `num_frames` into exactly `target` frames:
    uniform subsample when longer, last frame repeated when shorter."""
    if target <= 0 or num_frames == target:
        return list(range(num_frames))
    if num_frames > target:
        return np.linspace(0, num_frames - 1, target).round().astype(int).tolist()
    return list(range(num_frames)) + [num_frames - 1] * (target - num_frames)


def resample_video_frames(images, content: Dict, fixed_frames: int):
    """Resample the sample's frames to `fixed_frames`, keeping the chat template's
    metadata (`num_frames` / `timestamps`) in sync.

    `_convert_normal` returns `images` as `[frame_sequence]`; `_convert_online_video`
    returns the frame sequence itself. Both are handled and returned in their
    original shape.
    """
    num_frames = int(content["num_frames"])
    idx = resample_indices(num_frames, fixed_frames)
    if idx == list(range(num_frames)):
        return images

    nested = len(images) == 1 and num_frames != 1
    frames = images[0] if nested else images
    if isinstance(frames, np.ndarray):
        frames = frames[np.asarray(idx, dtype=int)]
    else:
        frames = [frames[j] for j in idx]

    timestamps = content.get("timestamps", None)
    if timestamps is not None:
        content["timestamps"] = [float(timestamps[j]) for j in idx]
    content["num_frames"] = len(idx)
    return [frames] if nested else frames


def build_range_ts_info(content: Dict, tokenizer) -> List[Tuple[int, List[int]]]:
    """(token length of the old "Time X.0s:" prefix, token ids of the new range string).

    The old string is reproduced exactly the way the chat template renders it
    (``'Time ' + ts|round(1)|string + 's:'``) so that
    ``prepare_inputs_labels_for_multimodal`` cuts back the right number of tokens.
    """
    timestamps = content.get("timestamps", None)
    # timestamps may be a list OR a numpy array (the decoder synthesizes per-frame
    # seconds when the annotation carries none) — avoid `not ndarray`.
    if timestamps is None or len(timestamps) == 0:
        return [(0, [])]
    ts_start, ts_end = float(timestamps[0]), float(timestamps[-1])
    old_ts_ids = tokenizer.encode(f"Time {round(ts_start, 1)}s:", add_special_tokens=False)
    new_ts_ids = tokenizer.encode(f"Time:{ts_start:.1f}s-{ts_end:.1f}s:", add_special_tokens=False)
    return [(len(old_ts_ids), new_ts_ids)]


class GlobalCompressorLazySupervisedDataset(LazySupervisedDataset):
    """Dataset that marks each sample's whole video for a single compression pass.

    Frames are decoded and run through the frozen encoder every step; the message
    assembly, frame budget, compression part, range timestamp and length check are
    shared with the still-image path.
    """

    def __init__(self, *args, fixed_frames: int = 0, model_args=None, qbase_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.fixed_frames = fixed_frames
        # Phase-2 pure-qbase replay: when True, this dataset's samples bypass the
        # unit split + fold and route straight through stage-1 (N*K qbase tokens).
        # Set per meta-JSON entry ("qbase_only": true); only Phase2FoldDataset acts
        # on it. docs/two_stage_compression_design.md §4 Phase 2.
        self.qbase_only = bool(qbase_only)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.list_data_dict[i]
            if self.online_mode:
                modal, images, messages, merge_size = self._convert_online_video(sample)
            else:
                modal, images, messages, merge_size = self._convert_normal(sample)
            is_still_image = modal == "image"
            if is_still_image:
                # Compress a still image as a one-frame video (see helper). The
                # pixels stay raw; only the content block and merge_size change.
                _rewrite_image_block_as_single_frame_video(messages)
                modal, merge_size = "video", self.data_args.video_merge_size
            assert modal == "video", "Compressor training currently only supports video data."
            content = get_video_content(messages)
            # fixed_frames is a temporal resample; a still image stays at T=1.
            if self.fixed_frames > 0 and not is_still_image:
                images = resample_video_frames(images, content, self.fixed_frames)
            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=self.return_label,
                return_tensors="pt",
            )
            # fp32 patches are what crosses worker -> main through /dev/shm; see
            # cast_pixel_values_.
            cast_pixel_values_(data_dict, getattr(self.data_args, "pixel_values_dtype", None))
            data_dict["modals"] = [modal] * len(images)

            total_frames = int(content["num_frames"])
            assert total_frames > 0, f"Sample {i} has no frames."

            # The sequence still carries the UNCOMPRESSED T x HW image tokens here
            # (compression happens inside the model), so it has to survive the
            # collator's model_max_length truncation intact -- otherwise the part
            # indexes past the end of the surviving image tokens.
            max_len = self.vlprocessor.tokenizer.model_max_length
            seq_len = int(data_dict["input_ids"].shape[-1])
            if seq_len > max_len:
                backup_idx = random.randint(0, len(self.list_data_dict) - 1)
                logger.warning(
                    "Sample %s: pre-compression length %d exceeds model_max_length %d (%d frames). "
                    "Lower --max_frames or --fixed_frames. Retrying with sample %s.",
                    i, seq_len, max_len, total_frames, backup_idx,
                )
                return self.__getitem__(backup_idx)

            image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            total_vision_tokens = int((data_dict["input_ids"] == image_token_id).sum().item())
            assert total_vision_tokens % total_frames == 0, (
                f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
            )
            # One part covering every vision token of the sample. Indices count image
            # tokens only, not sequence positions.
            data_dict["compression_parts"] = [[0, total_vision_tokens]]
            data_dict["compression_ts_info"] = build_range_ts_info(content, self.vlprocessor.tokenizer)

        except Exception:
            backup_idx = random.randint(0, len(self.list_data_dict) - 1)
            logger.exception("Failed to process sample %s. Fallback index: %s.", i, backup_idx)
            return self.__getitem__(backup_idx)
        return data_dict


def make_global_compressor_data_module(
    vlprocessor: transformers.ProcessorMixin,
    data_args,
    output_dir: Optional[str] = None,
    dataset_cls=None,
    model_args=None,
) -> Dict:
    if dataset_cls is None:
        dataset_cls = GlobalCompressorLazySupervisedDataset
    if data_args.multi_dataset:
        rank0_print("Use meta file to control datasets loading. Data path will use as meta path")
        ds_collection = json.loads(open(data_args.data_path[0]).read())
        collected_datasets = [
            dataset_cls(
                vlprocessor=vlprocessor,
                data_path=[dataset_cfg["annotation"]],
                data_args=data_args,
                dataset_name=dataset_name,
                dataset_root=dataset_cfg["data_root"],
                online_mode=dataset_cfg["online_mode"],
                prefix_captioning=dataset_cfg.get("prefix_captioning", False),
                fixed_frames=data_args.fixed_frames,
                model_args=model_args,
                qbase_only=dataset_cfg.get("qbase_only", False),
            )
            for dataset_name, dataset_cfg in ds_collection.items()
        ]
        train_dataset = ConcatDatasetWithLengths(collected_datasets)
    else:
        train_dataset = dataset_cls(
            vlprocessor=vlprocessor,
            data_path=data_args.data_path,
            data_args=data_args,
            fixed_frames=data_args.fixed_frames,
            model_args=model_args,
            qbase_only=getattr(data_args, "qbase_only", False),
        )

    if data_args.validation_split_rate > 0:
        n_total = len(train_dataset)
        n_val = max(1, int(round(n_total * data_args.validation_split_rate)))
        indices = list(range(n_total))
        random.shuffle(indices)
        val_indices = indices[n_total - n_val:]
        original_dataset = train_dataset
        eval_dataset = SubsetWithLengths(train_dataset, val_indices)
        train_dataset = SubsetWithLengths(train_dataset, indices[:n_total - n_val])
        if output_dir is not None and common.local_rank in (0, -1):
            val_paths = _collect_val_video_paths(original_dataset, val_indices)
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, "val_video_paths.txt")
            with open(out_path, "w") as f:
                f.write("\n".join(val_paths) + "\n")
            rank0_print(f"[INFO] Val dataset video paths ({len(val_paths)}) saved to {out_path}")
    else:
        eval_dataset = None

    data_collator = DataCollatorWithCompressor(vlprocessor=vlprocessor)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
