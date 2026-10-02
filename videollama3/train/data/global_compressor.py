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
from videollama3.model.compressor import adaptive_segment_count
from videollama3.train.data import common
from videollama3.train.data.common import cast_pixel_values_, logger, rank0_print
from videollama3.train.data.compressor import (
    DataCollatorWithCompressor,
    SubsetWithLengths,
    _collect_val_video_paths,
)
from videollama3.train.data.supervised import ConcatDatasetWithLengths, LazySupervisedDataset


class TooManySampleRejections(RuntimeError):
    """Raised when a dataset rejects `MAX_SAMPLE_RETRIES` samples in a row. Its own
    class so the per-sample `except Exception` handlers re-raise it instead of
    catching it and starting yet another retry chain around it."""


__all__ = [
    "TooManySampleRejections",
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

    def __init__(self, *args, fixed_frames: int = 0, model_args=None, qbase_only: bool = False,
                 raw_frames: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.fixed_frames = fixed_frames
        # LLM-SFT raw-frame stream: when True, this dataset's samples are fed to the LLM
        # UNCOMPRESSED (no compression part). Set per meta-JSON entry ("raw_frames":
        # true); only QbaseLLMSFTDataset (videollama3/train/qbase_llm_sft.py) acts on it.
        self.raw_frames = bool(raw_frames)
        # Phase-2 pure-qbase replay: when True, this dataset's samples bypass the
        # unit split + fold and route straight through stage-1 (N*K qbase tokens).
        # Set per meta-JSON entry ("qbase_only": true); only Phase2FoldDataset acts
        # on it. docs/two_stage_compression_design.md §4 Phase 2.
        self.qbase_only = bool(qbase_only)

    # Bound on the retry chain a rejected sample starts. Each rejection recurses into
    # another random index, so a registry where EVERY sample is rejected (all-thumbnail,
    # or model_max_length set far too low) would otherwise recurse until the stack dies
    # -- a confusing hang instead of a clear error.
    MAX_SAMPLE_RETRIES = 50

    def _backup_sample(self, i: int, retries: int) -> Dict[str, torch.Tensor]:
        """Draw a different sample in place of a rejected one."""
        if retries >= self.MAX_SAMPLE_RETRIES:
            raise TooManySampleRejections(
                f"{type(self).__name__}[{self.dataset_name}]: {retries} consecutive samples "
                f"rejected starting at index {i}. This is a dataset problem, not a flaky "
                f"sample -- check the warnings above (compressed output larger than the "
                f"sample's vision tokens, or pre-compression length over --model_max_length)."
            )
        backup_idx = random.randint(0, len(self.list_data_dict) - 1)
        return self.__getitem__(backup_idx, _retries=retries + 1)

    def _segment_count(self, n_frames: int) -> int:
        """``N`` for this sample -- Phase 1's fixed-frames rule, a pure function of
        the frame count and identical to the model's ``segment_count_for``, so no
        decode is needed here. Phase 2/3 override it."""
        ma = self.model_args
        tf = int(getattr(ma, "segment_target_frames", 4)) if ma is not None else 4
        fe = int(getattr(ma, "segment_force_every", 8)) if ma is not None else 8
        return adaptive_segment_count(int(n_frames), tf, fe)

    def _compressed_len(self, n_frames: int) -> int:
        """Rows the compressor will emit -- mirrors
        ``TransformerDecoderFlatCompressor.output_len_for``."""
        ma = self.model_args
        k = int(getattr(ma, "num_queries", 64)) if ma is not None else 64
        if ma is not None and not getattr(ma, "adaptive_segmentation", False):
            return k
        return self._segment_count(n_frames) * k

    def _reject_if_too_small(self, i: int, n_out: int, total_vision_tokens: int,
                             total_frames: int, what: str = "") -> bool:
        """True when this sample's compressed output cannot fit in the vision tokens
        it replaces, so the caller must skip it.

        ``compress_visual_tokens_with_compressor`` writes the compressed rows back
        INTO the part they replace (``replace_mask[ps : ps + n_out]``), so a sample
        whose grid holds fewer tokens than the compressor emits silently spills into
        the next part and dies on a shape mismatch. For a still image that is exactly
        ``num_queries``: one frame is ONE segment, so it emits ``K`` rows into a part
        holding only that image's own tokens, and anything under ~224x224 px is below
        it (measured: 27.6 % of videoxl ``pretrain.json`` are 100x100-240x160
        thumbnails). ``dataset_util/build_phase1_image_blend.py`` already drops those
        at build time and ``--vision_min_tokens K`` upscales the rest; this is the
        last-resort guard for a registry built without either."""
        if n_out <= total_vision_tokens:
            return False
        cls = type(self)
        n_seen = getattr(cls, "_too_small_count", 0) + 1
        cls._too_small_count = n_seen
        k = int(getattr(self.model_args, "num_queries", 64)) if self.model_args is not None else 64
        if n_seen == 1:
            # Spell out the remedy once; the rest are one-liners so a systematically
            # bad registry does not bury the rest of the log.
            logger.warning(
                "Sample %s of %s: the compressor emits %d rows%s but the sample holds only %d "
                "vision tokens (%d frame(s)), so its output would overflow the part it replaces. "
                "SKIPPING. A still image is ONE segment and therefore needs >= num_queries (%d) "
                "tokens, i.e. roughly 224x224 px. Rebuild the registry with "
                "dataset_util/build_phase1_image_blend.py (it drops sub-K images at native "
                "resolution) or pass --vision_min_tokens %d to upscale them instead. Further "
                "occurrences in this dataset are logged one line each, then every 100th.",
                i, self.dataset_name, n_out, f" ({what})" if what else "",
                total_vision_tokens, total_frames, k, k,
            )
        elif n_seen <= 20 or n_seen % 100 == 0:
            logger.warning("Sample %s of %s: only %d vision tokens < %d emitted -- skipped (#%d).",
                           i, self.dataset_name, total_vision_tokens, n_out, n_seen)
        return True

    def __getitem__(self, i, _retries: int = 0) -> Dict[str, torch.Tensor]:
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
                logger.warning(
                    "Sample %s: pre-compression length %d exceeds model_max_length %d (%d frames). "
                    "Lower --max_frames or --fixed_frames. Skipping.",
                    i, seq_len, max_len, total_frames,
                )
                return self._backup_sample(i, _retries)

            image_token_id = self.vlprocessor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
            total_vision_tokens = int((data_dict["input_ids"] == image_token_id).sum().item())
            assert total_vision_tokens % total_frames == 0, (
                f"Total vision tokens {total_vision_tokens} should be divisible by total frames {total_frames}."
            )
            # One part covering every vision token of the sample. Indices count image
            # tokens only, not sequence positions.
            n_out = self._compressed_len(total_frames)
            if self._reject_if_too_small(i, n_out, total_vision_tokens, total_frames,
                                         f"N={self._segment_count(total_frames)} x "
                                         f"K={self._compressed_len(1)}"):
                return self._backup_sample(i, _retries)

            data_dict["compression_parts"] = [[0, total_vision_tokens]]
            data_dict["compression_ts_info"] = build_range_ts_info(content, self.vlprocessor.tokenizer)
            # A still image has no timeline. The rewrite above dropped its timestamp,
            # so the chat template renders a bare `<image>` with no "Time X.0s:"
            # prefix and `build_range_ts_info` returns `(0, [])` -- nothing to cut
            # back and no range string to emit on this (single-stage) path. The flag
            # carries that to the model so the two-stage path does not emit a
            # degenerate `Time:0s-0s:` either (videollama3_arch.py).
            data_dict["compression_is_image"] = [is_still_image]

        except TooManySampleRejections:
            raise
        except Exception:
            logger.exception("Failed to process sample %s. Drawing a replacement.", i)
            return self._backup_sample(i, _retries)
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
                raw_frames=dataset_cfg.get("raw_frames", False),
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
