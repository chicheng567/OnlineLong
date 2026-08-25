#!/usr/bin/env python3
"""
Compressor finetuning with a SINGLE, whole-video compression window.

Same pipeline as ``videollama3_chat_finetune_compressor.py`` (frozen vision encoder
-> trainable token compressor -> LLM, CE on the assistant turns), with one change:
instead of sampling several fixed-size windows to compress
(``select_compression_parts`` / ``select_full_compression_parts``), **the whole video
is compressed by one compressor call into a single ``HW``-token summary**, whatever
its frame count.

    T frames x HW tokens  ->  compressor  ->  HW tokens  ->  mm_projector -> LLM

* One compression part per sample, so there is no partial/full split and no dual
  forward: the trainer runs a single CE forward (``use_dual_forward=False``).
* Every video yields exactly one part, so all ranks always take the compression path
  -- the NCCL desync the windowed script guards against (some ranks compressing, some
  not) cannot happen here.
* Timestamps still mark the *whole* span: the chat template's per-frame
  ``"Time X.0s:"`` text before the first frame is replaced by
  ``"Time:{first}s-{last}s:"``, and the per-frame timestamps inside the span are
  dropped along with the frame tokens they annotate (they sit between the first and
  last image token, which ``prepare_inputs_labels_for_multimodal`` cuts out).
* One video block per sample is assumed. Multi-turn online data (which
  ``preprocess_videollama3`` splits into one block per user turn) is out of scope and
  is skipped.

``--feat_meta`` switches the input from raw video to the **precomputed vision-encoder
features** written by ``dataset_util/extract_vision_features.py``: it takes that run's
``meta_with_vision_feat.json`` (each entry = the original annotation plus
``vision_feat_path`` / ``frame_timestamps`` / ``num_frames``) and reads the cached
``(T, HW, C)`` tensors instead of decoding video. Entries without
``frame_timestamps`` get a fabricated 1-FPS grid (see ``fake_fps1_timestamps``). The cached tokens ARE the frozen
encoder's output, so the vision encoder is replaced by an identity stand-in and never
runs; everything downstream (compressor -> mm_projector -> LLM) is unchanged. The cache
must have been extracted with the same geometry the compressor expects
(``VIDEO_MERGE_SIZE=2 FORCE_IMAGE_SIZE=448`` -> HW=256 == compress_image_w x _h), and
``--fps`` / ``--max_frames`` no longer apply (the cache is fixed at 1 FPS); use
``--fixed_frames`` to bound T.

``--fixed_frames N`` resamples every video to exactly N frames before the encoder
(uniform subsample when longer, last frame repeated when shorter). It is optional for
``transformer_decoder`` / ``local_attn_conv``, which take any T at runtime, and
**required (a power of two) for ``siglip_ae``**: its ``log2(N)`` stride-2 Conv3d
stages are built at construction time, and while the convs themselves accept any T,
the stack only lands on the T=1 that the HW-token output contract requires when
T == 2^stages; its ``DynamicTokenSynthesizer`` bias table is sized by N as well.

Sequence-length note: the compression happens *inside* the model, so the sequence the
collator builds still holds the uncompressed ``T x HW`` image tokens (256 per frame at
448px / merge_size 2). It must fit in ``model_max_length`` or the collator truncates
it and the compression part no longer lines up; samples that would overflow are
skipped with a warning. Bound it with ``--max_frames`` / ``--fixed_frames``.
"""
import copy
from dataclasses import dataclass, field
import json
import logging
import os
import pathlib
import random
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import transformers
from packaging import version

import torch.utils.data
sys.path.append("./")

from videollama3.constants import (  # noqa: E402
    DEFAULT_IMAGE_TOKEN,
    NUM_FRAMES,
    COMPRESSION_START_TOKEN,
    COMPRESSION_END_TOKEN,
)
from videollama3.model import Videollama3Qwen2Config, Videollama3Qwen2ForCausalLM  # noqa: E402
from videollama3.model.processor import Videollama3Processor  # noqa: E402
from videollama3.train.videollama3_chat_finetune_online import (  # noqa: E402
    ConcatDatasetWithLengths,
    LazySupervisedDataset,
    _is_trainable_lr,
    _set_module_trainable,
    find_all_linear_names,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
    set_seed,
)
from videollama3.train.videollama3_chat_finetune_compressor import (  # noqa: E402
    DataCollatorWithCompressor,
    SubsetWithLengths,
    _collect_val_video_paths,
    int_with_none,
)
from videollama3.train.videollama3_trainer import VideoLLaMA3Trainer  # noqa: E402
from functools import partial

logger = logging.getLogger(__name__)
torch.load = partial(torch.load, weights_only=False)
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum
    torch.serialization.add_safe_globals([LossScaler, ZeroStageEnum])
except ImportError:
    pass


def rank0_print(*args):
    if local_rank == 0:
        message = " ".join(str(arg) for arg in args)
        print(message)
        if logging.getLogger().hasHandlers():
            logging.info(message)


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
    if not timestamps:
        return [(0, [])]
    ts_start, ts_end = float(timestamps[0]), float(timestamps[-1])
    old_ts_ids = tokenizer.encode(f"Time {round(ts_start, 1)}s:", add_special_tokens=False)
    new_ts_ids = tokenizer.encode(f"Time:{ts_start:.1f}s-{ts_end:.1f}s:", add_special_tokens=False)
    return [(len(old_ts_ids), new_ts_ids)]


# ---------------------------------------------------------------------------
# Precomputed-feature path (--feat_meta)
# ---------------------------------------------------------------------------

class CachedFeatureEncoder(nn.Module):
    """Stands in for the SigLIP encoder when the features are already extracted.

    ``forward`` returns the cached tokens untouched (they are exactly what the real
    encoder would have produced, pre-mm_projector), while the attributes ``train()``
    reads off the encoder are carried over from the real one so the rest of the setup
    needs no branching.
    """

    def __init__(self, encoder):
        super().__init__()
        self.hidden_size = encoder.hidden_size
        self.image_size = encoder.image_size
        self.num_patches_per_side = encoder.num_patches_per_side
        self.image_processor = encoder.image_processor

    def forward(self, pixel_values, grid_sizes=None, merge_sizes=None, **kwargs):
        return pixel_values


def load_cached_feature(path: str, tokens_per_frame: int) -> torch.Tensor:
    """Read one extract_vision_features.py tensor as (T, HW, C)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for key in ("feat", "feature", "features", "vision_feat"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path}: expected a tensor, got {type(obj)}")
    if obj.dim() == 2:  # saved with --flatten
        assert obj.shape[0] % tokens_per_frame == 0, (
            f"{path}: flat length {obj.shape[0]} is not divisible by HW={tokens_per_frame}."
        )
        obj = obj.view(-1, tokens_per_frame, obj.shape[-1])
    assert obj.dim() == 3 and obj.shape[1] == tokens_per_frame, (
        f"{path}: expected (T, {tokens_per_frame}, C), got {tuple(obj.shape)}. The cache must be "
        f"extracted with the geometry the compressor expects (VIDEO_MERGE_SIZE=2, FORCE_IMAGE_SIZE=448)."
    )
    return obj


def fake_fps1_timestamps(num_frames: int) -> List[float]:
    """The timestamps extract_vision_features.py would have written at 1 FPS.

    Same convention as ``recaption_from_features.fake_fps_meta``: frame i stands for
    the clip [i, i+1), so its timestamp is that clip's midpoint i + 0.5. Used whenever
    a meta entry carries no ``frame_timestamps``. It is a deliberate lie about wall
    clock -- the extractor caps its output at max_frames, so a longer video's frames
    are spread over its whole runtime rather than the num_frames seconds claimed here.
    Fine while only frame order matters, wrong for second-referencing annotation
    (temporal grounding, dense captioning).
    """
    return [round(i + 0.5, 1) for i in range(num_frames)]


def messages_from_conversations(conversations: List[Dict], num_frames: int, timestamps: List[float]) -> List[Dict]:
    """conversations -> chat messages, with the video block carrying num_frames /
    timestamps instead of decoded pixels (the text half of `_convert_normal`)."""
    convs = copy.deepcopy(conversations)
    while convs and convs[0]["from"] not in ("human", "system"):
        convs = convs[1:]
    assert len(convs) > 1, "Invalid conversation"
    if all("<video>" not in c["value"] for c in convs):
        convs[0]["value"] = "<video>" + convs[0]["value"]

    video_block = {"type": "video", "num_frames": num_frames,
                   "timestamps": [float(t) for t in timestamps]}

    messages = []
    for conv in convs:
        if conv["from"] != "human":
            messages.append({"role": "assistant", "content": conv["value"]})
            continue
        chunks = conv["value"].split("<video>")
        content = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk = chunk.strip()
            if chunk:
                content.append({"type": "text", "text": chunk})
            if chunk_idx < len(chunks) - 1:
                content.append(dict(video_block))
        messages.append({"role": "user", "content": content})
    return messages


class CachedFeatureDataset(torch.utils.data.Dataset):
    """Samples from extract_vision_features.py's meta_with_vision_feat.json.

    Emits the same fields as the video-decoding dataset, except that `pixel_values`
    carries the cached vision tokens -- which is exactly what the model hands to
    `CachedFeatureEncoder`, so `DataCollatorWithCompressor` and the model forward are
    reused unchanged.
    """

    def __init__(self, meta_path: str, vlprocessor, merge_size: int, tokens_per_frame: int,
                 fixed_frames: int = 0, dtype: torch.dtype = torch.bfloat16):
        self.meta_path = meta_path
        self._warned_synthetic = False
        self.samples = json.loads(open(meta_path).read())
        assert isinstance(self.samples, list), (
            f"{meta_path}: expected extract_vision_features.py's meta list, got {type(self.samples)}."
        )
        self.vlprocessor = vlprocessor
        self.merge_size = merge_size
        self.tokens_per_frame = tokens_per_frame
        self.fixed_frames = fixed_frames
        self.dtype = dtype
        side = int(round(tokens_per_frame ** 0.5))
        assert side * side == tokens_per_frame, f"HW={tokens_per_frame} is not a perfect square."
        self.side = side

    def __len__(self):
        return len(self.samples)

    @property
    def lengths(self):
        return [
            (self.fixed_frames or int(s.get("num_frames") or 0)) * self.tokens_per_frame
            + sum(len(c["value"].split()) for c in s.get("conversations", []))
            for s in self.samples
        ]

    # every sample is a video, so grouped sampling sees positive lengths only
    modality_lengths = lengths

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.samples[i]
            feat = load_cached_feature(sample["vision_feat_path"], self.tokens_per_frame)
            timestamps = sample.get("frame_timestamps") or None
            if timestamps is None:
                timestamps = fake_fps1_timestamps(feat.shape[0])
                if not self._warned_synthetic:
                    self._warned_synthetic = True
                    logger.warning(
                        "No frame_timestamps in %s (first hit: sample %s); falling back to a "
                        "fabricated 1-FPS grid for every such entry.", self.meta_path, i,
                    )
            assert len(timestamps) == feat.shape[0], (
                f"Sample {i}: {len(timestamps)} timestamps for {feat.shape[0]} cached frames."
            )
            if self.fixed_frames > 0:
                idx = resample_indices(feat.shape[0], self.fixed_frames)
                feat = feat[torch.as_tensor(idx)]
                timestamps = [float(timestamps[j]) for j in idx]
            num_frames = feat.shape[0]

            messages = messages_from_conversations(sample["conversations"], num_frames, timestamps)
            content = get_video_content(messages)
            # process_text() only ever reads grid_sizes/merge_sizes, never pixels, so the
            # cached features skip the image processor entirely.
            grid_sizes = torch.tensor([[num_frames, self.side * self.merge_size, self.side * self.merge_size]])
            merge_sizes = torch.tensor([self.merge_size])
            data_dict = self.vlprocessor.process_text(
                messages,
                {"grid_sizes": grid_sizes, "merge_sizes": merge_sizes},
                return_labels=True,
                return_tensors="pt",
            )

            total_vision_tokens = num_frames * self.tokens_per_frame
            data_dict["pixel_values"] = feat.reshape(total_vision_tokens, -1).to(self.dtype)
            data_dict["grid_sizes"] = grid_sizes
            data_dict["merge_sizes"] = merge_sizes
            data_dict["modals"] = ["video"]
            data_dict["compression_parts"] = [[0, total_vision_tokens]]
            data_dict["compression_ts_info"] = build_range_ts_info(content, self.vlprocessor.tokenizer)
        except Exception:
            backup_idx = random.randint(0, len(self.samples) - 1)
            logger.exception("Failed to process sample %s. Fallback index: %s.", i, backup_idx)
            return self.__getitem__(backup_idx)
        return data_dict


class GlobalCompressorLazySupervisedDataset(LazySupervisedDataset):
    """Dataset that marks each sample's whole video for a single compression pass."""

    def __init__(self, *args, fixed_frames: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_frames = fixed_frames

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sample = self.list_data_dict[i]
            if self.online_mode:
                modal, images, messages, merge_size = self._convert_online_video(sample)
            else:
                modal, images, messages, merge_size = self._convert_normal(sample)
            assert modal == "video", "Compressor training currently only supports video data."

            content = get_video_content(messages)
            if self.fixed_frames > 0:
                images = resample_video_frames(images, content, self.fixed_frames)
            total_frames = int(content["num_frames"])
            assert total_frames > 0, f"Sample {i} has no frames."

            data_dict = self.vlprocessor(
                images=images,
                text=messages,
                merge_size=merge_size,
                return_labels=self.return_label,
                return_tensors="pt",
            )
            data_dict["modals"] = [modal] * len(images)

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


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="pretrained_models/videollama3_7b_local")
    tokenizer_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    mm_projector_type: Optional[str] = field(default="linear")
    vision_encoder: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_attn_implementation: Optional[str] = field(default="flash_attention_2")
    use_token_compression: Optional[bool] = field(default=True)
    # transformer_decoder | local_attn_conv | siglip_ae (siglip_ae needs --fixed_frames 2**k)
    compressor_type: str = field(default="transformer_decoder")
    compressor_num_layers: int = field(default=8)
    compressor_num_attention_heads: int = field(default=8)
    compressor_intermediate_size: Optional[int] = field(default=None)
    compressor_attention_dropout: float = field(default=0.0)
    compressor_layer_norm_eps: float = field(default=1e-6)
    compress_image_w: int = field(default=16)
    compress_image_h: int = field(default=16)
    pretrained_compressor_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional compressor_pretrained.pt from AE pretraining, loaded into token_compressor."},
    )


@dataclass
class DataArguments:
    data_path: List[str] = field(default=None)
    data_folder: Optional[str] = field(default=None)
    fps: Optional[int] = field(default=None)
    max_frames: Optional[int_with_none] = field(default=200)
    multi_dataset: bool = field(default=False)
    image_merge_size: Optional[int] = field(default=1)
    video_merge_size: Optional[int] = field(default=1)
    mm_max_length: Optional[int] = field(default=10240)
    image_aspect_ratio: str = "square"
    use_batch_flattening: bool = field(default=True)
    dataset_cache_dir: Optional[str] = field(default=None)
    force_image_size: Optional[int] = field(default=None)
    feat_meta: Optional[str] = field(
        default=None,
        metadata={"help": "extract_vision_features.py's meta_with_vision_feat.json. When set, the cached "
                          "vision-encoder features are used instead of decoding video."},
    )
    fixed_frames: int = field(
        default=0,
        metadata={"help": "Resample every video to exactly this many frames (0 = keep as decoded). "
                          "Must be a power of two >= 2 for compressor_type=siglip_ae."},
    )
    validation_split_rate: float = field(
        default=0,
        metadata={"help": "Percentage of the train set used as validation set."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    # Default to compressor-only training.
    vision_encoder_lr: Optional[float] = field(default=0.0)
    mm_projector_lr: Optional[float] = field(default=0.0)
    compressor_lr: Optional[float] = field(default=1e-4)
    llm_lr: Optional[float] = field(default=0.0)
    group_by_modality_length: bool = field(default=False)
    model_max_length: int = field(default=32768)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default=16)
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"


def make_global_compressor_data_module(
    vlprocessor: transformers.ProcessorMixin,
    data_args: DataArguments,
    output_dir: Optional[str] = None,
    tokens_per_frame: int = 256,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict:
    if data_args.feat_meta:
        train_dataset = CachedFeatureDataset(
            meta_path=data_args.feat_meta,
            vlprocessor=vlprocessor,
            merge_size=data_args.video_merge_size,
            tokens_per_frame=tokens_per_frame,
            fixed_frames=data_args.fixed_frames,
            dtype=dtype,
        )
    elif data_args.multi_dataset:
        rank0_print("Use meta file to control datasets loading. Data path will use as meta path")
        ds_collection = json.loads(open(data_args.data_path[0]).read())
        collected_datasets = [
            GlobalCompressorLazySupervisedDataset(
                vlprocessor=vlprocessor,
                data_path=[dataset_cfg["annotation"]],
                data_args=data_args,
                dataset_name=dataset_name,
                dataset_root=dataset_cfg["data_root"],
                online_mode=dataset_cfg["online_mode"],
                prefix_captioning=dataset_cfg.get("prefix_captioning", False),
                fixed_frames=data_args.fixed_frames,
            )
            for dataset_name, dataset_cfg in ds_collection.items()
        ]
        train_dataset = ConcatDatasetWithLengths(collected_datasets)
    else:
        train_dataset = GlobalCompressorLazySupervisedDataset(
            vlprocessor=vlprocessor,
            data_path=data_args.data_path,
            data_args=data_args,
            fixed_frames=data_args.fixed_frames,
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
        if output_dir is not None and local_rank in (0, -1) and not data_args.feat_meta:
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


def _build_token_compressor_config(
    model_config: Videollama3Qwen2Config, model_args: ModelArguments, data_args: DataArguments
) -> Dict:
    return {
        "compressor_type": model_args.compressor_type,
        "hidden_size": model_config.mm_hidden_size,
        "intermediate_size": model_args.compressor_intermediate_size or model_config.mm_hidden_size * 4,
        "num_layers": model_args.compressor_num_layers,
        "num_attention_heads": model_args.compressor_num_attention_heads,
        "attention_probs_dropout_prob": model_args.compressor_attention_dropout,
        "layer_norm_eps": model_args.compressor_layer_norm_eps,
        "compress_image_w": model_args.compress_image_w,
        "compress_image_h": model_args.compress_image_h,
        # siglip_ae builds log2(window_size) stride-2 stages and a per-frame bias table,
        # so it needs the exact frame count; the other two derive T from cu_seqlens.
        "window_size": data_args.fixed_frames if model_args.compressor_type == "siglip_ae" else 0,
    }


def train(attn_implementation=None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.compressor_type == "siglip_ae":
        n = data_args.fixed_frames
        if n < 2 or (n & (n - 1)) != 0:
            raise ValueError(
                "compressor_type='siglip_ae' halves T with log2(fixed_frames) stride-2 stages built "
                f"at construction time, and every window must end at T=1 to produce the "
                f"{model_args.compress_image_w * model_args.compress_image_h}-token output the model "
                f"scatters back. Set --fixed_frames to a power of two >= 2 (got {n})."
            )

    log_file = os.path.join(training_args.output_dir, "training.log")
    error_log_file = os.path.join(training_args.output_dir, "training_errors.log")
    os.makedirs(training_args.output_dir, exist_ok=True)

    log_formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(log_formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    error_handler = logging.FileHandler(error_log_file, mode="a")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_formatter)
    root_logger.addHandler(error_handler)

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    config = Videollama3Qwen2Config.from_pretrained(model_args.model_name_or_path)
    config._attn_implementation = attn_implementation
    config.mm_attn_implementation = attn_implementation
    config.use_token_compression = True
    config.trainable_mm_compressor = True
    if model_args.vision_encoder is not None:
        config.vision_encoder = model_args.vision_encoder

    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        do_sample=True,
    )
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    vision_encoder = model.get_vision_encoder()
    if data_args.feat_meta:
        # Features are precomputed: swap in a stand-in that returns them unchanged, so
        # the SigLIP weights never run and never reach the GPU.
        model.get_model().vision_encoder = CachedFeatureEncoder(vision_encoder)
        vision_encoder = model.get_vision_encoder()
        rank0_print(f"[INFO] Reading cached vision features from {data_args.feat_meta} (vision encoder disabled)")
    else:
        vision_encoder.to(dtype=compute_dtype, device=training_args.device)

    mm_projector = model.get_mm_projector()
    mm_projector.to(dtype=compute_dtype if training_args.bf16 else torch.float16, device=training_args.device)

    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.mm_hidden_size = vision_encoder.hidden_size
    model.config.token_compressor_config = _build_token_compressor_config(model.config, model_args, data_args)

    # Rebuild compressor with latest config dict.
    from videollama3.model.compressor import build_token_compressor

    model.get_model().token_compressor = build_token_compressor(model.config)
    if model.get_model().token_compressor is None:
        raise RuntimeError("Failed to build token_compressor. Check token_compressor_config.")
    if model_args.pretrained_compressor_path:
        state = torch.load(model_args.pretrained_compressor_path, map_location="cpu")
        state = state.get("compressor", state) if isinstance(state, dict) else state
        missing, unexpected = model.get_model().token_compressor.load_state_dict(state, strict=False)
        rank0_print(
            f"[INFO] Loaded pretrained compressor from {model_args.pretrained_compressor_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        if missing or unexpected:
            rank0_print(f"[WARN] missing keys: {missing}\n[WARN] unexpected keys: {unexpected}")
    model.get_model().token_compressor.to(dtype=compute_dtype, device=training_args.device)

    model.config.llm_lr = training_args.llm_lr
    model.config.vision_encoder_lr = training_args.vision_encoder_lr
    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.config.compressor_lr = training_args.compressor_lr

    llm_trainable = _is_trainable_lr(model.config.llm_lr)
    vision_trainable = _is_trainable_lr(model.config.vision_encoder_lr)
    projector_trainable = _is_trainable_lr(model.config.mm_projector_lr)
    compressor_trainable = _is_trainable_lr(model.config.compressor_lr)

    if training_args.lora_enable:
        # get_peft_model() already froze all base weights and enabled only LoRA params.
        # If llm_lr=0, also freeze the LoRA params themselves.
        if not llm_trainable:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = False
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)
    else:
        _set_module_trainable(model.get_model(), llm_trainable)
        _set_module_trainable(model.get_vision_encoder(), vision_trainable)
        _set_module_trainable(model.get_mm_projector(), projector_trainable)
        _set_module_trainable(getattr(model.get_model(), "token_compressor", None), compressor_trainable)

    total_param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_param_count == 0:
        raise RuntimeError(
            "No trainable parameters found. "
            "Please set at least one of llm_lr / vision_encoder_lr / mm_projector_lr / compressor_lr > 0."
        )
    if training_args.local_rank in (0, -1):
        trainable_ratio = 100.0 * trainable_param_count / total_param_count
        rank0_print(
            f"Trainable parameters: {trainable_param_count:,} / {total_param_count:,} "
            f"({trainable_ratio:.4f}%)"
        )

    model.config.max_frames = getattr(data_args, "max_frames", NUM_FRAMES)
    model.config.image_aspect_ratio = data_args.image_aspect_ratio if "avt" not in model_args.vision_encoder else "avt"
    model.config.image_size = data_args.image_size = vision_encoder.image_size
    model.config.image_token_length = data_args.image_token_length = mm_projector.cal_proj_size(
        vision_encoder.num_patches_per_side
    )
    old_vocabulary_size = len(tokenizer)
    new_tokens = tokenizer.add_tokens([COMPRESSION_START_TOKEN, COMPRESSION_END_TOKEN], special_tokens=True)
    if new_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
        if not llm_trainable:
            # Only the new rows need to be learned; old rows stay frozen. requires_grad
            # stays True on the whole embed_tokens tensor so ZeRO-2 assigns optimizer
            # state to it (a requires_grad parameter without optimizer state corrupts
            # AllReduce buckets -> NaN at step 2); a backward hook zeros the gradient of
            # the pre-existing rows. create_optimizer routes embed_tokens into the
            # compressor parameter group.
            _old_vocab = old_vocabulary_size

            def _zero_old_embed_rows(grad, _ov=_old_vocab):
                g = grad.clone()
                g[:_ov].zero_()
                return g

            embed = model.get_input_embeddings()
            embed.weight.requires_grad_(True)
            embed.weight.register_hook(_zero_old_embed_rows)

            # lm_head: compression tokens are always masked with IGNORE_INDEX in labels
            # so their rows never receive gradients; freeze it entirely.
            out_embed = model.get_output_embeddings()
            if out_embed is not None and out_embed.weight is not embed.weight:
                _set_module_trainable(out_embed, False)
    model.config.image_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    model.config.compression_start_token_id = tokenizer.convert_tokens_to_ids(COMPRESSION_START_TOKEN)
    model.config.compression_end_token_id = tokenizer.convert_tokens_to_ids(COMPRESSION_END_TOKEN)

    if data_args.force_image_size is not None:
        vision_encoder.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")
    vlprocessor = Videollama3Processor(vision_encoder.image_processor, tokenizer)

    assert data_args.use_batch_flattening, "Compressor training currently requires flattening mode (batch size 1 sequence)."
    assert model.config._attn_implementation == "flash_attention_2"
    assert version.parse(transformers.__version__) >= version.parse("4.44.0")

    rank0_print(
        f"[INFO] Whole-video compression: 1 window per sample -> "
        f"{model_args.compress_image_w * model_args.compress_image_h} tokens "
        f"(compressor_type={model_args.compressor_type}, "
        f"input={'cached features' if data_args.feat_meta else 'video @ max_frames=' + str(data_args.max_frames)}, "
        f"fixed_frames={data_args.fixed_frames or 'off'})"
    )

    data_module = make_global_compressor_data_module(
        vlprocessor=vlprocessor,
        data_args=data_args,
        output_dir=training_args.output_dir,
        tokens_per_frame=model_args.compress_image_w * model_args.compress_image_h,
        dtype=compute_dtype,
    )

    trainer = VideoLLaMA3Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        # Single compression mode -> single CE forward.
        use_dual_forward=False,
        partial_loss_weight=1.0,
        full_loss_weight=0.0,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            vlprocessor.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        if trainer.args.should_save:
            vlprocessor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
