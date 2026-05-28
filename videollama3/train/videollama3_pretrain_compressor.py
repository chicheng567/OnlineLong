"""
Autoencoder pretraining for the token compressor.

Pipeline (no LLM required):
  1. Vision encoder (frozen) → N × HW visual tokens  (N ≤ max_frames, HW = compress_wh)
  2. Compressor (trainable)  → HW compressed tokens   (single window over all N frames)
  3. Decoder   (trainable)   → 10 × HW decoded tokens  (fixed 10 frame-slots)
  4. Loss: uniformly sample N frame-slot indices from [0..9],
           MSE between sampled decoded frames and original N visual-token frames.

Spatial-token constraint:
  For LocalAttnConvCompressor the input must have exactly compress_image_wh tokens
  per frame (default 256).  Use video_merge_size=2 with 448×448 input to satisfy this.
"""
import gc
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput

sys.path.append("./")

from videollama3.mm_utils import read_frames_decord
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.compressor import (
    CompressorDecoder,
    LocalAttnConvCompressor,
    TransformerDecoderCompressor,
    Videollama3TokenCompressorConfig,
)
from videollama3.model.processor import Videollama3Processor

logger = logging.getLogger(__name__)
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(" ".join(str(a) for a in args))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_frame_indices(n_input: int, n_output: int = 10) -> List[int]:
    """
    Return n_input indices uniformly spread over [0, n_output-1].

    n_input == 1  → [n_output // 2]   (middle slot)
    n_input == n_output → identity
    """
    if n_input == 1:
        return [n_output // 2]
    return [round(i * (n_output - 1) / (n_input - 1)) for i in range(n_input)]


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="pretrained_models/videollama3_7b_local",
        metadata={"help": "Pretrained VideoLLaMA3 checkpoint (used to load vision encoder)."},
    )
    compressor_type: str = field(default="local_attn_conv")
    compressor_num_layers: int = field(default=8)
    compressor_num_attention_heads: int = field(default=8)
    compressor_intermediate_size: Optional[int] = field(default=None)
    compressor_attention_dropout: float = field(default=0.0)
    compressor_layer_norm_eps: float = field(default=1e-6)
    compress_image_w: int = field(default=16)
    compress_image_h: int = field(default=16)
    decoder_num_layers: Optional[int] = field(
        default=None,
        metadata={"help": "Decoder depth; defaults to compressor_num_layers."},
    )
    max_output_frames: int = field(default=10)


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JSON list: [{video: ...}, ...]"})
    data_root: Optional[str] = field(default=None)
    max_frames: int = field(default=10)
    video_merge_size: int = field(
        default=2,
        metadata={"help": "Spatial merge size; set to 2 → 256 tokens/frame for LocalAttnConv."},
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={
            "help": (
                "Force every frame to be resized to this square size before patching. "
                "Must satisfy force_image_size == compress_image_w * video_merge_size * patch_size "
                "(default 16 * 2 * 14 = 448) so that each frame produces exactly compress_image_wh tokens."
            ),
        },
    )
    min_frames: int = field(
        default=4,
        metadata={"help": "Skip videos that decode to fewer than this many frames."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    compressor_lr: float = field(default=1e-4)
    decoder_lr: float = field(default=1e-4)
    model_max_length: int = field(default=512)
    group_by_modality_length: bool = field(default=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoPretrainDataset(Dataset):
    """
    Loads videos and returns visual features in a format ready for the AE model.
    No conversation labels required.
    """

    def __init__(
        self,
        data_path: str,
        processor: Videollama3Processor,
        max_frames: int = 10,
        merge_size: int = 2,
        data_root: Optional[str] = None,
        min_frames: int = 4,
    ):
        with open(data_path) as f:
            raw = json.load(f)
        # Support both list and dict-of-datasets formats.
        if isinstance(raw, dict):
            items = []
            for _, ds_cfg in raw.items():
                ann_path = ds_cfg["annotation"]
                root = ds_cfg.get("data_root", data_root or "")
                with open(ann_path) as fa:
                    ann = json.load(fa)
                for entry in ann:
                    entry = dict(entry)
                    if "video" in entry and root:
                        entry["_data_root"] = root
                    items.append(entry)
            self.data = items
        else:
            self.data = raw

        self.processor = processor
        self.max_frames = max_frames
        self.merge_size = merge_size
        self.data_root = data_root
        self.min_frames = min_frames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> Dict:
        item = self.data[i]
        root = item.get("_data_root", self.data_root or "")
        video_path = os.path.join(root, item["video"]) if root else item["video"]

        try:
            frames, _ = read_frames_decord(
                video_path,
                num_frames=self.max_frames,
                return_timestamps=True,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s — retrying random sample", video_path, exc)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        n_frames = len(frames)
        if n_frames < self.min_frames:
            logger.warning(
                "Skipping %s: decoded %d frames < min_frames=%d",
                video_path, n_frames, self.min_frames,
            )
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # Use process_images directly — no text/chat-template processing needed
        # for AE pretraining.  This avoids the 'image_token is undefined' Jinja2
        # error that occurs when the model's chat_template.jinja uses image_token
        # as a template variable.
        try:
            data_dict = self.processor.process_images(
                images=[frames],
                merge_size=self.merge_size,
                return_tensors="pt",
            )
        except Exception as exc:
            logger.warning("Processor failed on %s: %s — retrying", video_path, exc)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        return {
            "pixel_values": data_dict["pixel_values"],
            "grid_sizes": data_dict["grid_sizes"],
            "merge_sizes": data_dict["merge_sizes"],
            "n_frames": torch.tensor(n_frames, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class DataCollatorForPretraining:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances], dim=0)
        batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances], dim=0)
        batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances], dim=0)
        batch["n_frames"] = torch.stack([x["n_frames"] for x in instances])
        return batch


# ---------------------------------------------------------------------------
# AE model output
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional

@_dataclass
class AEOutput(BaseModelOutput):
    loss: _Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# CompressorAutoEncoder
# ---------------------------------------------------------------------------

class CompressorAutoEncoder(nn.Module):
    """
    Wraps vision_encoder (frozen) + compressor + decoder.
    Forward returns AEOutput(loss=mse_loss).
    Compatible with HF Trainer (returns dict-like with 'loss').
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        compressor: nn.Module,
        decoder: CompressorDecoder,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.compressor = compressor
        self.decoder = decoder

        # Vision encoder is always frozen.
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: torch.Tensor,
        n_frames: torch.Tensor,
        **_kwargs,
    ) -> AEOutput:
        n_frames_list: List[int] = n_frames.tolist()
        B = len(n_frames_list)
        HW: int = self.compressor.compress_image_wh  # type: ignore[attr-defined]
        max_T: int = self.decoder.max_output_frames
        # Each input frame is matched 1-to-1 to a unique decoder output slot via
        # _uniform_frame_indices.  If n_input > max_T the resulting index list
        # contains duplicates and the same slot receives conflicting MSE targets,
        # silently corrupting training.
        assert all(n <= max_T for n in n_frames_list), (
            f"n_frames per sample must be <= max_output_frames ({max_T}); "
            f"got n_frames={n_frames_list}. Lower data_args.max_frames or raise "
            f"model_args.max_output_frames."
        )
        device = pixel_values.device

        # 1. Vision encoder (no grad).
        with torch.no_grad():
            visual_tokens = self.vision_encoder(
                pixel_values=pixel_values,
                grid_sizes=grid_sizes,
                merge_sizes=merge_sizes,
            )
        # visual_tokens: (total_tokens, hidden_size)  total = sum(n_i * HW)

        # Validate spatial token count per frame matches compressor expectation.
        expected_total = sum(n * HW for n in n_frames_list)
        if visual_tokens.shape[0] != expected_total:
            raise ValueError(
                f"Vision encoder output {visual_tokens.shape[0]} tokens, "
                f"expected {expected_total} (sum of n_i * HW={HW}). "
                "Check video_merge_size matches compress_image_w/h."
            )

        # 2. Build cu_seqlens for single-window compression per sample.
        cu_ends = [0]
        for n in n_frames_list:
            cu_ends.append(cu_ends[-1] + n * HW)
        cu_seqlens = torch.tensor(cu_ends, device=device, dtype=torch.int32)

        # 3. Compress: each sample's N frames → HW compressed tokens.
        compressed = self.compressor(visual_tokens, cu_seqlens)
        # compressed: (B * HW, hidden_size)

        # 4. Decode to 10 frame slots.
        decoded = self.decoder(compressed)
        # decoded: (B * max_output_frames * HW, hidden_size)

        # 5. MSE loss with uniform frame sampling.
        decoded_4d = decoded.view(B, max_T, HW, -1)

        loss = torch.tensor(0.0, device=device, dtype=visual_tokens.dtype)
        offset = 0
        for b, n in enumerate(n_frames_list):
            target = visual_tokens[offset : offset + n * HW].view(n, HW, -1).detach()
            offset += n * HW
            indices = _uniform_frame_indices(n, max_T)
            sampled = decoded_4d[b, indices]  # (n, HW, hidden_size)
            loss = loss + F.mse_loss(sampled, target)

        loss = loss / B
        return AEOutput(loss=loss)


# ---------------------------------------------------------------------------
# Custom Trainer (separate LR groups for compressor vs decoder)
# ---------------------------------------------------------------------------

class PretrainTrainer(Trainer):

    def create_optimizer(self):
        from transformers.trainer_pt_utils import get_parameter_names
        from videollama3.train.videollama3_trainer import ALL_LAYERNORM_LAYERS

        if self.optimizer is not None:
            return self.optimizer

        assert self.model is not None
        compressor_lr = getattr(self.args, "compressor_lr", 1e-4)
        decoder_lr = getattr(self.args, "decoder_lr", 1e-4)

        decay_params = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        compressor_names = {n for n, _ in trainable if "compressor" in n}
        decoder_names    = {n for n, _ in trainable if "decoder"    in n}

        def _groups(names, lr):
            decay   = [p for n, p in trainable if n in names and n in decay_params]
            nodecay = [p for n, p in trainable if n in names and n not in decay_params]
            out = []
            if decay:
                out.append({"params": decay,   "weight_decay": self.args.weight_decay, "lr": lr})
            if nodecay:
                out.append({"params": nodecay, "weight_decay": 0.0, "lr": lr})
            return out

        param_groups = _groups(compressor_names, compressor_lr) + _groups(decoder_names, decoder_lr)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_vision_encoder(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """
    Extract only the vision encoder from a VideoLLaMA3 checkpoint.
    Loads the full model temporarily; LLM weights are GC'd after extraction.
    """
    rank0_print(f"Loading vision encoder from {model_path} ...")
    full_model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    ve = full_model.get_vision_encoder()
    ve_hidden_size = ve.hidden_size
    config = full_model.config
    # Move encoder out before deleting the rest of the model.
    ve = ve.to("cpu")  # will be moved to correct device by Trainer
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    rank0_print(f"Vision encoder loaded. hidden_size={ve_hidden_size}")
    return ve, config, ve_hidden_size


def _build_compressor(
    model_args: ModelArguments,
    ve_hidden_size: int,
):
    intermediate = model_args.compressor_intermediate_size or ve_hidden_size * 4
    cfg = Videollama3TokenCompressorConfig(
        compressor_type=model_args.compressor_type,
        hidden_size=ve_hidden_size,
        intermediate_size=intermediate,
        num_layers=model_args.compressor_num_layers,
        num_attention_heads=model_args.compressor_num_attention_heads,
        attention_probs_dropout_prob=model_args.compressor_attention_dropout,
        layer_norm_eps=model_args.compressor_layer_norm_eps,
        compress_image_w=model_args.compress_image_w,
        compress_image_h=model_args.compress_image_h,
    )
    ct = cfg.compressor_type
    if "transformer_decoder" in ct:
        compressor: nn.Module = TransformerDecoderCompressor(config=cfg)
    elif ct == "local_attn_conv":
        compressor = LocalAttnConvCompressor(config=cfg)
    else:
        raise ValueError(f"Unknown compressor_type: {ct}")
    return compressor, cfg


def _build_decoder(model_args: ModelArguments, compressor_cfg: Videollama3TokenCompressorConfig) -> CompressorDecoder:
    # Allow independent decoder depth; default to same as encoder.
    decoder_cfg = Videollama3TokenCompressorConfig(
        compressor_type=model_args.compressor_type,
        hidden_size=compressor_cfg.hidden_size,
        intermediate_size=compressor_cfg.intermediate_size,
        num_layers=model_args.decoder_num_layers or compressor_cfg.num_layers,
        num_attention_heads=compressor_cfg.num_attention_heads,
        attention_probs_dropout_prob=compressor_cfg.attention_probs_dropout_prob,
        layer_norm_eps=compressor_cfg.layer_norm_eps,
        compress_image_w=compressor_cfg.compress_image_w,
        compress_image_h=compressor_cfg.compress_image_h,
    )
    return CompressorDecoder(decoder_cfg, max_output_frames=model_args.max_output_frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    assert data_args.max_frames <= model_args.max_output_frames, (
        f"data_args.max_frames ({data_args.max_frames}) must be <= "
        f"model_args.max_output_frames ({model_args.max_output_frames}); "
        f"otherwise _uniform_frame_indices produces duplicate decoder slot indices "
        f"and the MSE loss receives conflicting targets at the same slot."
    )

    # ---- Vision encoder ------------------------------------------------
    vision_encoder, _, ve_hidden_size = _load_vision_encoder(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    # ---- Compressor + Decoder ------------------------------------------
    compressor, compressor_cfg = _build_compressor(model_args, ve_hidden_size)
    decoder = _build_decoder(model_args, compressor_cfg)

    rank0_print(
        f"Compressor type={model_args.compressor_type} "
        f"layers={compressor_cfg.num_layers} "
        f"heads={compressor_cfg.num_attention_heads} "
        f"HW={compressor_cfg.compress_image_w}×{compressor_cfg.compress_image_h}"
    )
    rank0_print(f"Decoder layers={model_args.decoder_num_layers or compressor_cfg.num_layers} "
                f"max_output_frames={model_args.max_output_frames}")

    # ---- AE model ------------------------------------------------------
    ae_model = CompressorAutoEncoder(
        vision_encoder=vision_encoder,
        compressor=compressor,
        decoder=decoder,
    )

    # ---- Processor & Dataset -------------------------------------------
    processor = Videollama3Processor.from_pretrained(model_args.model_name_or_path)

    if data_args.force_image_size is not None:
        processor.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")
        # Sanity check: with force_image_size F, patch_size P and merge_size M,
        # every frame yields (F // (P * M))^2 tokens.  This must equal
        # compress_image_w * compress_image_h, or the encoder output count check
        # in CompressorAutoEncoder.forward will fail.
        patch_size = processor.image_processor.patch_size
        merge = data_args.video_merge_size
        side = data_args.force_image_size // (patch_size * merge)
        assert (
            side == compressor_cfg.compress_image_w == compressor_cfg.compress_image_h
        ), (
            f"force_image_size ({data_args.force_image_size}) / "
            f"(patch_size {patch_size} * video_merge_size {merge}) = {side}, "
            f"but compress_image_w/h = "
            f"{compressor_cfg.compress_image_w}/{compressor_cfg.compress_image_h}. "
            f"They must match so each frame produces compress_image_wh tokens."
        )

    train_dataset = VideoPretrainDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_frames=data_args.max_frames,
        merge_size=data_args.video_merge_size,
        data_root=data_args.data_root,
        min_frames=data_args.min_frames,
    )
    rank0_print(f"Dataset size: {len(train_dataset)}")

    collator = DataCollatorForPretraining()

    # ---- Trainer -------------------------------------------------------
    trainer = PretrainTrainer(
        model=ae_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save only compressor weights (decoder is a pretraining artifact).
    if training_args.local_rank in (-1, 0):
        out_dir = training_args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        compressor_state = {
            k[len("compressor."):]: v
            for k, v in ae_model.state_dict().items()
            if k.startswith("compressor.")
        }
        torch.save(compressor_state, os.path.join(out_dir, "compressor_pretrained.pt"))
        compressor_cfg.save_pretrained(out_dir)
        rank0_print(f"Compressor weights saved to {out_dir}/compressor_pretrained.pt")


if __name__ == "__main__":
    train()
