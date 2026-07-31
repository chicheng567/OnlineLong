"""
Standalone pretraining for the semantic-loss MLP (vision-encoder feature space
-> VQ-GAN latent space), decoupled from the compressor/decoder.

Why this exists
----------------
In the full AE pretrain pipeline (videollama3_pretrain_compressor.py), the
MLP that maps decoder-output tokens -> VQ-GAN latent is trained jointly with
the compressor+decoder, using ONLY decoder-reconstructed tokens as input and
a loose IV2-cosine semantic loss as its main signal. Diagnostics
(diagnostics/visualize_video_reconstruction.py, oracle mode) showed that MLP
collapses to a near-constant output regardless of its input — feeding it the
REAL (unreconstructed) vision-encoder tokens for 8 wildly different frames
still produced near-identical images (cross-frame pixel std 0.0039 vs 0.24
for the real frames).

This script trains that MLP in isolation, directly supervised: real frame ->
vision encoder (frozen) -> MLP (trainable) -> VQ-GAN decode (frozen) -> MSE
against that same real frame, resized to the VQ-GAN's output resolution. No
compressor, no decoder, no IV2, no temporal structure — every sampled frame
is an independent training example. A constant-output shortcut cannot reduce
pixel MSE against genuinely different target frames, so (unlike the IV2
cosine loss) this objective cannot be satisfied by collapsing.

The resulting MLP weights (raw nn.Sequential state dict, unprefixed — same
convention as compressor_pretrained.pt) can be loaded into
GlobalSemanticLoss.mlp / semantic_loss.mlp.* to give the full AE pretraining
run a non-degenerate decode head from the start.
"""
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.checkpoint import checkpoint
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput

sys.path.append("./")

from videollama3.model.processor import Videollama3Processor
from videollama3.model.vqgan_vendor import load_vqgan_decoder
from videollama3.train.videollama3_pretrain_compressor import (
    DataCollatorForPretraining,
    VideoPretrainDataset,
    _load_vision_encoder,
)

logger = logging.getLogger(__name__)
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(" ".join(str(a) for a in args))


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="pretrained_models/videollama3_7b_local")
    vqgan_state_dict: str = field(default="pretrained_models/vqgan/state_dict.pt")
    semantic_mlp_hidden: Optional[int] = field(
        default=None, metadata={"help": "Hidden width of the MLP; default 2x VQ-GAN z_channels (=512)."}
    )
    pixel_loss_weight: float = field(default=1.0)
    commit_loss_weight: float = field(
        default=0.25,
        metadata={
            "help": (
                "Weight for the VQ commitment loss ||z - sg(e_nearest)||^2, always on "
                "(unlike the AE pretrain script, this is not optional here) — keeps the "
                "MLP's output on the frozen VQ-GAN codebook manifold. 0.25 is the "
                "standard VQ-VAE commitment weight."
            ),
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JSON list: [{video: ...}, ...]"})
    data_root: Optional[str] = field(default=None)
    max_frames: int = field(
        default=8,
        metadata={
            "help": (
                "Frames sampled per video. There is no temporal structure here — every "
                "sampled frame becomes one independent training example, so this only "
                "controls how many (frame, target) pairs are drawn per video read."
            ),
        },
    )
    video_merge_size: int = field(default=2)
    force_image_size: Optional[int] = field(default=448)
    min_frames: int = field(default=1)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class PixelMLPOutput(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    pixel_loss: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    psnr: Optional[torch.Tensor] = None


class PixelMLPModel(nn.Module):
    """frozen vision_encoder -> trainable mlp -> frozen vqgan, MSE + commit loss."""

    def __init__(self, vision_encoder, vqgan, compressor_hw: int,
                 mlp_hidden: Optional[int] = None,
                 pixel_loss_weight: float = 1.0, commit_loss_weight: float = 0.25):
        super().__init__()
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.vqgan = vqgan
        for p in self.vqgan.parameters():
            p.requires_grad = False

        self.hw = compressor_hw
        self.pixel_loss_weight = pixel_loss_weight
        self.commit_loss_weight = commit_loss_weight

        z_channels = vqgan.embed_dim
        hidden = mlp_hidden if mlp_hidden is not None else 2 * z_channels
        self.mlp = nn.Sequential(
            nn.Linear(vision_encoder.hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_channels),
        )

    def _commit_loss(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Per-element MSE to the nearest frozen codebook entry (stop-gradient on
        the codebook side) — standard VQ-VAE commitment loss."""
        codebook = self.vqgan.quantize.embedding.weight
        z_fp32 = z_flat.float()
        cb_fp32 = codebook.float()
        with torch.no_grad():
            d = (
                z_fp32.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * z_fp32 @ cb_fp32.t()
                + cb_fp32.pow(2).sum(dim=1)
            )
            nearest_e = codebook[d.argmin(dim=1)].detach()
        return F.mse_loss(z_fp32, nearest_e.float())

    def forward(self, pixel_values, grid_sizes, merge_sizes, n_frames,
                vqgan_target_frames, **_kwargs) -> PixelMLPOutput:
        with torch.no_grad():
            visual_tokens = self.vision_encoder(
                pixel_values=pixel_values, grid_sizes=grid_sizes, merge_sizes=merge_sizes,
            )  # (total_frames * HW, hidden)

        total_frames = vqgan_target_frames.shape[0]
        assert visual_tokens.shape[0] == total_frames * self.hw, (
            f"visual_tokens has {visual_tokens.shape[0]} tokens, expected "
            f"total_frames({total_frames}) * HW({self.hw}) = {total_frames * self.hw}. "
            "Check video_merge_size / force_image_size match compressor_hw."
        )

        z_flat = self.mlp(visual_tokens)  # (total_frames * HW, z_channels)
        commit_loss = self._commit_loss(z_flat)

        z = (
            z_flat.view(total_frames, self.vqgan.latent_size, self.vqgan.latent_size, self.vqgan.embed_dim)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )
        # VQ-GAN decoder is deep (ResNet + attention across 5 resolution stages);
        # checkpoint it since it's frozen anyway (only trades compute for memory).
        imgs = checkpoint(self.vqgan.decode_from_continuous, z, use_reentrant=False)  # (total_frames,3,H,W) in [-1,1]

        pixel_loss = F.mse_loss(imgs.float(), vqgan_target_frames.to(imgs.device).float())
        loss = self.pixel_loss_weight * pixel_loss + self.commit_loss_weight * commit_loss

        with torch.no_grad():
            # Images live in [-1, 1] (peak-to-peak range 2), so MAX_I^2 = 4.
            psnr = 10.0 * torch.log10(4.0 / pixel_loss.clamp_min(1e-10))

        return PixelMLPOutput(
            loss=loss,
            pixel_loss=pixel_loss.detach(),
            commit_loss=commit_loss.detach(),
            psnr=psnr.detach(),
        )


# ---------------------------------------------------------------------------
# Trainer (just adds aux-loss averaging into the periodic log() call)
# ---------------------------------------------------------------------------

class PixelMLPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_running = {}
        self._aux_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        for key in ("pixel_loss", "commit_loss", "psnr"):
            v = getattr(outputs, key, None)
            if v is not None:
                self._aux_running[key] = self._aux_running.get(key, 0.0) + float(v.detach().float().item())
        self._aux_count += 1
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, *args, **kwargs):
        if self._aux_count > 0:
            for key, total in self._aux_running.items():
                logs[key] = total / self._aux_count
            self._aux_running.clear()
            self._aux_count = 0
        return super().log(logs, *args, **kwargs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    vision_encoder, _, ve_hidden_size = _load_vision_encoder(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    rank0_print(f"Loading frozen VQ-GAN from {model_args.vqgan_state_dict}")
    vqgan = load_vqgan_decoder(model_args.vqgan_state_dict, dtype=torch.float32)

    processor = Videollama3Processor.from_pretrained(model_args.model_name_or_path)
    if data_args.force_image_size is not None:
        processor.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")

    patch_size = processor.image_processor.patch_size
    compressor_hw = (data_args.force_image_size // (patch_size * data_args.video_merge_size)) ** 2
    rank0_print(
        f"compressor_hw={compressor_hw}  vqgan latent_size={vqgan.latent_size}  "
        f"z_channels={vqgan.embed_dim}  vqgan_image_size={vqgan.decoder.resolution}"
    )

    model = PixelMLPModel(
        vision_encoder=vision_encoder,
        vqgan=vqgan,
        compressor_hw=compressor_hw,
        mlp_hidden=model_args.semantic_mlp_hidden,
        pixel_loss_weight=model_args.pixel_loss_weight,
        commit_loss_weight=model_args.commit_loss_weight,
    )
    rank0_print(
        f"Semantic MLP: {sum(p.numel() for p in model.mlp.parameters()):,} trainable params  "
        f"pixel_loss_weight={model_args.pixel_loss_weight} commit_loss_weight={model_args.commit_loss_weight}"
    )

    train_dataset = VideoPretrainDataset(
        data_path=data_args.data_path,
        processor=processor,
        max_frames=data_args.max_frames,
        merge_size=data_args.video_merge_size,
        data_root=data_args.data_root,
        min_frames=data_args.min_frames,
        iv2_cache_dir=None,
        decode_pixel_targets=True,
        vqgan_image_size=vqgan.decoder.resolution,
    )
    rank0_print(f"Dataset size: {len(train_dataset)}")

    trainer = PixelMLPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForPretraining(),
    )
    trainer.train()

    if trainer.is_world_process_zero():
        out_dir = training_args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.mlp.state_dict(), os.path.join(out_dir, "semantic_mlp_pretrained.pt"))
        with open(os.path.join(out_dir, "semantic_mlp_config.json"), "w") as f:
            json.dump({
                "compressor_hidden": ve_hidden_size,
                "mlp_hidden": model_args.semantic_mlp_hidden or 2 * vqgan.embed_dim,
                "z_channels": vqgan.embed_dim,
            }, f, indent=2)
        rank0_print(f"MLP weights saved to {out_dir}/semantic_mlp_pretrained.pt")


if __name__ == "__main__":
    train()
