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
vision encoder (frozen) -> MLP (trainable) -> real VQ-GAN encoder latent for
that same frame. No compressor, no decoder, no IV2, no temporal structure —
every sampled frame is an independent training example.

Latent-space regression, computed end-to-end (no precompute/caching)
----------------------------------------------------------------------
This used to train against a pixel-MSE loss that decoded the MLP's output
through the full (frozen) VQGANDecoder every step — a deep 5-resolution-stage
ResNet+attention network, made slower still by gradient checkpointing
(recomputes the forward pass on backward). That made the training loop very
slow for what is, underneath, just a regression problem.

diagnostics/compare_vqgan_encoder_feature_space.py measured the mapping this
MLP has to learn (vision-encoder tokens -> real VQ-GAN encoder latent)
directly and found it learnable: a matched-architecture (Linear->GELU->Linear)
held-out probe reaches cosine=0.846 / R^2=0.77 against the REAL VQ-GAN encoder
latent, confirmed via a shuffled-pairing control (cosine collapses to 0.340,
R^2 negative) to be real signal, not a model-capacity artifact.

So this script now regresses directly against the REAL VQ-GAN ENCODER latent
instead of decoding to pixels — the frozen VQGANEncoder is run once per batch
under torch.no_grad() (no backward pass, so no gradient checkpointing is
needed either — that trick only pays for itself when a backward pass would
otherwise recompute the forward pass), producing the target on the fly. The
VQ-GAN DECODER is still never called anywhere in this script (only its
codebook, for the cheap commitment loss). No separate precompute step, no
cache directory to keep in sync with --max_frames — everything happens in
this one training run.

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
from transformers import Trainer
from transformers.modeling_outputs import BaseModelOutput

sys.path.append("./")

from videollama3.model.processor import Videollama3Processor
from videollama3.model.vqgan_vendor import load_vqgan_decoder, load_vqgan_encoder
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
    vqgan_state_dict: str = field(
        default="pretrained_models/vqgan/state_dict.pt",
        metadata={
            "help": (
                "Loads BOTH the real VQGANEncoder (to compute regression targets "
                "on the fly, forward-only under no_grad) and VQGANDecoder's "
                "codebook (.quantize, for the commitment loss) from this same "
                "file. VQGANDecoder's .decoder submodule is never called."
            ),
        },
    )
    semantic_mlp_hidden: Optional[int] = field(
        default=None, metadata={"help": "Hidden width of the MLP; default 2x VQ-GAN z_channels (=512)."}
    )
    latent_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight on the MSE(MLP(vision_tokens), real VQ-GAN encoder latent) loss."},
    )
    commit_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for the VQ commitment loss ||z - sg(e_nearest)||^2. Off by "
                "default now that latent_loss directly regresses to the real (already "
                "near-codebook) VQ-GAN encoder latent, which makes this largely redundant "
                "belt-and-suspenders regularization rather than a load-bearing signal. "
                "Set > 0 (0.25 is the standard VQ-VAE commitment weight) if the trained "
                "MLP's output will be quantized downstream (e.g. loaded into "
                "GlobalSemanticLoss, which does run it through VectorQuantizer2)."
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
class LatentMLPOutput(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    latent_loss: Optional[torch.Tensor] = None
    commit_loss: Optional[torch.Tensor] = None
    latent_cosine: Optional[torch.Tensor] = None


class LatentMLPModel(nn.Module):
    """frozen vision_encoder -> trainable mlp -> MSE against the REAL VQ-GAN
    encoder latent, computed ON THE FLY (forward-only, no_grad, no caching)
    every step, + commit loss. The VQ-GAN decoder is never called."""

    def __init__(self, vision_encoder, vqgan_encoder, vqgan_codebook_source, compressor_hw: int,
                 mlp_hidden: Optional[int] = None,
                 latent_loss_weight: float = 1.0, commit_loss_weight: float = 0.0):
        super().__init__()
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Computes the regression target live, forward-only (see forward()).
        self.vqgan_encoder = vqgan_encoder
        for p in self.vqgan_encoder.parameters():
            p.requires_grad = False

        # Held only for its frozen codebook (.quantize); .decoder is never called.
        self.vqgan_codebook_source = vqgan_codebook_source
        for p in self.vqgan_codebook_source.parameters():
            p.requires_grad = False

        self.hw = compressor_hw
        self.latent_size = vqgan_encoder.latent_size
        self.z_channels = vqgan_encoder.embed_dim
        self.latent_loss_weight = latent_loss_weight
        self.commit_loss_weight = commit_loss_weight

        z_channels = vqgan_encoder.embed_dim
        hidden = mlp_hidden if mlp_hidden is not None else 2 * z_channels
        self.mlp = nn.Sequential(
            nn.Linear(vision_encoder.hidden_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_channels),
        )

    def _commit_loss(self, z_flat: torch.Tensor) -> torch.Tensor:
        """Per-element MSE to the nearest frozen codebook entry (stop-gradient on
        the codebook side) — standard VQ-VAE commitment loss."""
        codebook = self.vqgan_codebook_source.quantize.embedding.weight
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
                vqgan_target_frames, **_kwargs) -> LatentMLPOutput:
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

        # Real target, computed live: forward-only through the frozen VQ-GAN
        # encoder, no backward pass -> no gradient-checkpointing trade-off to
        # make (that trick only pays off when a backward pass would otherwise
        # recompute this forward pass; here there simply isn't one).
        with torch.no_grad():
            target = self.vqgan_encoder.encode_to_continuous(
                vqgan_target_frames.to(
                    device=next(self.vqgan_encoder.parameters()).device,
                    dtype=next(self.vqgan_encoder.parameters()).dtype,
                )
            ).float()  # (total_frames, z_channels, latent_size, latent_size)

        z_flat = self.mlp(visual_tokens)  # (total_frames * HW, z_channels)
        commit_loss = self._commit_loss(z_flat)

        z = (
            z_flat.view(total_frames, self.latent_size, self.latent_size, self.z_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )  # (total_frames, z_channels, latent_size, latent_size)

        latent_loss = F.mse_loss(z, target.to(z.device))
        loss = self.latent_loss_weight * latent_loss + self.commit_loss_weight * commit_loss

        with torch.no_grad():
            latent_cosine = F.cosine_similarity(
                z.flatten(1), target.to(z.device).flatten(1), dim=1
            ).mean()

        return LatentMLPOutput(
            loss=loss,
            latent_loss=latent_loss.detach(),
            commit_loss=commit_loss.detach(),
            latent_cosine=latent_cosine.detach(),
        )


# ---------------------------------------------------------------------------
# Trainer (just adds aux-loss averaging into the periodic log() call)
# ---------------------------------------------------------------------------

class LatentMLPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_running = {}
        self._aux_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        for key in ("latent_loss", "commit_loss", "latent_cosine"):
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

    rank0_print(f"Loading real VQ-GAN encoder (live targets) from {model_args.vqgan_state_dict}")
    vqgan_encoder = load_vqgan_encoder(model_args.vqgan_state_dict, dtype=torch.float32, strict=True)
    rank0_print(f"Loading VQ-GAN codebook/config (decoder unused) from {model_args.vqgan_state_dict}")
    vqgan_decoder = load_vqgan_decoder(model_args.vqgan_state_dict, dtype=torch.float32)

    processor = Videollama3Processor.from_pretrained(model_args.model_name_or_path)
    if data_args.force_image_size is not None:
        processor.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")

    patch_size = processor.image_processor.patch_size
    compressor_hw = (data_args.force_image_size // (patch_size * data_args.video_merge_size)) ** 2
    rank0_print(
        f"compressor_hw={compressor_hw}  vqgan latent_size={vqgan_encoder.latent_size}  "
        f"z_channels={vqgan_encoder.embed_dim}  vqgan_image_size={vqgan_decoder.decoder.resolution}"
    )
    if compressor_hw != vqgan_encoder.latent_size ** 2:
        rank0_print(
            f"WARNING: compressor_hw ({compressor_hw}) != vqgan_encoder.latent_size^2 "
            f"({vqgan_encoder.latent_size ** 2}) — the per-frame token grid and the VQ-GAN "
            f"latent grid must have the same spatial size for the reshape in "
            f"LatentMLPModel.forward to be meaningful."
        )

    model = LatentMLPModel(
        vision_encoder=vision_encoder,
        vqgan_encoder=vqgan_encoder,
        vqgan_codebook_source=vqgan_decoder,
        compressor_hw=compressor_hw,
        mlp_hidden=model_args.semantic_mlp_hidden,
        latent_loss_weight=model_args.latent_loss_weight,
        commit_loss_weight=model_args.commit_loss_weight,
    )
    rank0_print(
        f"Semantic MLP: {sum(p.numel() for p in model.mlp.parameters()):,} trainable params  "
        f"latent_loss_weight={model_args.latent_loss_weight} commit_loss_weight={model_args.commit_loss_weight}"
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
        vqgan_image_size=vqgan_decoder.decoder.resolution,
    )
    rank0_print(f"Dataset size: {len(train_dataset)}")

    trainer = LatentMLPTrainer(
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
                "mlp_hidden": model_args.semantic_mlp_hidden or 2 * vqgan_encoder.embed_dim,
                "z_channels": vqgan_encoder.embed_dim,
            }, f, indent=2)
        rank0_print(f"MLP weights saved to {out_dir}/semantic_mlp_pretrained.pt")


if __name__ == "__main__":
    train()
