"""
DTS/SGM pretraining branch for the token compressor — Video-XL-Pro's ReCoT recipe
(arXiv:2503.18478), a separate branch of the plain-AE recipe in
videollama3_pretrain_compressor.py. See videollama3/model/dts.py for the full
provenance notes (paper Algorithm 1 transcription, SiglipAE reference-code
cross-check, and what is and is not ported).

Pipeline (no LLM required, mirrors videollama3_pretrain_compressor.py's shape):
  1. Vision encoder (frozen)      -> exactly `fixed_frames` (T) x HW visual tokens
                                      per sample. T is HARD-FIXED for every sample in
                                      the dataset (see VideoPretrainDatasetDTS) — this
                                      is a deliberate scoping decision (unlike the base
                                      AE-pretrain script's variable N <= max_output_frames),
                                      required by DTS's per-frame learnable temporal
                                      encoding (a (T, hidden) parameter) and by SGM's
                                      Algorithm 1, which shifts frame t-1 -> t across a
                                      single well-defined clip length.
  2. SGM (SemanticGuidedMasking)  -> per-token score from the RAW encoder tokens
                                      (Algorithm 1 operates on "Siglip(video)" directly,
                                      before any synthesis/encoding step), then the
                                      lowest-`mask_ratio`-fraction tokens are replaced
                                      with a learnable mask_token. Selection is a hard
                                      top-k forward / straight-through-Gumbel backward
                                      (see dts.py's SemanticGuidedMasking.apply_mask) so
                                      TempQuery/SpatialQuery get a real gradient instead
                                      of sitting dead at init.
  3. DTS (DynamicTokenSynthesizer)-> adds a learnable per-frame temporal_encoding to
                                      the (masked) tokens, matching SiglipAE.forward's
                                      unconditional `x = x + temporal_encoding`.
  4. Compressor (trainable)       -> T x HW tokens -> HW compressed tokens (single
                                      window, reusing compressor.py's real
                                      TransformerDecoderCompressor / LocalAttnConvCompressor
                                      so the pretrained weights plug directly into this
                                      project's actual compressor).
  5. Decoder (trainable, pretraining-only) -> HW -> T x HW reconstruction.
  6. Loss: MSE between the reconstruction and the ORIGINAL (pre-mask, pre-DTS) encoder
     tokens, on the masked positions by default (MAE-style; recon_masked_only=False
     scores all positions instead).

Scoping decisions made explicitly for this branch (see conversation / AskUserQuestion
record, not re-litigated here):
  - Video-only. The paper's headline DTS feature — replicate a SINGLE STATIC IMAGE's
    tokens T times and add the learnable temporal encoding to synthesize a pseudo-video,
    so abundant image data can be mixed into this pretraining — is NOT implemented.
    DynamicTokenSynthesizer is still used, exactly as SiglipAE uses it for a real clip:
    an additive per-frame learnable bias on top of real, distinct frames.
  - fixed_frames (T) defaults to 8, matching this project's existing
    MAX_OUTPUT_FRAMES convention (shell/pretrain_compressor.sh), not the paper's own
    T=4. Both satisfy CompressorDecoder's power-of-two constraint (decoder depth =
    log2(T)); T=8 was chosen over the paper's T=4 to match this project's existing
    default rather than the paper's.
  - Only `compressor.*` weights are saved to compressor_pretrained.pt (decoder, DTS,
    SGM are pretraining-only scaffolding), matching videollama3_pretrain_compressor.py's
    convention exactly so this branch's output is a drop-in alternative artifact.

Spatial-token constraint (same as the base AE-pretrain script):
  For LocalAttnConvCompressor the input must have exactly compress_image_wh tokens
  per frame (default 256). Use video_merge_size=2 with 448x448 input to satisfy this.
"""
import gc
import json
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
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
from videollama3.model.dts import DynamicTokenSynthesizer, SemanticGuidedMasking, SiglipAECompressor
from videollama3.model.processor import Videollama3Processor

logger = logging.getLogger(__name__)
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(" ".join(str(a) for a in args))


def _uniform_downsample_indices(n_available: int, n_keep: int) -> np.ndarray:
    """
    Return n_keep indices uniformly spread over [0, n_available-1] (n_available > n_keep).

    Same algorithm as mm_utils.get_frame_indices's sample="uniform" branch, so a
    precomputed-feature cache with more frames than fixed_frames is downsampled the
    same way read_frames_decord(sample="uniform") would have sampled the raw video.
    """
    return np.linspace(0, n_available - 1, n_keep).round().astype(int)


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
    # --- SGM masking ---
    mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Fraction of each window's T*HW tokens masked (lowest SGM tokenscore first)."},
    )
    recon_masked_only: bool = field(
        default=True,
        metadata={"help": "Compute the reconstruction MSE on masked positions only (MAE-style)."},
    )
    mask_temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "Straight-through relaxation temperature for SGM's mask selection "
                "(see SemanticGuidedMasking.apply_mask). Lower = closer to a true hard "
                "mask (sharper sigmoid, gradient concentrated tightly at the mask/keep "
                "boundary); higher = smoother, larger-magnitude gradient to "
                "TempQuery/SpatialQuery but a less faithful relaxation of the hard "
                "top-k. Does not affect which tokens are masked (forward is always the "
                "exact hard top-k) — only the backward gradient's shape."
            )
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to JSON list or dataset-registry dict: [{video: ...}, ...]"})
    data_root: Optional[str] = field(default=None)
    fixed_frames: int = field(
        default=8,
        metadata={
            "help": (
                "T: exact frame count sampled per video AND the decoder's "
                "max_output_frames (they are the same value in this branch — every "
                "window is fully reconstructed, no slot-subsampling). Must be a power "
                "of two (CompressorDecoder halves/doubles temporal length per layer)."
            )
        },
    )
    video_merge_size: int = field(
        default=2,
        metadata={"help": "Spatial merge size; set to 2 -> 256 tokens/frame for LocalAttnConv."},
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
    precomputed_feature_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory of precomputed frozen-vision-encoder features, one "
                "<video-stem>.pt per video, each a (T, HW, hidden) tensor with "
                "HW == compress_image_w * compress_image_h. T must be >= fixed_frames; "
                "if T > fixed_frames the cached frames are uniformly downsampled to "
                "exactly fixed_frames (same sampling as the raw-video path's "
                "read_frames_decord(sample='uniform')) since siglip_ae's depth is "
                "fixed at construction and every window must have exactly "
                "fixed_frames frames. T < fixed_frames is still rejected (dropped). "
                "When set, the dataset loads this cached tensor instead of decoding "
                "video frames, and the model skips the vision_encoder forward pass "
                "entirely. Samples lacking a cache file are dropped at dataset-load time."
            ),
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    compressor_lr: float = field(default=1e-4)
    decoder_lr: float = field(default=1e-4)
    dts_sgm_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "LR for DynamicTokenSynthesizer + SemanticGuidedMasking params; "
                "default -> compressor_lr. SGM's TempQuery/SpatialQuery only get "
                "gradient through the boundary-token straight-through relaxation "
                "(see dts.py), so their effective signal is weaker/noisier than a "
                "typical layer's; consider a lower value than compressor_lr if their "
                "gradnorm/sgm TensorBoard curve looks unstable."
            )
        },
    )
    model_max_length: int = field(default=512)
    group_by_modality_length: bool = field(default=False)
    ddp_find_unused_parameters: Optional[bool] = field(
        default=True,
        metadata={
            "help": (
                "SGM's TempQuery/SpatialQuery now receive a real gradient every step "
                "via the straight-through relaxation in "
                "SemanticGuidedMasking.apply_mask (see videollama3/model/dts.py), so "
                "this is no longer required for DDP correctness. Left True anyway: "
                "the extra unused-parameter traversal DDP does is cheap relative to "
                "this branch's per-step cost, and it stays a no-op safety net against "
                "any future masking-path edge case (e.g. k==0 or k==N) rather than a "
                "silent hang."
            )
        },
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoPretrainDatasetDTS(Dataset):
    """
    Loads videos and returns visual features ready for the DTS/SGM AE model.
    Every sample is sampled to EXACTLY `num_frames` frames (`sample="uniform"`
    guarantees this — see mm_utils.get_frame_indices — including for pathologically
    short source videos, which repeat frames rather than yielding fewer).
    """

    def __init__(
        self,
        data_path: str,
        processor: Videollama3Processor,
        num_frames: int,
        merge_size: int = 2,
        data_root: Optional[str] = None,
        precomputed_feature_dir: Optional[str] = None,
    ):
        with open(data_path) as f:
            raw = json.load(f)
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
            data = items
        else:
            data = raw

        self.processor = processor
        self.num_frames = num_frames
        self.merge_size = merge_size
        self.data_root = data_root
        self.precomputed_feature_dir = precomputed_feature_dir

        # If using precomputed vision-encoder features, drop samples without a
        # cache file (same pattern as videollama3_pretrain_compressor.py).
        if precomputed_feature_dir is not None:
            kept: List[Dict] = []
            cache_root = os.path.abspath(precomputed_feature_dir)
            for entry in data:
                v = entry.get("video")
                if not v:
                    continue
                if isinstance(v, (list, tuple)):
                    v = v[0]
                stem = os.path.splitext(os.path.basename(str(v)))[0]
                feat_path = os.path.join(cache_root, f"{stem}.pt")
                if not os.path.exists(feat_path):
                    continue
                entry = dict(entry)
                entry["_precomputed_feat_path"] = feat_path
                kept.append(entry)
            n_drop = len(data) - len(kept)
            if n_drop:
                logger.warning("Dropped %d/%d samples without precomputed feature under %s",
                               n_drop, len(data), precomputed_feature_dir)
            data = kept

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> Dict:
        item = self.data[i]

        if self.precomputed_feature_dir is not None:
            return self._getitem_precomputed(i, item)

        root = item.get("_data_root", self.data_root or "")
        video_path = os.path.join(root, item["video"]) if root else item["video"]

        try:
            frames, _ = read_frames_decord(
                video_path,
                num_frames=self.num_frames,
                sample="uniform",
                return_timestamps=True,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s — retrying random sample", video_path, exc)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # Defensive check for the branch's core requirement (should always hold under
        # sample="uniform"; see get_frame_indices).
        if len(frames) != self.num_frames:
            logger.warning(
                "Skipping %s: decoded %d frames != fixed_frames=%d",
                video_path, len(frames), self.num_frames,
            )
            return self.__getitem__(random.randint(0, len(self.data) - 1))

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
            "n_frames": torch.tensor(self.num_frames, dtype=torch.long),
        }

    def _getitem_precomputed(self, _i: int, item: Dict) -> Dict:
        """Load a cached (T, HW, hidden) vision-encoder feature tensor instead of
        decoding video + running the vision encoder. See DataArguments.precomputed_feature_dir."""
        feat_path = item["_precomputed_feat_path"]
        try:
            visual_tokens = torch.load(feat_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — retrying random sample", feat_path, exc)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        if visual_tokens.dim() != 3 or visual_tokens.shape[0] < self.num_frames:
            logger.warning(
                "%s: expected a (T>=%d, HW, hidden) tensor (fixed_frames=%d), got shape %s "
                "— retrying random sample",
                feat_path, self.num_frames, self.num_frames, tuple(visual_tokens.shape),
            )
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # siglip_ae's depth is fixed at construction, so every window must have
        # exactly fixed_frames frames. A cache built with a larger frame count is
        # uniformly downsampled to fixed_frames (mirrors the raw-video path, which
        # caps at fixed_frames via read_frames_decord(sample="uniform")); a cache
        # with fewer frames than fixed_frames is rejected above since there is no
        # equivalent "uniform" way to synthesize the missing frames.
        if visual_tokens.shape[0] > self.num_frames:
            idx = _uniform_downsample_indices(visual_tokens.shape[0], self.num_frames)
            visual_tokens = visual_tokens[idx]

        n_frames, HW, hidden = visual_tokens.shape
        return {
            "visual_tokens": visual_tokens.reshape(n_frames * HW, hidden),
            "n_frames": torch.tensor(n_frames, dtype=torch.long),
        }


class DataCollatorForDTSPretraining:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if "visual_tokens" in instances[0]:
            return {
                "visual_tokens": torch.cat([x["visual_tokens"] for x in instances], dim=0),
                "n_frames": torch.stack([x["n_frames"] for x in instances]),
            }
        return {
            "pixel_values": torch.cat([x["pixel_values"] for x in instances], dim=0),
            "grid_sizes": torch.cat([x["grid_sizes"] for x in instances], dim=0),
            "merge_sizes": torch.cat([x["merge_sizes"] for x in instances], dim=0),
            "n_frames": torch.stack([x["n_frames"] for x in instances]),
        }


# ---------------------------------------------------------------------------
# AE model output
# ---------------------------------------------------------------------------

@dataclass
class DTSOutput(BaseModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None
    mask_ratio_actual: Optional[torch.Tensor] = None
    # Collapse / scale-mismatch monitors (no grad) — same convention as
    # videollama3_pretrain_compressor.py, read by diagnostics/ tooling.
    recon_norm: Optional[torch.Tensor] = None
    recon_norm_std: Optional[torch.Tensor] = None
    target_norm: Optional[torch.Tensor] = None
    recon_cos: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# DTSCompressorAutoEncoder
# ---------------------------------------------------------------------------

class DTSCompressorAutoEncoder(nn.Module):
    """
    Wraps vision_encoder (frozen) + SGM masking + DTS temporal encoding +
    compressor + decoder. Forward returns DTSOutput(loss=recon_loss).
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        compressor: nn.Module,
        decoder: CompressorDecoder,
        dts: Optional[DynamicTokenSynthesizer],
        sgm: SemanticGuidedMasking,
        num_frames: int,
        mask_ratio: float = 0.75,
        recon_masked_only: bool = True,
        mask_temperature: float = 1.0,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.compressor = compressor
        self.decoder = decoder
        self.dts = dts
        self.sgm = sgm
        self.mask_temperature = mask_temperature
        self.num_frames = num_frames
        self.mask_ratio = mask_ratio
        self.recon_masked_only = recon_masked_only

        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        merge_sizes: Optional[torch.Tensor] = None,
        n_frames: torch.Tensor = None,  # type: ignore[assignment]
        visual_tokens: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> DTSOutput:
        n_frames_list: List[int] = n_frames.tolist()
        B = len(n_frames_list)
        T = self.num_frames
        HW: int = self.compressor.compress_image_wh  # type: ignore[attr-defined]
        assert all(n == T for n in n_frames_list), (
            f"DTS pretrain requires every sample to have exactly fixed_frames={T} "
            f"frames; got n_frames={n_frames_list}."
        )

        if visual_tokens is not None:
            # Precomputed-feature path (data_args.precomputed_feature_dir): the
            # vision_encoder forward pass is bypassed entirely.
            device = visual_tokens.device
        else:
            assert pixel_values is not None, (
                "forward() needs either pixel_values (+ grid_sizes/merge_sizes) or "
                "visual_tokens (precomputed-feature path)."
            )
            device = pixel_values.device
            # 1. Vision encoder (no grad).
            with torch.no_grad():
                visual_tokens = self.vision_encoder(
                    pixel_values=pixel_values,
                    grid_sizes=grid_sizes,
                    merge_sizes=merge_sizes,
                )
        assert visual_tokens is not None
        expected_total = B * T * HW
        if visual_tokens.shape[0] != expected_total:
            raise ValueError(
                f"Got {visual_tokens.shape[0]} visual tokens, expected "
                f"{expected_total} (= B={B} * T={T} * HW={HW}). Check video_merge_size "
                "matches compress_image_w/h."
            )

        C = visual_tokens.shape[-1]
        tokens_bthwc = visual_tokens.view(B, T, HW, C)

        # 2. SGM: score from the RAW encoder tokens (Algorithm 1 operates on
        #    "Siglip(video)" directly), then mask the lowest-scoring positions.
        scores = self.sgm.compute_scores(tokens_bthwc)
        masked_tokens, mask_bhw = self.sgm.apply_mask(
            tokens_bthwc, scores, self.mask_ratio, temperature=self.mask_temperature
        )

        # 3. DTS: inject the learnable per-frame temporal encoding after masking,
        #    matching SiglipAE (adds temporal_encoding unconditionally before encoding).
        #    Skipped when self.dts is None: compressor_type="siglip_ae" already does
        #    this internally (see SiglipAECompressor in dts.py) — applying it here too
        #    would add the additive bias twice.
        compressor_input = masked_tokens if self.dts is None else self.dts(masked_tokens)
        compressor_input_flat = compressor_input.reshape(B * T * HW, C)

        # 4. Compress: single T*HW-token window per sample -> HW compressed tokens.
        cu_seqlens = torch.arange(
            0, (B + 1) * T * HW, step=T * HW, device=device, dtype=torch.int32
        )
        compressed = self.compressor(compressor_input_flat, cu_seqlens)  # (B*HW, C)

        # 5. Decode back to T frames.
        decoded = self.decoder(compressed)  # (B*T*HW, C)
        decoded_bthwc = decoded.view(B, T, HW, C)

        # 6. Reconstruction loss vs the ORIGINAL (pre-mask, pre-DTS) tokens.
        target_bthwc = tokens_bthwc.detach()
        sel = mask_bhw if self.recon_masked_only else torch.ones_like(mask_bhw)
        selected_pred = decoded_bthwc[sel].float()
        selected_tgt = target_bthwc[sel].float()
        recon = F.mse_loss(selected_pred, selected_tgt)

        with torch.no_grad():
            s = decoded_bthwc.detach().float().reshape(-1, C)
            t = target_bthwc.float().reshape(-1, C)
            recon_norm_all = s.norm(dim=-1)
            recon_norm = recon_norm_all.mean()
            recon_norm_std = recon_norm_all.std()
            target_norm = t.norm(dim=-1).mean()
            recon_cos = F.cosine_similarity(s, t, dim=-1).mean()
            mask_ratio_actual = mask_bhw.float().mean()

        return DTSOutput(
            loss=recon,
            recon_loss=recon.detach(),
            mask_ratio_actual=mask_ratio_actual,
            recon_norm=recon_norm,
            recon_norm_std=recon_norm_std,
            target_norm=target_norm,
            recon_cos=recon_cos,
        )


# ---------------------------------------------------------------------------
# Per-module gradient-norm diagnostics
# ---------------------------------------------------------------------------

def _grad_group_key(name: str) -> str:
    """Coarse module bucket for a parameter name, used to localise which part of
    the compressor / decoder / DTS / SGM a gradient spike comes from."""
    if name.startswith("dts."):
        return "dts"
    if name.startswith("sgm."):
        return "sgm"
    if name.startswith("compressor."):
        if name.endswith(".query"):
            return "compressor.query"
        if ".layers." in name:
            idx = name.split(".layers.")[1].split(".")[0]
            return f"compressor.layer{idx}"
        return "compressor.other"
    if name.startswith("decoder."):
        if ".layers." in name:
            idx = name.split(".layers.")[1].split(".")[0]
            return f"decoder.layer{idx}"
        return "decoder.other"
    return "other"


# ---------------------------------------------------------------------------
# Custom Trainer (separate LR groups for compressor vs decoder vs dts/sgm)
# ---------------------------------------------------------------------------

class DTSPretrainTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aux_running: Dict[str, float] = {}
        self._aux_count: int = 0
        self._grad_norm_logs: Dict[str, float] = {}
        self._grad_norm_warn_threshold: float = float(os.environ.get("GRAD_NORM_WARN", "100"))

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            self._collect_grad_norms()
        return loss

    def _collect_grad_norms(self):
        log_every = int(getattr(self.args, "logging_steps", 0) or 0)
        if log_every and ((self.state.global_step + 1) % log_every != 0):
            return

        names: List[str] = []
        norms: List[torch.Tensor] = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            names.append(n)
            norms.append(p.grad.detach().norm(2))
        if not norms:
            return

        group_sq: Dict[str, torch.Tensor] = {}
        for n, t in zip(names, norms):
            key = _grad_group_key(n)
            sq = t * t
            group_sq[key] = sq if key not in group_sq else group_sq[key] + sq
        logs = {f"gradnorm/{k}": (v.item() ** 0.5) for k, v in group_sq.items()}

        stacked = torch.stack(norms)
        mx = torch.argmax(stacked)
        max_idx = int(mx.item())
        max_val = float(stacked[max_idx].item())
        max_name = names[max_idx]
        logs["gradnorm/max_param"] = max_val
        self._grad_norm_logs = logs

        if max_val > self._grad_norm_warn_threshold:
            rank0_print(
                f"[gradnorm] step {self.state.global_step}: "
                f"largest grad ‖{max_name}‖ = {max_val:.1f}"
            )

    def create_optimizer(self):
        from transformers.trainer_pt_utils import get_parameter_names
        from videollama3.train.videollama3_trainer import ALL_LAYERNORM_LAYERS

        if self.optimizer is not None:
            return self.optimizer

        assert self.model is not None
        compressor_lr = getattr(self.args, "compressor_lr", 1e-4)
        decoder_lr = getattr(self.args, "decoder_lr", 1e-4)
        dts_sgm_lr = getattr(self.args, "dts_sgm_lr", None) or compressor_lr

        decay_params = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        dts_sgm_names = {n for n, _ in trainable if n.startswith("dts.") or n.startswith("sgm.")}
        rest = [(n, p) for n, p in trainable if n not in dts_sgm_names]
        compressor_names = {n for n, _ in rest if "compressor" in n}
        decoder_names = {n for n, _ in rest if "decoder" in n}

        def _groups(names, lr):
            decay = [p for n, p in trainable if n in names and n in decay_params]
            nodecay = [p for n, p in trainable if n in names and n not in decay_params]
            out = []
            if decay:
                out.append({"params": decay, "weight_decay": self.args.weight_decay, "lr": lr})
            if nodecay:
                out.append({"params": nodecay, "weight_decay": 0.0, "lr": lr})
            return out

        param_groups = (
            _groups(compressor_names, compressor_lr)
            + _groups(decoder_names, decoder_lr)
            + _groups(dts_sgm_names, dts_sgm_lr)
        )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        for key in ("recon_loss", "mask_ratio_actual", "recon_norm", "recon_norm_std",
                    "target_norm", "recon_cos"):
            v = getattr(outputs, key, None)
            if v is None:
                continue
            self._aux_running[key] = self._aux_running.get(key, 0.0) + float(v.detach().float().item())
        self._aux_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        if self._aux_count > 0:
            for key, total in self._aux_running.items():
                logs[key] = total / self._aux_count
            self._aux_running.clear()
            self._aux_count = 0
        if self._grad_norm_logs:
            logs.update(self._grad_norm_logs)
            self._grad_norm_logs = {}
        return super().log(logs, *args, **kwargs)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_vision_encoder(model_path: str, dtype: torch.dtype = torch.bfloat16):
    """Extract only the vision encoder from a VideoLLaMA3 checkpoint."""
    rank0_print(f"Loading vision encoder from {model_path} ...")
    full_model = Videollama3Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    ve = full_model.get_vision_encoder()
    ve_hidden_size = ve.hidden_size
    ve = ve.to("cpu")  # will be moved to correct device by Trainer
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    rank0_print(f"Vision encoder loaded. hidden_size={ve_hidden_size}")
    return ve, ve_hidden_size


def _build_compressor(model_args: ModelArguments, ve_hidden_size: int, num_frames: int):
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
        # Only load-bearing for compressor_type="siglip_ae" (its depth is fixed at
        # construction as log2(window_size) — see SiglipAECompressor); harmless
        # metadata for the other two types, which handle T generically at runtime.
        window_size=num_frames,
    )
    ct = cfg.compressor_type
    if "transformer_decoder" in ct:
        compressor: nn.Module = TransformerDecoderCompressor(config=cfg)
    elif ct == "local_attn_conv":
        compressor = LocalAttnConvCompressor(config=cfg)
    elif ct == "siglip_ae":
        compressor = SiglipAECompressor(config=cfg)
    else:
        raise ValueError(f"Unknown compressor_type: {ct}")
    return compressor, cfg


def _build_decoder(compressor_cfg: Videollama3TokenCompressorConfig, num_frames: int) -> CompressorDecoder:
    return CompressorDecoder(compressor_cfg, max_output_frames=num_frames)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    T = data_args.fixed_frames
    assert T >= 1 and (T & (T - 1)) == 0, (
        f"data_args.fixed_frames must be a power of two (CompressorDecoder doubles "
        f"the temporal length per layer); got {T}."
    )

    # ---- Vision encoder ------------------------------------------------
    vision_encoder, ve_hidden_size = _load_vision_encoder(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    # ---- Compressor + Decoder + DTS + SGM -------------------------------
    compressor, compressor_cfg = _build_compressor(model_args, ve_hidden_size, T)
    decoder = _build_decoder(compressor_cfg, T)
    # compressor_type="siglip_ae" already applies its own temporal_encoding internally
    # (SiglipAECompressor reuses DynamicTokenSynthesizer — see dts.py); building a
    # second, external one here and applying it before the compressor would add the
    # additive bias TWICE. Skip it for that compressor type; DTSCompressorAutoEncoder
    # treats dts=None as "the compressor already handles this".
    dts = (
        None if model_args.compressor_type == "siglip_ae"
        else DynamicTokenSynthesizer(hidden_size=ve_hidden_size, num_frames=T)
    )
    sgm = SemanticGuidedMasking(hidden_size=ve_hidden_size)

    rank0_print(
        f"Compressor type={model_args.compressor_type} "
        f"layers={compressor_cfg.num_layers} "
        f"heads={compressor_cfg.num_attention_heads} "
        f"HW={compressor_cfg.compress_image_w}×{compressor_cfg.compress_image_h}"
    )
    rank0_print(f"Fixed frames T={T} (decoder depth=log2(T)={decoder.num_layers})")
    rank0_print(
        f"SGM mask_ratio={model_args.mask_ratio} recon_masked_only={model_args.recon_masked_only} "
        f"mask_temperature={model_args.mask_temperature}"
    )

    ae_model = DTSCompressorAutoEncoder(
        vision_encoder=vision_encoder,
        compressor=compressor,
        decoder=decoder,
        dts=dts,
        sgm=sgm,
        num_frames=T,
        mask_ratio=model_args.mask_ratio,
        recon_masked_only=model_args.recon_masked_only,
        mask_temperature=model_args.mask_temperature,
    )

    # ---- Processor & Dataset -------------------------------------------
    processor = Videollama3Processor.from_pretrained(model_args.model_name_or_path)

    if data_args.force_image_size is not None:
        processor.image_processor.force_size = [data_args.force_image_size] * 2
        rank0_print(f"Force set image size to be {data_args.force_image_size}")
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

    train_dataset = VideoPretrainDatasetDTS(
        data_path=data_args.data_path,
        processor=processor,
        num_frames=T,
        merge_size=data_args.video_merge_size,
        data_root=data_args.data_root,
        precomputed_feature_dir=data_args.precomputed_feature_dir,
    )
    rank0_print(f"Dataset size: {len(train_dataset)}")

    collator = DataCollatorForDTSPretraining()

    # ---- Trainer -------------------------------------------------------
    trainer = DTSPretrainTrainer(
        model=ae_model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    # Save only compressor weights (decoder/DTS/SGM are pretraining artifacts),
    # matching videollama3_pretrain_compressor.py's convention exactly.
    if trainer.is_world_process_zero():
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
