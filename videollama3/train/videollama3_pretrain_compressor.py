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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from PIL import Image
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
from videollama3.model.internvideo2_vendor import load_internvideo2_l, normalize_for_iv2
from videollama3.model.processor import Videollama3Processor
from videollama3.model.semantic_loss import GlobalSemanticLoss
from videollama3.model.vqgan_vendor import load_vqgan_decoder

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


def _uniform_downsample_indices(n_available: int, n_keep: int) -> np.ndarray:
    """
    Return n_keep indices uniformly spread over [0, n_available-1] (n_available > n_keep).

    Same algorithm as mm_utils.get_frame_indices's sample="uniform" branch, so a
    precomputed-feature cache with more frames than max_frames is downsampled the
    same way read_frames_decord(sample="uniform") would have sampled the raw video.
    """
    return np.linspace(0, n_available - 1, n_keep).round().astype(int)


def _frame_to_vqgan_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """PIL frame -> (3, size, size) tensor in [-1, 1] (taming-transformers convention)."""
    img = img.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    t = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
    return t.permute(2, 0, 1)


# Temporal arrangements the order-discrimination head must tell apart.
_ORDER_CLASSES = ("original", "reversed", "permuted")


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
    max_output_frames: int = field(
        default=8,
        metadata={
            "help": "Output frame count; must be a power of two. The decoder depth is "
            "fixed at log2(max_output_frames) — each layer doubles the temporal length."
        },
    )
    # --- Load a pretrained (possibly frozen) compressor ---
    load_compressor_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to a pretrained compressor state dict to load into the "
                "freshly-constructed compressor (must match compressor_type/layers/"
                "heads/compress_image_w/h exactly — strict=True). Accepts either the "
                "standalone compressor_pretrained.pt (raw keys) or a full training "
                "checkpoint's model.safetensors/pytorch_model.bin ('compressor.' "
                "prefixed keys, auto-stripped). Combine with --freeze_compressor to "
                "probe how well a FROZEN bottleneck can be reconstructed by a "
                "freshly-initialized decoder."
            ),
        },
    )
    freeze_compressor: bool = field(
        default=False,
        metadata={
            "help": (
                "Freeze the compressor (requires_grad=False) so only the decoder "
                "trains. Requires --load_compressor_path (freezing a random-init "
                "compressor is never intended). Use with plain recon loss "
                "(use_semantic_loss=False, the default) to test whether a fresh "
                "decoder can reconstruct a given pretrained compressor's bottleneck — "
                "decouples bottleneck information content from any one decoder's "
                "co-training history."
            ),
        },
    )
    # --- Global semantic loss (optional) ---
    use_semantic_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Decode each decoder output frame-slot through the frozen MLP -> "
                "VQ-GAN -> IV2 cycle (not the compressor bottleneck): restack the "
                "decoded frames into a video and compare its IV2 feature against "
                "the cached IV2 video feature (semantic_loss_weight), and optionally "
                "MSE the decoded frames against the real input frames "
                "(decoder_pixel_loss_weight)."
            ),
        },
    )
    semantic_loss_weight: float = field(default=0.1)
    decoder_pixel_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for pixel-space MSE between the VQ-GAN-decoded per-frame "
                "images (from the decoder's output, not the bottleneck) and the "
                "real input frames. Requires use_semantic_loss=True."
            ),
        },
    )
    vqgan_state_dict: Optional[str] = field(
        default="pretrained_models/vqgan/state_dict.pt",
        metadata={"help": "Plain state-dict extracted from vqgan-f16-16384."},
    )
    iv2_ckpt: Optional[str] = field(
        default="pretrained_models/iv2_L/pytorch_model.bin",
        metadata={"help": "InternVideo2-L Stage-2 checkpoint."},
    )
    semantic_mlp_hidden: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden width of the semantic-loss MLP; default 2× VQGAN z_channels (=512)."},
    )
    commit_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Weight for the VQ commitment loss ||z - sg(e_nearest)||² on the MLP "
                "projection that feeds VQ-GAN.  Keeps z close to the frozen codebook. "
                "Requires use_semantic_loss=True (the MLP only exists on that path)."
            ),
        },
    )
    # --- VideoMAE-style temporal tube masking (optional) ---
    use_tube_mask: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable VideoMAE-style temporal tube masking of the compressor input. "
                "A random subset of spatial positions is replaced by a learnable "
                "mask_token in *every* frame (a temporal tube), so the model cannot "
                "exploit temporal redundancy to trivially copy the masked content."
            ),
        },
    )
    mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Fraction of the HW spatial positions to mask (shared across all frames)."},
    )
    recon_masked_only: bool = field(
        default=True,
        metadata={
            "help": (
                "When use_tube_mask is set, compute the reconstruction MSE on the "
                "masked positions only (MAE-style).  Set False to keep the loss over "
                "all positions."
            ),
        },
    )
    # --- Feature-space reconstruction loss weight ---
    recon_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight on the feature-space reconstruction loss (decoder output vs. "
                "frozen vision-encoder tokens). This is the only loss term with no "
                "on/off flag, so it defaults to 1.0 (unweighted, previous behavior); "
                "lower it to let semantic_loss_weight/decoder_pixel_loss_weight "
                "dominate."
            ),
        },
    )
    # --- Motion-weighted reconstruction ---
    motion_loss_weight: float = field(
        default=1.0,
        metadata={
            "help": (
                "Weight λ on the *motion* component of the reconstruction loss. The "
                "per-position MSE is split into a temporal-mean (static) term and a "
                "deviation-from-mean (motion) term: recon = ‖m-m̂‖² + λ·‖δ-δ̂‖². λ=1 "
                "reproduces plain MSE exactly; λ>1 amplifies the motion gradient to "
                "help escape the temporal-mean collapse basin."
            ),
        },
    )
    # --- Order-discrimination auxiliary loss ---
    use_order_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "Add a classification head on the compressed tokens that must tell "
                "apart the temporal arrangement of the input (original / reversed / "
                "permuted), forcing the compressed feature to encode frame order "
                "instead of collapsing to the order-invariant temporal mean."
            ),
        },
    )
    order_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Scale for the order-discrimination cross-entropy loss."},
    )


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
    iv2_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory of precomputed IV2 video features (one .pt per video stem). "
                "Required when training_args.use_semantic_loss is True; samples lacking a "
                "cache file are dropped at dataset-load time."
            ),
        },
    )
    precomputed_feature_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Directory of precomputed frozen-vision-encoder features, one "
                "<video-stem>.pt per video, each a (T, HW, hidden) tensor with "
                "HW == compress_image_w * compress_image_h. T may be any value >= "
                "min_frames; if T > max_frames the cached frames are uniformly "
                "downsampled to max_frames (same sampling as the raw-video path's "
                "read_frames_decord(sample='uniform')). When set, the dataset loads "
                "this cached tensor instead of decoding video frames, and the model "
                "skips the vision_encoder forward pass entirely (the main cost "
                "precomputing removes). Samples lacking a cache file are dropped at "
                "dataset-load time. Incompatible with model_args.use_semantic_loss's "
                "decoder_pixel_loss path, which needs real frame pixels that this "
                "mode never decodes."
            ),
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    optim: str = field(default="adamw_torch")
    compressor_lr: float = field(default=1e-4)
    decoder_lr: float = field(default=1e-4)
    semantic_mlp_lr: Optional[float] = field(
        default=None,
        metadata={"help": "LR for the semantic-loss MLP head; default → compressor_lr."},
    )
    model_max_length: int = field(default=512)
    group_by_modality_length: bool = field(default=False)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoPretrainDataset(Dataset):
    """
    Loads videos and returns visual features in a format ready for the AE model.
    No conversation labels required.

    When ``iv2_cache_dir`` is given, samples whose precomputed IV2 feature is
    missing are filtered out at construction time (rather than being silently
    skipped at __getitem__), so length and __getitem__ stay consistent.
    """

    def __init__(
        self,
        data_path: str,
        processor: Videollama3Processor,
        max_frames: int = 10,
        merge_size: int = 2,
        data_root: Optional[str] = None,
        min_frames: int = 4,
        iv2_cache_dir: Optional[str] = None,
        decode_pixel_targets: bool = False,
        vqgan_image_size: int = 256,
        precomputed_feature_dir: Optional[str] = None,
    ):
        if precomputed_feature_dir is not None and decode_pixel_targets:
            raise ValueError(
                "precomputed_feature_dir provides cached vision-encoder features only "
                "(no raw pixels) — incompatible with decode_pixel_targets=True "
                "(model_args.use_semantic_loss's decoder_pixel_loss path needs real "
                "frame pixels this mode never decodes)."
            )
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
            data = items
        else:
            data = raw

        self.processor = processor
        self.max_frames = max_frames
        self.merge_size = merge_size
        self.data_root = data_root
        self.min_frames = min_frames
        self.iv2_cache_dir = iv2_cache_dir
        self.decode_pixel_targets = decode_pixel_targets
        self.vqgan_image_size = vqgan_image_size
        self.precomputed_feature_dir = precomputed_feature_dir

        # If using cached IV2 features, drop samples that don't have a cache file.
        if iv2_cache_dir is not None:
            kept: List[Dict] = []
            cache_root = os.path.abspath(iv2_cache_dir)
            for entry in data:
                v = entry.get("video")
                if not v:
                    continue
                if isinstance(v, (list, tuple)):
                    v = v[0]
                stem = os.path.splitext(os.path.basename(str(v)))[0]
                # honor explicit "iv2_feat_path" if precompute meta was passed in,
                # otherwise look it up by stem under cache_root.
                feat_path = entry.get("iv2_feat_path")
                if feat_path is None or not os.path.isabs(feat_path):
                    feat_path = os.path.join(cache_root, f"{stem}.pt")
                if not os.path.exists(feat_path):
                    continue
                entry = dict(entry)
                entry["_iv2_feat_path"] = feat_path
                kept.append(entry)
            n_drop = len(data) - len(kept)
            if n_drop:
                logger.warning("Dropped %d/%d samples without IV2 cache under %s",
                               n_drop, len(data), iv2_cache_dir)
            data = kept

        # If using precomputed vision-encoder features, drop samples without a
        # cache file too (same pattern as the IV2 filter above).
        if precomputed_feature_dir is not None:
            kept = []
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
                num_frames=self.max_frames,
                sample="uniform",
                return_timestamps=True,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s — retrying random sample", video_path, exc)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # sample="uniform" always returns exactly self.max_frames frames (indices
        # repeat only for pathologically short raw videos), so every sample in a
        # batch has the same frame count. min_frames is kept only as a guard
        # against those degenerate near-empty videos.
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

        out: Dict = {
            "pixel_values": data_dict["pixel_values"],
            "grid_sizes": data_dict["grid_sizes"],
            "merge_sizes": data_dict["merge_sizes"],
            "n_frames": torch.tensor(n_frames, dtype=torch.long),
        }
        if "_iv2_feat_path" in item:
            # Cached IV2 vector saved as fp16; collator/model casts later.
            vid_feat = torch.load(item["_iv2_feat_path"], map_location="cpu", weights_only=True)
            out["vid_feat"] = vid_feat.to(torch.float32)
        if self.decode_pixel_targets:
            # Ground truth for the decoder-end VQ-GAN pixel/semantic loss: the real
            # frames at the VQ-GAN's own output resolution, in its [-1, 1] convention.
            out["vqgan_target_frames"] = torch.stack(
                [_frame_to_vqgan_tensor(f, self.vqgan_image_size) for f in frames]
            )
        return out

    def _getitem_precomputed(self, _i: int, item: Dict) -> Dict:
        """Load a cached (T, HW, hidden) vision-encoder feature tensor instead of
        decoding video + running the vision encoder. See DataArguments.precomputed_feature_dir."""
        feat_path = item["_precomputed_feat_path"]
        try:
            visual_tokens = torch.load(feat_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.warning("Failed to load %s: %s — retrying random sample", feat_path, exc)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        if visual_tokens.dim() != 3:
            logger.warning(
                "%s: expected a (T, HW, hidden) tensor, got shape %s — retrying random sample",
                feat_path, tuple(visual_tokens.shape),
            )
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        n_frames, HW, hidden = visual_tokens.shape
        if n_frames < self.min_frames:
            logger.warning(
                "Skipping %s: cached %d frames < min_frames=%d",
                feat_path, n_frames, self.min_frames,
            )
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        # Cached feature may have been extracted with a different (larger) frame
        # count than this run's max_frames. Mirror the raw-video path — which caps
        # at max_frames via read_frames_decord(sample="uniform") — by uniformly
        # downsampling along T instead of feeding every cached frame through.
        if n_frames > self.max_frames:
            idx = _uniform_downsample_indices(n_frames, self.max_frames)
            visual_tokens = visual_tokens[idx]
            n_frames = self.max_frames

        out: Dict = {
            "visual_tokens": visual_tokens.reshape(n_frames * HW, hidden),
            "n_frames": torch.tensor(n_frames, dtype=torch.long),
        }
        if "_iv2_feat_path" in item:
            vid_feat = torch.load(item["_iv2_feat_path"], map_location="cpu", weights_only=True)
            out["vid_feat"] = vid_feat.to(torch.float32)
        return out


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class DataCollatorForPretraining:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = {}
        if "visual_tokens" in instances[0]:
            # Precomputed-feature path: no pixels/vision-encoder inputs to batch.
            batch["visual_tokens"] = torch.cat([x["visual_tokens"] for x in instances], dim=0)
        else:
            batch["pixel_values"] = torch.cat([x["pixel_values"] for x in instances], dim=0)
            batch["grid_sizes"] = torch.cat([x["grid_sizes"] for x in instances], dim=0)
            batch["merge_sizes"] = torch.cat([x["merge_sizes"] for x in instances], dim=0)
        batch["n_frames"] = torch.stack([x["n_frames"] for x in instances])
        if "vid_feat" in instances[0]:
            batch["vid_feat"] = torch.stack([x["vid_feat"] for x in instances])
        if "vqgan_target_frames" in instances[0]:
            batch["vqgan_target_frames"] = torch.cat(
                [x["vqgan_target_frames"] for x in instances], dim=0
            )
        return batch


# ---------------------------------------------------------------------------
# AE model output
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional

@_dataclass
class AEOutput(BaseModelOutput):
    loss: _Optional[torch.Tensor] = None
    recon_loss: _Optional[torch.Tensor] = None
    sem_loss: _Optional[torch.Tensor] = None
    pixel_loss: _Optional[torch.Tensor] = None
    commit_loss: _Optional[torch.Tensor] = None
    # Static (temporal-mean) vs motion (deviation-from-mean) split of the recon loss.
    recon_static: _Optional[torch.Tensor] = None
    recon_motion: _Optional[torch.Tensor] = None
    # Order-discrimination auxiliary loss and its train accuracy (chance = 1/3).
    order_loss: _Optional[torch.Tensor] = None
    order_acc: _Optional[torch.Tensor] = None
    # Collapse / scale-mismatch monitors (no grad). recon_norm/target_norm track the
    # output scale (should approach target_norm); recon_norm_std → 0 flags norm
    # collapse; recon_cos → 0 flags directional collapse (output ignores the input).
    recon_norm: _Optional[torch.Tensor] = None
    recon_norm_std: _Optional[torch.Tensor] = None
    target_norm: _Optional[torch.Tensor] = None
    recon_cos: _Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# CompressorAutoEncoder
# ---------------------------------------------------------------------------

class CompressorAutoEncoder(nn.Module):
    """
    Wraps vision_encoder (frozen) + compressor + decoder.
    Forward returns AEOutput(loss=recon_loss).
    Compatible with HF Trainer (returns dict-like with 'loss').
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        compressor: nn.Module,
        decoder: CompressorDecoder,
        semantic_loss: Optional[GlobalSemanticLoss] = None,
        semantic_loss_weight: float = 0.0,
        decoder_pixel_loss_weight: float = 0.0,
        commit_loss_weight: float = 0.0,
        recon_loss_weight: float = 1.0,
        use_tube_mask: bool = False,
        mask_ratio: float = 0.75,
        recon_masked_only: bool = True,
        motion_loss_weight: float = 1.0,
        use_order_loss: bool = False,
        order_loss_weight: float = 0.0,
        freeze_compressor: bool = False,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.compressor = compressor
        self.decoder = decoder
        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.decoder_pixel_loss_weight = decoder_pixel_loss_weight
        self.commit_loss_weight = commit_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.use_tube_mask = use_tube_mask
        self.mask_ratio = mask_ratio
        self.recon_masked_only = recon_masked_only
        self.motion_loss_weight = motion_loss_weight
        self.use_order_loss = use_order_loss
        self.order_loss_weight = order_loss_weight

        # Learnable mask embedding for VideoMAE-style tube masking. Created only
        # when masking is enabled so it does not show up in the saved state dict
        # otherwise. Lives at the AE top level; create_optimizer routes it into
        # the compressor LR group.
        if use_tube_mask:
            hidden: int = compressor.hidden_size  # type: ignore[attr-defined]
            self.mask_token = nn.Parameter(torch.zeros(hidden))
            nn.init.normal_(self.mask_token, std=0.02)

        # Order-discrimination head: a per-token 3-way classifier (original /
        # reversed / permuted) on the compressed feature. Created only when enabled
        # so it stays out of the saved state dict otherwise; create_optimizer routes
        # it into the compressor LR group.
        if use_order_loss:
            hidden_o: int = compressor.hidden_size  # type: ignore[attr-defined]
            self.order_head = nn.Sequential(
                nn.Linear(hidden_o, hidden_o // 4),
                nn.GELU(),
                nn.Linear(hidden_o // 4, len(_ORDER_CLASSES)),
            )

        # Vision encoder is always frozen.
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Optionally freeze the compressor too (decoder-only probe training).
        if freeze_compressor:
            for p in self.compressor.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------
    def _apply_tube_mask(
        self,
        visual_tokens: torch.Tensor,
        n_frames_list: List[int],
        HW: int,
    ):
        """
        VideoMAE-style temporal tube masking, random.

        For each sample ``round(mask_ratio * HW)`` spatial positions are chosen
        uniformly at random and masked in *every* frame (a temporal tube), so the
        same spatial columns are hidden across the whole clip. Masked tokens are
        replaced by the learnable ``mask_token``; visible tokens are untouched.

        Returns
        -------
        masked_tokens : (total_tokens, hidden)  compressor input with mask_token
            substituted at the masked tube positions.
        spatial_masks : list of (HW,) bool tensors — the masked positions per
            sample, reused to restrict the reconstruction loss.
        """
        device = visual_tokens.device
        # Keep at least one visible and one masked position.
        k = int(round(self.mask_ratio * HW))
        k = min(max(k, 1), HW - 1)

        spatial_masks: List[torch.Tensor] = []
        token_mask_parts: List[torch.Tensor] = []
        for n in n_frames_list:
            bm = torch.zeros(HW, dtype=torch.bool, device=device)
            # Random spatial positions, masked across all n frames (the "tube").
            idx = torch.randperm(HW, device=device)[:k]
            bm[idx] = True
            spatial_masks.append(bm)
            # Same spatial mask repeated for each of the n frames (the "tube").
            token_mask_parts.append(bm.unsqueeze(0).expand(n, HW).reshape(-1))
        token_mask = torch.cat(token_mask_parts)  # (total_tokens,)

        mask_tok = self.mask_token.to(visual_tokens.dtype)
        masked_tokens = torch.where(
            token_mask.unsqueeze(-1), mask_tok.unsqueeze(0), visual_tokens
        )
        return masked_tokens, spatial_masks

    # ------------------------------------------------------------------
    @staticmethod
    def _recon_static_motion(sampled: torch.Tensor, target: torch.Tensor):
        """
        Split the per-position reconstruction MSE into a temporal-mean (static) term
        and a deviation-from-mean (motion) term.  ``sampled`` / ``target`` are
        ``(n, P, hidden)`` over the n frames at P spatial positions.

        Writing x = m + δ with m = mean_t x and δ = x − m (so Σ_t δ = 0), the cross
        term vanishes and the plain MSE decomposes orthogonally:
            mean_t ‖x − x̂‖²  =  ‖m − m̂‖²  +  mean_t ‖δ − δ̂‖²
        Returned as (static, motion), each mean-reduced; static + 1·motion equals the
        plain MSE exactly.
        """
        s_mean = sampled.mean(dim=0, keepdim=True)
        t_mean = target.mean(dim=0, keepdim=True)
        static = F.mse_loss(s_mean, t_mean)
        motion = F.mse_loss(sampled - s_mean, target - t_mean)
        return static, motion

    # ------------------------------------------------------------------
    def _order_loss(self, visual_tokens: torch.Tensor, n_frames_list: List[int], HW: int):
        """
        Temporal-order discrimination on the compressed feature.

        For every window with >= 3 frames (so original / reversed / a non-trivial
        permutation are all distinct) we build the three arrangements of its frozen-
        encoder tokens, run the *trainable* compressor on all of them in one packed
        call, and the per-token head must classify which arrangement produced each
        compressed feature (cross-entropy, chance = 1/3).

        A collapsed compressor outputs the order-invariant temporal mean, so the head
        cannot beat chance and the gradient pushes the compressed feature to encode
        frame order.  Returns (loss, accuracy), or None when no window is eligible.
        """
        device = visual_tokens.device
        hidden = visual_tokens.shape[-1]

        slices, off = [], 0
        for n in n_frames_list:
            if n >= 3:
                slices.append((off, n))
            off += n * HW
        if not slices:
            return None

        arr_tokens: List[torch.Tensor] = []
        cu: List[int] = [0]
        labels: List[int] = []
        for cls, kind in enumerate(_ORDER_CLASSES):
            for s, n in slices:
                frames = visual_tokens[s : s + n * HW].view(n, HW, hidden)
                ident = torch.arange(n, device=device)
                rev = torch.arange(n - 1, -1, -1, device=device)
                if kind == "original":
                    perm = ident
                elif kind == "reversed":
                    perm = rev
                else:
                    perm = torch.randperm(n, device=device)
                    # keep the label unambiguous: a permutation must differ from both
                    # the identity and the reverse.
                    while torch.equal(perm, ident) or torch.equal(perm, rev):
                        perm = torch.randperm(n, device=device)
                arr_tokens.append(frames[perm].reshape(-1, hidden))
                cu.append(cu[-1] + n * HW)
                labels.append(cls)

        tokens = torch.cat(arr_tokens, dim=0)
        cu_seqlens = torch.tensor(cu, device=device, dtype=torch.int32)
        labels_t = torch.tensor(labels, device=device, dtype=torch.long)

        compressed = self.compressor(tokens, cu_seqlens)  # (num_arr * HW, hidden)
        num_arr = labels_t.numel()
        # Per-token logits averaged over the HW grid: an ensemble of per-position
        # classifiers, robust to order being coded in only some spatial positions.
        logits = self.order_head(compressed).view(num_arr, HW, -1).mean(dim=1)  # (num_arr, C)
        loss = F.cross_entropy(logits, labels_t)
        with torch.no_grad():
            acc = (logits.argmax(dim=-1) == labels_t).float().mean()
        return loss, acc

    # ------------------------------------------------------------------
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        grid_sizes: Optional[torch.Tensor] = None,
        merge_sizes: Optional[torch.Tensor] = None,
        n_frames: torch.Tensor = None,  # type: ignore[assignment]
        vid_feat: Optional[torch.Tensor] = None,
        vqgan_target_frames: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
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

        if visual_tokens is not None:
            # Precomputed-feature path (data_args.precomputed_feature_dir): the
            # vision_encoder forward pass — the expensive, per-step-repeated part
            # this mode exists to skip — is bypassed entirely.
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
        # visual_tokens: (total_tokens, hidden_size)  total = sum(n_i * HW)

        # Validate spatial token count per frame matches compressor expectation.
        expected_total = sum(n * HW for n in n_frames_list)
        if visual_tokens.shape[0] != expected_total:
            raise ValueError(
                f"Got {visual_tokens.shape[0]} visual tokens, "
                f"expected {expected_total} (sum of n_i * HW={HW}). "
                "Check video_merge_size matches compress_image_w/h (pixel path) or "
                "that the cached (T, HW, hidden) feature's HW matches (precomputed path)."
            )

        # 1b. Optional VideoMAE-style temporal tube masking. A random subset of
        #     spatial positions is masked in *every* frame, so the compressor must
        #     infer them from visible spatial neighbours instead of copying the
        #     temporally-redundant content. The reconstruction target stays the
        #     *unmasked* encoder output; only the compressor input is corrupted.
        spatial_masks: Optional[List[torch.Tensor]] = None
        if self.use_tube_mask:
            compressor_input, spatial_masks = self._apply_tube_mask(
                visual_tokens, n_frames_list, HW
            )
        else:
            compressor_input = visual_tokens

        # 2. Build cu_seqlens for single-window compression per sample.
        cu_ends = [0]
        for n in n_frames_list:
            cu_ends.append(cu_ends[-1] + n * HW)
        cu_seqlens = torch.tensor(cu_ends, device=device, dtype=torch.int32)

        # 3. Compress: each sample's N frames → HW compressed tokens.
        compressed = self.compressor(compressor_input, cu_seqlens)
        # compressed: (B * HW, hidden_size)

        # 4. Decode to 10 frame slots.
        decoded = self.decoder(compressed)
        # decoded: (B * max_output_frames * HW, hidden_size)

        # 5. Reconstruction loss with uniform frame sampling. Targets are always the
        #    original (unmasked) tokens; with tube masking the loss defaults to the
        #    masked positions only (MAE-style).
        decoded_4d = decoded.view(B, max_T, HW, -1)

        recon = torch.zeros((), device=device, dtype=torch.float32)
        static_sum = torch.zeros((), device=device, dtype=torch.float32)
        motion_sum = torch.zeros((), device=device, dtype=torch.float32)
        # Per-token monitors (no grad) over all sampled positions, regardless of
        # masking — used to detect collapse / output-scale mismatch in TensorBoard.
        mon_recon_norms: List[torch.Tensor] = []
        mon_target_norms: List[torch.Tensor] = []
        mon_cos: List[torch.Tensor] = []

        # Decoder-end semantic/pixel supervision: decode each sample's *decoder*
        # output frame-slots (not the compressor bottleneck) through the frozen
        # MLP -> VQ-GAN cycle, then (a) restack the decoded frames into a video and
        # compare its IV2 feature against the cached target, and/or (b) MSE the
        # decoded frames against the real input frames.
        need_semantic_path = (
            self.semantic_loss is not None
            and vid_feat is not None
            and (
                self.semantic_loss_weight > 0.0
                or self.decoder_pixel_loss_weight > 0.0
                or self.commit_loss_weight > 0.0
            )
        )
        sem_sum = torch.zeros((), device=device, dtype=torch.float32) if (need_semantic_path and self.semantic_loss_weight > 0.0) else None
        pixel_sum = torch.zeros((), device=device, dtype=torch.float32) if (need_semantic_path and self.decoder_pixel_loss_weight > 0.0) else None
        commit_sum = torch.zeros((), device=device, dtype=torch.float32) if (need_semantic_path and self.commit_loss_weight > 0.0) else None

        offset = 0
        frame_offset = 0
        for b, n in enumerate(n_frames_list):
            target = visual_tokens[offset : offset + n * HW].view(n, HW, -1).detach()
            offset += n * HW
            indices = _uniform_frame_indices(n, max_T)
            sampled = decoded_4d[b, indices]  # (n, HW, hidden_size)
            if spatial_masks is not None and self.recon_masked_only:
                m = spatial_masks[b]  # (HW,) bool — masked positions
                static, motion = self._recon_static_motion(
                    sampled[:, m, :].float(), target[:, m, :].float()
                )
            else:
                static, motion = self._recon_static_motion(sampled.float(), target.float())
            # recon = ‖m-m̂‖² + λ·‖δ-δ̂‖²; λ=1 reproduces plain MSE exactly.
            recon = recon + static + self.motion_loss_weight * motion
            static_sum = static_sum + static.detach()
            motion_sum = motion_sum + motion.detach()
            with torch.no_grad():
                s = sampled.detach().float().reshape(-1, sampled.shape[-1])
                t = target.float().reshape(-1, target.shape[-1])
                mon_recon_norms.append(s.norm(dim=-1))
                mon_target_norms.append(t.norm(dim=-1))
                mon_cos.append(F.cosine_similarity(s, t, dim=-1))

            if need_semantic_path:
                assert self.semantic_loss is not None and vid_feat is not None  # implied by need_semantic_path
                z_flat = self.semantic_loss.project_to_latent(sampled.reshape(n * HW, -1))
                need_imgs = pixel_sum is not None or sem_sum is not None
                imgs = self.semantic_loss.decode_images(z_flat, n) if need_imgs else None
                # (n, 3, H, W) in [-1, 1], decoded from this sample's decoder output.

                if pixel_sum is not None and vqgan_target_frames is not None:
                    assert imgs is not None  # need_imgs is True whenever pixel_sum is not None
                    tgt_imgs = vqgan_target_frames[frame_offset : frame_offset + n]
                    pixel_sum = pixel_sum + F.mse_loss(imgs.float(), tgt_imgs.to(device).float())

                if sem_sum is not None:
                    assert imgs is not None  # need_imgs is True whenever sem_sum is not None
                    resized = F.interpolate(
                        imgs.float(),
                        size=(self.semantic_loss.iv2_image_size, self.semantic_loss.iv2_image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).to(imgs.dtype)
                    resized = normalize_for_iv2(resized)
                    video_in = resized.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 3, n, H, W)
                    feat = self.semantic_loss.iv2_forward(video_in).squeeze(0)  # (768,)
                    a = F.normalize(feat.float(), dim=-1)
                    tgt = F.normalize(vid_feat[b].float().to(device).detach(), dim=-1)
                    sem_sum = sem_sum + (1.0 - (a * tgt).sum())

                if commit_sum is not None:
                    commit_sum = commit_sum + self.semantic_loss.commit_loss(z_flat)

            frame_offset += n

        recon = recon / B
        recon_static = static_sum / B
        recon_motion = motion_sum / B
        sem = (sem_sum / B) if sem_sum is not None else None
        pixel = (pixel_sum / B) if pixel_sum is not None else None
        commit = (commit_sum / B) if commit_sum is not None else None

        with torch.no_grad():
            rn = torch.cat(mon_recon_norms)
            recon_norm = rn.mean()
            recon_norm_std = rn.std()
            target_norm = torch.cat(mon_target_norms).mean()
            recon_cos = torch.cat(mon_cos).mean()

        # 6b. Optional temporal-order discrimination on the compressed tokens.
        order: Optional[torch.Tensor] = None
        order_acc: Optional[torch.Tensor] = None
        if self.use_order_loss and self.order_loss_weight > 0.0:
            order_out = self._order_loss(visual_tokens, n_frames_list, HW)
            if order_out is not None:
                order, order_acc = order_out

        # Combine everything in fp32 so the optimized loss never round-trips
        # through bf16, regardless of the component dtypes.
        loss = self.recon_loss_weight * recon
        if sem is not None and self.semantic_loss_weight > 0.0:
            loss = loss + self.semantic_loss_weight * sem.float()
        if commit is not None and self.commit_loss_weight > 0.0:
            loss = loss + self.commit_loss_weight * commit.float()
        if order is not None:
            loss = loss + self.order_loss_weight * order.float()
        if pixel is not None and self.decoder_pixel_loss_weight > 0.0:
            loss = loss + self.decoder_pixel_loss_weight * pixel.float()
        return AEOutput(
            loss=loss,
            recon_loss=recon.detach(),
            sem_loss=sem.detach() if sem is not None else None,
            pixel_loss=pixel.detach() if pixel is not None else None,
            commit_loss=commit.detach() if commit is not None else None,
            recon_static=recon_static,
            recon_motion=recon_motion,
            order_loss=order.detach() if order is not None else None,
            order_acc=order_acc.detach() if order_acc is not None else None,
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
    the compressor / decoder a gradient spike comes from."""
    if name.startswith("semantic_loss"):
        return "semantic"
    if name == "mask_token":
        return "mask_token"
    if name.startswith("order_head"):
        return "order_head"
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
# Custom Trainer (separate LR groups for compressor vs decoder)
# ---------------------------------------------------------------------------

class PretrainTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Running sums for auxiliary (per-component) losses; flushed by self.log().
        self._aux_running: Dict[str, float] = {}
        self._aux_count: int = 0
        # Latest per-module grad norms, injected into the next self.log() call.
        self._grad_norm_logs: Dict[str, float] = {}
        # Warn with the exact tensor name when a single param's grad norm crosses this.
        self._grad_norm_warn_threshold: float = float(os.environ.get("GRAD_NORM_WARN", "100"))

    # ------------------------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Collect per-module grad norms on the optimizer-step micro-batch, where
        # gradients are accumulated and all-reduced but not yet clipped (HF clips
        # after training_step returns), so the logged norms are pre-clip.
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self.accelerator.sync_gradients:
            self._collect_grad_norms()
        return loss

    # ------------------------------------------------------------------
    def _collect_grad_norms(self):
        """Group unclipped gradient norms by module and record the single
        largest-grad parameter. Only runs on logged steps (aligned to
        logging_steps) to limit the per-tensor norm computation.
        """
        log_every = int(getattr(self.args, "logging_steps", 0) or 0)
        if log_every and ((self.state.global_step + 1) % log_every != 0):
            return

        names: List[str] = []
        norms: List[torch.Tensor] = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            names.append(n)
            norms.append(p.grad.detach().norm(2))  # GPU scalar, no host sync yet
        if not norms:
            return

        # Accumulate squared norms per group on-device, then a single sync per group.
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
        semantic_mlp_lr = getattr(self.args, "semantic_mlp_lr", None) or compressor_lr

        decay_params = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

        # ``semantic_loss.mlp.*`` must be matched first to avoid being caught by
        # the broader ``compressor`` / ``decoder`` filters (which use substring
        # tests against the full dotted path).
        semantic_names = {n for n, _ in trainable if n.startswith("semantic_loss.")}
        rest = [(n, p) for n, p in trainable if n not in semantic_names]
        # The learnable mask_token and the order-discrimination head live at the AE
        # top level (no "compressor"/"decoder" substring), so route them explicitly
        # into the compressor group.
        compressor_names = {
            n for n, _ in rest
            if "compressor" in n or n == "mask_token" or n.startswith("order_head")
        }
        decoder_names    = {n for n, _ in rest if "decoder"    in n}

        def _groups(names, lr):
            decay   = [p for n, p in trainable if n in names and n in decay_params]
            nodecay = [p for n, p in trainable if n in names and n not in decay_params]
            out = []
            if decay:
                out.append({"params": decay,   "weight_decay": self.args.weight_decay, "lr": lr})
            if nodecay:
                out.append({"params": nodecay, "weight_decay": 0.0, "lr": lr})
            return out

        param_groups = (
            _groups(compressor_names, compressor_lr)
            + _groups(decoder_names, decoder_lr)
            + _groups(semantic_names, semantic_mlp_lr)
        )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        for key in ("recon_loss", "sem_loss", "pixel_loss", "commit_loss",
                    "recon_static", "recon_motion", "order_loss", "order_acc",
                    "recon_norm", "recon_norm_std", "target_norm", "recon_cos"):
            v = getattr(outputs, key, None)
            if v is None:
                continue
            self._aux_running[key] = self._aux_running.get(key, 0.0) + float(v.detach().float().item())
        self._aux_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs):
        # Inject averaged auxiliary losses whenever HF Trainer logs.
        if self._aux_count > 0:
            for key, total in self._aux_running.items():
                logs[key] = total / self._aux_count
            self._aux_running.clear()
            self._aux_count = 0
        # Inject the most recent per-module grad norms.
        if self._grad_norm_logs:
            logs.update(self._grad_norm_logs)
            self._grad_norm_logs = {}
        return super().log(logs, *args, **kwargs)


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


def _load_pretrained_compressor(compressor: nn.Module, path: str) -> None:
    """Load a pretrained compressor state dict (raw or 'compressor.'-prefixed) in place."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=False)
    if any(k.startswith("compressor.") for k in state.keys()):
        state = {k[len("compressor."):]: v for k, v in state.items() if k.startswith("compressor.")}
    compressor.load_state_dict(state, strict=True)
    rank0_print(f"Loaded pretrained compressor weights from {path} ({len(state)} tensors)")


def _build_decoder(model_args: ModelArguments, compressor_cfg: Videollama3TokenCompressorConfig) -> CompressorDecoder:
    # Decoder depth is not configurable: CompressorDecoder fixes it at
    # log2(max_output_frames) (each layer doubles the temporal length). num_layers in
    # this config carrier is unused by the decoder.
    decoder_cfg = Videollama3TokenCompressorConfig(
        compressor_type=model_args.compressor_type,
        hidden_size=compressor_cfg.hidden_size,
        intermediate_size=compressor_cfg.intermediate_size,
        num_layers=compressor_cfg.num_layers,
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

    if model_args.commit_loss_weight > 0.0 and not model_args.use_semantic_loss:
        raise ValueError(
            "commit_loss_weight > 0 requires use_semantic_loss=True — the "
            "commitment loss is computed on the MLP projection that only exists "
            "on the semantic-loss path."
        )

    if model_args.decoder_pixel_loss_weight > 0.0 and not model_args.use_semantic_loss:
        raise ValueError(
            "decoder_pixel_loss_weight > 0 requires use_semantic_loss=True — the "
            "MLP -> VQ-GAN decode it reads from only exists on the semantic-loss path."
        )

    if data_args.precomputed_feature_dir is not None and model_args.decoder_pixel_loss_weight > 0.0:
        raise ValueError(
            "data_args.precomputed_feature_dir provides cached vision-encoder "
            "features only (no raw pixels) — incompatible with "
            "decoder_pixel_loss_weight > 0, which needs real frame pixels "
            "(data_args.decode_pixel_targets can't produce them in this mode)."
        )

    if model_args.use_tube_mask and not (0.0 < model_args.mask_ratio < 1.0):
        raise ValueError(
            f"mask_ratio must be in (0, 1) when use_tube_mask=True; "
            f"got {model_args.mask_ratio}."
        )

    if model_args.freeze_compressor and not model_args.load_compressor_path:
        raise ValueError(
            "freeze_compressor=True but load_compressor_path is not set — freezing "
            "a randomly-initialized compressor is almost certainly not intended."
        )

    # ---- Vision encoder ------------------------------------------------
    vision_encoder, _, ve_hidden_size = _load_vision_encoder(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    # ---- Compressor + Decoder ------------------------------------------
    compressor, compressor_cfg = _build_compressor(model_args, ve_hidden_size)
    if model_args.load_compressor_path:
        _load_pretrained_compressor(compressor, model_args.load_compressor_path)
    decoder = _build_decoder(model_args, compressor_cfg)

    rank0_print(
        f"Compressor type={model_args.compressor_type} "
        f"layers={compressor_cfg.num_layers} "
        f"heads={compressor_cfg.num_attention_heads} "
        f"HW={compressor_cfg.compress_image_w}×{compressor_cfg.compress_image_h}"
    )
    if model_args.freeze_compressor:
        rank0_print(
            f"Compressor FROZEN (probe mode) — only the decoder trains. "
            f"{sum(p.numel() for p in compressor.parameters()):,} compressor params "
            f"excluded from the optimizer."
        )
    rank0_print(f"Decoder layers={decoder.num_layers} (=log2 max_output_frames) "
                f"max_output_frames={model_args.max_output_frames}")
    if model_args.use_tube_mask:
        rank0_print(
            f"Tube masking enabled: mask_ratio={model_args.mask_ratio} "
            f"recon_masked_only={model_args.recon_masked_only}"
        )
    if model_args.motion_loss_weight != 1.0:
        rank0_print(f"Motion-weighted recon enabled: λ={model_args.motion_loss_weight}")
    if model_args.use_order_loss:
        rank0_print(
            f"Order-discrimination loss enabled: weight={model_args.order_loss_weight}"
        )

    # ---- Optional global semantic loss --------------------------------
    semantic_loss_module: Optional[GlobalSemanticLoss] = None
    if model_args.use_semantic_loss:
        if data_args.iv2_cache_dir is None:
            raise ValueError(
                "model_args.use_semantic_loss=True but data_args.iv2_cache_dir is not set. "
                "Run shell/precompute_iv2.sh first and pass --iv2_cache_dir."
            )
        if not model_args.vqgan_state_dict or not model_args.iv2_ckpt:
            raise ValueError(
                "use_semantic_loss=True requires --vqgan_state_dict and --iv2_ckpt."
            )
        rank0_print(f"Loading frozen VQ-GAN from {model_args.vqgan_state_dict}")
        vqgan = load_vqgan_decoder(
            model_args.vqgan_state_dict,
            dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        )
        rank0_print(f"Loading frozen InternVideo2-L from {model_args.iv2_ckpt}")
        iv2 = load_internvideo2_l(
            model_args.iv2_ckpt,
            dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        )
        semantic_loss_module = GlobalSemanticLoss(
            compressor_hidden=ve_hidden_size,
            vqgan=vqgan,
            iv2=iv2,
            mlp_hidden=model_args.semantic_mlp_hidden,
            use_commit_loss=(model_args.commit_loss_weight > 0.0),
        )
        rank0_print(
            f"Decoder-end semantic loss enabled (reads decoder per-frame output, not "
            f"the bottleneck). semantic_loss_weight={model_args.semantic_loss_weight} "
            f"decoder_pixel_loss_weight={model_args.decoder_pixel_loss_weight} "
            f"trainable MLP params={sum(p.numel() for p in semantic_loss_module.mlp.parameters()):,}"
        )

    # ---- AE model ------------------------------------------------------
    ae_model = CompressorAutoEncoder(
        vision_encoder=vision_encoder,
        compressor=compressor,
        decoder=decoder,
        semantic_loss=semantic_loss_module,
        semantic_loss_weight=model_args.semantic_loss_weight,
        decoder_pixel_loss_weight=model_args.decoder_pixel_loss_weight,
        commit_loss_weight=model_args.commit_loss_weight,
        recon_loss_weight=model_args.recon_loss_weight,
        use_tube_mask=model_args.use_tube_mask,
        mask_ratio=model_args.mask_ratio,
        recon_masked_only=model_args.recon_masked_only,
        motion_loss_weight=model_args.motion_loss_weight,
        use_order_loss=model_args.use_order_loss,
        order_loss_weight=model_args.order_loss_weight,
        freeze_compressor=model_args.freeze_compressor,
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
        iv2_cache_dir=data_args.iv2_cache_dir if model_args.use_semantic_loss else None,
        decode_pixel_targets=(
            model_args.use_semantic_loss and data_args.precomputed_feature_dir is None
        ),
        vqgan_image_size=(
            semantic_loss_module.vqgan.decoder.resolution
            if semantic_loss_module is not None else 256
        ),
        precomputed_feature_dir=data_args.precomputed_feature_dir,
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

    # Auto-resume: pick up the newest checkpoint-* already in output_dir
    # (HF Trainer never reads args.resume_from_checkpoint on its own).
    from transformers.trainer_utils import get_last_checkpoint

    resume_ckpt = None
    if os.path.isdir(training_args.output_dir):
        resume_ckpt = get_last_checkpoint(training_args.output_dir)
    if resume_ckpt is not None:
        rank0_print(f"Resuming from checkpoint: {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # Save only compressor weights (decoder is a pretraining artifact).
    # Use the global process rank (not local_rank) so that on multi-node runs
    # with a shared filesystem only one process writes the checkpoint instead of
    # every node's local_rank 0 racing to overwrite the same file.
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
