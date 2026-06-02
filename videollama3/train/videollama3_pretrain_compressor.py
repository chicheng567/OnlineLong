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
from videollama3.model.internvideo2_vendor import load_internvideo2_l
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
    # --- Global semantic loss (optional) ---
    use_semantic_loss: bool = field(
        default=False,
        metadata={"help": "Add MLP → VQ-GAN → IV2 cycle loss against cached IV2 video features."},
    )
    semantic_loss_weight: float = field(default=0.1)
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
            data = items
        else:
            data = raw

        self.processor = processor
        self.max_frames = max_frames
        self.merge_size = merge_size
        self.data_root = data_root
        self.min_frames = min_frames
        self.iv2_cache_dir = iv2_cache_dir

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
            self.data = kept
        else:
            self.data = data

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
        return out


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
        if "vid_feat" in instances[0]:
            batch["vid_feat"] = torch.stack([x["vid_feat"] for x in instances])
        return batch


# ---------------------------------------------------------------------------
# AE model output
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional

@_dataclass
class AEOutput(BaseModelOutput):
    loss: _Optional[torch.Tensor] = None
    mse_loss: _Optional[torch.Tensor] = None
    sem_loss: _Optional[torch.Tensor] = None
    commit_loss: _Optional[torch.Tensor] = None


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
        semantic_loss: Optional[GlobalSemanticLoss] = None,
        semantic_loss_weight: float = 0.0,
        commit_loss_weight: float = 0.0,
        use_tube_mask: bool = False,
        mask_ratio: float = 0.75,
        recon_masked_only: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.compressor = compressor
        self.decoder = decoder
        self.semantic_loss = semantic_loss
        self.semantic_loss_weight = semantic_loss_weight
        self.commit_loss_weight = commit_loss_weight
        self.use_tube_mask = use_tube_mask
        self.mask_ratio = mask_ratio
        self.recon_masked_only = recon_masked_only

        # Learnable mask embedding for VideoMAE-style tube masking. Created only
        # when masking is enabled so it does not show up in the saved state dict
        # otherwise. Lives at the AE top level; create_optimizer routes it into
        # the compressor LR group.
        if use_tube_mask:
            hidden: int = compressor.hidden_size  # type: ignore[attr-defined]
            self.mask_token = nn.Parameter(torch.zeros(hidden))
            nn.init.normal_(self.mask_token, std=0.02)

        # Vision encoder is always frozen.
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def _apply_tube_mask(
        self,
        visual_tokens: torch.Tensor,
        n_frames_list: List[int],
        HW: int,
    ):
        """
        VideoMAE-style temporal tube masking.

        For each sample an independent set of ``round(mask_ratio * HW)`` spatial
        positions is chosen and masked in *every* frame (a temporal tube), so the
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
    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_sizes: torch.Tensor,
        merge_sizes: torch.Tensor,
        n_frames: torch.Tensor,
        vid_feat: Optional[torch.Tensor] = None,
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

        # 5. MSE loss with uniform frame sampling. Targets are always the original
        #    (unmasked) tokens; with tube masking the loss defaults to the masked
        #    positions only (MAE-style).
        decoded_4d = decoded.view(B, max_T, HW, -1)

        mse = torch.tensor(0.0, device=device, dtype=visual_tokens.dtype)
        offset = 0
        for b, n in enumerate(n_frames_list):
            target = visual_tokens[offset : offset + n * HW].view(n, HW, -1).detach()
            offset += n * HW
            indices = _uniform_frame_indices(n, max_T)
            sampled = decoded_4d[b, indices]  # (n, HW, hidden_size)
            if spatial_masks is not None and self.recon_masked_only:
                m = spatial_masks[b]  # (HW,) bool — masked positions
                mse = mse + F.mse_loss(sampled[:, m, :], target[:, m, :])
            else:
                mse = mse + F.mse_loss(sampled, target)
        mse = mse / B

        # 6. Optional global semantic loss (frozen VQ-GAN → frozen IV2 cycle)
        #    and/or VQ commitment loss on the MLP projection.
        sem: Optional[torch.Tensor] = None
        commit: Optional[torch.Tensor] = None
        need_semantic_path = (
            self.semantic_loss is not None
            and vid_feat is not None
            and (self.semantic_loss_weight > 0.0 or self.commit_loss_weight > 0.0)
        )
        if need_semantic_path:
            sem, commit = self.semantic_loss(compressed, vid_feat.to(device))

        loss = mse
        if sem is not None and self.semantic_loss_weight > 0.0:
            loss = loss + self.semantic_loss_weight * sem.to(mse.dtype)
        if commit is not None and self.commit_loss_weight > 0.0:
            loss = loss + self.commit_loss_weight * commit.to(mse.dtype)
        return AEOutput(
            loss=loss,
            mse_loss=mse.detach(),
            sem_loss=sem.detach() if sem is not None else None,
            commit_loss=commit.detach() if commit is not None else None,
        )


# ---------------------------------------------------------------------------
# Custom Trainer (separate LR groups for compressor vs decoder)
# ---------------------------------------------------------------------------

class PretrainTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Running sums for auxiliary (per-component) losses; flushed by self.log().
        self._aux_running: Dict[str, float] = {}
        self._aux_count: int = 0

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
        # The learnable mask_token lives at the AE top level (no "compressor"/
        # "decoder" substring), so route it explicitly into the compressor group.
        compressor_names = {n for n, _ in rest if "compressor" in n or n == "mask_token"}
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
        for key in ("mse_loss", "sem_loss", "commit_loss"):
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

    if model_args.commit_loss_weight > 0.0 and not model_args.use_semantic_loss:
        raise ValueError(
            "commit_loss_weight > 0 requires use_semantic_loss=True — the "
            "commitment loss is computed on the MLP projection that only exists "
            "on the semantic-loss path."
        )

    if model_args.use_tube_mask and not (0.0 < model_args.mask_ratio < 1.0):
        raise ValueError(
            f"mask_ratio must be in (0, 1) when use_tube_mask=True; "
            f"got {model_args.mask_ratio}."
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
    if model_args.use_tube_mask:
        rank0_print(
            f"Tube masking enabled: mask_ratio={model_args.mask_ratio} "
            f"recon_masked_only={model_args.recon_masked_only}"
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
            f"Semantic loss enabled. weight={model_args.semantic_loss_weight} "
            f"trainable MLP params={sum(p.numel() for p in semantic_loss_module.mlp.parameters()):,}"
        )

    # ---- AE model ------------------------------------------------------
    ae_model = CompressorAutoEncoder(
        vision_encoder=vision_encoder,
        compressor=compressor,
        decoder=decoder,
        semantic_loss=semantic_loss_module,
        semantic_loss_weight=model_args.semantic_loss_weight,
        commit_loss_weight=model_args.commit_loss_weight,
        use_tube_mask=model_args.use_tube_mask,
        mask_ratio=model_args.mask_ratio,
        recon_masked_only=model_args.recon_masked_only,
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
