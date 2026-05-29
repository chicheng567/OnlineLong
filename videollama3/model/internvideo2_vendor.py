"""
Minimal frozen InternVideo2 Stage-2 ViT-L (image + video → 768-d CLIP-aligned vec).

Vendored from https://github.com/OpenGVLab/InternVideo (commit ~main, Stage2
single-modality vision tower).  Only the modules needed for inference are
kept: ``PatchEmbed3D``, ``Block`` (×24), ``ClipProjector``.  All other
heads (``clip_decoder`` / ``final_clip_decoder`` used for Stage-1 mask
distillation) are intentionally not built — their weights are silently
discarded at load time.

Architecture (L variant)
------------------------
embed_dim=1024, depth=24, num_heads=16, head_dim=64, mlp_ratio=4.
QKV linear has **no bias**; output proj has bias.
``q_norm`` / ``k_norm`` are per-head LayerNorm(bias=False) on q/k.
``norm1`` / ``norm2`` are LayerNorm(bias=False).
``ls1`` / ``ls2`` are LayerScale parameters (``gamma`` only).
``patch_embed`` is a 3D convolution with temporal kernel=1 (spatial 14×14),
so image (T=1) and video (T=N) share the same backbone weights and are
distinguished only by positional embedding length and pos lookup.

Inputs
------
- Image: ``(B, 3, H, W)`` with ``H=W=224``.
- Video: ``(B, 3, T, H, W)`` with ``T<=8`` for default ``pos_embed`` of
  length 2049 (= 1 cls + 8 × 16 × 16).

Output
------
- ``(B, 768)``, the CLIP-aligned global feature from ``clip_projector``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _layer_norm(dim: int) -> nn.LayerNorm:
    """LayerNorm without bias (matches IV2 state-dict which stores only .weight)."""
    return nn.LayerNorm(dim, eps=1e-6, bias=False)


class LayerScale(nn.Module):
    """``x * gamma``.  ``gamma`` is the only stored parameter (key suffix ``.gamma``)."""

    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ---------------------------------------------------------------------------
# 3-D patch embedding (T=1 conv kernel → image-and-video sharing weights)
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    def __init__(self, *, in_channels: int = 3, embed_dim: int = 1024, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T, H, W) → (B, embed_dim, T, H/P, W/P) → (B, T*H'*W', embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Attention block (EVA-02 style: qkv no bias, QK-norm, output proj has bias)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Per-head LayerNorm on q/k (state-dict only stores .weight)
        self.q_norm = _layer_norm(dim)
        self.k_norm = _layer_norm(dim)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)  # (3, B, N, C)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Native SDPA (FlashAttention path on CUDA + bf16/fp16)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 layerscale_init: float = 1e-5):
        super().__init__()
        self.norm1 = _layer_norm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values=layerscale_init)

        self.norm2 = _layer_norm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values=layerscale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# ClipProjector: cross-attention pool with cls as query, all tokens as KV.
# ---------------------------------------------------------------------------

class _ClipCrossAttn(nn.Module):
    """
    Cross-attention with **separate** q/k/v projections (each with its own
    learnable bias parameter stored apart from the linear weight — matches
    state-dict keys ``q.weight + q_bias``).
    """

    def __init__(self, dim: int, out_dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear without bias (bias is a separate Parameter, applied manually).
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.k_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))

        self.proj = nn.Linear(dim, out_dim, bias=True)

    def forward(self, q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        B, Nq, C = q_in.shape
        Nkv = k_in.shape[1]

        q = self.q(q_in) + self.q_bias
        k = self.k(k_in) + self.k_bias
        v = self.v(v_in) + self.v_bias

        q = q.reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class ClipProjector(nn.Module):
    """
    Pre-norm separately on q-/k-/v-inputs, then a single cross-attention head
    that pools over all tokens with the cls token (token 0) as the query.

    Output dim: 768 (CLIP-aligned).
    """

    def __init__(self, dim: int = 1024, out_dim: int = 768, num_heads: int = 16):
        super().__init__()
        # NB: these LayerNorms DO have bias in the state-dict (norm1_q.bias …).
        self.norm1_q = nn.LayerNorm(dim, eps=1e-6, bias=True)
        self.norm1_k = nn.LayerNorm(dim, eps=1e-6, bias=True)
        self.norm1_v = nn.LayerNorm(dim, eps=1e-6, bias=True)
        self.cross_attn = _ClipCrossAttn(dim, out_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1+N, dim).  Q = cls; KV = all tokens (including cls).
        q_in = self.norm1_q(x[:, :1])
        k_in = self.norm1_k(x)
        v_in = self.norm1_v(x)
        out = self.cross_attn(q_in, k_in, v_in)  # (B, 1, out_dim)
        return out.squeeze(1)


# ---------------------------------------------------------------------------
# Full backbone
# ---------------------------------------------------------------------------

class InternVideo2L(nn.Module):
    """
    Stage-2 backbone for InternVideo2-L.  ``forward`` returns the 768-d
    CLIP-aligned global feature.  Supports both image input (B,3,H,W) and
    video input (B,3,T,H,W).  H = W = 224, T ≤ 8 by default (controlled by
    the size of ``pos_embed``/``clip_pos_embed`` loaded from the checkpoint).
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 14,
        image_size: int = 224,
        num_frames: int = 8,
        clip_out_dim: int = 768,
        layerscale_init: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_frames = num_frames
        self.tokens_per_frame = (image_size // patch_size) ** 2  # 16*16 = 256
        self.max_tokens = self.tokens_per_frame * num_frames     # 2048
        self.clip_out_dim = clip_out_dim

        self.patch_embed = PatchEmbed3D(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens + 1, embed_dim))
        # clip_pos_embed is the one consumed in the CLIP-aligned forward path.
        self.clip_pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, layerscale_init=layerscale_init)
            for _ in range(depth)
        ])

        self.clip_projector = ClipProjector(dim=embed_dim, out_dim=clip_out_dim, num_heads=num_heads)

    # ------------------------------------------------------------------
    def _pos_embed_for(self, num_tokens: int) -> torch.Tensor:
        """
        Return the slice of ``clip_pos_embed`` of length ``1+num_tokens``
        (cls + num_tokens visual tokens).
        For image (T=1) → use cls + first frame's 256 tokens.
        For video (T=N≤num_frames) → use cls + first N*256 tokens.
        """
        return self.clip_pos_embed[:, : 1 + num_tokens, :]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) → treated as T=1, or (B, 3, T, H, W) for video.
        Returns: (B, clip_out_dim) global feature.
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (B, 3, 1, H, W)
        elif x.dim() != 5:
            raise ValueError(f"InternVideo2L: expected 4-D or 5-D input, got {x.dim()}-D")

        B = x.shape[0]
        x = self.patch_embed(x)                  # (B, T*HW, embed_dim)
        N = x.shape[1]

        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        x = torch.cat([cls, x], dim=1)           # (B, 1+N, embed_dim)
        x = x + self._pos_embed_for(N)

        for blk in self.blocks:
            x = blk(x)

        return self.clip_projector(x)            # (B, clip_out_dim)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_internvideo2_l(
    ckpt_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> InternVideo2L:
    """
    Load ``pytorch_model.bin`` from a Stage-2 InternVideo2-L checkpoint.
    ``clip_decoder.*`` and ``final_clip_decoder.*`` (Stage-1 distillation
    heads) are silently discarded.
    """
    model = InternVideo2L()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Drop Stage-1 distillation heads.
    sd = {k: v for k, v in sd.items()
          if not (k.startswith("clip_decoder.") or k.startswith("final_clip_decoder."))}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"InternVideo2L: missing keys after load: {missing}")
    if unexpected:
        raise RuntimeError(f"InternVideo2L: unexpected keys: {unexpected}")

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if device is not None:
        model = model.to(device=device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return model


# Mean/std for input normalization.  InternVideo2's own ViT encoders (the
# vision_encoder name contains "internvideo"/"vit"/"umt") use **ImageNet**
# normalization — NOT the OpenAI-CLIP norm.  The CLIP norm is only used by IV2
# when the encoder is an actual OpenAI-CLIP tower (name contains "clip").
# Verified against the IV2 multi_modality dataset transforms (norm picked by
# encoder name) and the official demo (v_mean/v_std = the values below).
IV2_IMAGE_MEAN = (0.485, 0.456, 0.406)
IV2_IMAGE_STD = (0.229, 0.224, 0.225)


def normalize_for_iv2(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize images from ``[-1, 1]`` (taming-transformers output convention)
    into the ImageNet normalization expected by the IV2 ViT encoder.

    Accepts ``(B, 3, H, W)`` or ``(B, 3, T, H, W)``.
    """
    if images.dim() == 4:
        shape = (1, 3, 1, 1)
    elif images.dim() == 5:
        shape = (1, 3, 1, 1, 1)
    else:
        raise ValueError(f"normalize_for_iv2: expected 4-D or 5-D, got {images.dim()}-D")
    mean = torch.tensor(IV2_IMAGE_MEAN, device=images.device, dtype=images.dtype).view(*shape)
    std = torch.tensor(IV2_IMAGE_STD, device=images.device, dtype=images.dtype).view(*shape)
    # [-1, 1] → [0, 1]
    images = (images.clamp(-1.0, 1.0) + 1.0) * 0.5
    return (images - mean) / std
