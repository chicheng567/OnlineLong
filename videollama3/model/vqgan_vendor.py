"""
Minimal frozen VQ-GAN decoder + quantizer (taming-transformers compatible).

Vendored from https://github.com/CompVis/taming-transformers
(commit 24268930bf1dce879235a7fddd0b2355b84d7ea6) — only the inference-time
modules required to take a *continuous* latent `z` and produce an RGB image:

    z (B, embed_dim, H, W)
      → VectorQuantizer2 (straight-through)
      → post_quant_conv (1×1)
      → Decoder
      → image (B, 3, H*f, W*f)

Encoder / LPIPS / discriminator are intentionally excluded.

Compatible with the public ``vqgan-f16-16384`` checkpoint
(``ch=128, ch_mult=[1,1,2,2,4], attn_resolutions=[16], z_channels=embed_dim=256,
n_embed=16384``). Use ``load_vqgan_decoder`` to construct + load weights from a
pre-extracted plain state-dict (see scripts/extract_vqgan_state_dict.py).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (verbatim arithmetic from taming-transformers; renamed only
# where it improves readability).
# ---------------------------------------------------------------------------

def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)  # Swish / SiLU


def _normalize(channels: int) -> nn.Module:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = _normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(_nonlinearity(self.norm2(h))))
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = _normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)

        b, c, hh, ww = q.shape
        q = q.reshape(b, c, hh * ww).permute(0, 2, 1)  # (b, hw, c)
        k = k.reshape(b, c, hh * ww)                   # (b, c, hw)
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = attn.softmax(dim=-1)
        v = v.reshape(b, c, hh * ww).permute(0, 2, 1)  # (b, hw, c)
        out = torch.bmm(attn, v).permute(0, 2, 1).reshape(b, c, hh, ww)
        return x + self.proj_out(out)


# ---------------------------------------------------------------------------
# Decoder sub-modules (named subclasses so attribute access is type-checkable;
# state-dict key layout is preserved).
# ---------------------------------------------------------------------------

class _Mid(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels, dropout=dropout)
        self.attn_1 = AttnBlock(channels)
        self.block_2 = ResnetBlock(channels, channels, dropout=dropout)


class _UpStage(nn.Module):
    upsample: Optional[Upsample]

    def __init__(self):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.upsample = None


class Decoder(nn.Module):
    """
    Mirror of taming-transformers ``Decoder`` for ``vqgan-f16-16384``:
      ch=128, ch_mult=(1,1,2,2,4), num_res_blocks=2, attn_resolutions=(16,),
      z_channels=256, out_ch=3, resolution=256.
    Forward: ``z (B, z_channels, H, W)`` → ``image (B, out_ch, H * 2**n_down, W * 2**n_down)``.
    """

    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        attn_resolutions: List[int],
        dropout: float,
        in_channels: int,   # unused at inference (kept for arch parity)
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # compute lowest-resolution channel count
        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z → block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = _Mid(block_in, dropout)

        # upsampling: taming indexes self.up[0..n-1] from coarsest to finest
        # but the forward loop iterates in reverse.  We keep the original layout
        # so the state-dict loads cleanly.
        self.up: nn.ModuleList = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            stage = _UpStage()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                stage.block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    stage.attn.append(AttnBlock(block_in))
            if i_level != 0:
                stage.upsample = Upsample(block_in)
                curr_res *= 2
            # ``self.up.insert(0, ...)`` so up[0] is the finest stage in the
            # ModuleList layout but iteration during forward goes high→low.
            self.up.insert(0, stage)

        # end
        self.norm_out = _normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # iterate from low-res (high channels) to high-res (low channels)
        for i_level in reversed(range(self.num_resolutions)):
            stage: _UpStage = self.up[i_level]  # type: ignore[assignment]
            for i_block in range(self.num_res_blocks + 1):
                h = stage.block[i_block](h)
                if len(stage.attn) > 0:
                    h = stage.attn[i_block](h)
            if stage.upsample is not None:
                h = stage.upsample(h)

        h = self.conv_out(_nonlinearity(self.norm_out(h)))
        return h


# ---------------------------------------------------------------------------
# VectorQuantizer2 — straight-through, no EMA, no legacy reshape.
# ---------------------------------------------------------------------------

class VectorQuantizer2(nn.Module):
    """
    Straight-through vector quantizer. Codebook is registered as a plain
    ``nn.Embedding`` so the standard taming key ``quantize.embedding.weight``
    matches without remapping.
    """

    def __init__(self, n_embed: int, embed_dim: int):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(n_embed, embed_dim)
        # Init range matches taming (uniform(-1/n_embed, 1/n_embed)); overridden
        # by load_state_dict so the value here only matters in unit tests.
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, C, H, W) — C must equal embed_dim
        b, c, h, w = z.shape
        assert c == self.embed_dim, f"VectorQuantizer2: got channel={c}, expected {self.embed_dim}"
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embed_dim)  # (BHW, C)

        # squared L2 distance to every codebook entry
        # d_ij = ||z_i||^2 + ||e_j||^2 - 2 z_i . e_j
        d = (
            (z_flat ** 2).sum(dim=1, keepdim=True)
            - 2 * z_flat @ self.embedding.weight.t()
            + (self.embedding.weight ** 2).sum(dim=1)
        )
        indices = d.argmin(dim=1)
        z_q_flat = self.embedding(indices)  # (BHW, C)
        z_q = z_q_flat.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()

        # straight-through: forward = z_q, backward gradient = grad of z
        return z + (z_q - z).detach()


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class VQGANDecoder(nn.Module):
    """
    Frozen-only wrapper exposing the path
        continuous z → quantize → post_quant_conv → decoder → image.

    All parameters are set ``requires_grad=False`` after weight load.
    """

    def __init__(
        self,
        *,
        n_embed: int = 16384,
        embed_dim: int = 256,
        z_channels: int = 256,
        ch: int = 128,
        ch_mult: Sequence[int] = (1, 1, 2, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Sequence[int] = (16,),
        resolution: int = 256,
        out_ch: int = 3,
        in_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.z_channels = z_channels
        self.latent_size = resolution // 2 ** (len(ch_mult) - 1)  # 16 for default

        self.quantize = VectorQuantizer2(n_embed=n_embed, embed_dim=embed_dim)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, kernel_size=1)
        self.decoder = Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=list(ch_mult),
            num_res_blocks=num_res_blocks,
            attn_resolutions=list(attn_resolutions),
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
        )

    def decode_from_continuous(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (B, embed_dim, H, W).  Returns RGB image in [-1, 1] range
        (taming-transformers convention).
        """
        z_q = self.quantize(z)
        z_q = self.post_quant_conv(z_q)
        return self.decoder(z_q)


def load_vqgan_decoder(
    state_dict_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    strict: bool = False,
) -> VQGANDecoder:
    """
    Build a VQGANDecoder with vqgan-f16-16384 defaults and load the plain
    state-dict produced by scripts/extract_vqgan_state_dict.py.

    ``strict=False`` because the on-disk state-dict still contains encoder
    and quant_conv tensors which this wrapper does not own.
    """
    model = VQGANDecoder()
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected_keep = [k for k in unexpected
                       if not (k.startswith("encoder.") or k.startswith("quant_conv."))]
    if strict and (missing or unexpected_keep):
        raise RuntimeError(f"Unexpected missing={missing} unexpected={unexpected_keep}")
    if missing:
        raise RuntimeError(f"VQGANDecoder missing weights: {missing}")
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if device is not None:
        model = model.to(device=device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)
    return model
