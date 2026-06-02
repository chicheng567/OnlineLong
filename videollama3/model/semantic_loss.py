"""
Global semantic loss for compressor pretraining.

Path
----
    compressed (B·HW, C_vec)        ← compressor output
        │
        ▼  MLP
    z (B·HW, z_channels)
        │
        ▼  reshape → (B, z_channels, latent_h, latent_w)
    VQ-GAN.quantize (frozen codebook, STE)
        │
        ▼
    VQ-GAN.post_quant_conv + Decoder (frozen)
        │
        ▼  image (B, 3, 256, 256) in [-1, 1]
    bilinear resize → 224
        │
        ▼  CLIP normalization
    InternVideo2-L image branch (frozen)
        │
        ▼  img_feat (B, 768)
    1 − cosine(img_feat, cached_vid_feat)

Notes
-----
* Cached ``vid_feat`` is detached/has-no-grad by construction (loaded from
  disk), so the loss only sends gradient back through the left half of the
  cycle: IV2 → VQ-GAN → MLP → compressor.
* All three frozen models (VQ-GAN, IV2) are stored as submodules so they
  move with ``.to(device)`` / ``.to(dtype)`` along with the trainable MLP.
  Their parameters are kept ``requires_grad=False``; only the MLP trains.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .internvideo2_vendor import InternVideo2L, normalize_for_iv2
from .vqgan_vendor import VQGANDecoder


class GlobalSemanticLoss(nn.Module):
    """
    Trainable MLP head + frozen VQ-GAN decoder + frozen IV2-L; computes the
    cosine semantic loss against a precomputed ``vid_feat``.

    Parameters
    ----------
    compressor_hidden : int
        Hidden size of the compressor output (= vision encoder hidden size).
    vqgan : VQGANDecoder
        Already-loaded (and frozen) VQ-GAN decoder wrapper.
    iv2 : InternVideo2L
        Already-loaded (and frozen) InternVideo2-L backbone.
    mlp_hidden : int, optional
        Hidden width of the two-layer MLP head.  Defaults to
        ``2 * vqgan.embed_dim``.
    iv2_image_size : int
        Input resolution for IV2; image is bilinear-resized to this before
        normalization.
    """

    def __init__(
        self,
        compressor_hidden: int,
        vqgan: VQGANDecoder,
        iv2: InternVideo2L,
        *,
        mlp_hidden: Optional[int] = None,
        iv2_image_size: int = 224,
        use_commit_loss: bool = False,
    ):
        super().__init__()
        self.vqgan = vqgan
        self.iv2 = iv2
        self.latent_size = vqgan.latent_size       # 16 for vqgan-f16-16384
        self.z_channels = vqgan.embed_dim          # 256
        self.tokens_per_image = self.latent_size ** 2  # 256
        self.iv2_image_size = iv2_image_size
        self.use_commit_loss = use_commit_loss

        hidden = mlp_hidden if mlp_hidden is not None else 2 * self.z_channels
        self.mlp = nn.Sequential(
            nn.Linear(compressor_hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.z_channels),
        )

        # Belt-and-suspenders: ensure frozen submodules stay frozen.
        for p in self.vqgan.parameters():
            p.requires_grad = False
        for p in self.iv2.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    def _reshape_to_latent_grid(self, z_flat: torch.Tensor, batch_size: int) -> torch.Tensor:
        """``(B*HW, C) → (B, C, H, W)`` matching VQ-GAN's expected layout."""
        H = W = self.latent_size
        return (
            z_flat.view(batch_size, H, W, self.z_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    # ------------------------------------------------------------------
    def _compute_commit_loss(self, z_flat: torch.Tensor) -> torch.Tensor:
        """
        Per-element MSE between ``z_flat`` and its nearest codebook entry
        (stop-gradient on the codebook side — standard VQ-VAE commitment).

        Argmin is done in fp32 in ``no_grad``; only the final MSE participates
        in autograd, so only the MLP (and upstream compressor) receives
        gradient.  The frozen codebook is not updated.
        """
        codebook = self.vqgan.quantize.embedding.weight  # (n_embed, C)
        z_fp32 = z_flat.float()
        cb_fp32 = codebook.float()
        with torch.no_grad():
            d = (
                z_fp32.pow(2).sum(dim=1, keepdim=True)
                - 2.0 * z_fp32 @ cb_fp32.t()
                + cb_fp32.pow(2).sum(dim=1)
            )
            nearest_idx = d.argmin(dim=1)
            nearest_e = codebook[nearest_idx].detach()
        # Commitment loss in fp32 to match the rest of the loss path.
        return F.mse_loss(z_flat.float(), nearest_e.float())

    # ------------------------------------------------------------------
    def forward(self, compressed: torch.Tensor, vid_feat: torch.Tensor):
        """
        Parameters
        ----------
        compressed : (B*HW, compressor_hidden)
            Per-window compressed tokens.  HW must equal
            ``self.tokens_per_image`` (= 256 for the default config).
        vid_feat : (B, iv2_out_dim)
            Cached InternVideo2-L global features (CLIP-aligned).  Treated as
            the supervision target; gradient is not propagated into it.

        Returns
        -------
        sem_loss : scalar tensor — mean over the batch of ``1 − cosine``.
        commit_loss : scalar tensor or None
            ``||z - sg(e_nearest)||²`` (per-element MSE) when
            ``self.use_commit_loss`` is True, else None.
        """
        B = vid_feat.shape[0]
        if compressed.shape[0] != B * self.tokens_per_image:
            raise ValueError(
                f"GlobalSemanticLoss: got compressed.shape[0]={compressed.shape[0]}, "
                f"expected B*HW = {B}*{self.tokens_per_image} = {B*self.tokens_per_image}."
            )

        # 1. MLP → VQ-GAN latent space
        z_flat = self.mlp(compressed)  # (B*HW, C)

        # Optional commitment loss is computed on the flat z BEFORE reshape,
        # matching the layout of the codebook entries.
        commit_loss = self._compute_commit_loss(z_flat) if self.use_commit_loss else None

        z = self._reshape_to_latent_grid(z_flat, B)

        # 2. VQ-GAN: quantize (STE) → post_quant_conv → decoder
        # decode_from_continuous handles all three steps and yields image
        # tensor in [-1, 1] at (B, 3, latent_size*f, latent_size*f) — for the
        # default vqgan-f16 that is (B, 3, 256, 256).
        img = self.vqgan.decode_from_continuous(z)

        # 3. Resize to IV2 input size + CLIP normalize.
        if img.shape[-1] != self.iv2_image_size:
            img = F.interpolate(
                img.float(),
                size=(self.iv2_image_size, self.iv2_image_size),
                mode="bilinear",
                align_corners=False,
            ).to(img.dtype)
        img = normalize_for_iv2(img)

        # 4. IV2 image branch.
        img_feat = self.iv2(img)  # (B, 768)

        # 5. Cosine similarity loss in fp32 for numerical stability.
        a = F.normalize(img_feat.float(), dim=-1)
        b = F.normalize(vid_feat.float().detach(), dim=-1)
        cos = (a * b).sum(dim=-1)
        sem_loss = (1.0 - cos).mean()
        return sem_loss, commit_loss
