"""
SigLIP-AE compressor — a port of Video-XL-Pro's `SiglipAE` encoder (arXiv:2503.18478),
wired into this project as a selectable ``compressor_type="siglip_ae"`` (see
``compressor.py::build_token_compressor``). It is trained with the LLM cross-entropy
recipe like every other compressor here — ``videollama3/train/compressor_pretrain_with_videollama3.py``.

Ported from referenceCode/Video-XL/Video-XL-Pro/videoxlpro/videoxlpro/model/sae.py
(``SiglipAE`` + ``sae_utils.py``) and, for ``TemporalAttention``'s relative-position
machinery, ``attention_temporal_videoae.py`` (``RelativePosition`` / ``QKVAttention``).
Trimmed to only the code paths ``SiglipAE`` actually exercises (e.g. ``QKVAttention``'s
mask / causal branches, never used by ``TemporalAttention``, are dropped) and rewritten
with plain ``.permute()`` / ``.reshape()`` instead of ``einops.rearrange`` — the tensor
algebra is unchanged.

``SiglipAE`` realises the paper's "merge tokens from four frames into one" with a
learnable ``nn.Parameter(torch.randn((4, 1152)))`` (``temporal_encoding``) added
unconditionally at the top of the encoder forward (``x = x + temporal_encoding``),
followed by ``log2(T)`` stride-2 Conv3d stages (T: 4 -> 2 -> 1). Here the additive
per-frame bias is factored into ``DynamicTokenSynthesizer`` and each stage is
``AttnBlock3D`` (per-frame spatial self-attn) -> ``TemporalAttention`` (per-position
temporal self-attn) -> ``SamePadConv3d`` (stride-2 temporal downsample).

Unlike ``TransformerDecoderCompressor`` / ``LocalAttnConvCompressor``, this
architecture's depth is fixed at construction (``log2(window_size)`` stages), so every
compression window handed to ``forward`` must contain exactly ``config.window_size``
frames. The per-window spatial grid ``(h, w)`` is still dynamic.

Deliberate deviations from the literal reference code (none affect the paper's stated
algorithm — only numerically-motivated choices the reference leaves unspecified):
  - ``temporal_encoding`` init is ``torch.randn(...) * 0.02`` (this project's
    learned-query init scale), not the paper's raw ``torch.randn`` (std=1) — an
    unscaled std=1 additive bias into SigLIP-scale features is disruptively large at
    init.
  - ``SamePadConv3d`` runs its whole forward (padding + conv) in the conv's own
    parameter dtype rather than the ambient autocast dtype: 3-D convs are unreliable in
    fp16/bf16 on CUDA.
  - ``GroupNorm`` groups is asserted at the paper's fixed 32 (fail-loud for a
    ``hidden_size`` not divisible by 32 rather than a silent fallback).
  - ``TemporalAttention``'s ``num_heads`` is hardcoded to the paper's default (1), not
    wired to ``config.num_attention_heads`` (a knob for the other compressor types).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTokenSynthesizer(nn.Module):
    """Learnable additive per-frame temporal encoding.

    Reference: ``SiglipAE.temporal_encoding`` in sae.py,
    ``nn.Parameter(torch.randn((4, 1152)))``, added once at the top of the encoder's
    forward. ``SiglipAECompressor`` holds one of these and applies it before its
    conv/attention stages.
    """

    def __init__(self, hidden_size: int, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_encoding = nn.Parameter(torch.randn(num_frames, hidden_size) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, HW, C), T must equal self.num_frames (SiglipAECompressor fixes
        # every window at exactly num_frames).
        _, T, _, C = tokens.shape
        assert T == self.num_frames, (
            f"DynamicTokenSynthesizer: expected T={self.num_frames}, got {T}."
        )
        return tokens + self.temporal_encoding.to(tokens.dtype).view(1, T, 1, C)


# ---------------------------------------------------------------------------
# SiglipAE encoder building blocks.
# ---------------------------------------------------------------------------

def _zero_module(module: nn.Module) -> nn.Module:
    """Zero out a module's parameters in place. Ported verbatim (utils_encoder.py):
    SiglipAE zero-inits TemporalAttention's qkv/proj_out convs so that sub-layer starts
    as an identity function (output ~= 0, so `x + out` ~= x at init)."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def _group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """Ported (sae_utils.py::Normalize, norm_type='group' — SiglipAE's only usage)."""
    assert channels % num_groups == 0, (
        f"SiglipAE-style GroupNorm requires channels ({channels}) divisible by "
        f"num_groups ({num_groups}); this project's default hidden_size=1152 "
        f"satisfies this (1152 / 32 = 36), but a custom hidden_size might not."
    )
    return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)


class SamePadConv3d(nn.Module):
    """Same-padding Conv3d with replicate padding (ported verbatim, sae_utils.py).

    Deviation from the reference: the ENTIRE forward (padding + conv), not just the
    padding, runs in the conv's own parameter dtype regardless of the ambient autocast
    dtype — 3-D (transpose-)convs are unreliable in fp16/bf16 on CUDA.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding_type="replicate"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # Assumes the input shape is divisible by stride (true here: T is always a
        # power of two and H/W never shrink since their stride is always 1).
        total_pad = tuple(k - s for k, s in zip(kernel_size, stride))
        pad_input = []
        for p in total_pad[::-1]:  # reverse: F.pad starts from the last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        self.pad_input = sum(pad_input, tuple())
        self.padding_type = padding_type
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        # Run in the conv's OWN parameter dtype, not unconditionally in fp32: the
        # compressor may be cast wholesale to bf16, and feeding an fp32 input to bf16
        # weights raises "Input type (float) and bias type (c10::BFloat16) should be
        # the same".
        compute_dtype = self.conv.weight.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x_padded = F.pad(x.to(compute_dtype), self.pad_input, mode=self.padding_type)
            out = self.conv(x_padded)
        return out.to(out_dtype)


class RelativePosition(nn.Module):
    """Learned relative-position embedding table (ported verbatim,
    attention_temporal_videoae.py; original credit in that file's docstring:
    https://github.com/evelinehong/Transformer_Relative_Position_PyTorch)."""

    def __init__(self, num_units: int, max_relative_position: int):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.zeros(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q: int, length_k: int) -> torch.Tensor:
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = (distance_mat_clipped + self.max_relative_position).long()
        return self.embeddings_table[final_mat]  # (length_q, length_k, num_units)


class _TemporalQKVAttention(nn.Module):
    """QKV attention with a relative-position bias added to both attention weights
    and values (ported from attention_temporal_videoae.py::QKVAttention, trimmed to
    the mask=None / non-causal path — the only one TemporalAttention ever uses)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor, rp) -> torch.Tensor:
        # qkv: (bs, 3 * n_heads * ch, length)
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))

        q_s = (q * scale).view(bs * self.n_heads, ch, length)
        k_s = (k * scale).view(bs * self.n_heads, ch, length)
        weight = torch.einsum("bct,bcs->bts", q_s, k_s)

        k_rp, v_rp = rp  # each (length, length, ch)
        weight = weight + torch.einsum("bct,tsc->bst", q_s, k_rp)
        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)

        v_flat = v.reshape(bs * self.n_heads, ch, length)
        out = torch.einsum("bts,bcs->bct", weight, v_flat)
        out = out + torch.einsum("bts,tsc->btc", weight, v_rp).transpose(1, 2)
        return out.reshape(bs, -1, length)


class AttnBlock3D(nn.Module):
    """Per-frame spatial self-attention over all H*W positions within each frame
    (despite the "3D" name, attention itself is 2-D/per-frame — only the surrounding
    tensor is 5-D; ported verbatim, sae_utils.py). No positional bias: plain full
    attention over the H*W positions treated as an unordered set."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = _group_norm(channels)
        self.q = nn.Conv3d(channels, channels, kernel_size=1)
        self.k = nn.Conv3d(channels, channels, kernel_size=1)
        self.v = nn.Conv3d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, t, h, w = q.shape

        q = q.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)   # (b t) (h w) c
        k = k.permute(0, 2, 1, 3, 4).reshape(b * t, c, h * w)   # (b t) c (h w)
        v = v.permute(0, 2, 1, 3, 4).reshape(b * t, c, h * w)

        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = torch.softmax(attn, dim=2)                        # (b t) (h w) (h w)
        out = torch.bmm(v, attn.permute(0, 2, 1))                 # (b t) c (h w)
        out = out.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()  # b c t h w

        out = self.proj_out(out)
        return x + out


class TemporalAttention(nn.Module):
    """Per-spatial-position multi-head temporal self-attention with a learned
    relative-position bias (ported verbatim, sae_utils.py). `num_heads` is hardcoded
    to the paper's literal default (1) — see the module docstring's "deliberate
    deviations" note for why this isn't wired to `config.num_attention_heads`."""

    def __init__(self, channels: int, num_heads: int = 1, max_temporal_length: int = 64):
        super().__init__()
        assert channels % num_heads == 0, (
            f"TemporalAttention: channels ({channels}) must be divisible by "
            f"num_heads ({num_heads})."
        )
        self.num_heads = num_heads
        head_dim = channels // num_heads

        self.norm = _group_norm(channels)
        self.qkv = _zero_module(nn.Conv1d(channels, channels * 3, kernel_size=1))
        self.attention = _TemporalQKVAttention(num_heads)
        self.relative_position_k = RelativePosition(head_dim, max_temporal_length)
        self.relative_position_v = RelativePosition(head_dim, max_temporal_length)
        self.proj_out = _zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        out = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, c, t)  # (b h w) c t

        qkv = self.qkv(self.norm(out))
        k_rp = self.relative_position_k(t, t)
        v_rp = self.relative_position_v(t, t)
        out = self.attention(qkv, rp=(k_rp, v_rp))
        out = self.proj_out(out)

        out = out.view(b, h, w, c, t).permute(0, 3, 4, 1, 2).contiguous()  # b c t h w
        return x + out


class SiglipAEStage(nn.Module):
    """One SiglipAE encoder stage: spatial self-attn -> temporal self-attn -> stride-2
    temporal downsample. `SiglipAE.encoder` in the reference is exactly two of these
    stacked (T=4 -> 2 -> 1); `SiglipAECompressor` stacks `log2(window_size)` of them."""

    def __init__(self, channels: int):
        super().__init__()
        self.attn_spatial = AttnBlock3D(channels)
        self.attn_temporal = TemporalAttention(channels)
        self.downsample = SamePadConv3d(channels, channels, kernel_size=3, stride=(2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_spatial(x)
        x = self.attn_temporal(x)
        x = self.downsample(x)
        return x


class SiglipAECompressor(nn.Module):
    """
    Faithful port of Video-XL-Pro's `SiglipAE` (arXiv:2503.18478), adapted to this
    project's flatten + `cu_seqlens` compressor calling convention
    (`forward(kv, compression_cu_seqlens) -> compressed`, matching
    `TransformerDecoderCompressor`/`LocalAttnConvCompressor` in compressor.py) so it is
    a real, selectable `compressor_type="siglip_ae"`.

    Pipeline per window (matches `SiglipAE.forward` exactly):
        x = x + temporal_encoding                          # DynamicTokenSynthesizer
        x = [AttnBlock3D -> TemporalAttention -> SamePadConv3d(stride=(2,1,1))] * L
    where `L = log2(window_size)`, so T: window_size -> ... -> 1, spatial H/W unchanged
    (pure temporal compression, matching `LocalAttnConvCompressor`'s framing, not
    `TransformerDecoderCompressor`'s learned-query cross-attention framing).

    Constraint (unlike TransformerDecoderCompressor): `config.window_size` must be a
    power of two >= 2, and EVERY window passed to `forward` must contain exactly
    `window_size` frames — this architecture's depth is fixed at construction, it
    cannot adapt to a different T per call. Set `window_size` to match whatever T you
    actually compress per window (e.g. `--fixed_frames`, or `compression_window_size`
    in real inference — see `Videollama3TokenCompressorConfig`).

    Unlike T, the per-window spatial grid (h, w) IS dynamic: `forward` accepts an
    optional `grid_hws` (one (h, w) per window) and reshapes/runs each window with its
    own actual grid rather than the constructor's `config.compress_image_h/w` (which is
    now only the fallback used when `grid_hws` is omitted). Since this architecture is a
    pure spatial pass-through, the output grid always equals that window's input grid —
    see `output_hw_for`. Because windows may now differ in (h, w), `forward` processes
    each window through the conv/attention stages individually (no batched B>1 conv
    call) rather than reshaping all windows into one `(B, T, H, W, C)` tensor.
    """

    def __init__(self, config):
        super().__init__()
        C = config.hidden_size
        T = getattr(config, "window_size", None)
        assert T is not None and T >= 2 and (T & (T - 1)) == 0, (
            f"SiglipAECompressor requires config.window_size to be a power of two "
            f">= 2 (each stage halves T via a stride-2 Conv3d, matching Video-XL-Pro's "
            f"SiglipAE); got window_size={T}. Set it to the exact number of frames "
            f"every compression window will contain."
        )
        self.hidden_size = C
        self.window_size = T
        self.compress_image_w = config.compress_image_w
        self.compress_image_h = config.compress_image_h
        self.compress_image_wh = self.compress_image_w * self.compress_image_h
        self.num_stages = T.bit_length() - 1  # log2(T)

        self.temporal_encoding = DynamicTokenSynthesizer(hidden_size=C, num_frames=T)
        self.stages = nn.ModuleList([SiglipAEStage(C) for _ in range(self.num_stages)])

    def output_hw_for(self, h: int, w: int):
        # Pure spatial pass-through (only T is reduced) — the output grid always
        # equals the window's own input grid. See TransformerDecoderCompressor's
        # output_hw_for for the contrasting fixed-output-grid case.
        return h, w

    def forward(self, kv: torch.Tensor, compression_cu_seqlens: torch.Tensor, grid_hws=None) -> torch.Tensor:
        # kv: (total_tokens, hidden_size) or (1, total_tokens, hidden_size), frame-major
        # per window: [frame0_p0, .., frame0_p{HW_i-1}, frame1_p0, ...] (same convention
        # as LocalAttnConvCompressor).
        # grid_hws: optional list of (h, w) pairs, one per window — the ACTUAL input
        # frame grid for that window (may differ window to window). Defaults to
        # compress_image_h/w for every window, matching the old fixed-grid behavior.
        if kv.dim() == 3:
            kv = kv.squeeze(0)
        device = kv.device
        compression_cu_seqlens = compression_cu_seqlens.to(device=device, dtype=torch.int32)

        window_lens = (compression_cu_seqlens[1:] - compression_cu_seqlens[:-1]).long()
        B = window_lens.shape[0]
        if grid_hws is None:
            grid_hws = [(self.compress_image_h, self.compress_image_w)] * B
        assert len(grid_hws) == B, (
            f"SiglipAECompressor: grid_hws must have one (h, w) per window ({B} "
            f"windows), got {len(grid_hws)}."
        )

        T, C = self.window_size, self.hidden_size

        # Windows may have different (h, w), so each is reshaped/run through the
        # conv/attention stages individually rather than batched as one (B, T, H, W, C)
        # tensor — see the class docstring.
        outputs = []
        for i in range(B):
            H_i, W_i = grid_hws[i]
            HW_i = H_i * W_i
            expected = T * HW_i
            got = window_lens[i].item()
            assert got == expected, (
                f"SiglipAECompressor: window {i} must contain exactly "
                f"window_size={T} frames at its grid_hws (h={H_i}, w={W_i}) "
                f"-> {expected} tokens; got {got}. Unlike LocalAttnConvCompressor/"
                f"TransformerDecoderCompressor, this architecture's depth is fixed at "
                f"construction time (matching Video-XL-Pro's SiglipAE), so every window "
                f"must share the same frame count — see the class docstring."
            )

            s = compression_cu_seqlens[i].item()
            e = compression_cu_seqlens[i + 1].item()
            # frame-major (T*HW_i, C) -> (1, T, H_i, W_i, C).
            x_bthwc = kv[s:e].view(1, T, H_i, W_i, C)

            # DTS: unconditional additive per-frame temporal bias, matching
            # SiglipAE.forward's `x = x + temporal_encoding`. DynamicTokenSynthesizer
            # expects (B, T, HW, C), so flatten H,W for this call only.
            x_bthwc = self.temporal_encoding(x_bthwc.view(1, T, HW_i, C)).view(1, T, H_i, W_i, C)

            x = x_bthwc.permute(0, 4, 1, 2, 3).contiguous()  # (1, C, T, H_i, W_i)
            for stage in self.stages:
                x = stage(x)

            assert x.shape[2] == 1, (
                f"SiglipAECompressor: expected T=1 after {self.num_stages} stride-2 "
                f"stages (log2(window_size={self.window_size})), got T={x.shape[2]}."
            )
            x = x.squeeze(2)                        # (1, C, H_i, W_i)
            x = x.permute(0, 2, 3, 1).contiguous()   # (1, H_i, W_i, C)
            outputs.append(x.reshape(HW_i, C))

        return torch.cat(outputs, dim=0)
