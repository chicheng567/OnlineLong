"""
Stage-2 segment aggregator — a Mamba-2 accumulator with bounded state.

Context
-------
Stage 1 (the pretrained token compressor) turns each scene segment of a long video
into a *fixed* number of tokens ``K`` (e.g. 64). This module is stage 2: it folds the
temporally-ordered stream of per-segment summaries

    seg_0 (K tokens), seg_1 (K tokens), ...  seg_{N-1} (K tokens)

into a *fixed* ``M``-token video-level summary (e.g. 32), independent of ``N``.

Why Mamba-2 and not TTM
-----------------------
The requirement is "an infinitely mergeable module with bounded state". A gated
linear-attention / SSM layer *is* exactly an associative fold: a fixed-size recurrent
state ``(nheads, headdim, d_state)`` that you update once per token and never grows.
Unlike Token Turing Machines it has a chunk-parallel training form (the SSD scan
below), so it trains like a Transformer instead of a deep per-step BPTT loop.

Two entry points, same recurrence
---------------------------------
* ``forward(segment_tokens)`` — training / offline. Flattens ``[segments ; M learned
  summary queries]`` into one causal sequence, runs the chunk-parallel SSD scan, and
  returns the last ``M`` positions. One batched call, fixed ``N`` per batch.
* ``init_state`` / ``update`` / ``readout`` — streaming / unbounded. ``update`` folds
  one segment into the running state via the step recurrence; ``readout`` runs the
  ``M`` summary queries from a *copy* of the state (non-destructive), so you can keep
  calling ``update`` forever. ``reduce(...)`` is the convenience wrapper.

The two paths implement the identical linear recurrence + identical depthwise causal
conv, so their outputs match to ~1e-3 (checked in ``__main__``).

Kernels
-------
Pure PyTorch + einops, CPU-testable, no ``mamba_ssm`` / Triton / flash-attn needed.
The SSD scan (``_ssd_chunk_scan``) is the reference "minimal SSD" from the Mamba-2
paper. For production training you can drop in the fused ``mamba_ssm.Mamba2`` kernel
(same math, ~same weight layout) — not required and not imported here.

Not wired into any training script — this file only builds the module.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

__all__ = ["SegmentAggregatorConfig", "SegmentAggregator", "Mamba2Mixer", "AggregatorState"]


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


class RMSNormGated(nn.Module):
    """Mamba-2's output norm: RMSNorm(x * silu(z))."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        if z is not None:
            x = x * F.silu(z.float())
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x.to(dtype))


# ---------------------------------------------------------------------------
# Minimal SSD (state-space dual) chunk scan — Mamba-2 reference algorithm
# ---------------------------------------------------------------------------

def _segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment-sum: out[..., i, j] = sum_{j < k <= i} x_k  (i >= j), else -inf."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, float("-inf"))
    return x_segsum


def _ssd_chunk_scan(
    X: torch.Tensor,          # (b, l, h, p)   already dt-scaled
    A: torch.Tensor,          # (b, l, h)      already dt-scaled log-decay per step
    B: torch.Tensor,          # (b, l, h, n)   groups pre-expanded to heads
    C: torch.Tensor,          # (b, l, h, n)
    chunk_size: int,
    initial_states: Optional[torch.Tensor] = None,   # (b, 1, h, p, n)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (Y : (b, l, h, p), final_state : (b, h, p, n)).  All math in fp32."""
    b, seqlen, h, p = X.shape
    pad = (chunk_size - seqlen % chunk_size) % chunk_size
    if pad:
        X = F.pad(X, (0, 0, 0, 0, 0, pad))
        A = F.pad(A, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    X, A, B, C = (rearrange(t, "b (c l) ... -> b c l ...", l=chunk_size) for t in (X, A, B, C))
    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. intra-chunk (diagonal) contribution
    Lmat = torch.exp(_segsum(A))
    Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, Lmat, X)

    # 2. each chunk's end state
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
    states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)

    # 3. inter-chunk recurrence
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, :1])
    states = torch.cat([initial_states, states], dim=1)
    decay_chunk = torch.exp(_segsum(F.pad(A_cumsum[..., -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. state -> output (off-diagonal) contribution
    state_decay_out = torch.exp(A_cumsum)
    Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)

    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y[:, :seqlen], final_state


# ---------------------------------------------------------------------------
# Mamba-2 mixer
# ---------------------------------------------------------------------------

class Mamba2Mixer(nn.Module):
    """
    Faithful Mamba-2 token mixer (scalar-decay SSD). Supports:
      * ``forward(u, ssm_state=None, return_state=False)`` — chunk-parallel scan.
      * ``step(u_t, conv_state, ssm_state)``               — single-token recurrence.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        headdim: int = 64,
        ngroups: int = 1,
        d_conv: int = 4,
        expand: int = 2,
        chunk_size: int = 128,
        conv_bias: bool = True,
        proj_bias: bool = False,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        A_init_range: Tuple[float, float] = (1.0, 16.0),
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.ngroups = ngroups
        self.d_conv = d_conv
        self.chunk_size = chunk_size

        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // headdim
        assert self.nheads % ngroups == 0, "nheads must be divisible by ngroups"
        self.conv_dim = self.d_inner + 2 * ngroups * d_state

        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=proj_bias)

        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=d_conv,
            groups=self.conv_dim, padding=d_conv - 1, bias=conv_bias,
        )
        self.act = nn.SiLU()

        # dt bias = inverse-softplus of a log-uniform sample in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        a_lo, a_hi = A_init_range
        A = torch.empty(self.nheads).uniform_(a_lo, a_hi)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.D._no_weight_decay = True

        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=proj_bias)

    # -- chunk-parallel (training / offline) -------------------------------
    def forward(
        self,
        u: torch.Tensor,                              # (b, l, d_model)
        ssm_state: Optional[torch.Tensor] = None,     # (b, nheads, headdim, d_state)
        return_state: bool = False,
    ):
        b, seqlen, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :seqlen].transpose(1, 2)
        xBC = self.act(xBC)
        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )

        A = -torch.exp(self.A_log.float())                       # (nheads,)
        dt = F.softplus(dt.float() + self.dt_bias.float())       # (b, l, nheads)
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim).float()
        B = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups).float()
        C = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups).float()
        B = repeat(B, "b l g n -> b l (g r) n", r=self.nheads // self.ngroups)
        C = repeat(C, "b l g n -> b l (g r) n", r=self.nheads // self.ngroups)

        X = x * rearrange(dt, "b l h -> b l h 1")
        A_dt = A * dt                                            # (b, l, nheads)
        init = None if ssm_state is None else ssm_state.float().unsqueeze(1)
        Y, final_state = _ssd_chunk_scan(X, A_dt, B, C, self.chunk_size, initial_states=init)
        Y = Y + x * rearrange(self.D.float(), "h -> h 1")
        Y = rearrange(Y, "b l h p -> b l (h p)").to(u.dtype)
        Y = self.norm(Y, z)
        out = self.out_proj(Y)
        if return_state:
            return out, final_state.to(u.dtype)
        return out

    # -- single-step recurrence (streaming) ------------------------------
    def step(
        self,
        u_t: torch.Tensor,            # (b, d_model)
        conv_state: torch.Tensor,     # (b, conv_dim, d_conv)
        ssm_state: torch.Tensor,      # (b, nheads, headdim, d_state)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zxbcdt = self.in_proj(u_t)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )

        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 k -> d k"), dim=-1)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC)

        x, B, C = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1
        )
        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt.float() + self.dt_bias.float())        # (b, nheads)
        dA = torch.exp(dt * A)                                    # (b, nheads)

        x = rearrange(x, "b (h p) -> b h p", p=self.headdim).float()
        B = rearrange(B, "b (g n) -> b g n", g=self.ngroups).float()
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups).float()
        B = repeat(B, "b g n -> b (g r) n", r=self.nheads // self.ngroups)
        C = repeat(C, "b g n -> b (g r) n", r=self.nheads // self.ngroups)

        dBx = torch.einsum("bh,bhn,bhp->bhpn", dt, B, x)
        ssm_state = ssm_state.float() * rearrange(dA, "b h -> b h 1 1") + dBx
        y = torch.einsum("bhpn,bhn->bhp", ssm_state, C)
        y = y + rearrange(self.D.float(), "h -> h 1") * x
        y = rearrange(y, "b h p -> b (h p)").to(u_t.dtype)
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out, conv_state, ssm_state.to(u_t.dtype)


# ---------------------------------------------------------------------------
# Block (pre-norm residual, optional MLP) + streaming state container
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Mamba2Block(nn.Module):
    def __init__(self, cfg: "SegmentAggregatorConfig"):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = Mamba2Mixer(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            chunk_size=cfg.chunk_size,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.mlp = None
        if cfg.mlp_ratio and cfg.mlp_ratio > 0:
            self.norm2 = RMSNorm(cfg.d_model)
            self.mlp = _MLP(cfg.d_model, cfg.mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.norm(x)))
        if self.mlp is not None:
            x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

    def step(self, x_t: torch.Tensor, state: List[torch.Tensor]):
        conv_state, ssm_state = state
        y, conv_state, ssm_state = self.mixer.step(self.norm(x_t), conv_state, ssm_state)
        x_t = x_t + y
        if self.mlp is not None:
            x_t = x_t + self.mlp(self.norm2(x_t))
        return x_t, [conv_state, ssm_state]

    def init_state(self, batch: int, device, dtype) -> List[torch.Tensor]:
        m = self.mixer
        return [
            torch.zeros(batch, m.conv_dim, m.d_conv, device=device, dtype=dtype),
            torch.zeros(batch, m.nheads, m.headdim, m.d_state, device=device, dtype=dtype),
        ]


@dataclass
class AggregatorState:
    layer_states: List[List[torch.Tensor]]
    n_seen: int = 0


# ---------------------------------------------------------------------------
# Config + top-level module
# ---------------------------------------------------------------------------

@dataclass
class SegmentAggregatorConfig:
    d_input: int = 3584            # dim of a stage-1 compressor token (LLM hidden for Qwen2-7B)
    d_model: int = 1024           # aggregator working width
    d_output: Optional[int] = None  # None -> d_input (project back so it drops into the LLM stream)
    tokens_per_segment: int = 64  # K
    n_summary_tokens: int = 32    # M
    n_layers: int = 4

    # Mamba-2 mixer
    d_state: int = 128
    headdim: int = 64
    ngroups: int = 1
    d_conv: int = 4
    expand: int = 2
    chunk_size: int = 128

    mlp_ratio: float = 0.0        # 0 = pure Mamba stack; e.g. 4.0 for interleaved MLP
    dropout: float = 0.0
    input_norm: bool = True

    # segment-level temporal encoding, broadcast over the K tokens of a segment
    time_embed: str = "index_sincos"   # "index_sincos" | "seconds_mlp" | "none"


class SegmentAggregator(nn.Module):
    """
    (B, N, K, d_input) stream of per-segment compressor tokens  ->  (B, M, d_output)
    video-level summary, via an N-independent Mamba-2 fold.

    Offline:   ``forward(segment_tokens, segment_seconds=None)``
    Streaming: ``s = init_state(...); s = update(s, seg); ...; summary = readout(s)``
               or ``reduce([seg0, seg1, ...])``
    """

    def __init__(self, cfg: SegmentAggregatorConfig):
        super().__init__()
        self.cfg = cfg
        d_out = cfg.d_output or cfg.d_input

        self.input_proj = (
            nn.Identity() if cfg.d_input == cfg.d_model else nn.Linear(cfg.d_input, cfg.d_model)
        )
        self.input_norm = RMSNorm(cfg.d_model) if cfg.input_norm else nn.Identity()

        if cfg.time_embed == "seconds_mlp":
            self.time_mlp = nn.Sequential(
                nn.Linear(3, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model)
            )
        else:
            self.time_mlp = None

        self.summary_tokens = nn.Parameter(torch.randn(cfg.n_summary_tokens, cfg.d_model) * 0.02)
        self.layers = nn.ModuleList(Mamba2Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.output_proj = (
            nn.Identity() if cfg.d_model == d_out else nn.Linear(cfg.d_model, d_out)
        )

    # -- temporal encoding ------------------------------------------------
    def _index_sincos(self, n: int, device, dtype) -> torch.Tensor:
        d = self.cfg.d_model
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
        pe = torch.zeros(n, d, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.to(dtype)

    def _seg_time_embed(
        self, n_seg: int, device, dtype, segment_seconds: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Return (n_seg, d_model) or (B, n_seg, d_model), added to every token of a segment."""
        mode = self.cfg.time_embed
        if mode == "none":
            return None
        if mode == "index_sincos":
            return self._index_sincos(n_seg, device, dtype)
        if mode == "seconds_mlp":
            assert segment_seconds is not None, "time_embed='seconds_mlp' needs segment_seconds"
            s = segment_seconds.float()
            if s.dim() == 2:                       # (B, n_seg) -> treat as start; synth end/dur = 0
                s = torch.stack([s, s, torch.zeros_like(s)], dim=-1)
            elif s.shape[-1] == 2:                 # (B, n_seg, 2) start,end
                s = torch.cat([s, (s[..., 1:] - s[..., :1])], dim=-1)
            return self.time_mlp(s.to(dtype))
        raise ValueError(f"unknown time_embed {mode!r}")

    # -- offline / training ---------------------------------------------
    def forward(
        self,
        segment_tokens: torch.Tensor,                 # (B, N, K, d_input) or (B, N*K, d_input)
        segment_seconds: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        K = self.cfg.tokens_per_segment
        if segment_tokens.dim() == 3:
            B, S, _ = segment_tokens.shape
            assert S % K == 0, f"flat length {S} not a multiple of tokens_per_segment {K}"
            segment_tokens = segment_tokens.view(B, S // K, K, -1)
        B, N, k, _ = segment_tokens.shape
        assert k == K, f"expected K={K} tokens/segment, got {k}"

        x = self.input_norm(self.input_proj(segment_tokens))          # (B, N, K, D)
        te = self._seg_time_embed(N, x.device, x.dtype, segment_seconds)
        if te is not None:
            x = x + (te[:, :, None, :] if te.dim() == 3 else te[None, :, None, :])

        seq = rearrange(x, "b n k d -> b (n k) d")
        sm = self.summary_tokens.to(seq.dtype).expand(B, -1, -1)
        seq = torch.cat([seq, sm], dim=1)                            # (B, N*K + M, D)
        for blk in self.layers:
            seq = blk(seq)
        summary = self.final_norm(seq[:, -self.cfg.n_summary_tokens:])
        out = self.output_proj(summary)
        return (out, seq) if return_hidden else out

    # -- streaming / unbounded ----------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None) -> AggregatorState:
        device = device or self.summary_tokens.device
        dtype = dtype or self.summary_tokens.dtype
        return AggregatorState(
            layer_states=[blk.init_state(batch_size, device, dtype) for blk in self.layers],
            n_seen=0,
        )

    def update(
        self,
        state: AggregatorState,
        segment_tokens: torch.Tensor,                 # (B, K, d_input)
        segment_seconds: Optional[torch.Tensor] = None,
    ) -> AggregatorState:
        K = self.cfg.tokens_per_segment
        assert segment_tokens.dim() == 3 and segment_tokens.shape[1] == K
        x = self.input_norm(self.input_proj(segment_tokens))          # (B, K, D)

        if self.cfg.time_embed == "index_sincos":
            x = x + self._index_sincos(state.n_seen + 1, x.device, x.dtype)[-1]      # (D,)
        elif self.cfg.time_embed == "seconds_mlp":
            secs = None if segment_seconds is None else segment_seconds[:, None]     # (B, 1, ·)
            te = self._seg_time_embed(1, x.device, x.dtype, secs)                    # (B, 1, D)
            if te is not None:
                x = x + te

        for t in range(K):
            x_t = x[:, t]
            for i, blk in enumerate(self.layers):
                x_t, state.layer_states[i] = blk.step(x_t, state.layer_states[i])
        state.n_seen += 1
        return state

    def readout(self, state: AggregatorState) -> torch.Tensor:
        """Run the M summary queries from a COPY of the state (does not consume it)."""
        B = state.layer_states[0][0].shape[0]
        copy = [[cs.clone(), ss.clone()] for cs, ss in state.layer_states]
        sm = self.summary_tokens.to(copy[0][1].dtype)
        outs = []
        for m in range(self.cfg.n_summary_tokens):
            x_t = sm[m].unsqueeze(0).expand(B, -1)
            for i, blk in enumerate(self.layers):
                x_t, copy[i] = blk.step(x_t, copy[i])
            outs.append(x_t)
        summary = self.final_norm(torch.stack(outs, dim=1))
        return self.output_proj(summary)

    def reduce(
        self,
        segments,                                     # list of (B, K, d_input)  OR  (B, N, K, d_input)
        segment_seconds=None,
    ) -> torch.Tensor:
        if torch.is_tensor(segments) and segments.dim() == 4:
            segments = [segments[:, i] for i in range(segments.shape[1])]
        state = self.init_state(segments[0].shape[0], segments[0].device, segments[0].dtype)
        for i, seg in enumerate(segments):
            secs = None if segment_seconds is None else segment_seconds[:, i]
            state = self.update(state, seg, secs)
        return self.readout(state)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)  # noqa: A001  (unbuffered for background runs)
    torch.manual_seed(0)
    # CPU by default: the streaming path is a Python loop over tiny tensors, so kernel
    # launch overhead dominates on GPU. Set AGG_SMOKE_DEVICE=cuda to override.
    dev = os.environ.get("AGG_SMOKE_DEVICE", "cpu")

    cfg = SegmentAggregatorConfig(
        d_input=512, d_model=256, tokens_per_segment=16, n_summary_tokens=8,
        n_layers=3, d_state=64, headdim=32, chunk_size=32, time_embed="index_sincos",
    )
    agg = SegmentAggregator(cfg).to(dev).eval()
    n_params = sum(p.numel() for p in agg.parameters())
    print(f"params: {n_params/1e6:.2f}M   d_model={cfg.d_model} layers={cfg.n_layers} "
          f"K={cfg.tokens_per_segment} M={cfg.n_summary_tokens}")

    B, N, K = 2, 12, cfg.tokens_per_segment
    segs = torch.randn(B, N, K, cfg.d_input, device=dev)

    with torch.no_grad():
        y_offline = agg(segs)
        y_stream = agg.reduce(segs)
    assert y_offline.shape == (B, cfg.n_summary_tokens, cfg.d_input), y_offline.shape
    max_diff = (y_offline - y_stream).abs().max().item()
    rel = max_diff / y_offline.abs().max().item()
    print(f"offline vs streaming  max|Δ|={max_diff:.2e}  rel={rel:.2e}")
    assert rel < 5e-3, f"offline and streaming recurrence disagree (rel={rel:.2e})"

    # unbounded: fold many segments through a fixed-size state
    n_long = int(os.environ.get("AGG_SMOKE_LONG", "400"))
    with torch.no_grad():
        st = agg.init_state(B, dev)
        for _ in range(n_long):
            st = agg.update(st, torch.randn(B, K, cfg.d_input, device=dev))
        long_summary = agg.readout(st)
    state_bytes = sum(s.numel() * s.element_size() for ls in st.layer_states for s in ls)
    print(f"folded {n_long} segments  summary={tuple(long_summary.shape)}  "
          f"recurrent state={state_bytes/1024:.1f} KiB (constant in N)")
    assert torch.isfinite(long_summary).all()

    # gradients flow to the summary queries and the mixer
    agg.train()
    loss = agg(segs).square().mean()
    loss.backward()
    g_sm = agg.summary_tokens.grad
    g_mix = agg.layers[0].mixer.in_proj.weight.grad
    assert g_sm is not None and torch.isfinite(g_sm).all() and g_sm.abs().sum() > 0
    assert g_mix is not None and torch.isfinite(g_mix).all() and g_mix.abs().sum() > 0
    print(f"backward ok  |grad summary_tokens|={g_sm.norm():.3e}  |grad mixer.in_proj|={g_mix.norm():.3e}")
    print("all checks passed")
