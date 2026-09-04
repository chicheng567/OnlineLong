import contextlib
import math

from torch.nn import LayerNorm
import torch
from transformers.activations import GELUTanh
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from .videollama3_encoder.modeling_videollama3_encoder import VisionRotaryEmbedding, apply_rotary_pos_emb_vision
from .dts import SiglipAECompressor
from .segment_aggregator import SegmentAggregator, SegmentAggregatorConfig


def _build_2d_rotary_pos_emb(rotary_pos_emb_module, w, h):
    device = rotary_pos_emb_module.inv_freq.device
    hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w).reshape(-1)
    wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1).reshape(-1)
    pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
    rotary_pos_emb_full = rotary_pos_emb_module(max(h, w))
    return rotary_pos_emb_full[pos_ids].flatten(1)


def _build_sinusoidal_position_encoding(num_positions: int, dim: int) -> torch.Tensor:
    """
    Classic (Vaswani et al.) additive sin/cos positional encoding, directly encoding
    the flat index 0..num_positions-1 -- NOT RoPE (no rotation, added straight to the
    embedding). Returns (num_positions, dim), fp32 (cast by the caller).

        PE[pos, 2i]   = sin(pos / 10000^(2i/dim))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/dim))
    """
    assert dim % 2 == 0, f"_build_sinusoidal_position_encoding requires an even dim, got {dim}."
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)  # (N, 1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))  # (dim/2,)
    pe = torch.zeros(num_positions, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def _build_factorized_rotary(inv_freq: torch.Tensor, coords_list, dims_list) -> torch.Tensor:
    """
    Build a factorized (multi-axis) rotary frequency table for cross-attention.

    Each axis ``a`` gets a contiguous slice of ``inv_freq`` of width ``dims_list[a]``
    and is multiplied by its per-token coordinate ``coords_list[a]`` (shape ``(N,)``).
    The concatenation has shape ``(N, sum(dims_list))`` which must equal
    ``(N, head_dim // 2)`` so that ``apply_rotary_pos_emb_vision`` can duplicate it to
    ``head_dim`` and pair channel ``i`` with ``i + head_dim // 2``.

    This realizes M-RoPE-style position encoding: e.g. for ``coords_list=[t, h, w]``
    each frequency band rotates a disjoint set of channels, so the same dot product
    encodes relative temporal *and* spatial offsets simultaneously.  A 1-D variant
    (``coords_list=[t]``) dedicates the whole budget to the temporal axis.

    ``inv_freq`` is the buffer of a ``VisionRotaryEmbedding(dim=head_dim)`` module, so
    ``len(inv_freq) == head_dim // 2``.
    """
    parts = []
    start = 0
    for coords, d in zip(coords_list, dims_list):
        band = inv_freq[start:start + d]
        parts.append(torch.outer(coords.to(band.dtype), band))
        start += d
    return torch.cat(parts, dim=-1)  # (N, sum(dims_list)) == (N, head_dim // 2)


def prune_kv_by_common_component(kv, compression_cu_seqlens, grid_hws, ratio, min_tokens=0):
    """Training-time redundancy pruning of the compressor's KV.

    Per window, drop the ``ratio`` fraction of tokens whose direction is closest
    (cosine) to that window's *common component* -- the L2-normalised mean of its
    tokens. A token that is almost entirely the common component adds little unique
    signal, so it is the cheapest to remove.

    Survivors keep their ORIGINAL flat position within the window, so
    ``_build_cross_rotary_3d`` / ``_build_cross_rotary_kv`` give each the same
    ``(t, h, w)`` (and hence the same RoPE angle) it would have in the un-pruned
    window; the dropped tokens simply never enter the KV.

    Returns ``(kv_pruned, cu_pruned, kept_idx)`` where ``kept_idx[i]`` is a sorted
    1-D LongTensor of survivor flat indices into window i's dense ``[0, L_i)`` range
    (or ``None`` when that window was left intact). Deterministic (topk, no RNG).
    """
    device = kv.device
    cu = compression_cu_seqlens.tolist()
    parts, kept, new_cu = [], [], [0]
    for i in range(len(cu) - 1):
        w = kv[cu[i]:cu[i + 1]]
        length = w.shape[0]
        hw = int(grid_hws[i][0]) * int(grid_hws[i][1])
        floor = max(1, int(min_tokens) if min_tokens and min_tokens > 0 else hw)
        n_keep = max(floor, length - int(round(length * ratio)))
        if n_keep >= length:
            parts.append(w); kept.append(None); new_cu.append(new_cu[-1] + length)
            continue
        with torch.no_grad():
            x = w.float()
            c = torch.nn.functional.normalize(x.mean(dim=0), dim=0)
            sim = torch.nn.functional.normalize(x, dim=1) @ c
            keep_i = torch.topk(sim, n_keep, largest=False).indices.sort().values
        parts.append(w.index_select(0, keep_i))
        kept.append(keep_i)
        new_cu.append(new_cu[-1] + n_keep)
    kv_pruned = torch.cat(parts, dim=0)
    cu_pruned = torch.tensor(new_cu, device=device, dtype=compression_cu_seqlens.dtype)
    return kv_pruned, cu_pruned, kept


def _match_encoder_scale(compressed, ref_mean, ref_std, gamma, beta):
    """Option A -- put the compressor output back on the frozen-encoder scale.

    Affine-maps the compressed set's per-dim mean / std onto the encoder tokens'
    (``ref_mean`` / ``ref_std``, taken from the compressor's OWN KV input this
    forward -- i.e. the very tokens the frozen ``mm_projector`` was trained on),
    then applies a learnable per-channel ``gamma`` / ``beta`` (identity at init).

    Deterministic and runs at train *and* inference, so the compressed tokens
    stop drifting ~26x above the encoder-token norm the projector expects. Grad
    flows through ``compressed``; ``ref_*`` are detached.
    """
    x = compressed.float()
    m = x.mean(0, keepdim=True)
    s = x.std(0, keepdim=True).clamp_min(1e-6)
    x = (x - m) / s * ref_std + ref_mean
    x = x * gamma.float() + beta.float()
    return x.to(compressed.dtype)


def _distribution_match_loss(compressed, ref_rows):
    """Option B -- CORAL-style pull of the compressed token cloud onto the frozen
    encoder-token manifold, designed to COMPOSE with ``_match_encoder_scale``
    (which only fixes the per-dim mean / variance, i.e. the covariance diagonal).

    Three terms, each a RELATIVE error bounded roughly in ``[0, 1]`` (denominator
    detached, so it acts as a constant scale) -- this keeps the weight sane whether
    or not Option A is also on:
      * ``l_mean`` : ``||E[c] - E[r]||^2``                       (centroid)
      * ``l_cov``  : ``||cov(c) - cov(r)||_F^2``                 (off-diagonal
                     correlation structure -- the PCA-spectrum / effective-rank
                     mismatch that the diagonal affine cannot touch)
      * ``l_norm`` : ``(mean||c|| - mean||r||)^2``               (the headline
                     compressed/raw norm gap)

    ``ref_rows`` is a detached ``(Nr, d)`` sample of the encoder tokens;
    ``compressed`` is ``(Nc, d)`` and carries grad.
    """
    eps = 1e-8
    r = ref_rows.float()
    c = compressed.float()
    mr, mc = r.mean(0), c.mean(0)
    l_mean = (mc - mr).pow(2).mean() / (mr.pow(2).mean() + mc.detach().pow(2).mean() + eps)
    rc = r - mr
    cc = c - mc
    cov_r = (rc.t() @ rc) / max(r.shape[0] - 1, 1)
    cov_c = (cc.t() @ cc) / max(c.shape[0] - 1, 1)
    l_cov = (cov_c - cov_r).pow(2).mean() / (
        cov_r.pow(2).mean() + cov_c.detach().pow(2).mean() + eps)
    nr = r.norm(dim=1).mean()
    nc = c.norm(dim=1).mean()
    l_norm = (nc - nr).pow(2) / (nr.pow(2) + nc.detach().pow(2) + eps)
    return l_mean + l_cov + l_norm


def _init_encoder_scale_match(module, config):
    """Shared __init__ tail for the transformer_decoder* compressors: read the
    Option A / Option B knobs off ``config`` and, for A, register the learnable
    per-channel ``out_gamma`` / ``out_beta`` (identity init). Both default off so
    existing checkpoints load and behave unchanged."""
    module.match_encoder_scale = bool(getattr(config, "match_encoder_scale", False))
    if module.match_encoder_scale:
        module.out_gamma = nn.Parameter(torch.ones(config.hidden_size))
        module.out_beta = nn.Parameter(torch.zeros(config.hidden_size))
    # Option B: the trainer reads distr_loss_weight + _last_distr_loss off this
    # module and adds distr_loss_weight * _last_distr_loss to the CE loss. 0 = off.
    module.distr_loss_weight = float(getattr(config, "distr_loss_weight", 0.0) or 0.0)
    module.distr_loss_max_ref_tokens = int(getattr(config, "distr_loss_max_ref_tokens", 4096) or 4096)
    module._last_distr_loss = None


def _capture_ref_stats(module, kv):
    """Per-dim mean/std (and, when Option B is active in training, a row sample) of
    the compressor's raw KV input -- the frozen encoder tokens. Called BEFORE any
    KV pruning so the target is the full window. Returns (ref_mean, ref_std,
    ref_rows); any element is None when not needed."""
    need_scale = getattr(module, "match_encoder_scale", False)
    need_aux = module.training and getattr(module, "distr_loss_weight", 0.0) > 0.0
    if not (need_scale or need_aux):
        return None, None, None
    with torch.no_grad():
        rk = kv.detach().float()
        ref_mean = rk.mean(0, keepdim=True)
        ref_std = rk.std(0, keepdim=True).clamp_min(1e-6)
        ref_rows = None
        if need_aux:
            cap = module.distr_loss_max_ref_tokens
            if rk.shape[0] > cap:
                sel = torch.randperm(rk.shape[0], device=rk.device)[:cap]
                ref_rows = rk.index_select(0, sel)
            else:
                ref_rows = rk
    return ref_mean, ref_std, ref_rows


def _finalize_compressed(module, query, ref_mean, ref_std, ref_rows):
    """Shared forward tail: apply Option A's scale match (if enabled) and stash
    Option B's aux loss on the module (if enabled in training). Returns the
    (possibly rescaled) query."""
    if getattr(module, "match_encoder_scale", False) and ref_mean is not None:
        query = _match_encoder_scale(query, ref_mean, ref_std, module.out_gamma, module.out_beta)
    if module.training and getattr(module, "distr_loss_weight", 0.0) > 0.0 and ref_rows is not None:
        module._last_distr_loss = _distribution_match_loss(query, ref_rows)
    else:
        module._last_distr_loss = None
    return query


class mlp(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = GELUTanh();
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Attention(nn.Module):
    def __init__(self, n_head, embed_dim, dropout=0.1, causal=True):
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.num_heads = n_head
        self.head_dim = embed_dim // n_head
        self.scale = self.head_dim ** -0.5
        self.dropout_rate = dropout
        assert self.head_dim * n_head == embed_dim, "embed_dim must be divisible by n_head"
        self.w_q = nn.Linear(self.embed_dim, n_head * self.head_dim, bias=False)
        self.w_k = nn.Linear(self.embed_dim, n_head * self.head_dim, bias=False)
        self.w_v = nn.Linear(self.embed_dim, n_head * self.head_dim, bias=False)
        self.w_o = nn.Linear(n_head * self.head_dim, self.embed_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        self.causal = causal
    def forward(self):
        raise NotImplementedError
    
class CrossFlashAttention2(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, x_q, x_kv, cu_seqlens_q, cu_seqlens_kv,
                rotary_pos_emb_q: torch.Tensor = None,
                rotary_pos_emb_kv: torch.Tensor = None):
        # x_q should be of shape (batch_size * seq_len_q, d_model)
        # x_kv should be of shape (batch_size * seq_len_kv, d_model)
        # cu_seqlens_q should be of shape (batch_size + 1,) like (0, 4, 7, 9, 32, 33, ...)
        # cu_seqlens_kv should be of shape (batch_size + 1,) like (0, 4, 7, 9, 32, 33, ...)
        # rotary_pos_emb_{q,kv}: optional (total_tokens, head_dim // 2) RoPE tables.
        #   Only q and k are rotated (never v).  Leaving rotary_pos_emb_q=None is
        #   equivalent to placing every query at coordinate 0 — used by
        #   LocalAttnConvCompressor so each query sits at temporal slot 0 while the
        #   keys carry their real frame index, encoding the relative offset -t.
        drop_rate = self.dropout_rate if self.training else 0.0
        q = self.w_q(x_q).view(-1, self.n_head, self.head_dim)
        k = self.w_k(x_kv).view(-1, self.n_head, self.head_dim)
        v = self.w_v(x_kv).view(-1, self.n_head, self.head_dim)
        if rotary_pos_emb_q is not None:
            q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb_q).squeeze(0)
        if rotary_pos_emb_kv is not None:
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb_kv).squeeze(0)
        assert cu_seqlens_q[0].item() == 0 and cu_seqlens_kv[0].item() == 0
        assert cu_seqlens_q[-1].item() == q.shape[0], (cu_seqlens_q[-1].item(), q.shape[0])
        assert cu_seqlens_kv[-1].item() == k.shape[0], (cu_seqlens_kv[-1].item(), k.shape[0])
        max_len_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
        max_len_kv = (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
        # output shape: (total_tokens_q, n_head, d_kv)
        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_len_q,
            max_seqlen_k=max_len_kv,
            dropout_p=drop_rate,
            causal=self.causal,
        )
        output = output.reshape(-1, self.n_head * self.head_dim)
        output = self.dropout_layer(self.w_o(output))
        return output

class selfFlashAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        q_len, _ = hidden_states.size()
        drop_rate = self.dropout_rate if self.training else 0.0
        query_states = self.w_q(hidden_states).view(q_len, self.n_head, self.head_dim)
        key_states = self.w_k(hidden_states).view(q_len, self.n_head, self.head_dim)
        value_states = self.w_v(hidden_states).view(q_len, self.n_head, self.head_dim)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(q_len, self.n_head, self.head_dim)
        key_states = key_states.view(q_len, self.n_head, self.head_dim)
        value_states = value_states.view(q_len, self.n_head, self.head_dim)
        parts_count = cu_seqlens.size(0) - 1
        query_states = query_states.view(parts_count, -1, self.n_head, self.head_dim)
        key_states = key_states.view(parts_count, -1, self.n_head, self.head_dim)
        # Apply rotary positional embeddings. rotary_pos_emb=None skips rotation
        # entirely (matches CrossFlashAttention2's same convention) -- used by
        # compressors whose queries carry positional info additively instead of via
        # RoPE (e.g. TransformerDecoderFlatCompressor's sin/cos encoding).
        if rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb_vision(query_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
            key_states = apply_rotary_pos_emb_vision(key_states.unsqueeze(0), rotary_pos_emb).squeeze(0)
        query_states = query_states.view(-1, self.n_head, self.head_dim)
        key_states = key_states.view(-1, self.n_head, self.head_dim)
        assert cu_seqlens[0].item() == 0
        assert cu_seqlens[-1].item() == query_states.shape[0], (cu_seqlens[-1].item(), query_states.shape[0])

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states, 
            key_states, 
            value_states, 
            cu_seqlens_q=cu_seqlens, 
            cu_seqlens_k=cu_seqlens, 
            max_seqlen_q=max_seqlen, 
            max_seqlen_k=max_seqlen,
            dropout_p=drop_rate,
            causal=self.causal).reshape(q_len, -1)
        attn_output = self.w_o(attn_output)
        
        return attn_output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn = CrossFlashAttention2(embed_dim=config.hidden_size, n_head=config.num_attention_heads, dropout=config.attention_probs_dropout_prob, causal=False)
        self.self_attn = selfFlashAttention(embed_dim=config.hidden_size, n_head=config.num_attention_heads, dropout=config.attention_probs_dropout_prob, causal=False)
        self.embed_dim = config.hidden_size
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm3 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = mlp(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)
    def forward(self, q, kv, cu_seqlens_q, cu_seqlens_kv, rotary_pos_emb,
                cross_rotary_q=None, cross_rotary_kv=None):
        q = q + self.self_attn(self.layer_norm1(q), cu_seqlens_q, rotary_pos_emb)
        q = q + self.cross_attn(self.layer_norm2(q), kv, cu_seqlens_q, cu_seqlens_kv,
                                cross_rotary_q, cross_rotary_kv)
        q = q + self.mlp(self.layer_norm3(q))
        return q

class TransformerDecoderCompressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config.num_layers
        head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.num_head = config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)
        # 3D (t, h, w) rotary for the cross-attention onto the input video tokens.
        # Its inv_freq holds head_dim // 2 frequencies, split into temporal / height /
        # width bands so the query↔key dot product encodes relative spatial *and*
        # temporal offsets — without it the cross-attention is permutation-invariant
        # over the flattened T×HW token set and loses all frame ordering.
        self.cross_rotary = VisionRotaryEmbedding(dim=head_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.compress_image_w = config.compress_image_w
        self.compress_image_h = config.compress_image_h
        self.compress_image_wh = self.compress_image_w * self.compress_image_h
        # Learned query tokens (small-scale init, std=0.02).
        self.query = nn.Parameter(torch.randn(1, self.compress_image_w * self.compress_image_h, config.hidden_size) * 0.02)
        self.window_size = getattr(config, "window_size", 1)
        # Training-only common-component KV pruning (see prune_kv_by_common_component).
        # 0 disables; applied only in .train() mode so inference always sees the full KV.
        self.token_prune_ratio = float(getattr(config, "token_prune_ratio", 0.0) or 0.0)
        self.token_prune_min_tokens = int(getattr(config, "token_prune_min_tokens", 0) or 0)
        # Option A (encoder-scale match) + Option B (distribution-match aux loss).
        _init_encoder_scale_match(self, config)

    def _build_query_rotary_pos_emb(self, w, h) -> torch.Tensor:
        return _build_2d_rotary_pos_emb(self.rotary_pos_emb, w, h)

    def output_hw_for(self, h: int, w: int):
        """The compressor's output grid is a fixed, learned query — it does not
        depend on the input's (h, w) at all (unlike SiglipAECompressor's spatial
        pass-through). Exposed so callers (arch.py) can size placeholder tokens
        per compression part without hardcoding compress_image_h/w themselves."""
        return self.compress_image_h, self.compress_image_w

    def _build_cross_rotary_3d(self, compression_cu_seqlens, device, grid_hws, kept_idx=None):
        """
        Build 3-D (t, h, w) rotary tables for the cross-attention.

        Queries: one HW grid per window, all anchored at temporal slot 0 with their
        2-D spatial coordinates (compressor's own fixed output grid,
        compress_image_h × compress_image_w). Keys: the window's input tokens —
        temporal coordinate = frame index, spatial coordinate = position within the
        frame, using PER-WINDOW (h_i, w_i) from `grid_hws`.

        By default (`kept_idx=None`) window i is assumed dense frame-major
        (T_i×h_i×w_i tokens, in order). If `kept_idx` is given (one 1-D LongTensor
        per window, flat indices into that dense layout), the KV already holds only
        those tokens and each is given the (t, h, w) of its ORIGINAL flat index —
        so a surviving token's RoPE angle is byte-identical to the un-pruned clip;
        the dropped tokens simply never enter the KV. Returns (q_freqs, kv_freqs).
        """
        window_lens = (compression_cu_seqlens[1:] - compression_cu_seqlens[:-1]).long()
        B = window_lens.shape[0]
        assert len(grid_hws) == B, (
            f"TransformerDecoderCompressor: grid_hws must have one (h, w) per window "
            f"({B} windows), got {len(grid_hws)}."
        )
        if kept_idx is not None:
            assert len(kept_idx) == B, (
                f"TransformerDecoderCompressor: kept_idx must have one entry per window "
                f"({B}), got {len(kept_idx)}."
            )

        inv_freq = self.cross_rotary.inv_freq  # (head_dim // 2,)
        D = inv_freq.shape[0]
        d_t = D // 3
        d_h = (D - d_t) // 2
        d_w = D - d_t - d_h
        dims = [d_t, d_h, d_w]

        # Keys: derive each token's (t, h, w) from its flat index in the dense
        # frame-major layout — dense == arange, pruned == the kept indices.
        kv_t, kv_h, kv_w = [], [], []
        for i in range(B):
            h_i, w_i = grid_hws[i]
            hw_i = h_i * w_i
            ki = None if kept_idx is None else kept_idx[i]
            if ki is None:
                assert window_lens[i].item() % hw_i == 0, (
                    f"TransformerDecoderCompressor: window {i}'s token count "
                    f"({window_lens[i].item()}) must be divisible by its grid_hws "
                    f"h*w={hw_i} (h={h_i}, w={w_i})."
                )
                idx_i = torch.arange(window_lens[i].item(), device=device)
            else:
                idx_i = ki.to(device=device, dtype=torch.long)
                assert idx_i.numel() == window_lens[i].item(), (
                    f"TransformerDecoderCompressor: window {i} kept_idx has {idx_i.numel()} "
                    f"entries but {window_lens[i].item()} tokens were passed for it."
                )
            rem = idx_i % hw_i
            kv_t.append(idx_i // hw_i)
            kv_h.append(rem // w_i)
            kv_w.append(rem % w_i)
        kv_t = torch.cat(kv_t); kv_h = torch.cat(kv_h); kv_w = torch.cat(kv_w)

        # Queries: B grids at the compressor's own fixed output resolution, all at t=0.
        H, W, HW = self.compress_image_h, self.compress_image_w, self.compress_image_wh
        hpos_q = torch.arange(H, device=device).unsqueeze(1).expand(-1, W).reshape(-1)
        wpos_q = torch.arange(W, device=device).unsqueeze(0).expand(H, -1).reshape(-1)
        q_t = torch.zeros(B * HW, device=device)
        q_h = hpos_q.repeat(B); q_w = wpos_q.repeat(B)

        kv_freqs = _build_factorized_rotary(inv_freq, [kv_t, kv_h, kv_w], dims)
        q_freqs = _build_factorized_rotary(inv_freq, [q_t, q_h, q_w], dims)
        return q_freqs, kv_freqs

    def forward(self, kv, compression_cu_seqlens, grid_hws=None, kept_idx=None):
        # kv: (1, total_tokens, hidden_size)
        # grid_hws: optional list of (h, w) pairs, one per window — the ACTUAL input
        # frame grid for that window (may differ from compress_image_h/w and from
        # window to window). Defaults to compress_image_h/w for every window.
        # kept_idx: optional list (one per window) of flat indices into that window's
        # dense frame-major layout; when given, kv already holds only those tokens
        # and the cross-RoPE gives each survivor its original (t, h, w). Normally left
        # None — the compressor prunes its own KV below when token_prune_ratio > 0.
        compression_parts = compression_cu_seqlens.size(0) - 1
        if kv.dim() == 3:
            kv = kv.squeeze(0) # (total_tokens, hidden_size)
        # Encoder-token reference stats (Options A/B), captured before KV pruning.
        ref_mean, ref_std, ref_rows = _capture_ref_stats(self, kv)
        B = compression_parts
        if grid_hws is None:
            grid_hws = [(self.compress_image_h, self.compress_image_w)] * B
        query = self.query.expand(B, -1, -1).contiguous().view(-1, kv.size(-1))  # (B * compress_image_wh, hidden_size)
        cu_seqlens_q = torch.arange(
            0,
            (B + 1) * self.compress_image_wh,
            step=self.compress_image_wh,
            device=kv.device,
            dtype=torch.int32,
        ).contiguous()
        compression_cu_seqlens = compression_cu_seqlens.to(device=kv.device, dtype=torch.int32).contiguous()
        # Training-only redundancy pruning of the KV (inference always sees the full set).
        if kept_idx is None and self.training and self.token_prune_ratio > 0.0:
            kv, compression_cu_seqlens, kept_idx = prune_kv_by_common_component(
                kv, compression_cu_seqlens, grid_hws, self.token_prune_ratio, self.token_prune_min_tokens
            )
            compression_cu_seqlens = compression_cu_seqlens.to(dtype=torch.int32).contiguous()
        rotary_pos_emb = self._build_query_rotary_pos_emb(self.compress_image_w, self.compress_image_h)
        cross_rotary_q, cross_rotary_kv = self._build_cross_rotary_3d(compression_cu_seqlens, kv.device, grid_hws, kept_idx)
        for layer in self.layers:
            query = layer(query, kv, cu_seqlens_q, compression_cu_seqlens, rotary_pos_emb,
                          cross_rotary_q, cross_rotary_kv)
        return _finalize_compressed(self, query, ref_mean, ref_std, ref_rows)


class TransformerDecoderFlatCompressor(nn.Module):
    """
    Variant of TransformerDecoderCompressor whose compressed OUTPUT is a flat set of
    `num_queries` learned tokens (default 32) instead of a 2-D
    compress_image_h x compress_image_w spatial grid, positioned with a classic
    (Vaswani et al.) ADDITIVE sin/cos positional encoding over the flat index range
    [0, num_queries) — NOT RoPE. Everything else (cross-attention onto the T x
    (h_i x w_i) input tokens with per-window dynamic (h, w) via grid_hws, the
    TransformerDecoderLayer stack) is identical to TransformerDecoderCompressor.

    Because the output has no spatial grid, the queries carry no (h, w) coordinate for
    cross-attention either: cross_rotary_q is left None, which CrossFlashAttention2
    treats as "every query sits at coordinate 0" (see its forward() docstring) — the
    same convention LocalAttnConvCompressor's cross-attn already relies on. Only the
    input KV tokens' real (t, h_i, w_i) coordinates get a rotary table; self-attention
    among the queries drops RoPE entirely (rotary_pos_emb=None — see
    selfFlashAttention's guard) since positional identity is already baked additively
    into the query embedding before the layer stack runs.
    """

    def __init__(self, config):
        super().__init__()
        num_layers = config.num_layers
        head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.num_head = config.num_attention_heads
        self.num_queries = getattr(config, "num_queries", 32)

        # KV-side (input tokens) 3-D (t, h, w) rotary — same role as
        # TransformerDecoderCompressor.cross_rotary, query side unused (see above).
        self.cross_rotary = VisionRotaryEmbedding(dim=head_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(num_layers)])
        self.num_layers = num_layers

        # Learned per-slot content (small-scale init, std=0.02) + fixed sin/cos
        # positional encoding over the flat index, added together once before the
        # layer stack (not re-applied per layer — the residual connections in
        # TransformerDecoderLayer carry it forward, same as a standard Transformer
        # decoder's input embedding + positional encoding).
        self.query = nn.Parameter(torch.randn(1, self.num_queries, config.hidden_size) * 0.02)
        self.register_buffer(
            "pos_encoding",
            _build_sinusoidal_position_encoding(self.num_queries, config.hidden_size),
            persistent=False,
        )
        self.window_size = getattr(config, "window_size", 1)
        # Training-only common-component KV pruning (see prune_kv_by_common_component).
        self.token_prune_ratio = float(getattr(config, "token_prune_ratio", 0.0) or 0.0)
        self.token_prune_min_tokens = int(getattr(config, "token_prune_min_tokens", 0) or 0)
        # Option A (encoder-scale match) + Option B (distribution-match aux loss).
        _init_encoder_scale_match(self, config)

    def output_hw_for(self, h: int, w: int):
        # Flat output, no 2-D grid — (1, num_queries) so callers' oh*ow arithmetic
        # (arch.py) still yields the right total token count.
        return 1, self.num_queries

    def _build_cross_rotary_kv(self, compression_cu_seqlens, device, grid_hws, kept_idx=None):
        """KV-side-only counterpart of TransformerDecoderCompressor._build_cross_rotary_3d
        (see its docstring, including the `kept_idx` token-pruning contract) — no
        query-side coordinates are built here since queries carry no rotary at all."""
        window_lens = (compression_cu_seqlens[1:] - compression_cu_seqlens[:-1]).long()
        B = window_lens.shape[0]
        assert len(grid_hws) == B, (
            f"TransformerDecoderFlatCompressor: grid_hws must have one (h, w) per "
            f"window ({B} windows), got {len(grid_hws)}."
        )
        if kept_idx is not None:
            assert len(kept_idx) == B, (
                f"TransformerDecoderFlatCompressor: kept_idx must have one entry per "
                f"window ({B}), got {len(kept_idx)}."
            )

        inv_freq = self.cross_rotary.inv_freq  # (head_dim // 2,)
        D = inv_freq.shape[0]
        d_t = D // 3
        d_h = (D - d_t) // 2
        d_w = D - d_t - d_h
        dims = [d_t, d_h, d_w]

        kv_t, kv_h, kv_w = [], [], []
        for i in range(B):
            h_i, w_i = grid_hws[i]
            hw_i = h_i * w_i
            ki = None if kept_idx is None else kept_idx[i]
            if ki is None:
                assert window_lens[i].item() % hw_i == 0, (
                    f"TransformerDecoderFlatCompressor: window {i}'s token count "
                    f"({window_lens[i].item()}) must be divisible by its grid_hws "
                    f"h*w={hw_i} (h={h_i}, w={w_i})."
                )
                idx_i = torch.arange(window_lens[i].item(), device=device)
            else:
                idx_i = ki.to(device=device, dtype=torch.long)
                assert idx_i.numel() == window_lens[i].item(), (
                    f"TransformerDecoderFlatCompressor: window {i} kept_idx has "
                    f"{idx_i.numel()} entries but {window_lens[i].item()} tokens were passed."
                )
            rem = idx_i % hw_i
            kv_t.append(idx_i // hw_i)
            kv_h.append(rem // w_i)
            kv_w.append(rem % w_i)
        kv_t = torch.cat(kv_t); kv_h = torch.cat(kv_h); kv_w = torch.cat(kv_w)
        return _build_factorized_rotary(inv_freq, [kv_t, kv_h, kv_w], dims)

    def forward(self, kv, compression_cu_seqlens, grid_hws, kept_idx=None):
        # kv: (1, total_tokens, hidden_size) or (total_tokens, hidden_size)
        # grid_hws: required (one (h, w) per window) — this compressor has no fixed
        # compress_image_h/w to fall back on.
        # kept_idx: optional list (one per window) of flat indices into that window's
        # dense frame-major layout; kv then holds only those tokens, each keeping its
        # original (t, h, w) in the cross-RoPE.
        if kv.dim() == 3:
            kv = kv.squeeze(0)
        # Encoder-token reference stats (Options A/B), captured before KV pruning.
        ref_mean, ref_std, ref_rows = _capture_ref_stats(self, kv)
        B = compression_cu_seqlens.size(0) - 1
        if grid_hws is None:
            raise ValueError(
                "TransformerDecoderFlatCompressor has no 2-D output grid to fall back "
                "on — grid_hws (one (h, w) per window) is required, not optional."
            )

        query = self.query + self.pos_encoding.to(dtype=self.query.dtype)  # (1, num_queries, hidden)
        query = query.expand(B, -1, -1).contiguous().view(-1, kv.size(-1))
        cu_seqlens_q = torch.arange(
            0, (B + 1) * self.num_queries, step=self.num_queries,
            device=kv.device, dtype=torch.int32,
        ).contiguous()
        compression_cu_seqlens = compression_cu_seqlens.to(device=kv.device, dtype=torch.int32).contiguous()
        # Training-only redundancy pruning of the KV (inference always sees the full set).
        if kept_idx is None and self.training and self.token_prune_ratio > 0.0:
            kv, compression_cu_seqlens, kept_idx = prune_kv_by_common_component(
                kv, compression_cu_seqlens, grid_hws, self.token_prune_ratio, self.token_prune_min_tokens
            )
            compression_cu_seqlens = compression_cu_seqlens.to(dtype=torch.int32).contiguous()
        cross_rotary_kv = self._build_cross_rotary_kv(compression_cu_seqlens, kv.device, grid_hws, kept_idx)
        for layer in self.layers:
            query = layer(
                query, kv, cu_seqlens_q, compression_cu_seqlens,
                rotary_pos_emb=None, cross_rotary_q=None, cross_rotary_kv=cross_rotary_kv,
            )
        return _finalize_compressed(self, query, ref_mean, ref_std, ref_rows)


# ---------------------------------------------------------------------------
# LocalAttnConvCompressor
#
# Each query corresponds to one spatial position (i, j) in the output grid.
# compress_image_w × compress_image_h == patches per frame.
# Each compression window must contain exactly T × compress_image_wh tokens
# (T frames in frame-major order), so this compressor performs pure *temporal*
# compression while preserving spatial resolution.
#
# Layer order per transformer block:
#   1. Spatial self-attention (all H×W queries within a window communicate)
#   2. Local cross-attention  (query p attends only to the T tokens at position p)
#   3. MLP
# All sub-layers use pre-norm + residual connection.
#
# Self-attention runs FIRST (not last) deliberately, matching TransformerDecoderLayer's
# self-attn -> cross-attn -> mlp convention (and the standard Transformer-decoder /
# DETR ordering). Measured (see diagnostics/query_attention_health.py): a positional
# self-attention is an averaging/mixing operation over the H×W queries, and whichever
# sub-layer runs LAST has the final say on cross-position diversity (MLP is position-
# wise and can't restore it). With self-attn last, its branch output was found to
# outscale the residual stream ~16x and homogenize all H×W outputs to near-identical
# vectors, erasing the real per-position signal cross-attn had just injected — even
# though the residual connection is intact. Running self-attn first (while queries are
# still maximally diverse, straight from the learned per-position init) and cross-attn
# last (small-magnitude, content-differentiated, residual-dominated — confirmed healthy
# in TransformerDecoderCompressor) avoids this.
# ---------------------------------------------------------------------------

class LocalAttnConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cross_attn = CrossFlashAttention2(
            embed_dim=config.hidden_size,
            n_head=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=False,
        )
        # Spatial self-attention over the output query grid — no causal ordering.
        self.self_attn = selfFlashAttention(
            embed_dim=config.hidden_size,
            n_head=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=False,
        )
        self.embed_dim = config.hidden_size
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm3 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = mlp(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)

    def forward(self, q, kv_local, cu_seqlens_q_local, cu_seqlens_kv_local, cu_seqlens_q_self, rotary_pos_emb,
                cross_rotary_kv=None):
        # 1. Spatial self-attention: queries within a window exchange spatial context,
        #    while they are still maximally differentiated (straight from the learned
        #    per-position query init / the previous layer's cross-attn-injected content).
        q = q + self.self_attn(self.layer_norm1(q), cu_seqlens_q_self, rotary_pos_emb)
        # 2. Local cross-attention: query at position p sees only position p's T frame tokens.
        #    The query stays at temporal slot 0 (rotary_pos_emb_q=None → identity) while the
        #    keys carry their frame index via cross_rotary_kv, so the dot product encodes the
        #    relative offset -t and the T frames are no longer exchangeable. Runs LAST (after
        #    self-attn) so the per-position content it injects isn't mixed back across
        #    positions by anything downstream (MLP is position-wise).
        q = q + self.cross_attn(self.layer_norm2(q), kv_local, cu_seqlens_q_local, cu_seqlens_kv_local,
                                None, cross_rotary_kv)
        # 3. MLP.
        q = q + self.mlp(self.layer_norm3(q))
        return q


class LocalAttnConvCompressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config.num_layers
        head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.num_head = config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)
        # 1-D temporal rotary for the local cross-attention.  Spatial position is
        # already fixed by the local routing (query p only sees position p's frames),
        # so the whole frequency budget (head_dim // 2) is dedicated to the temporal
        # axis — this is what lets the compressor tell the T frames apart instead of
        # treating them as an unordered set.
        self.cross_rotary = VisionRotaryEmbedding(dim=head_dim)
        self.layers = nn.ModuleList([LocalAttnConvLayer(config) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.compress_image_w = config.compress_image_w
        self.compress_image_h = config.compress_image_h
        self.compress_image_wh = self.compress_image_w * self.compress_image_h
        # One learned query per spatial position; shared across windows and batch
        # (small-scale init, std=0.02).
        self.query = nn.Parameter(torch.randn(1, self.compress_image_wh, config.hidden_size) * 0.02)
        self.window_size = getattr(config, "window_size", 1)

    def _build_query_rotary_pos_emb(self, w, h) -> torch.Tensor:
        return _build_2d_rotary_pos_emb(self.rotary_pos_emb, w, h)

    def output_hw_for(self, h: int, w: int):
        # Fixed output grid, matching TransformerDecoderCompressor's convention — see
        # its output_hw_for docstring. Dynamic input (h, w) is NOT yet supported here
        # (unlike TransformerDecoderCompressor/SiglipAECompressor); this compressor
        # still requires every window's actual grid to equal compress_image_h/w.
        return self.compress_image_h, self.compress_image_w

    def forward(self, kv, compression_cu_seqlens):
        # kv: (total_tokens, hidden_size) or (1, total_tokens, hidden_size)
        # Each window i spans kv[compression_cu_seqlens[i] : compression_cu_seqlens[i+1]]
        # and must contain exactly T_i * compress_image_wh tokens (frame-major order).
        if kv.dim() == 3:
            kv = kv.squeeze(0)

        device = kv.device
        compression_cu_seqlens = compression_cu_seqlens.to(device=device, dtype=torch.int32)
        B = compression_cu_seqlens.size(0) - 1
        HW = self.compress_image_wh

        # Queries: (B * HW, hidden_size)
        query = self.query.expand(B, -1, -1).contiguous().view(-1, self.hidden_size)

        # Validate and derive number of frames per window.
        window_lens = (compression_cu_seqlens[1:] - compression_cu_seqlens[:-1]).long()
        assert (window_lens % HW == 0).all(), (
            f"LocalAttnConvCompressor: token count per window must be divisible by "
            f"spatial grid size HW={HW}. Got window lengths: {window_lens.tolist()}"
        )
        T_per_window = window_lens // HW  # (B,)

        # Rearrange each window's KV from frame-major to spatial-major layout so that
        # position p's T tokens are contiguous:
        #   input  window i: [frame0_p0, frame0_p1, ..., frame0_pHW-1, frame1_p0, ...]
        #   output window i: [p0_f0..fT, p1_f0..fT, ..., pHW-1_f0..fT]
        kv_local_parts = []
        for i in range(B):
            s = compression_cu_seqlens[i].item()
            e = compression_cu_seqlens[i + 1].item()
            T_i = T_per_window[i].item()
            kv_i = kv[s:e].view(T_i, HW, self.hidden_size).permute(1, 0, 2).reshape(-1, self.hidden_size)
            kv_local_parts.append(kv_i)
        kv_local = torch.cat(kv_local_parts, dim=0)  # (sum_i(T_i * HW), hidden_size)

        # cu_seqlens for local cross-attention
        #   Q side: every query is its own single-token group → [0, 1, 2, ..., B*HW]
        #   K side: group (i*HW + j) corresponds to window i, position j → T_i tokens
        cu_seqlens_q_local = torch.arange(0, B * HW + 1, device=device, dtype=torch.int32)
        T_repeated = T_per_window.to(device=device, dtype=torch.int32).repeat_interleave(HW)  # (B*HW,)
        cu_seqlens_kv_local = torch.cat([
            torch.zeros(1, device=device, dtype=torch.int32),
            T_repeated.cumsum(0).to(torch.int32),
        ])

        # cu_seqlens for spatial self-attention: window i owns queries [i*HW, (i+1)*HW)
        cu_seqlens_q_self = torch.arange(0, (B + 1) * HW, step=HW, device=device, dtype=torch.int32)

        rotary_pos_emb = self._build_query_rotary_pos_emb(self.compress_image_w, self.compress_image_h)

        # 1-D temporal rotary on the keys.  kv_local is position-major within each
        # window ([p0_f0..f{T-1}, p1_f0.., ...]), so the per-key frame index is
        # arange(T_i) tiled HW times per window.  Queries stay at temporal slot 0
        # (rotary_pos_emb_q=None in the layer), encoding the relative offset -t.
        kv_t = torch.cat([
            torch.arange(int(t), device=device).repeat(HW) for t in T_per_window.tolist()
        ])
        inv_freq = self.cross_rotary.inv_freq
        cross_rotary_kv = _build_factorized_rotary(inv_freq, [kv_t], [inv_freq.shape[0]])

        for layer in self.layers:
            query = layer(query, kv_local, cu_seqlens_q_local, cu_seqlens_kv_local, cu_seqlens_q_self, rotary_pos_emb,
                          cross_rotary_kv)

        return query


class CompressorDecoderLayer(nn.Module):
    """
    Fixed AE-decoder block.  Doubles the temporal length, then refines it with a
    temporal + spatial transformer.  The structure is hard-wired (four steps, in
    this order):

        x = conv_up(x) + copy2x(x)        # T → 2T  (learnable ConvTranspose3d main + NN-copy residual)
        x = x + temporal_attn(LN(x))      # per spatial position, self-attn over the 2T frames
        x = x + spatial_attn(LN(x))       # per frame, self-attn over the HW positions
        x = x + mlp(LN(x))                # position-wise feed-forward

    Because every layer doubles T, the number of layers is fixed by the target
    frame count (``log2(max_output_frames)``) and is not configurable — see
    ``CompressorDecoder``.

    Tensors flow as the channel-last 5-D layout ``(B, T, H, W, dim)``.  Each
    attention sub-layer flattens to the packed ``(N, dim)`` form flash-attention
    expects: *temporal-major* (each ``(b, h, w)``'s ``T`` frames contiguous) for the
    temporal attention, *frame-major* (each ``(b, t)``'s ``HW`` positions contiguous)
    for the spatial attention.  RoPE is shared from the parent decoder: a 1-D
    temporal table (rebuilt per layer for the current ``T``) and a constant 2-D
    spatial table.
    """

    def __init__(self, config):
        super().__init__()
        dim = config.hidden_size
        self.embed_dim = dim
        self.upsample_factor = 2  # every decoder layer doubles the temporal length

        # ── Temporal 2x upsample (T → 2T) ──
        #   MAIN path  : a learnable depthwise ConvTranspose3d (temporal stride 2,
        #     kernel 2 so T_out == 2*T_in exactly) + pointwise 1x1x1 channel mix. The
        #     two output frames of each input frame come from DIFFERENT kernel taps, so
        #     they carry different content — this is what breaks the temporal symmetry
        #     and lets the decoder reconstruct motion.
        #   RESIDUAL path: the parameter-free nearest-neighbour copy (repeat_interleave),
        #     an identity highway at init.
        #   The conv always runs in fp32 (3-D (transpose-)convs are unreliable in
        #     fp16/bf16 on CUDA); see _upsample_conv_fp32.
        self.layer_norm_conv = LayerNorm(dim, eps=config.layer_norm_eps)
        self.depthwise = nn.ConvTranspose3d(
            dim, dim,
            kernel_size=(2, 3, 3),
            stride=(2, 1, 1),
            padding=(0, 1, 1),
            groups=dim,
        )
        self.pointwise = nn.Conv3d(dim, dim, kernel_size=1)
        self.act = GELUTanh()

        # ── Temporal self-attention (each spatial position over its T frames) ──
        self.temporal_attn = selfFlashAttention(
            embed_dim=dim,
            n_head=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=False,
        )
        self.layer_norm_temporal = LayerNorm(dim, eps=config.layer_norm_eps)

        # ── Spatial self-attention (each frame over its HW positions) ──
        self.spatial_attn = selfFlashAttention(
            embed_dim=dim,
            n_head=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=False,
        )
        self.layer_norm_spatial = LayerNorm(dim, eps=config.layer_norm_eps)

        # ── MLP ──
        self.layer_norm_mlp = LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = mlp(hidden_size=dim, intermediate_size=config.intermediate_size)

    def _upsample_conv_fp32(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, dim) → (B, 2T, H, W, dim).  3-D (transpose-)convs are
        # numerically unreliable in fp16/bf16 on CUDA, so the whole conv sub-layer runs
        # in fp32: disable autocast and feed fp32 activations.  The conv params are kept
        # in fp32 (AE pretraining uses AMP autocast, which leaves master weights in
        # fp32), so the op runs entirely in fp32; the result is cast back.
        out_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = self.layer_norm_conv(x.float())              # (B, T, H, W, dim)
            h = h.permute(0, 4, 1, 2, 3).contiguous()        # (B, dim, T, H, W)
            h = self.depthwise(h)                            # (B, dim, 2T, H, W)
            h = self.pointwise(h)
            h = self.act(h)
            h = h.permute(0, 2, 3, 4, 1).contiguous()        # (B, 2T, H, W, dim)
        return h.to(out_dtype)

    def forward(self, x: torch.Tensor, spatial_rotary: torch.Tensor,
                temporal_rotary_module) -> torch.Tensor:
        # x: (B, T, H, W, dim)
        B, T, H, W, dim = x.shape
        device = x.device

        # 1. Temporal 2x upsample (T → 2T): learnable ConvTranspose3d as the MAIN path
        #    (its per-frame-distinct kernel taps make the two new frames differ, breaking
        #    the temporal symmetry) + parameter-free nearest-neighbour copy as the
        #    RESIDUAL skip.  Both branches read the original (pre-upsample) x.
        x = x.repeat_interleave(2, dim=1) + self._upsample_conv_fp32(x)
        T = T * 2
        HW = H * W

        # 2. Temporal self-attention: temporal-major pack so each (b, h, w)'s T frames
        #    are contiguous; 1-D temporal RoPE (arange(T)) shared across all positions.
        temporal_rotary = temporal_rotary_module(T)                       # (T, head_dim // 2)
        xt = x.permute(0, 2, 3, 1, 4).reshape(B * HW * T, dim)
        cu_t = torch.arange(0, B * HW * T + 1, step=T, device=device, dtype=torch.int32)
        xt = xt + self.temporal_attn(self.layer_norm_temporal(xt), cu_t, temporal_rotary)
        x = xt.view(B, H, W, T, dim).permute(0, 3, 1, 2, 4).contiguous()  # → (B, T, H, W, dim)

        # 3. Spatial self-attention: frame-major pack so each (b, t)'s HW positions are
        #    contiguous; 2-D spatial RoPE shared across all frames.
        xs = x.reshape(B * T * HW, dim)
        cu_s = torch.arange(0, B * T * HW + 1, step=HW, device=device, dtype=torch.int32)
        xs = xs + self.spatial_attn(self.layer_norm_spatial(xs), cu_s, spatial_rotary)

        # 4. MLP.
        xs = xs + self.mlp(self.layer_norm_mlp(xs))
        return xs.view(B, T, H, W, dim)


class CompressorDecoder(nn.Module):
    """
    AE decoder: a stack of fixed ``CompressorDecoderLayer`` blocks that expand the
    single compressed frame back to ``max_output_frames`` frames.

    Each layer doubles the temporal length (upsample 2x → temporal attention →
    spatial attention → MLP), so the seed frame grows 1 → 2 → 4 → … .  The number
    of layers is therefore **fixed by the output length**: it is exactly
    ``log2(max_output_frames)`` and is *not* configurable.  ``max_output_frames``
    must be a power of two.

    Input  : ``compressed_tokens`` (B * HW, hidden_size)   — one HW grid per sample.
    Output : (B * max_output_frames * HW, hidden_size)     — frame-major (B, T, H, W).

    Parameters
    ----------
    config : Videollama3TokenCompressorConfig
    max_output_frames : int  (power of two, default 8)
    """

    def __init__(self, config, max_output_frames: int = 8):
        super().__init__()
        H   = config.compress_image_h
        W   = config.compress_image_w
        dim = config.hidden_size
        head_dim = dim // config.num_attention_heads

        assert max_output_frames >= 1 and (max_output_frames & (max_output_frames - 1)) == 0, (
            f"max_output_frames must be a power of two (each decoder layer doubles the "
            f"temporal length); got {max_output_frames}."
        )

        self.hidden_size       = dim
        self.H                 = H
        self.W                 = W
        self.HW                = H * W
        self.max_output_frames = max_output_frames
        self.num_layers        = max_output_frames.bit_length() - 1  # log2(max_output_frames)

        # Shared RoPE: 2-D spatial (h/w bands) for the spatial attention and 1-D
        # temporal for the temporal attention.  Each table's last dim is head_dim//2,
        # as apply_rotary_pos_emb_vision expects.
        self.spatial_rotary  = VisionRotaryEmbedding(dim=head_dim // 2)
        self.temporal_rotary = VisionRotaryEmbedding(dim=head_dim)

        self.layers = nn.ModuleList([
            CompressorDecoderLayer(config) for _ in range(self.num_layers)
        ])

    # ------------------------------------------------------------------
    def _build_spatial_rotary_pos_emb(self) -> torch.Tensor:
        return _build_2d_rotary_pos_emb(self.spatial_rotary, self.W, self.H)

    # ------------------------------------------------------------------
    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_tokens : (B * HW, hidden_size)
        Returns:
            (B * max_output_frames * HW, hidden_size)  — frame-major (B, T, H, W).
        """
        B   = compressed_tokens.shape[0] // self.HW
        dim = self.hidden_size

        spatial_rotary = self._build_spatial_rotary_pos_emb()

        # Seed the temporal axis with the single compressed frame:
        # (B * HW, dim) → (B, 1, H, W, dim)
        x = compressed_tokens.view(B, 1, self.H, self.W, dim)

        for layer in self.layers:
            x = layer(x, spatial_rotary, self.temporal_rotary)

        # (B, T, H, W, dim) → frame-major (B*T*HW, dim)
        T = x.shape[1]
        assert T == self.max_output_frames, (T, self.max_output_frames)
        return x.reshape(B * T * self.HW, dim)  # (B * max_output_frames * HW, hidden_size)


def _load_flat_compressor_state_dict(path: str) -> dict:
    """Read a `transformer_decoder_flat` (qbase) state dict from either

    * a plain ``.pt`` / ``.bin`` file (bare compressor, or ``{"compressor"|"state_dict": ...}``), or
    * an HF checkpoint dir — every ``*.safetensors`` shard is scanned for
      ``…token_compressor.<k>`` keys (but NOT ``…token_compressor.stage2.<k>``).

    Keys are returned relative to the compressor module (leading
    ``model.token_compressor.`` / ``token_compressor.`` / ``stage1.`` stripped), ready
    for ``TransformerDecoderFlatCompressor.load_state_dict(..., strict=False)``.
    """
    import os

    def _strip(k: str) -> str:
        for pref in ("model.token_compressor.", "token_compressor.", "stage1."):
            if k.startswith(pref):
                k = k[len(pref):]
        return k

    if os.path.isdir(path):
        from safetensors import safe_open
        shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if not shards:
            raise FileNotFoundError(f"no *.safetensors under {path}")
        sd = {}
        for shard in shards:
            with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if ".token_compressor." in k and ".token_compressor.stage2." not in k:
                        sd[_strip(k)] = f.get_tensor(k)
        if not sd:
            raise KeyError(f"no *.token_compressor.* tensors in the shards under {path}")
        return sd

    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        obj = obj.get("compressor", obj.get("state_dict", obj))
    return {_strip(k): v for k, v in obj.items()}


class TwoStageCompressor(nn.Module):
    """Stage-1 per-segment query compressor + Stage-2 Mamba-2 segment fold.

    ``compress_windows(kv, compression_cu_seqlens, grid_hws)`` compresses each window
    (= one readout unit / one ``compression_part``) to ``M`` tokens::

        window's  T x h x w  input tokens
          └ split into ceil(T / frames_per_segment) segments (<= frames_per_segment frames each)
              └ Stage-1 flat compressor, per segment  -> N_u x K tokens
                  └ Stage-2 fold (fresh state per window) -> M tokens

    Output is ``(sum_windows M, hidden)`` in input-window order, so
    ``compress_visual_tokens_with_compressor`` scatters it exactly like the other
    compressors' output. ``output_hw_for`` returns ``(1, M)``.

    Stage-1 (``.stage1``, a ``TransformerDecoderFlatCompressor``) is warm-started from
    the pretrained qbase and, for Stage-2a, frozen via ``freeze_stage1()`` — its
    params get ``requires_grad=False`` and it is pinned to ``eval()`` (no dropout, no
    Option-B distr-loss stash) even when the parent is in ``train()``. Only
    ``.stage2`` (the ``SegmentAggregator``) trains.
    """

    def __init__(self, config):
        super().__init__()
        self.stage1 = TransformerDecoderFlatCompressor(config)
        self.tokens_per_segment = int(getattr(config, "num_queries", 64))                       # K
        self.n_summary_tokens = int(getattr(config, "stage2_n_summary_tokens", self.tokens_per_segment))  # M
        self.frames_per_segment = int(getattr(config, "stage2_frames_per_segment", 4))
        hidden = int(config.hidden_size)
        agg_cfg = SegmentAggregatorConfig(
            d_input=hidden,
            d_output=hidden,
            d_model=int(getattr(config, "stage2_d_model", 1024)),
            tokens_per_segment=self.tokens_per_segment,
            n_summary_tokens=self.n_summary_tokens,
            n_layers=int(getattr(config, "stage2_n_layers", 4)),
            d_state=int(getattr(config, "stage2_d_state", 128)),
            headdim=int(getattr(config, "stage2_headdim", 64)),
            ngroups=int(getattr(config, "stage2_ngroups", 1)),
            d_conv=int(getattr(config, "stage2_d_conv", 4)),
            expand=int(getattr(config, "stage2_expand", 2)),
            chunk_size=int(getattr(config, "stage2_chunk_size", 128)),
            mlp_ratio=float(getattr(config, "stage2_mlp_ratio", 0.0)),
            dropout=float(getattr(config, "stage2_dropout", 0.0)),
            input_norm=bool(getattr(config, "stage2_input_norm", True)),
            time_embed=str(getattr(config, "stage2_time_embed", "index_sincos")),
        )
        self.stage2 = SegmentAggregator(agg_cfg)
        self.stage1_frozen = False

    # -- API parity with the single-stage compressors -------------------------
    def output_hw_for(self, h: int, w: int):
        return 1, self.n_summary_tokens

    def freeze_stage1(self):
        for p in self.stage1.parameters():
            p.requires_grad_(False)
        self.stage1.eval()
        self.stage1_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.stage1_frozen:
            self.stage1.eval()
        return self

    def _segment_cu_seqlens(self, n_frames: int, hw: int, device) -> "tuple[torch.Tensor, int]":
        fps = self.frames_per_segment
        n_seg = max(1, (n_frames + fps - 1) // fps)
        lens = [min(fps, n_frames - s * fps) * hw for s in range(n_seg)]
        cu = torch.zeros(n_seg + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(lens, dtype=torch.int32, device=device).cumsum(0)
        return cu, n_seg

    def compress_windows(self, kv, compression_cu_seqlens, grid_hws):
        if kv.dim() == 3:
            kv = kv.squeeze(0)                                   # (total_tokens, hidden)
        cu = compression_cu_seqlens.to(device=kv.device, dtype=torch.long).tolist()
        W = len(cu) - 1
        assert grid_hws is not None and len(grid_hws) == W, (
            f"TwoStageCompressor: need one (h, w) per window ({W}), got "
            f"{None if grid_hws is None else len(grid_hws)}"
        )
        outs = []
        for i in range(W):
            win = kv[cu[i]:cu[i + 1]]                            # (T*h*w, hidden)
            h, w = int(grid_hws[i][0]), int(grid_hws[i][1])
            hw = h * w
            assert win.shape[0] % hw == 0, (
                f"TwoStageCompressor: window {i} has {win.shape[0]} tokens, not a multiple "
                f"of h*w={hw}"
            )
            n_frames = win.shape[0] // hw
            seg_cu, n_seg = self._segment_cu_seqlens(n_frames, hw, kv.device)
            seg_grid = [(h, w)] * n_seg
            ctx = torch.no_grad() if self.stage1_frozen else contextlib.nullcontext()
            with ctx:
                k_tok = self.stage1(win, seg_cu, seg_grid)       # (n_seg * K, hidden)
            k_tok = k_tok.reshape(1, n_seg, self.tokens_per_segment, -1)
            m_tok = self.stage2(k_tok)                           # (1, M, hidden)
            outs.append(m_tok[0])
        return torch.cat(outs, dim=0)                            # (W * M, hidden)

    def forward(self, kv, compression_cu_seqlens, grid_hws=None, kept_idx=None):
        return self.compress_windows(kv, compression_cu_seqlens, grid_hws)

    def load_stage1_pretrained(self, path: str, verbose: bool = True):
        sd = _load_flat_compressor_state_dict(path)
        missing, unexpected = self.stage1.load_state_dict(sd, strict=False)
        if verbose:
            print(
                f"[TwoStageCompressor] loaded stage-1 qbase from {path} "
                f"({len(sd)} tensors; missing={len(missing)}, unexpected={len(unexpected)})"
            )
            if unexpected:
                print(f"[TwoStageCompressor]   unexpected: {sorted(unexpected)[:8]}")
        return missing, unexpected


from transformers import PretrainedConfig

class Videollama3TokenCompressorConfig(PretrainedConfig):
    model_type = "videollama3_token_compressor"

    # compressor_type: "transformer_decoder" | "transformer_decoder_flat" |
    #                  "local_attn_conv" | "siglip_ae"
    #   "siglip_ae" is a faithful port of Video-XL-Pro's SiglipAE (see dts.py) — unlike
    #   the other two, its depth is fixed at construction from `window_size`
    #   (log2(window_size) stride-2 Conv3d stages), so every compression window given
    #   to it must contain exactly `window_size` frames; `window_size` must be set
    #   explicitly (a power of two >= 2) when selecting this type.
    #   "transformer_decoder_flat" outputs `num_queries` flat tokens (sin/cos
    #   positional encoding over their flat index) instead of the 2-D
    #   compress_image_h x compress_image_w grid the other transformer_decoder variant
    #   and local_attn_conv use — see TransformerDecoderFlatCompressor's docstring.
    def __init__(
        self,
        compressor_type="transformer_decoder",
        hidden_size=1152,
        intermediate_size=4304,
        num_layers=8,
        num_attention_heads=4,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        compress_image_w=16,
        compress_image_h=16,
        window_size=1,
        num_queries=32,
        token_prune_ratio=0.0,
        token_prune_min_tokens=0,
        match_encoder_scale=False,
        distr_loss_weight=0.0,
        distr_loss_max_ref_tokens=4096,
        # Stage-2 fold (compressor_type "…+mamba" -> TwoStageCompressor). K is
        # num_queries (the stage-1 qbase's query count); these size .stage2, the
        # SegmentAggregator. See docs/two_stage_compression_design.md.
        stage2_n_summary_tokens=64,     # M (tie to K unless told otherwise)
        stage2_frames_per_segment=4,    # frames per stage-1 segment, clamp [1, 8]
        stage2_d_model=1024,
        stage2_n_layers=4,
        stage2_d_state=128,
        stage2_headdim=64,
        stage2_ngroups=1,
        stage2_d_conv=4,
        stage2_expand=2,
        stage2_chunk_size=128,
        stage2_mlp_ratio=0.0,
        stage2_dropout=0.0,
        stage2_input_norm=True,
        stage2_time_embed="index_sincos",
        **kwargs,
    ):

        super().__init__(**kwargs)
        self.compressor_type = compressor_type
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.compress_image_w = compress_image_w
        self.compress_image_h = compress_image_h
        self.window_size = window_size
        self.num_queries = num_queries
        # Training-only common-component KV pruning; only transformer_decoder /
        # transformer_decoder_flat read these (0 = disabled).
        self.token_prune_ratio = token_prune_ratio
        self.token_prune_min_tokens = token_prune_min_tokens
        # Option A: affine-match the compressor output's per-dim mean/std onto the
        # frozen-encoder tokens + a learnable per-channel gamma/beta (applies at
        # train AND inference). Option B: CORAL-style distribution-match aux loss
        # (weight; the trainer adds it to CE). transformer_decoder* only.
        self.match_encoder_scale = match_encoder_scale
        self.distr_loss_weight = distr_loss_weight
        self.distr_loss_max_ref_tokens = distr_loss_max_ref_tokens
        # Stage-2 fold knobs (only read when compressor_type endswith "+mamba").
        self.stage2_n_summary_tokens = stage2_n_summary_tokens
        self.stage2_frames_per_segment = stage2_frames_per_segment
        self.stage2_d_model = stage2_d_model
        self.stage2_n_layers = stage2_n_layers
        self.stage2_d_state = stage2_d_state
        self.stage2_headdim = stage2_headdim
        self.stage2_ngroups = stage2_ngroups
        self.stage2_d_conv = stage2_d_conv
        self.stage2_expand = stage2_expand
        self.stage2_chunk_size = stage2_chunk_size
        self.stage2_mlp_ratio = stage2_mlp_ratio
        self.stage2_dropout = stage2_dropout
        self.stage2_input_norm = stage2_input_norm
        self.stage2_time_embed = stage2_time_embed

def build_token_compressor(config):
    compressor = getattr(config, 'token_compressor_config', None)
    if compressor is None:
        compressor = getattr(config, 'token_compressor', None)
    if compressor is None:
        return None
    if isinstance(compressor, Videollama3TokenCompressorConfig):
        pass
    elif hasattr(compressor, "to_dict"):
        compressor = Videollama3TokenCompressorConfig(**compressor.to_dict())
    elif isinstance(compressor, dict):
        compressor = dict(compressor)
        # Normalize legacy keys to canonical names before constructing config.
        if "compress_w" in compressor and "compress_image_w" not in compressor:
            compressor["compress_image_w"] = compressor.pop("compress_w")
        if "compress_h" in compressor and "compress_image_h" not in compressor:
            compressor["compress_image_h"] = compressor.pop("compress_h")
        if "hidden_size" not in compressor:
            compressor["hidden_size"] = config.hidden_size
        if "num_attention_heads" not in compressor:
            compressor["num_attention_heads"] = config.num_attention_heads
        compressor = Videollama3TokenCompressorConfig(**compressor)
    if isinstance(compressor, Videollama3TokenCompressorConfig):
        ct = compressor.compressor_type
        # "transformer_decoder_flat+mamba" -> Stage-1 flat qbase + Stage-2 Mamba-2
        # fold. Checked BEFORE the substring test below (which would otherwise pick
        # the grid variant).
        if ct.endswith("+mamba"):
            base = ct[: -len("+mamba")]
            assert base == "transformer_decoder_flat", (
                f"two-stage compressor only supports a 'transformer_decoder_flat' base, got {base!r}"
            )
            return TwoStageCompressor(config=compressor)
        if ct == "transformer_decoder_flat":
            return TransformerDecoderFlatCompressor(config=compressor)
        if "transformer_decoder" in ct:
            return TransformerDecoderCompressor(config=compressor)
        if ct == "local_attn_conv":
            return LocalAttnConvCompressor(config=compressor)
        if ct == "siglip_ae":
            return SiglipAECompressor(config=compressor)
    raise ValueError(f"Unknown token compressor type: {getattr(compressor, 'compressor_type', None)}")
