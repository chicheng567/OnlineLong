import contextlib
import math

from torch.nn import LayerNorm
import torch
from transformers.activations import GELUTanh
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from .videollama3_encoder.modeling_videollama3_encoder import VisionRotaryEmbedding, apply_rotary_pos_emb_vision
from .siglip_ae import SiglipAECompressor
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


def adaptive_segment_count(n_frames: int, target_frames: int = 4) -> int:
    """N = n_frames // target_frames + 1 — the Phase-1 segment count. A pure
    function of the frame count, so the collator / arch can reserve exactly
    ``N * num_queries`` placeholder tokens before the encoder runs. Kept in one
    place so ``output_len_for`` and the segmenter cannot disagree."""
    n_frames = int(n_frames)
    if n_frames <= 1:
        return 1
    return max(1, min(n_frames // max(1, int(target_frames)) + 1, n_frames))


def adaptive_segment_lengths(
    per_frame_feat: torch.Tensor,      # (T, C) frozen-encoder per-frame mean feature
    target_frames: int = 4,
    force_every: int = 8,
    sample_tau: float = 0.0,           # >0 -> Gumbel-top-k draw (per-epoch augmentation)
    generator=None,
) -> "list[int]":
    """Fixed-count, adaptively-placed segmentation (design doc §4 Phase 1, validated
    in ``eval_ablation/segmenter_validate.py``).

    Returns segment frame-lengths, ``sum == T`` and ``len == adaptive_segment_count``.
    Boundaries = a forced cut every ``force_every`` frames + the remaining budget on
    the largest ``1 - cos(f_i, f_{i-1})`` positions. Every length lands in
    ``[1, force_every]``.
    """
    T = int(per_frame_feat.shape[0])
    if T <= 1:
        return [T] if T == 1 else []
    N = adaptive_segment_count(T, target_frames)
    n_cuts = N - 1
    if n_cuts <= 0:
        return [T]

    f = torch.nn.functional.normalize(per_frame_feat.float(), dim=-1)
    diff = 1.0 - (f[1:] * f[:-1]).sum(-1)                     # (T-1,)
    forced = [c for c in range(force_every, T, force_every)]
    forced_set = set(forced)
    cuts = set(forced)
    budget = n_cuts - len(forced)
    if budget > 0:
        cand = [j + 1 for j in range(T - 1) if (j + 1) not in forced_set]
        scores = diff[torch.tensor([c - 1 for c in cand], device=diff.device)]
        if sample_tau and sample_tau > 0.0:
            z = (scores - scores.mean()) / (scores.std() + 1e-6)
            u = torch.rand(len(cand), generator=generator, device=scores.device).clamp_min(1e-12)
            gumbel = -torch.log(-torch.log(u))
            order = torch.argsort(-(z / sample_tau + gumbel))
        else:
            order = torch.argsort(-scores)
        for i in order[:budget].tolist():
            cuts.add(cand[i])
    elif budget < 0:                                          # only with force_every small
        keep = sorted(forced, key=lambda c: float(diff[c - 1]), reverse=True)[:n_cuts]
        cuts = set(keep)

    bounds = sorted(cuts)
    # Safety: no segment longer than force_every (a no-op for the default 4/8 config,
    # since consecutive forced cuts are exactly force_every apart).
    fixed, prev = [], 0
    for c in bounds:
        while c - prev > force_every:
            prev += force_every
            fixed.append(prev)
        fixed.append(c)
        prev = c
    while T - prev > force_every:
        prev += force_every
        fixed.append(prev)
    bounds = sorted(b for b in set(fixed) if 0 < b < T)

    lens = [bounds[0]] + [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)] + [T - bounds[-1]]
    return [int(x) for x in lens]


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
        # Phase-1 fixed-count adaptive segmenter (design doc §4 Phase 1). When on,
        # forward() subdivides one whole-video window into N segments and emits
        # N * num_queries tokens; output_len_for tells the arch the count.
        self.adaptive_segmentation = bool(getattr(config, "adaptive_segmentation", False))
        self.segment_target_frames = int(getattr(config, "segment_target_frames", 4) or 4)
        self.segment_force_every = int(getattr(config, "segment_force_every", 8) or 8)
        self.segment_sample_tau = float(getattr(config, "segment_sample_tau", 0.0) or 0.0)

    def output_hw_for(self, h: int, w: int):
        # Flat output, no 2-D grid — (1, num_queries) so callers' oh*ow arithmetic
        # (arch.py) still yields the right total token count.
        return 1, self.num_queries

    def output_len_for(self, n_frames: int, h: int, w: int) -> int:
        """Compressed token count for a window of ``n_frames`` frames — what the arch
        reserves as placeholder slots. Fixed ``num_queries`` unless the adaptive
        segmenter is on, then ``N * num_queries`` with ``N`` from
        ``adaptive_segment_count`` (a pure function of the frame count)."""
        if not self.adaptive_segmentation:
            return int(self.num_queries)
        return adaptive_segment_count(n_frames, self.segment_target_frames) * int(self.num_queries)

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

    def forward(self, kv, compression_cu_seqlens, grid_hws, kept_idx=None, seed=None,
                sample_segmentation: "bool | None" = None):
        # kv: (1, total_tokens, hidden_size) or (total_tokens, hidden_size)
        # grid_hws: required (one (h, w) per window) — this compressor has no fixed
        # compress_image_h/w to fall back on.
        # kept_idx: optional list (one per window) of flat indices into that window's
        # dense frame-major layout; kv then holds only those tokens, each keeping its
        # original (t, h, w) in the cross-RoPE.
        # seed: int|list|None. When the segmenter samples, its Gumbel-top-k draw is
        # reproducible from this; per window wi uses seed[wi]/seed. After forward,
        # ``self._last_seg_lens`` holds the chosen frame-length list per window (so
        # TwoStageCompressor can reuse the exact cut without recomputing).
        # sample_segmentation: None -> gate the per-epoch boundary jitter on
        # self.training (the Phase-1 single-stage behaviour). Pass an explicit bool
        # so a caller whose train-state differs from this module's can drive it --
        # TwoStageCompressor.freeze_stage1() pins stage1 to eval(), which must NOT
        # also silently disable the jitter the fold-variance thesis relies on.
        if kv.dim() == 3:
            kv = kv.squeeze(0)
        self._last_seg_lens = None
        # Encoder-token reference stats (Options A/B), captured before KV pruning /
        # segmentation so the target is the whole window's encoder tokens.
        ref_mean, ref_std, ref_rows = _capture_ref_stats(self, kv)
        if grid_hws is None:
            raise ValueError(
                "TransformerDecoderFlatCompressor has no 2-D output grid to fall back "
                "on — grid_hws (one (h, w) per window) is required, not optional."
            )

        # Phase-1 fixed-count adaptive segmenter: subdivide EACH input window (one
        # per video) into N_i per-segment sub-windows, then run the ordinary
        # multi-window path (Σ N_i * num_queries tokens). N_i matches
        # output_len_for(T_i) by construction (both via adaptive_segment_count), so
        # the arch's per-part placeholder count lines up. Handles the packed
        # multi-video batch (per_device_train_batch_size > 1) too.
        if self.adaptive_segmentation and kept_idx is None:
            assert kept_idx is None, "adaptive_segmentation is incompatible with KV token pruning"
            cu_in = compression_cu_seqlens.to("cpu").tolist()
            W = len(cu_in) - 1
            assert len(grid_hws) == W, (
                f"adaptive_segmentation: {W} windows but {len(grid_hws)} grid_hws"
            )
            _do_sample = self.training if sample_segmentation is None else bool(sample_segmentation)
            tau = self.segment_sample_tau if _do_sample else 0.0
            new_cu = [0]
            new_grid = []
            seg_lens_per_window = []
            C = kv.size(-1)
            for wi in range(W):
                a, b = int(cu_in[wi]), int(cu_in[wi + 1])
                hw = int(grid_hws[wi][0]) * int(grid_hws[wi][1])
                span = b - a
                assert span % hw == 0, (
                    f"adaptive_segmentation: window {wi} has {span} tokens, not a multiple of h*w={hw}"
                )
                gen = None
                if tau > 0.0:
                    gen = torch.Generator(device=kv.device)
                    if seed is None:
                        gen.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
                    else:
                        sv = seed[wi] if isinstance(seed, (list, tuple)) else seed
                        gen.manual_seed((int(sv) * 1_000_003 + wi) % (2 ** 63 - 1))
                with torch.no_grad():
                    per_frame = kv[a:b].view(span // hw, hw, C).float().mean(1)
                    seg_lens = adaptive_segment_lengths(
                        per_frame, self.segment_target_frames, self.segment_force_every, tau, gen
                    )
                seg_lens_per_window.append([int(x) for x in seg_lens])
                for L in seg_lens:
                    new_cu.append(new_cu[-1] + L * hw)
                    new_grid.append(grid_hws[wi])
            self._last_seg_lens = seg_lens_per_window
            compression_cu_seqlens = torch.tensor(
                new_cu, device=kv.device, dtype=compression_cu_seqlens.dtype
            )
            grid_hws = new_grid

        B = compression_cu_seqlens.size(0) - 1

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

    ``compress_windows(kv, compression_cu_seqlens, grid_hws, retained_counts, seed)``
    takes one **whole-video** window per ``compression_part`` and splits it, model
    side, into content-adaptive segments and then into ``U`` content-adaptive
    readout **units**::

        window's  T x h x w  input tokens
          └ adaptive_segment_lengths  -> N segments (1..8 frames each, forced-8 + top-diff)
              └ Stage-1 flat compressor, per segment -> N x K tokens
                  └ U-1 unit boundaries at largest inter-segment feature diff (+Gumbel)
                      └ per unit u (N_u segments): SegmentAggregator fold -> M readout tokens
                          + r_u retained segments' raw K tokens, merged by RoPE slot

    Returns ``(compressed, unit_meta)``:

      * ``compressed`` : ``(sum_u n_out_u, hidden)``, unit-major. Each unit's rows are
        emitted in ascending RoPE-slot order; when ``r_u > 0`` (Phase 3) the
        retained-K rows are physically interleaved among the M readout rows at that
        order and ``pos_offsets`` carries the same order. Phase 2 runs ``r_u == 0``
        so a unit is just its M readout rows (``n_out_u = M``).
      * ``unit_meta``  : list of dicts, one per unit across all windows in window
        order, ``{a_frame, b_frame, n_out, pos_offsets (LongTensor[n_out], slot
        units relative to unit start), unit_span (= N_u*K), window}``. The arch uses
        it for the ``Time:`` range string, the per-unit placeholder block size and
        the strided ``position_ids``.

    ``retained_counts`` is a list aligned to windows; ``retained_counts[w]`` is the
    per-unit ``r_u`` list (its length == U for that window). ``None`` -> one unit per
    window, ``r_u = 0`` (whole-video fold; used by the feature probes). The
    ``r_u``-shortest-segment pick + slot interleave stay guarded behind ``r_u > 0``
    and are inert in Phase 2 (retained is a Phase-3 mechanism).

    ``qbase_only`` is a list of bools aligned to windows; ``qbase_only[w]`` routes
    window ``w`` straight through stage-1 (``N*K`` qbase tokens, stride-1 RoPE slots,
    no unit split, no fold) -- the Phase-2 pure-qbase replay path.

    ``seed`` (int or None): when set **and** ``self.training``, seeds the Gumbel-top-k
    draws for the segment and unit boundaries and the retained pick, so a resume
    reproduces the partition. ``None`` / eval -> deterministic top-diff.

    Stage-1 (``.stage1``, a ``TransformerDecoderFlatCompressor``) is warm-started from
    the pretrained qbase; ``.stage1.adaptive_segmentation`` is forced **on** here (the
    fold consumes per-segment qbase tokens, so stage-1 must self-segment and report
    the cut via ``_last_seg_lens``). ``freeze_stage1()``
    pins it to ``eval()`` + ``requires_grad=False``. Option A/B on the fold readout
    live on this wrapper (``out_gamma``/``out_beta``, ``_last_distr_loss``,
    ``distr_loss_weight``) so the trainer's ``_compressor_distr_loss`` picks them up
    unchanged; ``_last_distr_loss`` also folds in ``.stage1``'s own term when stage-1
    is trainable.
    """

    def __init__(self, config):
        super().__init__()
        self.stage1 = TransformerDecoderFlatCompressor(config)
        # Stage-1 does the content-adaptive segmentation (same code path as Phase 1);
        # it reports the cut via ``stage1._last_seg_lens`` and this wrapper groups
        # those segments into units. The fold consumes per-segment qbase tokens, so
        # adaptive segmentation is required here — turn it on if the config left it off.
        self.stage1.adaptive_segmentation = True
        self.tokens_per_segment = int(getattr(config, "num_queries", 64))                       # K
        self.n_summary_tokens = int(getattr(config, "stage2_n_summary_tokens", self.tokens_per_segment))  # M
        # N = adaptive_segment_count(T) is a pure function of the frame count, kept in
        # sync with the collator/dataset via stage-1's own segment_target_frames.
        self.segment_target_frames = int(getattr(self.stage1, "segment_target_frames", 4))
        # RoPE-slot scale for the readout (design doc §1 / §7). "ratio" -> readout
        # token m at round(m * N_u*K / M), unit span N_u*K (Phase-1 footprint);
        # a float S -> seconds scale, unit span round((b-a)*S).
        rss = getattr(config, "stage2_rope_slot_scale", "ratio")
        self.rope_slot_scale = rss if rss == "ratio" else float(rss)
        hidden = int(config.hidden_size)
        # The fold runs BEFORE mm_projector, in the compressor's own hidden space
        # (NOT the LLM hidden): d_input == d_output == the compressor hidden, and
        # SegmentAggregator.output_proj (d_model -> d_output) is the learned
        # readout DECODE -- the M summary tokens come out of the fold's working
        # space (stage2_d_model, a bottleneck below `hidden`) and output_proj maps
        # them onto the encoder scale the SHARED frozen mm_projector expects
        # (Option A/B then pull them onto its manifold). One projector, no
        # separate fold projector; the SSM state stays its own (nheads, headdim,
        # d_state) object, distinct from this readout.
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
            final_norm=str(getattr(config, "stage2_final_norm", "rmsnorm")),
            time_embed=str(getattr(config, "stage2_time_embed", "index_sincos")),
        )
        self.stage2 = SegmentAggregator(agg_cfg)
        self.stage1_frozen = False
        # Option A (out_gamma/out_beta) + Option B (distr_loss_weight, _last_distr_loss)
        # for the fold readout — same helper the single-stage compressors use.
        _init_encoder_scale_match(self, config)

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

    # -- unit boundary placement (content-aware, model-side) -----------------
    def _place_unit_boundaries(self, seg_feat: torch.Tensor, U: int, gen) -> "list[int]":
        """Return ``[0, c_1, ..., c_{U-1}, N]`` — ``U`` contiguous segment groups.
        Cuts land on the largest inter-segment ``1 - cos`` (Gumbel-jittered when
        ``gen`` is set), spaced at least ``min_gap`` segments apart; falls back to an
        even split when the spread is too tight to place ``U-1`` valid cuts."""
        N = int(seg_feat.shape[0])
        even = [round(N * u / U) for u in range(U + 1)]
        if U <= 1 or N <= U:
            return even
        f = torch.nn.functional.normalize(seg_feat.float(), dim=-1)
        diff = 1.0 - (f[1:] * f[:-1]).sum(-1)                    # (N-1,)  j -> cut before seg j+1
        scores = (diff - diff.mean()) / (diff.std() + 1e-6)
        if gen is not None:
            u = torch.rand(scores.shape, generator=gen, device=scores.device).clamp_min(1e-12)
            scores = scores / 0.5 + (-torch.log(-torch.log(u)))
        order = torch.argsort(-scores).tolist()
        min_gap = max(1, N // (2 * U))
        chosen: "list[int]" = []
        for j in order:
            c = j + 1
            if c < min_gap or (N - c) < min_gap:
                continue
            if all(abs(c - x) >= min_gap for x in chosen):
                chosen.append(c)
                if len(chosen) == U - 1:
                    break
        if len(chosen) < U - 1:
            return even
        return [0] + sorted(chosen) + [N]

    def compress_windows(self, kv, compression_cu_seqlens, grid_hws,
                         retained_counts=None, seed=None, qbase_only=None):
        if kv.dim() == 3:
            kv = kv.squeeze(0)                                   # (total_tokens, hidden)
        cu = compression_cu_seqlens.to(device=kv.device, dtype=torch.long).tolist()
        W = len(cu) - 1
        assert grid_hws is not None and len(grid_hws) == W, (
            f"TwoStageCompressor: need one (h, w) per window ({W}), got "
            f"{None if grid_hws is None else len(grid_hws)}"
        )
        if retained_counts is None:
            retained_counts = [None] * W
        assert len(retained_counts) == W, (
            f"TwoStageCompressor: retained_counts has {len(retained_counts)} entries, {W} windows"
        )
        if qbase_only is None:
            qbase_only = [False] * W
        assert len(qbase_only) == W, (
            f"TwoStageCompressor: qbase_only has {len(qbase_only)} entries, {W} windows"
        )
        K, M = self.tokens_per_segment, self.n_summary_tokens
        _has_seed = seed is not None and not (isinstance(seed, (list, tuple)) and len(seed) == 0)
        # Per-epoch re-partitioning is governed by THIS wrapper's train state + the
        # qbase's segment_sample_tau, NOT stage1.training -- freeze_stage1() pins
        # stage1 to eval() and that must not also disable the segment/unit jitter
        # (docs/two_stage_compression_design.md §4 Phase 2). tau == 0 -> a true
        # fixed-partition ablation: segments AND unit boundaries deterministic.
        seg_tau = float(getattr(self.stage1, "segment_sample_tau", 0.0) or 0.0)
        sample_partition = bool(self.training and seg_tau > 0.0)
        do_gumbel = bool(sample_partition and _has_seed)

        # -- Stage-1: one call over ALL windows; it does the content-adaptive
        #    segmentation and reports the cut via ``_last_seg_lens`` (list per window).
        cc = compression_cu_seqlens.to(device=kv.device, dtype=torch.int32).contiguous()
        ctx = torch.no_grad() if self.stage1_frozen else contextlib.nullcontext()
        with ctx:
            k_all = self.stage1(kv, cc, list(grid_hws),
                                seed=(seed if do_gumbel else None),
                                sample_segmentation=sample_partition)
        seg_lens_all = self.stage1._last_seg_lens
        assert seg_lens_all is not None and len(seg_lens_all) == W, (
            "TwoStageCompressor needs stage1.adaptive_segmentation on (it reports _last_seg_lens)"
        )
        k_all = k_all.reshape(-1, K, k_all.shape[-1])               # (Σ N_i, K, hidden)

        distr_terms: "list[torch.Tensor]" = []
        seg_row = 0                                              # running segment offset into k_all

        # -- pass 1: per-window bookkeeping + unit boundaries; DEFER the actual
        #    fold call so every unit in this call (across all W windows) can be
        #    folded in ONE batched SegmentAggregator call instead of one per unit.
        #    ("items" keeps window-then-unit order -- outs/unit_meta must end up
        #    in exactly that order, see compress_visual_tokens_with_compressor's
        #    contiguous per-window replace_mask slicing.)
        items: "list[tuple]" = []       # ("ready", rows, meta) | ("fold", task_dict)
        need_seg_secs = self.stage2.cfg.time_embed == "rel_gap_mlp"

        for wi in range(W):
            win = kv[cu[wi]:cu[wi + 1]]                          # (T*h*w, hidden)
            h, w = int(grid_hws[wi][0]), int(grid_hws[wi][1])
            hw = h * w

            seg_lens = seg_lens_all[wi]
            N = len(seg_lens)
            fstart = [0]
            for L in seg_lens:
                fstart.append(fstart[-1] + L)                    # segment i -> frames [fstart[i], fstart[i+1])
            k_tok = k_all[seg_row: seg_row + N]                  # (N, K, hidden) — stage1's qbase output
            seg_row += N

            # -- qbase-only passthrough (Phase-1 layout: no units, no fold) ----
            # A whole-video replay window that must exercise the raw-qbase ->
            # mm_projector -> LLM path, so the (unfrozen) projector and, once
            # unfrozen, the qbase stay anchored to the encoder-scale qbase
            # manifold and cannot overfit the fold-readout statistics. Emits the
            # N*K qbase tokens at stride-1 RoPE slots as one unit_meta entry; no
            # SegmentAggregator call. docs/two_stage_compression_design.md
            # §4 Phase 2 (qbase-only replay) / §5 item 12.
            if qbase_only[wi]:
                rows = k_tok.reshape(N * K, k_tok.shape[-1])
                items.append(("ready", rows, {
                    "a_frame": 0,
                    "b_frame": int(fstart[-1]),
                    "n_out": int(N * K),
                    "pos_offsets": torch.arange(N * K, dtype=torch.long, device=kv.device),
                    "unit_span": int(N * K),
                    "window": wi,
                }))
                continue

            gen = None
            if do_gumbel:
                sv = seed[wi] if isinstance(seed, (list, tuple)) else seed
                gen = torch.Generator(device=kv.device)
                gen.manual_seed((int(sv) * 1_000_003 + wi + 7) % (2 ** 63 - 1))

            # -- unit boundaries from the SEGMENT (qbase) features ------------
            # Group segments by what the fold will actually consume: each
            # segment's K stage-1 qbase tokens mean-pooled to one vector. NOT the
            # raw frozen-encoder feature (whose ~0.94 common component dominates
            # the cosine and whose geometry the fold never sees).
            # docs/two_stage_compression_design.md §4 Phase 2.
            rc = retained_counts[wi]
            U = 1 if rc is None else len(rc)
            U = max(1, min(U, N))
            with torch.no_grad():
                seg_feat = k_tok.float().mean(1)                 # (N, hidden) — qbase tokens / segment
                sf = torch.nn.functional.normalize(seg_feat, dim=-1)
                seg_diff = torch.cat([                            # (N,)  seg_diff[i] = 1 - cos(i, i-1); [0] = sentinel
                    seg_feat.new_tensor([1.0e4]),
                    1.0 - (sf[1:] * sf[:-1]).sum(-1),
                ])
            bounds = self._place_unit_boundaries(seg_feat, U, gen)

            for u in range(U):
                s_a, s_b = bounds[u], bounds[u + 1]
                N_u = s_b - s_a
                k_u = k_tok[s_a:s_b]                             # (N_u, K, hidden)
                # rel_gap_mlp time embed: per-segment (gap_from_prev_start, duration)
                # in seconds (fps = 1 -> frame count). Gap resets at the unit start
                # (each unit is an independent fold with a fresh state).
                seg_secs = None
                if need_seg_secs:
                    durs = [float(seg_lens[s_a + i]) for i in range(N_u)]
                    gaps = [0.0] + durs[:-1]
                    seg_secs = k_u.new_tensor(list(zip(gaps, durs)))            # (N_u, 2)
                items.append(("fold", {
                    "wi": wi, "u": u, "win": win, "h": h, "w": w, "hw": hw, "fstart": fstart,
                    "seg_lens": seg_lens, "seg_diff": seg_diff, "rc": rc, "gen": gen,
                    "s_a": s_a, "s_b": s_b, "N_u": N_u, "k_u": k_u, "seg_secs": seg_secs,
                }))

        # -- pass 2: ONE padded + masked batched fold call for every "fold" item
        #    (arbitrary N_u per item -- front-pad to N_max, mask the padding to a
        #    state no-op; see Mamba2Mixer.forward / SegmentAggregator.forward).
        #    Falls back to nothing (empty tensor) when this call has no fold units
        #    at all (e.g. every window here is qbase_only).
        fold_idx = [i for i, it in enumerate(items) if it[0] == "fold"]
        if fold_idx:
            tasks = [items[i][1] for i in fold_idx]
            T = len(tasks)
            N_max = max(t["N_u"] for t in tasks)
            hidden = k_all.shape[-1]
            padded = k_all.new_zeros(T, N_max, K, hidden)
            seg_mask = torch.zeros(T, N_max, dtype=torch.bool, device=kv.device)
            secs_padded = k_all.new_zeros(T, N_max, 2) if need_seg_secs else None
            for i, t in enumerate(tasks):
                n = t["N_u"]
                padded[i, N_max - n:] = t["k_u"]
                seg_mask[i, N_max - n:] = True
                if need_seg_secs:
                    secs_padded[i, N_max - n:] = t["seg_secs"]
            m_tok_all = self.stage2(padded, segment_seconds=secs_padded,
                                    segment_valid_mask=seg_mask)               # (T, M, hidden)
            for i, t in enumerate(tasks):
                t["m_tok"] = m_tok_all[i]

        # -- pass 3: Option A/B + retained + RoPE-slot interleave, in original order.
        outs: "list[torch.Tensor]" = []
        unit_meta: "list[dict]" = []
        for kind, *rest in items:
            if kind == "ready":
                rows, meta = rest
                outs.append(rows)
                unit_meta.append(meta)
                continue

            t = rest[0]
            wi, win, h, w, hw = t["wi"], t["win"], t["h"], t["w"], t["hw"]
            fstart, seg_lens, seg_diff, rc, gen = t["fstart"], t["seg_lens"], t["seg_diff"], t["rc"], t["gen"]
            s_a, s_b, N_u, k_u = t["s_a"], t["s_b"], t["N_u"], t["k_u"]
            m_tok = t["m_tok"]                                    # (M, hidden)

            # Option A/B on the readout — ref = this unit's encoder tokens.
            need_scale = getattr(self, "match_encoder_scale", False)
            need_aux = bool(self.training and getattr(self, "distr_loss_weight", 0.0) > 0.0)
            if need_scale or need_aux:
                win_u = win[fstart[s_a] * hw: fstart[s_b] * hw]
                r_mean, r_std, r_rows = _capture_ref_stats(self, win_u)
                if need_scale and r_mean is not None:
                    m_tok = _match_encoder_scale(m_tok, r_mean, r_std, self.out_gamma, self.out_beta)
                if need_aux and r_rows is not None:
                    distr_terms.append(_distribution_match_loss(m_tok, r_rows))

            # Retained subset: r_u shortest segments (frame count), tie-break
            # -seg_diff, Gumbel over the shortest 2*r_u.
            r_u = 0 if rc is None else max(0, min(int(rc[t["u"]]), N_u))
            retained_local: "list[int]" = []
            if r_u > 0:
                loc = list(range(N_u))
                loc.sort(key=lambda i: (seg_lens[s_a + i], -float(seg_diff[s_a + i])))
                pool = loc[:min(N_u, 2 * r_u)]
                if gen is not None and len(pool) > r_u:
                    gg = torch.rand(len(pool), generator=gen, device=kv.device)
                    pick = [pool[i] for i in torch.argsort(-gg)[:r_u].tolist()]
                else:
                    pick = pool[:r_u]
                retained_local = sorted(pick)

            # Merge readout + retained-K rows by RoPE slot (unit span = N_u*K
            # for "ratio"; round((b-a)*S) for a seconds scale S).
            if self.rope_slot_scale == "ratio":
                span = N_u * K
            else:
                span = max(N_u, int(round((fstart[s_b] - fstart[s_a]) * self.rope_slot_scale)))
            entries: "list[tuple[int, torch.Tensor]]" = []
            for m in range(M):
                entries.append((int(round(m * span / M)), m_tok[m]))
            for li in retained_local:
                base = int(round(li * span / N_u))
                for kk in range(K):
                    entries.append((min(base + kk, span - 1), k_u[li, kk]))
            entries.sort(key=lambda e: e[0])
            pos_offsets = torch.tensor([e[0] for e in entries], dtype=torch.long, device=kv.device)
            unit_rows = torch.stack([e[1] for e in entries], dim=0)   # (M + r_u*K, hidden)

            outs.append(unit_rows)
            unit_meta.append({
                "a_frame": int(fstart[s_a]),
                "b_frame": int(fstart[s_b]),
                "n_out": int(unit_rows.shape[0]),
                "pos_offsets": pos_offsets,
                "unit_span": int(span),
                "window": wi,
            })

        # Option B: sum the fold-readout terms and fold in stage-1's own term
        # (present only when stage-1 is trainable and in training).
        last = torch.stack(distr_terms).mean() if distr_terms else None
        s1 = getattr(self.stage1, "_last_distr_loss", None)
        if s1 is not None:
            last = s1 if last is None else last + s1
        self._last_distr_loss = last

        compressed = torch.cat(outs, dim=0) if outs else kv.new_zeros(0, kv.shape[-1])
        return compressed, unit_meta

    def forward(self, kv, compression_cu_seqlens, grid_hws=None, kept_idx=None,
                retained_counts=None, seed=None, qbase_only=None):
        return self.compress_windows(kv, compression_cu_seqlens, grid_hws,
                                     retained_counts=retained_counts, seed=seed,
                                     qbase_only=qbase_only)

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
    #   "siglip_ae" is a faithful port of Video-XL-Pro's SiglipAE (see siglip_ae.py) — unlike
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
        # Phase-1 fixed-count adaptive segmenter (transformer_decoder_flat only).
        # When on, one whole-video compression window is subdivided model-side into
        # N = n_frames // segment_target_frames + 1 segments (a pure function of the
        # frame count, so the collator predicts the compressed length without the
        # features); boundaries land on the largest consecutive-frame encoder-feature
        # cosine distances, with a forced cut every segment_force_every frames. Each
        # segment -> num_queries qbase tokens; output is N * num_queries tokens.
        # See docs/two_stage_compression_design.md §4 Phase 1.
        adaptive_segmentation=False,
        segment_target_frames=4,
        segment_force_every=8,
        segment_sample_tau=0.0,          # >0: Gumbel-top-k boundary draw (train only)
        # Stage-2 fold (compressor_type "…+mamba" -> TwoStageCompressor). K is
        # num_queries (the stage-1 qbase's query count); these size .stage2, the
        # SegmentAggregator. See docs/two_stage_compression_design.md.
        stage2_n_summary_tokens=64,     # M (tie to K unless told otherwise)
        stage2_frames_per_segment=4,    # frames per stage-1 segment, clamp [1, 8]
        stage2_d_model=1024,            # fold working width (bottleneck below hidden); output_proj decodes back to hidden
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
        stage2_final_norm="rmsnorm",     # readout norm: rmsnorm|layernorm|scale|none
        stage2_time_embed="index_sincos",
        stage2_rope_slot_scale="ratio",  # "ratio" (readout stride N_u*K/M) or float S (slots/sec)
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
        self.adaptive_segmentation = adaptive_segmentation
        self.segment_target_frames = segment_target_frames
        self.segment_force_every = segment_force_every
        self.segment_sample_tau = segment_sample_tau
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
        self.stage2_final_norm = stage2_final_norm
        self.stage2_time_embed = stage2_time_embed
        self.stage2_rope_slot_scale = stage2_rope_slot_scale

def bake_time_tokens(config, tokenizer, max_seconds: int = 4096):
    """Pre-tokenise the pieces the arch assembles ``Time:{a}s-{b}s:`` from and stash
    them on ``config`` (``time_tok_open`` / ``time_tok_mid`` / ``time_tok_close`` /
    ``time_tok_digits``), so ``prepare_inputs_labels_for_multimodal`` needs no
    tokenizer at forward time (design doc §5 item 11). Verifies the fragment
    concatenation reproduces the full-string encoding for a spread of ``(a, b)``;
    raises if the tokenizer's digit tokens are context-dependent."""
    enc = lambda s: list(tokenizer.encode(s, add_special_tokens=False))
    max_seconds = int(max_seconds)
    config.time_tok_open = enc("Time:")
    config.time_tok_mid = enc("s-")
    config.time_tok_close = enc("s:")
    config.time_tok_digits = [enc(str(k)) for k in range(max_seconds + 1)]
    step = max(1, max_seconds // 50)
    checks = sorted({0, 1, 2, 9, 10, 11, 59, 60, 61, 99, 100, 101, 999, 1000, max_seconds}
                    | set(range(0, max_seconds + 1, step)))
    bad = []
    for a in checks:
        if a > max_seconds:
            continue
        for b in {a, min(a + 1, max_seconds), min(a + 7, max_seconds), max_seconds}:
            got = (config.time_tok_open + config.time_tok_digits[a] + config.time_tok_mid
                   + config.time_tok_digits[b] + config.time_tok_close)
            want = enc(f"Time:{a}s-{b}s:")
            if got != want:
                bad.append((a, b, got, want))
    if bad:
        raise RuntimeError(
            f"bake_time_tokens: fragment concat != full encode for {len(bad)} (a,b) pairs; "
            f"first a={bad[0][0]} b={bad[0][1]} got={bad[0][2]} want={bad[0][3]}. "
            f"This tokenizer's digit tokens are not context-independent — bake full "
            f"'Time:{{a}}s-{{b}}s:' strings on a coarser grid instead."
        )
    return config


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
