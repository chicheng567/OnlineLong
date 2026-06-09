from torch.nn import LayerNorm
from typing import List
import torch
from transformers.activations import GELUTanh
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from .videollama3_encoder.modeling_videollama3_encoder import VisionRotaryEmbedding, apply_rotary_pos_emb_vision
from torch.nn import functional as F


def _build_2d_rotary_pos_emb(rotary_pos_emb_module, w, h):
    device = rotary_pos_emb_module.inv_freq.device
    hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w).reshape(-1)
    wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1).reshape(-1)
    pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
    rotary_pos_emb_full = rotary_pos_emb_module(max(h, w))
    return rotary_pos_emb_full[pos_ids].flatten(1)


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
        # Apply rotary positional embeddings
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
        # Final norm on the compressor output (feeds the AE decoder and the LLM).
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_layers = num_layers
        self.compress_image_w = config.compress_image_w
        self.compress_image_h = config.compress_image_h
        self.compress_image_wh = self.compress_image_w * self.compress_image_h
        # Learned query tokens (small-scale init, std=0.02).
        self.query = nn.Parameter(torch.randn(1, self.compress_image_w * self.compress_image_h, config.hidden_size) * 0.02)
        self.window_size = getattr(config, "window_size", 1)

    def _build_query_rotary_pos_emb(self, w, h) -> torch.Tensor:
        return _build_2d_rotary_pos_emb(self.rotary_pos_emb, w, h)

    def _build_cross_rotary_3d(self, compression_cu_seqlens, device):
        """
        Build 3-D (t, h, w) rotary tables for the cross-attention.

        Queries: one HW grid per window, all anchored at temporal slot 0 with their
        2-D spatial coordinates.  Keys: the window's T×HW input tokens in frame-major
        order — temporal coordinate = frame index, spatial coordinate = position
        within the frame.  Returns (q_freqs, kv_freqs), each (N, head_dim // 2).
        """
        H, W, HW = self.compress_image_h, self.compress_image_w, self.compress_image_wh
        # Per-frame spatial coords, matching the _build_2d_rotary_pos_emb convention.
        hpos = torch.arange(H, device=device).unsqueeze(1).expand(-1, W).reshape(-1)  # (HW,)
        wpos = torch.arange(W, device=device).unsqueeze(0).expand(H, -1).reshape(-1)  # (HW,)

        window_lens = (compression_cu_seqlens[1:] - compression_cu_seqlens[:-1]).long()
        assert (window_lens % HW == 0).all(), (
            f"TransformerDecoderCompressor: token count per window must be divisible "
            f"by HW={HW}. Got window lengths: {window_lens.tolist()}"
        )
        T_per = (window_lens // HW).tolist()
        B = len(T_per)

        # Keys: frame-major layout [frame0_p0..p_{HW-1}, frame1_p0.., ...]
        kv_t, kv_h, kv_w = [], [], []
        for Ti in T_per:
            kv_t.append(torch.arange(Ti, device=device).repeat_interleave(HW))
            kv_h.append(hpos.repeat(Ti))
            kv_w.append(wpos.repeat(Ti))
        kv_t = torch.cat(kv_t); kv_h = torch.cat(kv_h); kv_w = torch.cat(kv_w)

        # Queries: B grids, all at t=0.
        q_t = torch.zeros(B * HW, device=device)
        q_h = hpos.repeat(B); q_w = wpos.repeat(B)

        inv_freq = self.cross_rotary.inv_freq  # (head_dim // 2,)
        D = inv_freq.shape[0]
        d_t = D // 3
        d_h = (D - d_t) // 2
        d_w = D - d_t - d_h
        dims = [d_t, d_h, d_w]
        kv_freqs = _build_factorized_rotary(inv_freq, [kv_t, kv_h, kv_w], dims)
        q_freqs = _build_factorized_rotary(inv_freq, [q_t, q_h, q_w], dims)
        return q_freqs, kv_freqs

    def forward(self, kv, compression_cu_seqlens):
        # kv: (1, total_tokens, hidden_size)
        compression_parts = compression_cu_seqlens.size(0) - 1
        if kv.dim() == 3:
            kv = kv.squeeze(0) # (total_tokens, hidden_size)
        B = compression_parts
        query = self.query.expand(B, -1, -1).contiguous().view(-1, kv.size(-1))  # (B * compress_image_wh, hidden_size)
        cu_seqlens_q = torch.arange(
            0,
            (B + 1) * self.compress_image_wh,
            step=self.compress_image_wh,
            device=kv.device,
            dtype=torch.int32,
        ).contiguous()
        compression_cu_seqlens = compression_cu_seqlens.to(device=kv.device, dtype=torch.int32).contiguous()
        rotary_pos_emb = self._build_query_rotary_pos_emb(self.compress_image_w, self.compress_image_h)
        cross_rotary_q, cross_rotary_kv = self._build_cross_rotary_3d(compression_cu_seqlens, kv.device)
        for layer in self.layers:
            query = layer(query, kv, cu_seqlens_q, compression_cu_seqlens, rotary_pos_emb,
                          cross_rotary_q, cross_rotary_kv)
        query = self.post_layernorm(query)
        return query


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
#   1. Local cross-attention  (query p attends only to the T tokens at position p)
#   2. Spatial self-attention (all H×W queries within a window communicate)
#   3. MLP
# All sub-layers use pre-norm + residual connection.
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
        # 1. Local cross-attention: query at position p sees only position p's T frame tokens.
        #    The query stays at temporal slot 0 (rotary_pos_emb_q=None → identity) while the
        #    keys carry their frame index via cross_rotary_kv, so the dot product encodes the
        #    relative offset -t and the T frames are no longer exchangeable.
        q = q + self.cross_attn(self.layer_norm1(q), kv_local, cu_seqlens_q_local, cu_seqlens_kv_local,
                                None, cross_rotary_kv)
        # 2. Spatial self-attention: queries within a window exchange spatial context.
        q = q + self.self_attn(self.layer_norm2(q), cu_seqlens_q_self, rotary_pos_emb)
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
        # Final norm on the compressor output (feeds the AE decoder and the LLM).
        self.post_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
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

        query = self.post_layernorm(query)
        return query


class CompressorDecoderLayer(nn.Module):
    """
    Transformer-encoder AE-decoder block, optionally followed by a temporal-
    upsampling 3-D conv.

    Every layer is a standard pre-norm transformer encoder block (two residual
    sub-layers):

        x = x + self_attn(LN(x))             # per-frame spatial self-attention
        x = x + mlp(LN(x))                   # position-wise feed-forward

    On the layers selected for temporal upsampling (``upsample_factor > 1``) a third
    pre-norm residual sub-layer is appended *after the MLP*:

        x = up(x) + conv(LN(x))              # depthwise ConvTranspose3d + pointwise Conv3d

    The conv's temporal stride equals ``upsample_factor`` and its kernel == factor, so
    the output length is exactly ``T_in * factor``; the residual branch is upsampled in
    time by a simple linear layer (``T → T*factor``, init = nearest-neighbour repeat)
    so the two shapes match.  When ``upsample_factor == 1`` the block is a plain
    transformer encoder layer with no conv and no temporal change (temporal mixing is
    delegated entirely to the upsample layers' conv kernels).

    ⚠️ The conv sub-layer is always run in **fp32**: 3-D (transpose-)convolutions are
    numerically unreliable in fp16/bf16 on CUDA and were observed to cause a large
    reconstruction-loss regression.  ``_upsample_conv_fp32`` disables autocast and
    feeds fp32 activations; the conv params stay fp32 (AE pretraining uses AMP
    autocast, which keeps master weights in fp32).

    All tensors are 5-D ``(B, dim, T, H, W)``; the transformer sub-layers flatten to
    frame-major ``(B*T*HW, dim)`` internally and use the shared 2-D spatial RoPE.
    """

    def __init__(self, config, upsample_factor: int = 1):
        super().__init__()
        dim = config.hidden_size
        self.embed_dim = dim
        self.upsample_factor = upsample_factor

        # ── Transformer encoder: per-frame spatial self-attention + MLP ──
        self.self_attn = selfFlashAttention(
            embed_dim=dim,
            n_head=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=False,
        )
        self.layer_norm_attn = LayerNorm(dim, eps=config.layer_norm_eps)
        self.layer_norm_mlp = LayerNorm(dim, eps=config.layer_norm_eps)
        self.mlp = mlp(hidden_size=dim, intermediate_size=config.intermediate_size)

        # ── Temporal-upsampling 3-D conv sub-layer (upsample layers only) ──
        # Depthwise ConvTranspose3d: temporal stride = factor (kernel = factor so
        # T_out == T_in * factor exactly), spatial 3x3 local mixing kept
        # resolution-preserving via padding 1.  Pointwise 1x1x1 mixes channels.
        if upsample_factor > 1:
            self.layer_norm_conv = LayerNorm(dim, eps=config.layer_norm_eps)
            self.depthwise = nn.ConvTranspose3d(
                dim, dim,
                kernel_size=(upsample_factor, 3, 3),
                stride=(upsample_factor, 1, 1),
                padding=(0, 1, 1),
                groups=dim,
            )
            self.pointwise = nn.Conv3d(dim, dim, kernel_size=1)
            self.act = GELUTanh()

            # Residual-branch temporal upsampling. A simple linear layer expands each
            # frame's channels into `factor` frames (T → T*factor) so the residual
            # matches the conv output length.  Initialised to nearest-neighbour repeat
            # (stacked identity blocks) so the residual is an identity highway at init
            # and only learns to deviate from a plain copy.
            self.residual_up = nn.Linear(dim, dim * upsample_factor)
            with torch.no_grad():
                self.residual_up.weight.copy_(torch.eye(dim).repeat(upsample_factor, 1))
                self.residual_up.bias.zero_()

    def _upsample_conv_fp32(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, T, H, W).  3-D (transpose-)convs are numerically unreliable in
        # fp16/bf16 on CUDA, so the whole conv sub-layer runs in fp32: disable
        # autocast and feed fp32 activations.  The conv params are kept in fp32 (AE
        # pretraining uses AMP autocast, which leaves master weights in fp32), so the
        # op runs entirely in fp32; the result is cast back to the autocast dtype.
        out_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            # channel-wise LayerNorm: (B, dim, T, H, W) → (B, T, H, W, dim) → norm → back.
            h = self.layer_norm_conv(x.permute(0, 2, 3, 4, 1).float())
            h = h.permute(0, 4, 1, 2, 3).contiguous()
            h = self.depthwise(h)
            h = self.pointwise(h)
            h = self.act(h)
        return h.to(out_dtype)

    def forward(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, T, H, W)
        B, dim, T, H, W = x.shape
        HW = H * W
        device = x.device

        # ── Transformer encoder: per-frame spatial self-attention + MLP ──
        # frame-major flatten: (B, dim, T, H, W) → (B*T*HW, dim)
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B * T * HW, dim)
        cu = torch.arange(0, B * T * HW + 1, step=HW, device=device, dtype=torch.int32)
        tokens = tokens + self.self_attn(self.layer_norm_attn(tokens), cu, rotary_pos_emb)
        tokens = tokens + self.mlp(self.layer_norm_mlp(tokens))
        x = tokens.view(B, T, H, W, dim).permute(0, 4, 1, 2, 3).contiguous()

        if self.upsample_factor <= 1:
            return x

        # ── Temporal-upsampling 3-D conv after the MLP (pre-norm residual, fp32) ──
        # Learned temporal upsample (T → T*f) on the residual branch: per-frame linear
        # channel expansion reshaped into f new frames, ordered t*f + j to match the
        # conv output.  (Run in the ambient autocast dtype — only the conv needs fp32.)
        f = self.upsample_factor
        r = self.residual_up(x.permute(0, 2, 3, 4, 1))                  # (B, T, H, W, f*dim)
        r = r.view(B, T, H, W, f, dim)                                  # split channels → f frames
        r = r.permute(0, 1, 4, 2, 3, 5).reshape(B, T * f, H, W, dim)    # interleave: t*f + j
        residual = r.permute(0, 4, 1, 2, 3).contiguous()               # (B, dim, T*f, H, W)
        x = residual + self._upsample_conv_fp32(x)
        return x


def _temporal_factors(n: int) -> List[int]:
    """
    Decompose ``n`` into a sequence of small integer factors (>= 2) for staged
    temporal upsampling.  Each returned factor becomes the temporal stride of one
    upsampling ``CompressorDecoderLayer`` (assigned by
    ``_tail_upsample_factors``) and the product equals ``n``.

    Examples
    --------
    >>> _temporal_factors(10)   # 1 → 2 → 10
    [2, 5]
    >>> _temporal_factors(8)    # 1 → 2 → 4 → 8
    [2, 2, 2]
    >>> _temporal_factors(16)   # 1 → 2 → 4 → 8 → 16
    [2, 2, 2, 2]
    >>> _temporal_factors(11)   # prime — no decomposition possible
    [11]
    """
    if n <= 1:
        return []
    factors: List[int] = []
    remaining = n
    for d in (2, 3, 5, 7, 11, 13):
        while remaining % d == 0 and remaining > 1:
            factors.append(d)
            remaining //= d
        if remaining == 1:
            break
    if remaining > 1:
        factors.append(remaining)
    return factors


def _tail_upsample_factors(num_layers: int, target_frames: int) -> List[int]:
    """
    Assign a temporal-upsampling stride to each of ``num_layers`` decoder layers,
    concentrating all of the upsampling on the final layers.

    Returns ``layer_factors`` of length ``num_layers`` whose product equals
    ``target_frames``.  The "fine" factors from ``_temporal_factors(target_frames)``
    are placed (in order) on the last ``len(factors)`` layers, so all temporal
    expansion happens progressively at the tail of the stack; every earlier layer
    gets stride 1 (a plain 3-D conv refinement, no temporal change).  If there are
    fewer layers than factors, trailing factors are merged so the count fits the
    depth.

    Examples
    --------
    >>> _tail_upsample_factors(8, 10)   # factors [2, 5] on the last 2 layers
    [1, 1, 1, 1, 1, 1, 2, 5]
    >>> _tail_upsample_factors(2, 8)    # factors [2, 2, 2] merged to fit 2 layers
    [2, 4]
    >>> _tail_upsample_factors(4, 1)    # target_frames == 1 → no upsampling
    [1, 1, 1, 1]
    """
    assert num_layers >= 1, "decoder needs at least one layer"
    factors = _temporal_factors(target_frames)  # product == target_frames
    if not factors:                              # target_frames == 1
        return [1] * num_layers
    # Too many upsample stages for the available depth → merge trailing factors.
    while len(factors) > num_layers:
        f = factors.pop()
        factors[-1] *= f
    layer_factors = [1] * num_layers
    # Concentrate the k upsample stages on the final k layers, keeping their order.
    layer_factors[num_layers - len(factors):] = factors
    return layer_factors


class CompressorDecoder(nn.Module):
    """
    AE decoder: a stack of transformer-encoder ``CompressorDecoderLayer`` blocks that
    expand the single compressed frame back to ``max_output_frames`` frames.

    Each layer is a standard pre-norm transformer encoder block (per-frame spatial
    self-attention + MLP).  The final layers additionally append a temporal-upsampling
    3-D conv after their MLP, with a stride chosen by ``_tail_upsample_factors`` so the
    strides' product equals ``max_output_frames``.  All temporal expansion is
    concentrated on the tail of the stack: the earlier layers are plain transformer
    refinements on the single seed frame, and the last few layers progressively
    upsample in time (the conv always runs in fp32 — see ``CompressorDecoderLayer``).

    Input  : ``compressed_tokens`` (B * HW, hidden_size)   — one HW grid per sample.
    Output : (B * max_output_frames * HW, hidden_size)     — frame-major (B, T, H, W).

    A final LayerNorm normalizes the decoded output.  The layers are pre-norm
    residual, so the residual stream is otherwise never normalized; without this norm
    training was observed to diverge (faster gradient blow-up and a U-shaped loss).

    Parameters
    ----------
    config : Videollama3TokenCompressorConfig
    max_output_frames : int  (default 10)
    """

    def __init__(self, config, max_output_frames: int = 10):
        super().__init__()
        H   = config.compress_image_h
        W   = config.compress_image_w
        dim = config.hidden_size
        head_dim = dim // config.num_attention_heads

        self.hidden_size       = dim
        self.H                 = H
        self.W                 = W
        self.HW                = H * W
        self.max_output_frames = max_output_frames

        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)

        # Per-layer temporal strides; product == max_output_frames, concentrated on
        # the final layers.
        layer_factors = _tail_upsample_factors(config.num_layers, max_output_frames)
        prod = 1
        for f in layer_factors:
            prod *= f
        assert prod == max_output_frames, (
            f"_tail_upsample_factors({config.num_layers}, {max_output_frames}) "
            f"= {layer_factors} has product {prod}, expected {max_output_frames}."
        )
        self.layer_factors = layer_factors
        self.layers = nn.ModuleList([
            CompressorDecoderLayer(config, upsample_factor=f) for f in layer_factors
        ])
        # Final norm on the decoded output. The pre-norm residual stream is otherwise
        # left unnormalized; removing this norm was observed to diverge (faster
        # gradient blow-up and a U-shaped loss).
        self.post_layernorm = LayerNorm(dim, eps=config.layer_norm_eps)

    # ------------------------------------------------------------------
    def _build_spatial_rotary_pos_emb(self) -> torch.Tensor:
        return _build_2d_rotary_pos_emb(self.rotary_pos_emb, self.W, self.H)

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

        rotary_pos_emb = self._build_spatial_rotary_pos_emb()

        # Seed the temporal axis with the single compressed frame:
        # (B * HW, dim) → (B, H, W, dim) → (B, dim, H, W) → (B, dim, 1, H, W)
        x = compressed_tokens.view(B, self.H, self.W, dim).permute(0, 3, 1, 2).unsqueeze(2)

        for layer in self.layers:
            x = layer(x, rotary_pos_emb)

        # (B, dim, T, H, W) → frame-major (B, T, H, W, dim) → (B*T*HW, dim)
        T = x.shape[2]
        assert T == self.max_output_frames, (T, self.max_output_frames)
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T * self.HW, dim)
        x = self.post_layernorm(x)
        return x  # (B * max_output_frames * HW, hidden_size)


from transformers import PretrainedConfig

class Videollama3TokenCompressorConfig(PretrainedConfig):
    model_type = "videollama3_token_compressor"

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
        if "transformer_decoder" in ct:
            return TransformerDecoderCompressor(config=compressor)
        if ct == "local_attn_conv":
            return LocalAttnConvCompressor(config=compressor)
    raise ValueError(f"Unknown token compressor type: {getattr(compressor, 'compressor_type', None)}")
