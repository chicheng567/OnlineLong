from torch.nn import LayerNorm
from typing import List, cast
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
        self.num_layers = num_layers
        self.compress_image_w = config.compress_image_w
        self.compress_image_h = config.compress_image_h
        self.compress_image_wh = self.compress_image_w * self.compress_image_h
        self.query = nn.Parameter(torch.randn(1,self.compress_image_w * self.compress_image_h, config.hidden_size))
        self.window_size = getattr(config, "window_size", 1)
        self.compression_decoder = None

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
        return query

    def decode_tokens(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        assert self.compression_decoder is not None, "compression_decoder is not defined, cannot decode tokens."
        if compressed_tokens.dim() == 2:
            compressed_tokens = compressed_tokens.view(-1, 1, self.compress_image_w, self.compress_image_h, self.hidden_size)
            compressed_tokens = compressed_tokens.permute(0, 4, 1, 2, 3).contiguous() # (B, hidden_size, T, w, h)
        assert compressed_tokens.dim() == 5, f"Expected compressed_tokens to have 5 dimensions (B, hidden_size, T, w, h), but got {compressed_tokens.shape}"
        decoded_tokens = self.compression_decoder(compressed_tokens)
        decoded_tokens = decoded_tokens.view(-1, self.hidden_size)
        return decoded_tokens


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
        self.num_layers = num_layers
        self.compress_image_w = config.compress_image_w
        self.compress_image_h = config.compress_image_h
        self.compress_image_wh = self.compress_image_w * self.compress_image_h
        # One learned query initialiser per spatial position; shared across windows and batch.
        self.query = nn.Parameter(torch.randn(1, self.compress_image_wh, config.hidden_size))
        self.window_size = getattr(config, "window_size", 1)
        self.compression_decoder = None

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

        return query

    def decode_tokens(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        assert self.compression_decoder is not None, "compression_decoder is not defined, cannot decode tokens."
        if compressed_tokens.dim() == 2:
            compressed_tokens = compressed_tokens.view(-1, 1, self.compress_image_w, self.compress_image_h, self.hidden_size)
            compressed_tokens = compressed_tokens.permute(0, 4, 1, 2, 3).contiguous()
        assert compressed_tokens.dim() == 5, (
            f"Expected compressed_tokens with 5 dims (B, hidden_size, T, w, h), got {compressed_tokens.shape}"
        )
        decoded_tokens = self.compression_decoder(compressed_tokens)
        decoded_tokens = decoded_tokens.view(-1, self.hidden_size)
        return decoded_tokens


class CompressorDecoderLayer(nn.Module):
    """
    Single decoder block: non-causal spatial self-attention (per frame slot) →
    global cross-attention to compressed tokens → MLP.  All sub-layers use
    pre-norm + residual.  Mirrors TransformerDecoderLayer but with causal=False.
    """
    def __init__(self, config):
        super().__init__()
        self.self_attn = selfFlashAttention(
            embed_dim=config.hidden_size,
            n_head=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            causal=False,
        )
        self.cross_attn = CrossFlashAttention2(
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

    def forward(self, q, kv, cu_seqlens_self, cu_seqlens_q_cross, cu_seqlens_kv_cross, rotary_pos_emb):
        q = q + self.self_attn(self.layer_norm1(q), cu_seqlens_self, rotary_pos_emb)
        q = q + self.cross_attn(self.layer_norm2(q), kv, cu_seqlens_q_cross, cu_seqlens_kv_cross)
        q = q + self.mlp(self.layer_norm3(q))
        return q


def _temporal_factors(n: int) -> List[int]:
    """
    Decompose ``n`` into a sequence of small integer factors (>= 2) for staged
    temporal upsampling.  Each returned factor is the per-stage stride used by
    one ``TemporalUpsampleBlock`` and the product equals ``n``.

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


class TemporalUpsampleBlock(nn.Module):
    """
    One stage of staged temporal upsampling.

    Pipeline (operates on 5-D ``(B, dim, T, H, W)`` tensors):
        1. Depthwise ``ConvTranspose3d`` — temporal stride = factor, spatial 3×3
           local mixing.  Channel-wise to keep the parameter count small.
        2. Pointwise ``Conv3d`` 1×1×1 — cross-channel mixing.
        3. ``LayerNorm`` over the channel dimension.
        4. ``GELU`` activation — gives the staged design its non-linear
           refinement between hops.
    """

    def __init__(self, dim: int, factor: int, layer_norm_eps: float = 1e-6):
        super().__init__()
        self.factor = factor
        self.depthwise = nn.ConvTranspose3d(
            dim, dim,
            kernel_size=(factor, 3, 3),
            stride=(factor, 1, 1),
            padding=(0, 1, 1),
            groups=dim,
        )
        self.pointwise = nn.Conv3d(dim, dim, kernel_size=1)
        self.norm = LayerNorm(dim, eps=layer_norm_eps)
        self.act = GELUTanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, T_in, H, W)  →  (B, dim, T_in * factor, H, W)
        x = self.depthwise(x)
        x = self.pointwise(x)
        # LayerNorm over channel dim: (B, dim, T, H, W) → (B, T, H, W, dim) → norm → back.
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x


class CompressorDecoder(nn.Module):
    """
    AE decoder combining attention with staged 3-D transposed convolution.

    Three-stage pipeline
    --------------------
    1. Pre-attention (num_layers // 2 layers):
       Spatial self-attention + MLP operating on the B×HW compressed tokens.
       No cross-attention — this refines the compressed representation before
       the conv upsampling sees it.

    2. Staged temporal upsampling (one ``TemporalUpsampleBlock`` per factor in
       ``_temporal_factors(max_output_frames)``).  For ``max_output_frames=10``
       this is two stages 1 → 2 → 10; each stage is depthwise ConvTranspose3d +
       pointwise Conv3d + LayerNorm + GELU.  Replaces the original single-shot
       ``1 → max_output_frames`` ConvTranspose, which created checkerboard
       artifacts and gave the model no non-linear refinement between hops.

    3. Post-attention (remaining layers):
       Per-frame spatial self-attention + cross-attention to the (pre-attn-refined)
       compressed tokens + MLP.  Cross-attention ties every decoded frame back to
       the encoder output so the reconstruction is grounded in the compressed
       representation.

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
        HW  = H * W
        head_dim = dim // config.num_attention_heads

        self.hidden_size      = dim
        self.H                = H
        self.W                = W
        self.HW               = HW
        self.max_output_frames = max_output_frames

        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)

        n_pre  = config.num_layers // 2
        n_post = config.num_layers - n_pre

        # Stage 1: pre-attention layers (self-attn + MLP only)
        self.pre_layers = nn.ModuleList([CompressorDecoderLayer(config) for _ in range(n_pre)])

        # Stage 2: staged temporal upsampling.
        #   For max_output_frames=10 → factors=[2, 5] → two blocks (1→2→10);
        #   for max_output_frames=8  → factors=[2, 2, 2] → three blocks (1→2→4→8).
        #   Each block is depthwise ConvTranspose3d + pointwise Conv3d + LN + GELU.
        upsample_factors = _temporal_factors(max_output_frames)
        prod = 1
        for f in upsample_factors:
            prod *= f
        assert prod == max_output_frames, (
            f"_temporal_factors({max_output_frames}) = {upsample_factors} "
            f"has product {prod}, expected {max_output_frames}."
        )
        self.upsample_factors = upsample_factors
        self.upsample_blocks = nn.ModuleList([
            TemporalUpsampleBlock(dim, factor=f, layer_norm_eps=config.layer_norm_eps)
            for f in upsample_factors
        ])

        # Stage 3: post-attention layers (self-attn + cross-attn to compressed + MLP)
        self.post_layers = nn.ModuleList([CompressorDecoderLayer(config) for _ in range(n_post)])

    # ------------------------------------------------------------------
    def _build_spatial_rotary_pos_emb(self) -> torch.Tensor:
        return _build_2d_rotary_pos_emb(self.rotary_pos_emb, self.W, self.H)

    # ------------------------------------------------------------------
    def forward(self, compressed_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            compressed_tokens : (B * HW, hidden_size)
        Returns:
            (B * max_output_frames * HW, hidden_size)
        """
        B      = compressed_tokens.shape[0] // self.HW
        device = compressed_tokens.device
        T      = self.max_output_frames
        HW     = self.HW
        dim    = self.hidden_size

        rotary_pos_emb = self._build_spatial_rotary_pos_emb()

        # ── Stage 1: pre-attention in compressed space ──────────────────
        # B groups, each HW tokens
        cu_pre = torch.arange(0, B * HW + 1, step=HW, device=device, dtype=torch.int32)
        x = compressed_tokens
        for _layer in self.pre_layers:
            layer = cast(CompressorDecoderLayer, _layer)
            # Use only self_attn + mlp (skip cross_attn for pre-layers)
            x = x + layer.self_attn(layer.layer_norm1(x), cu_pre, rotary_pos_emb)
            x = x + layer.mlp(layer.layer_norm3(x))

        # Save the refined compressed tokens for cross-attention in stage 3.
        compressed_refined = x  # (B * HW, dim)

        # ── Stage 2: staged temporal upsampling ─────────────────────────
        # (B * HW, dim) → (B, H, W, dim) → (B, dim, H, W) → (B, dim, 1, H, W)
        x_5d = x.view(B, self.H, self.W, dim).permute(0, 3, 1, 2).unsqueeze(2)
        # Each block multiplies temporal dim by its factor, with non-linear
        # refinement (LN + GELU) between hops.  Final shape: (B, dim, T, H, W).
        for block in self.upsample_blocks:
            x_5d = block(x_5d)
        # Back to flat token sequence: (B, T, H, W, dim) → (B*T*HW, dim)
        x = x_5d.permute(0, 2, 3, 4, 1).reshape(B * T * HW, dim)

        # ── Stage 3: post-attention over expanded sequence ───────────────
        # Self-attn: per-frame HW groups (preserves 2-D spatial RoPE shape)
        cu_self = torch.arange(0, B * T * HW + 1, step=HW, device=device, dtype=torch.int32)
        # Cross-attn: per-batch groups
        #   Q  group i  → T * HW tokens for batch item i
        #   KV group i  → HW compressed_refined tokens for batch item i
        cu_q_cross  = torch.arange(0, (B + 1) * T * HW, step=T * HW, device=device, dtype=torch.int32)
        cu_kv_cross = torch.arange(0, (B + 1) * HW,     step=HW,     device=device, dtype=torch.int32)

        for layer in self.post_layers:
            x = layer(x, compressed_refined, cu_self, cu_q_cross, cu_kv_cross, rotary_pos_emb)

        return x  # (B * T * HW, hidden_size)


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
