from torch.nn import LayerNorm
import torch
from transformers.activations import GELUTanh
from torch import nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from .videollama3_encoder.modeling_videollama3_encoder import VisionRotaryEmbedding, apply_rotary_pos_emb_vision
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
        
    def forward(self, x_q, x_kv, cu_seqlens_q, cu_seqlens_kv):
        # x_q should be of shape (batch_size * seq_len_q, d_model)
        # x_kv should be of shape (batch_size * seq_len_kv, d_model)
        # cu_seqlens_q should be of shape (batch_size + 1,) like (0, 4, 7, 9, 32, 33, ...)
        # cu_seqlens_kv should be of shape (batch_size + 1,) like (0, 4, 7, 9, 32, 33, ...)
        drop_rate = self.dropout_rate if self.training else 0.0
        q = self.w_q(x_q).view(-1, self.n_head, self.head_dim)
        k = self.w_k(x_kv).view(-1, self.n_head, self.head_dim)
        v = self.w_v(x_kv).view(-1, self.n_head, self.head_dim)
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
        self.self_attn = selfFlashAttention(embed_dim=config.hidden_size, n_head=config.num_attention_heads, dropout=config.attention_probs_dropout_prob, causal=True)
        self.embed_dim = config.hidden_size
        self.layer_norm1 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm2 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.layer_norm3 = LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = mlp(hidden_size=config.hidden_size, intermediate_size=config.intermediate_size)
    def forward(self, q, kv, cu_seqlens_q, cu_seqlens_kv, rotary_pos_emb):
        q = q + self.self_attn(self.layer_norm1(q), cu_seqlens_q, rotary_pos_emb)
        q = q + self.cross_attn(self.layer_norm2(q), kv, cu_seqlens_q, cu_seqlens_kv)
        q = q + self.mlp(self.layer_norm3(q))
        return q
    
class TransformerDecoderCompressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = config.num_layers
        head_dim = config.hidden_size // config.num_attention_heads
        self.head_dim = head_dim
        self.num_head = config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(dim=head_dim // 2)
        self.layers = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.compress_image_wh = config.compress_image_wh
        self.query = nn.Parameter(torch.randn(1,self.compress_image_wh, config.hidden_size))

    def _build_query_rotary_pos_emb(self, seq_len: int) -> torch.Tensor:
        # Keep the same indexing style as vision encoder: build (x, y), then flatten.
        h = int(seq_len ** 0.5)
        while h > 1 and seq_len % h != 0:
            h -= 1
        w = seq_len // h
        device = self.rotary_pos_emb.inv_freq.device
        hpos_ids = torch.arange(h, device=device).unsqueeze(1).expand(-1, w).reshape(-1)
        wpos_ids = torch.arange(w, device=device).unsqueeze(0).expand(h, -1).reshape(-1)
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)
        max_grid_size = max(h, w)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        return rotary_pos_emb_full[pos_ids].flatten(1)

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
        rotary_pos_emb = self._build_query_rotary_pos_emb(self.compress_image_wh)
        for layer in self.layers:
            query = layer(query, kv, cu_seqlens_q, compression_cu_seqlens, rotary_pos_emb)
        query = query.view(B, self.compress_image_wh, -1)
        return query
    
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
        compress_image_wh=256,
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
        self.compress_image_wh = compress_image_wh

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
        if "hidden_size" not in compressor:
            compressor["hidden_size"] = config.hidden_size
        if "num_attention_heads" not in compressor:
            compressor["num_attention_heads"] = config.num_attention_heads
        compressor = Videollama3TokenCompressorConfig(**compressor)
    if isinstance(compressor, Videollama3TokenCompressorConfig) and "transformer_decoder" in compressor.compressor_type:
        return TransformerDecoderCompressor(config=compressor)
    raise ValueError(f"Unknown token compressor type: {getattr(compressor, 'compressor_type', None)}")
