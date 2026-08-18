"""
Dynamic Token Synthesizer (DTS) + Semantic-Guided Masking (SGM) from Video-XL-Pro's
ReCoT self-supervised pretraining recipe (arXiv:2503.18478). Used by
`videollama3/train/videollama3_pretrain_compressor_dts.py` as a separate compressor
pretrain branch alongside the plain-AE recipe in `videollama3_pretrain_compressor.py`.

Cross-checked against the official code (referenceCode/Video-XL/Video-XL-Pro,
`videoxlpro/videoxlpro/model/sae.py::SiglipAE`), which is the only piece of ReCoT the
public repo actually ships (the SGM masking / DTS-synthesis code itself is not present
there — this project's `token_compress.py` equivalent only exists as a stray
`__pycache__/token_compress.cpython-310.pyc` with no matching source). `SiglipAE`
confirms the paper's stated "merge tokens from four frames into one" via a learnable
`nn.Parameter(torch.randn((4, 1152)))` named `temporal_encoding`, added unconditionally
at the top of the encoder's forward (`x = x + temporal_encoding`) before two
temporal_stride=2 Conv3d downsamples (4 -> 2 -> 1 frames).

This project already has its own compressor architectures (`compressor.py`'s
`TransformerDecoderCompressor` / `LocalAttnConvCompressor` + `CompressorDecoder`), so
DTS's `SiglipAE` conv/attention *architecture* is not reproduced here (it would be a
disconnected, throwaway module whose weights don't feed the real compressor). What is
ported is the two *training-recipe* pieces that are otherwise architecture-agnostic and
squarely aimed at this project's known open problem (see the
compressor-pretrain-collapse-diagnosis memory: every plain-recon-MSE / motion-weighted /
order-CE / semantic / tube-mask variant tried leaves the compressed bottleneck
perfectly order-invariant, cos(real, shuffled) = 1.0000) — DTS's learnable temporal
signal and SGM's frame-aware masking are a genuinely different, order-sensitive
self-supervision signal, not yet tried:

  DynamicTokenSynthesizer — the paper's learnable per-frame temporal encoding. The
    paper's headline use is synthesizing T pseudo-frames from a SINGLE static image
    (replicate its tokens T times, add this encoding, so images can be used for
    masked-video pretraining too); this project's DTS branch is video-only per an
    explicit scoping decision (see videollama3_pretrain_compressor_dts.py's module
    docstring), so here the module is used exactly as `SiglipAE` uses it for a REAL
    clip: an additive per-frame learnable bias injected before compression.

  SemanticGuidedMasking — the paper's "Algorithm 1", transcribed line-for-line from
    the arXiv HTML source (the reference repo does not include SGM's code):

        videofeature = Siglip(video)                     # (B, T, HW, C)
        V_{t-1}      = cat(zeros, videofeature[:-1])      # previous-frame tokens; frame 0 -> zeros
        V_t          = mean(videofeature, dim=HW)         # (B, T, C) per-frame "semantic average"
        S_temp       = TempQuery(V_{t-1})                 # (B, T, HW, C)
        S_spa        = SpatialQuery(V_t)                  # (B, T, C)
        score_temp[t, p] = videofeature[t, p] . S_temp[t, p]   # position-aligned dot product
        score_spa[t, p]  = videofeature[t, p] . S_spa[t]       # dot with the frame's own gist
        tokenscore   = score_temp + score_spa
    "under the guidance of the token scores, we randomly mask low-scoring tokens" is
    the paper's only description of the selection step; it gives no mask-ratio value.
    This implementation reads "randomly ... guided by" as literally both: Gumbel(0,1)
    noise perturbs `tokenscore` before ranking (the "randomly"), then the lowest
    `mask_ratio` fraction of the perturbed score is masked (the "guided by" / "low-
    scoring") — flagged here as an interpretation, not a verbatim paper detail, since
    the paper does not fully specify it. See the "Gradient note" below for why this
    (rather than plain deterministic bottom-k) is also what makes `TempQuery`/
    `SpatialQuery` trainable.

  Gradient note (revised): a hard top-k over `tokenscore` is not differentiable, so a
  literal `torch.topk` + `scatter_`/`where` implementation gives `TempQuery`/
  `SpatialQuery` no path to the loss at all — not "a small gradient", exactly zero,
  every step, forever (verified: `temp_query.weight.grad` stays `None` through
  training). The paper does not resolve this either (no loss is defined on the scores,
  no mention of Gumbel/straight-through/REINFORCE anywhere in Section 3.2) but two
  `nn.Linear` modules that structurally cannot receive gradient under any input is not
  a defensible reading of "these are learnable queries" — so `apply_mask` below uses a
  **straight-through Gumbel-top-k relaxation**, standard for differentiable subset
  selection (the same trick underlies Concrete/Gumbel-Softmax masking and MoE hard
  routing):
    1. Perturb `tokenscore` with Gumbel(0,1) noise (training only) before ranking —
       this is what makes the selection "random[ly] mask low-scoring tokens" rather
       than deterministic bottom-k, per the paper's wording.
    2. The *forward* mask is still the exact hard top-k of the noisy score (bit-identical
       token content to the old implementation — `mask_token` at masked positions,
       original token elsewhere).
    3. The *backward* pass substitutes the gradient of a temperature-scaled sigmoid
       soft-assignment centered at the k-th smallest noisy score, so perturbing
       `TempQuery`/`SpatialQuery` changes which tokens sit near the mask/keep boundary
       and that change now reaches the reconstruction MSE loss. Tokens far from the
       boundary contribute ~0 gradient (correct — their rank is not sensitive to a
       small perturbation); the boundary token(s) always exist by construction
       (`1 <= k <= N-1`), so the queries get a genuine, never-vanishing gradient every
       step, not merely "sometimes".
  See `apply_mask`'s docstring for the exact formula.
"""

import torch
import torch.nn as nn


class DynamicTokenSynthesizer(nn.Module):
    """Learnable additive per-frame temporal encoding (DTS's core learned component).

    Reference: `SiglipAE.temporal_encoding` in sae.py, `nn.Parameter(torch.randn((4, 1152)))`,
    added once at the top of the encoder's forward, unconditionally on real or
    (in the paper's image branch, not implemented here) synthesized frames.
    """

    def __init__(self, hidden_size: int, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        self.temporal_encoding = nn.Parameter(torch.randn(num_frames, hidden_size) * 0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, T, HW, C), T must equal self.num_frames (this DTS branch fixes
        # every window at exactly num_frames — see the training script's dataset).
        _, T, _, C = tokens.shape
        assert T == self.num_frames, (
            f"DynamicTokenSynthesizer: expected T={self.num_frames}, got {T}."
        )
        return tokens + self.temporal_encoding.to(tokens.dtype).view(1, T, 1, C)


class SemanticGuidedMasking(nn.Module):
    """Algorithm 1 (Semantic-Guided Masking): per-token score + bottom-k masking."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.temp_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.spatial_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

    def compute_scores(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, T, HW, C) raw (unmasked, pre-DTS) SigLIP tokens.
        Returns tokenscore: (B, T, HW).
        """
        B, _, HW, C = tokens.shape
        zero_frame = tokens.new_zeros(B, 1, HW, C)
        v_prev = torch.cat([zero_frame, tokens[:, :-1]], dim=1)   # V_{t-1}: (B,T,HW,C)
        v_cur_mean = tokens.mean(dim=2)                            # V_t:     (B,T,C)

        s_temp = self.temp_query(v_prev)        # (B,T,HW,C)
        s_spa = self.spatial_query(v_cur_mean)   # (B,T,C)

        score_temp = torch.einsum("bthc,bthc->bth", tokens, s_temp)
        score_spa = torch.einsum("bthc,btc->bth", tokens, s_spa)
        return score_temp + score_spa            # (B,T,HW)

    def apply_mask(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        mask_ratio: float,
        temperature: float = 1.0,
        add_noise: bool = True,
    ):
        """
        Replace the lowest-`mask_ratio`-fraction-scoring positions with `mask_token`.

        tokens/scores: (B, T, HW, C) / (B, T, HW). Masking ranks GLOBALLY over each
        sample's T*HW tokens (matches Algorithm 1's single flat "all tokens score"
        output, not a per-frame ratio).

        The forward value is an exact hard top-k mask (bit-identical token content to
        a plain `torch.where`); the backward pass uses a straight-through estimator
        (STE) so `scores` — and therefore `temp_query`/`spatial_query` — receive a
        real gradient. See the module docstring's "Gradient note" for why a plain hard
        top-k cannot do this. Mechanics:
          1. `noisy = scores + Gumbel(0,1)` (training only) — the paper's "randomly
             mask low-scoring tokens": still guided by score, but not deterministic.
          2. hard mask = exact top-k(`noisy`, k, smallest) -> identical to before.
          3. soft mask = sigmoid((threshold - noisy) / (temperature * scale)), a
             continuous relaxation of "is this token below the k-th smallest noisy
             score", `scale` = per-sample std of `scores` (detached) so the sigmoid's
             effective steepness is invariant to the raw dot-product magnitude
             (~sqrt(hidden_size)) instead of saturating/vanishing as hidden_size or
             training scale drifts.
          4. `mask = hard + (soft - soft.detach())` — forward equals `hard`,
             `d(mask)/d(scores) = d(soft)/d(scores)`.

        Returns (masked_tokens, mask_bool); mask_bool is the hard (bool) mask, True at
        masked positions — used unchanged by callers for recon-loss selection / logging.
        """
        B, T, HW, C = tokens.shape
        N = T * HW
        k = int(round(mask_ratio * N))
        k = min(max(k, 1), N - 1)  # keep >=1 visible and >=1 masked position

        flat_scores = scores.reshape(B, N)
        if add_noise and self.training:
            u = torch.rand_like(flat_scores).clamp_(min=1e-9, max=1.0 - 1e-9)
            noisy_scores = flat_scores + (-torch.log(-torch.log(u)))
        else:
            noisy_scores = flat_scores

        topk = torch.topk(noisy_scores, k, dim=1, largest=False)
        hard_mask_flat = torch.zeros(B, N, dtype=torch.bool, device=tokens.device)
        hard_mask_flat.scatter_(1, topk.indices, True)

        threshold = topk.values[:, -1:].detach()  # k-th smallest noisy score, (B,1)
        scale = flat_scores.detach().std(dim=1, keepdim=True).clamp_min(1e-6)
        soft_mask_flat = torch.sigmoid((threshold - noisy_scores) / (temperature * scale))
        mask_flat = hard_mask_flat.to(soft_mask_flat.dtype) + (soft_mask_flat - soft_mask_flat.detach())

        mask_bhw = mask_flat.view(B, T, HW).to(tokens.dtype)  # forward == hard, grad == soft
        mask_bool_bhw = hard_mask_flat.view(B, T, HW)

        mask_tok = self.mask_token.to(tokens.dtype).view(1, 1, 1, C)
        masked_tokens = mask_bhw.unsqueeze(-1) * mask_tok + (1 - mask_bhw.unsqueeze(-1)) * tokens
        return masked_tokens, mask_bool_bhw
