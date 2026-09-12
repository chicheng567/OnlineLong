"""multi-window smoke: adaptive segmenter must handle a packed multi-video batch."""
import sys, torch
sys.path.insert(0, "/root/OnlineLong")
from videollama3.model.compressor import (
    adaptive_segment_count, Videollama3TokenCompressorConfig, build_token_compressor,
)

cfg = Videollama3TokenCompressorConfig(
    compressor_type="transformer_decoder_flat", hidden_size=1152, num_attention_heads=8,
    num_layers=2, num_queries=64, match_encoder_scale=True, distr_loss_weight=0.05,
    adaptive_segmentation=True, segment_target_frames=4, segment_force_every=8,
    segment_sample_tau=0.5,
)
comp = build_token_compressor(type("C", (), {"token_compressor_config": cfg.to_dict(),
                                             "hidden_size": 1152, "num_attention_heads": 8})())
comp = comp.cuda().to(torch.bfloat16).train()
hw = 16 * 16

for Ts in ([400], [40, 137, 12, 211], [8, 9, 400], [16]):
    spans = [t * hw for t in Ts]
    kv = torch.randn(sum(spans), 1152, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(spans).cumsum(0).tolist()], device="cuda")
    grid = [(16, 16)] * len(Ts)
    out = comp(kv, cu, grid_hws=grid)
    exp = sum(adaptive_segment_count(t, 4) for t in Ts) * 64
    # arch-side reservation: sum of per-part output_len_for
    arch = sum(comp.output_len_for(t, 16, 16) for t in Ts)
    ok = out.shape[0] == exp == arch
    print(f"Ts={Ts}  out={tuple(out.shape)}  Σ N_i*64={exp}  Σ arch={arch}  {'OK' if ok else 'MISMATCH'}")
