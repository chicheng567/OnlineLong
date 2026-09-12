"""Follow-up: (1) segment-length histogram for topk_forced8, (2) stochastic top-k
keeps N exact AND varies placement across epochs (seeds)."""
from __future__ import annotations
import json, os, sys, random, math
import numpy as np
import torch

sys.path.insert(0, "/root/OnlineLong")
from eval_ablation.common import load_model, load_processor, DEFAULT_FORCE_IMAGE_SIZE
from videollama3.mm_utils import load_video

MODEL = "/root/OnlineLong/pretrained_models/compressor_pretrain_video_norm"
VID_DIR = "/share/dataset/internVid"
MAX_FRAMES = 320


def per_frame_feats(model, proc, path):
    frames, ts = load_video(path, fps=1, max_frames=MAX_FRAMES)
    T = len(frames)
    if T < 8:
        return None
    conv = [{"role": "user", "content": [
        {"type": "video", "timestamps": [float(t) for t in ts], "num_frames": T},
        {"type": "text", "text": "x"}]}]
    inp = proc(images=[frames], text=conv, merge_size=2, return_tensors="pt")
    pv = inp["pixel_values"].to("cuda:0", torch.bfloat16)
    gs = inp["grid_sizes"].to("cuda:0"); ms = inp["merge_sizes"].to("cuda:0")
    with torch.no_grad():
        tok = model.get_model().get_vision_encoder()(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)
    tok = tok.float().cpu()
    T2 = T
    tpf = tok.shape[0] // T2
    return tok.view(T2, tpf, -1).mean(1), T2


def cosdiff(f):
    fn = torch.nn.functional.normalize(f, dim=-1)
    return (1.0 - (fn[1:] * fn[:-1]).sum(-1)).numpy()


def seglens(cuts, T):
    b = sorted(set([0] + [int(c) for c in cuts] + [T]))
    return np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])


def forced8(diff, T, gen=None, temp=None):
    target_N = max(1, T // 4) + 1
    forced = set(range(8, T, 8))
    budget = target_N - 1 - len(forced)
    cand = [j + 1 for j in range(len(diff)) if (j + 1) not in forced]
    if budget > 0:
        if gen is None:
            order = sorted(cand, key=lambda p: -diff[p - 1])
            forced.update(order[:budget])
        else:                                   # stochastic: Gumbel-top-k over softmax(diff/temp)
            d = np.array([diff[p - 1] for p in cand], float)
            d = (d - d.mean()) / (d.std() + 1e-6)
            logits = d / temp + gen.gumbel(size=len(cand))
            forced.update(cand[i] for i in np.argsort(-logits)[:budget])
    cuts = sorted(forced)
    out, prev = [], 0
    for c in cuts:
        while c - prev > 8:
            prev += 8; out.append(prev)
        out.append(c); prev = c
    while T - prev > 8:
        prev += 8; out.append(prev)
    return sorted(set(x for x in out if 0 < x < T))


def main():
    random.seed(1)
    dur = [d for d in json.load(open("/root/OnlineLong/anno_data/internVid_durations.json"))
           if d.get("ok") and 30 <= d["duration_sec"] < 420]
    random.shuffle(dur)
    model = load_model(MODEL, device="cuda:0")
    proc = load_processor(MODEL, force_image_size=DEFAULT_FORCE_IMAGE_SIZE)

    all_L = []
    jaccard_pairs = []      # placement variability of stochastic across seeds
    n_exact = 0; n = 0
    for d in dur:
        if n >= 30:
            break
        path = os.path.join(VID_DIR, d["video_id"] + ".mp4")
        if not os.path.exists(path):
            continue
        try:
            r = per_frame_feats(model, proc, path)
        except Exception as e:
            print("skip", d["video_id"], e); continue
        if r is None:
            continue
        f, T = r
        diff = cosdiff(f)
        tgtN = max(1, T // 4) + 1

        L = seglens(forced8(diff, T), T)
        all_L.append(L)
        if len(L) == tgtN:
            n_exact += 1

        s1 = set(forced8(diff, T, gen=np.random.default_rng(10), temp=0.5))
        s2 = set(forced8(diff, T, gen=np.random.default_rng(11), temp=0.5))
        L1 = seglens(s1, T); L2 = seglens(s2, T)
        jac = len(s1 & s2) / max(1, len(s1 | s2))
        jaccard_pairs.append((jac, len(L1) == tgtN and len(L2) == tgtN))
        n += 1

    L = np.concatenate(all_L)
    print(f"\ntopk_forced8 over {n} clips, {len(L)} segments")
    print("N == floor(T/4)+1 exactly:", n_exact, "/", n)
    print("segment-length histogram (frames):")
    for k in range(1, 9):
        c = int((L == k).sum())
        print(f"  {k}f: {c:5d}  {100*c/len(L):5.1f}%  " + "#" * int(60 * c / len(L)))
    print(f"  mean={L.mean():.2f}  cv={L.std()/L.mean():.3f}  max={L.max()}  >8f={(L>8).sum()}")
    js = np.array([j for j, ok in jaccard_pairs])
    print(f"\nstochastic top-k (temp=0.5): boundary Jaccard between two seeds "
          f"mean={js.mean():.2f} (lower = more per-epoch variation)")
    print("  N still exact for both seeds on every clip:", all(ok for _, ok in jaccard_pairs))


if __name__ == "__main__":
    main()
