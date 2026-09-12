"""
Small-scale validation of the proposed qbase segmenter:

    diff[i] = 1 - cos(f[i], f[i-1])          # f = per-frame mean-pooled encoder feature
    k       = floor(T / 4)                    # fixed budget  -> N = k + 1  (predictable)
    cuts    = sort(topk(diff, k))             # adaptive PLACEMENT

Question: are the resulting segments "uniform enough" (no degenerate clustering,
[1,8] clamp respectable), while still landing cuts on real changes?

Compares: uniform-4 | topk-plain | topk-centered (common-comp removed) |
          topk+forced8 (clamp-enforced) | threshold-tau (to show N variance)

Runs the real frozen SigLIP-NaViT on real InternVid clips.
"""
from __future__ import annotations
import json, os, sys, random, math, time
import numpy as np
import torch

sys.path.insert(0, "/root/OnlineLong")
from eval_ablation.common import load_model, load_processor, DEFAULT_FORCE_IMAGE_SIZE
from videollama3.mm_utils import load_video
from videollama3.constants import DEFAULT_IMAGE_TOKEN

MODEL = "/root/OnlineLong/pretrained_models/compressor_pretrain_video_norm"
VID_DIR = "/share/dataset/internVid"
FPS = 1
MAX_FRAMES = 320          # cap decode cost; covers Phase-1/2 length range
MERGE = 2
N_CLIPS = 40
SEED = 0


def per_frame_feats(model, proc, path):
    frames, ts = load_video(path, fps=FPS, max_frames=MAX_FRAMES)
    T = len(frames)
    if T < 8:
        return None
    conv = [{"role": "user", "content": [
        {"type": "video", "timestamps": [float(t) for t in ts], "num_frames": T},
        {"type": "text", "text": "x"}]}]
    inp = proc(images=[frames], text=conv, merge_size=MERGE, return_tensors="pt")
    pv = inp["pixel_values"].to("cuda:0", torch.bfloat16)
    gs = inp["grid_sizes"].to("cuda:0")
    ms = inp["merge_sizes"].to("cuda:0")
    with torch.no_grad():
        tok = model.get_model().get_vision_encoder()(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)
    tok = tok.float().cpu()                       # (N, C)
    N, C = tok.shape
    assert N % T == 0, (N, T)
    tpf = N // T
    f = tok.view(T, tpf, C).mean(1)              # (T, C) per-frame mean-pooled
    return f, T, tpf


def cosdiff(f):
    fn = torch.nn.functional.normalize(f, dim=-1)
    c = (fn[1:] * fn[:-1]).sum(-1)               # cos(f[i], f[i-1]), i=1..T-1
    return (1.0 - c).numpy()                     # length T-1, index j -> boundary before frame j+1


def seg_lengths_from_cuts(cuts, T):
    """cuts: sorted list of frame indices where a new segment starts (in 1..T-1)."""
    b = [0] + sorted(int(x) for x in cuts) + [T]
    b = sorted(set(b))
    return np.array([b[i + 1] - b[i] for i in range(len(b) - 1)])


def uniform4(diff, T):
    return list(range(4, T, 4))


def topk_plain(diff, T, k=None):
    k = k if k is not None else max(1, T // 4)
    k = min(k, T - 1)
    idx = np.argsort(-diff)[:k] + 1             # +1: diff[j] is the cut before frame j+1
    return sorted(idx.tolist())


def topk_forced8(diff, T):
    """Force a boundary every 8 frames, spend the rest of the N=floor(T/4)+1 budget
    on the biggest remaining diffs, enforce [1,8]."""
    target_N = max(1, T // 4) + 1
    forced = set(range(8, T, 8))
    budget = target_N - 1 - len(forced)
    if budget > 0:
        order = [j + 1 for j in np.argsort(-diff).tolist() if (j + 1) not in forced]
        forced.update(order[:budget])
    cuts = sorted(forced)
    # enforce max-8 (in case T//4 budget < forced count is impossible, but guard anyway)
    out, prev = [], 0
    for c in cuts:
        while c - prev > 8:
            prev += 8
            out.append(prev)
        out.append(c); prev = c
    while T - prev > 8:
        prev += 8; out.append(prev)
    return sorted(set(x for x in out if 0 < x < T))


def threshold_tau(diff, T, tau_pct=60):
    tau = np.percentile(diff, tau_pct)
    cuts, run = [], 0
    for j in range(len(diff)):
        run += 1
        if diff[j] > tau or run >= 8:
            cuts.append(j + 1); run = 0
    return sorted(set(c for c in cuts if 0 < c < T))


def metrics(cuts, diff, T):
    L = seg_lengths_from_cuts(cuts, T)
    N = len(L)
    allj = np.arange(len(diff))
    bset = set(int(c) - 1 for c in cuts)          # diff indices used as cuts
    bmask = np.array([j in bset for j in allj])
    d_at = diff[bmask].mean() if bmask.any() else float("nan")
    d_in = diff[~bmask].mean() if (~bmask).any() else float("nan")
    return dict(
        N=N, Lmin=int(L.min()), Lmax=int(L.max()), Lmean=round(float(L.mean()), 2),
        Lcv=round(float(L.std() / L.mean()), 3),
        pct_gt8=round(100 * float((L > 8).mean()), 1),
        pct_eq1=round(100 * float((L == 1).mean()), 1),
        frac_frames_in_top10pct_segs=round(
            float(np.sort(L)[::-1][:max(1, N // 10)].sum()) / T, 3),
        boundary_vs_interior_diff=round(float(d_at / d_in), 2) if d_in and not math.isnan(d_in) else None,
    )


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dur = {d["video_id"]: d for d in json.load(open("/root/OnlineLong/anno_data/internVid_durations.json"))}
    caps = set()
    with open("/root/OnlineLong/recaption/qwen3vl/captions.jsonl") as fh:
        for i, line in enumerate(fh):
            if i > 300000:
                break
            caps.add(json.loads(line)["video_id"])
    # sample across buckets: 15-60s, 60-180s, 180-420s
    buckets = {"15-60": [], "60-180": [], "180-420": []}
    for vid, d in dur.items():
        if not d.get("ok") or vid not in caps:
            continue
        s = d["duration_sec"]
        if 15 <= s < 60:
            buckets["15-60"].append(vid)
        elif 60 <= s < 180:
            buckets["60-180"].append(vid)
        elif 180 <= s < 420:
            buckets["180-420"].append(vid)
    pick = []
    for b, lst in buckets.items():
        random.shuffle(lst)
        pick += [(b, v) for v in lst[:N_CLIPS // 3 + 4]]
    random.shuffle(pick)

    model = load_model(MODEL, device="cuda:0")
    proc = load_processor(MODEL, force_image_size=DEFAULT_FORCE_IMAGE_SIZE)

    segmenters = ["uniform4", "topk_plain", "topk_centered", "topk_forced8", "threshold_tau"]
    agg = {s: [] for s in segmenters}
    done = 0
    for b, vid in pick:
        if done >= N_CLIPS:
            break
        path = os.path.join(VID_DIR, vid + ".mp4")
        if not os.path.exists(path):
            continue
        try:
            r = per_frame_feats(model, proc, path)
        except Exception as e:
            print(f"skip {vid}: {type(e).__name__} {e}")
            continue
        if r is None:
            continue
        f, T, tpf = r
        diff = cosdiff(f)
        fc = f - f.mean(0, keepdim=True)                 # common-component removed
        diff_c = cosdiff(fc)
        cutfns = {
            "uniform4": uniform4(diff, T),
            "topk_plain": topk_plain(diff, T),
            "topk_centered": topk_plain(diff_c, T),
            "topk_forced8": topk_forced8(diff, T),
            "threshold_tau": threshold_tau(diff, T),
        }
        row = {"vid": vid, "bucket": b, "T": T}
        for s in segmenters:
            m = metrics(cutfns[s], diff if s != "topk_centered" else diff_c, T)
            agg[s].append(m)
            row[s] = m
        done += 1
        print(f"[{done:2d}] {vid[:12]:12s} {b:7s} T={T:3d}  "
              + "  ".join(f"{s}:N={row[s]['N']},Lmax={row[s]['Lmax']},>8={row[s]['pct_gt8']}%,"
                          f"cv={row[s]['Lcv']},b/i={row[s]['boundary_vs_interior_diff']}"
                          for s in ["topk_plain", "topk_forced8"]))

    print("\n==================  AGGREGATE over", done, "clips  ==================")
    hdr = f"{'segmenter':16s} {'N(mean)':>8s} {'Lmax(mean)':>10s} {'Lmax(p95)':>9s} " \
          f"{'>8f %':>7s} {'==1f %':>7s} {'Lcv':>6s} {'top10%frac':>10s} {'bnd/int diff':>12s}"
    print(hdr); print("-" * len(hdr))
    for s in segmenters:
        A = agg[s]
        def col(key):
            return np.array([x[key] for x in A], float)
        bi = np.array([x["boundary_vs_interior_diff"] for x in A
                       if x["boundary_vs_interior_diff"] is not None], float)
        print(f"{s:16s} {col('N').mean():8.1f} {col('Lmax').mean():10.2f} "
              f"{np.percentile(col('Lmax'),95):9.1f} {col('pct_gt8').mean():7.1f} "
              f"{col('pct_eq1').mean():7.1f} {col('Lcv').mean():6.3f} "
              f"{col('frac_frames_in_top10pct_segs').mean():10.3f} "
              f"{bi.mean():12.2f}")

    # worst-case clustering for the plain rule
    worst = sorted(agg["topk_plain"], key=lambda x: -x["Lmax"])[:5]
    print("\ntopk_plain worst-5 by Lmax:",
          [(w["N"], w["Lmax"], w["pct_gt8"], w["Lcv"]) for w in worst])
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmenter_results.json")
    json.dump({s: agg[s] for s in segmenters}, open(out_path, "w"))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
