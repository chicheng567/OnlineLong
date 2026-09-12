#!/usr/bin/env bash
# =============================================================================
# Plan X — Phase 1 post-training analysis.
#
# Runs three probes on the finished qbase checkpoint vs. the whole-video-qbase
# baseline it was warm-started from, all in the Phase-1 canonical config
# (one whole-video compression part, dynamic HW, fps=1):
#
#   1. struct_probe.py         feature health / collapse gate
#                              (struct_rho vs encoder, xcos_cen, cc_frac)
#   2. feature_distribution.py norm ratio, PCA participation ratio / eff rank,
#                              token collapse, compressed/raw norm ratio
#   3. caption_eval.py         ROUGE/BLEU vs the InternVid Qwen3-VL reference,
#                              degeneration signals, phase1-vs-baseline agreement
#
# then stitches everything into  $OUT/REPORT.md .
#
# USAGE
#   bash eval_ablation/phase1_analyze.sh
#   CKPT=work_dirs/phase1_qbase_internvid/checkpoint-2000 DEVICE=cuda:3 \
#     bash eval_ablation/phase1_analyze.sh
#
# Every ALL-CAPS var is overridable from the environment.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

RUN_DIR="${RUN_DIR:-work_dirs/phase1_qbase_internvid}"
BASELINE="${BASELINE:-pretrained_models/compressor_pretrain_video_norm}"
MANIFEST="${MANIFEST:-eval_ablation/manifest_internvid_heldout.json}"
SHAREGPT_MANIFEST="${SHAREGPT_MANIFEST:-eval_ablation/manifest.json}"   # used only if its videos exist
OUT="${OUT:-$RUN_DIR/analysis}"
NUM_VIDEOS="${NUM_VIDEOS:-20}"
MAX_FRAMES="${MAX_FRAMES:-180}"
NATIVE_MAX_TOKENS="${NATIVE_MAX_TOKENS:-16384}"   # == Plan-X training dynamic-HW budget
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# ---- resolve the checkpoint: final save at RUN_DIR root, else newest checkpoint-* ----
if [[ -n "${CKPT:-}" ]]; then
  :
elif [[ -f "$RUN_DIR/model.safetensors.index.json" || -f "$RUN_DIR/model.safetensors" ]]; then
  CKPT="$RUN_DIR"
else
  CKPT="$(ls -d "$RUN_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)"
fi
[[ -n "${CKPT:-}" && -e "$CKPT" ]] || { echo "[analyze] no checkpoint found under $RUN_DIR" >&2; exit 1; }

# a checkpoint-* dir has no tokenizer/processor/config for the LLM; fall back to RUN_DIR root,
# and if that is also bare, to the base model the run trained from.
CKPT_HAS_CFG=0; [[ -f "$CKPT/config.json" ]] && CKPT_HAS_CFG=1

# ---- pick the least-used visible GPU unless DEVICE is set ----
if [[ -z "${DEVICE:-}" ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
         | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')"
  DEVICE="cuda:${GPU:-0}"
fi

mkdir -p "$OUT"
echo "[analyze] ckpt=$CKPT  (has config.json=$CKPT_HAS_CFG)"
echo "[analyze] baseline=$BASELINE"
echo "[analyze] device=$DEVICE  manifest=$MANIFEST  num_videos=$NUM_VIDEOS  out=$OUT"

MODELS=(phase1="$CKPT" baseline="$BASELINE")

run () { echo; echo "=========== $1 ==========="; shift; "$@"; echo "[exit $?]"; }

# 1. feature health / collapse gate --------------------------------------------
run "struct_probe (feature health / gate)" \
  python eval_ablation/struct_probe.py \
    --models "${MODELS[@]}" \
    --manifest "$MANIFEST" --num_videos "$NUM_VIDEOS" \
    --max_frames "$MAX_FRAMES" --fps 1 --merge_size 2 \
    --device "$DEVICE" --out "$OUT/struct"

# 2. feature-distribution comparison ----------------------------------------------
run "feature_distribution (norm ratio / PCA / collapse)" \
  python eval_ablation/feature_distribution.py \
    --models "${MODELS[@]}" \
    --manifest "$MANIFEST" --num_videos "$NUM_VIDEOS" \
    --whole_video --fps 1 --max_frames "$MAX_FRAMES" \
    --force_image_size 0 --native_max_tokens "$NATIVE_MAX_TOKENS" \
    --device "$DEVICE" --out "$OUT/feature_distribution"

# 3. caption eval vs the InternVid Qwen3-VL reference ---------------------------
run "caption_eval (InternVid held-out, whole-video)" \
  python eval_ablation/caption_eval.py \
    --models "${MODELS[@]}" \
    --manifest "$MANIFEST" --num_videos "$NUM_VIDEOS" \
    --whole_video --fps 1 --max_frames "$MAX_FRAMES" \
    --force_image_size 0 --native_max_tokens "$NATIVE_MAX_TOKENS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" --out "$OUT/caption_eval"

# 3b. optional: same on a ShareGPT4V slice if those videos are on this box -------
if python3 -c "import json,os,sys; d=json.load(open('$SHAREGPT_MANIFEST')); sys.exit(0 if any(os.path.exists(e['video']) for e in d) else 1)" 2>/dev/null; then
  run "caption_eval (ShareGPT4V, whole-video)" \
    python eval_ablation/caption_eval.py \
      --models "${MODELS[@]}" \
      --manifest "$SHAREGPT_MANIFEST" --num_videos "$NUM_VIDEOS" \
      --whole_video --fps 1 --max_frames "$MAX_FRAMES" \
      --force_image_size 0 --native_max_tokens "$NATIVE_MAX_TOKENS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --device "$DEVICE" --out "$OUT/caption_eval_sharegpt4v"
else
  echo "[analyze] ShareGPT4V manifest videos not on disk — skipping that slice"
fi

# ---- stitch REPORT.md -------------------------------------------------------
{
  echo "# Plan X — Phase 1 analysis"
  echo
  echo "- checkpoint: \`$CKPT\`"
  echo "- baseline (warm-start): \`$BASELINE\`"
  echo "- config: whole-video part, dynamic HW (native_max_tokens=$NATIVE_MAX_TOKENS), fps=1, max_frames=$MAX_FRAMES, $NUM_VIDEOS videos"
  echo "- data: \`$MANIFEST\` (InternVid held-out, Qwen3-VL reference captions)"
  echo
  echo "## Training loss (tail)"
  echo '```'
  grep -aE "\{'loss':" "$RUN_DIR/run.log" 2>/dev/null | tail -8
  echo '```'
  echo
  for f in "$OUT/struct/struct.md" \
           "$OUT/feature_distribution/feature_distribution.md" \
           "$OUT/caption_eval/caption_eval.md" \
           "$OUT/caption_eval_sharegpt4v/caption_eval.md"; do
    [[ -f "$f" ]] && { echo; echo "---"; echo; cat "$f"; echo; }
  done
  echo
  echo "---"
  echo "_full JSON + per-video side-by-side captions under \`$OUT/\`_"
} > "$OUT/REPORT.md"

echo
echo "[analyze] done -> $OUT/REPORT.md"
