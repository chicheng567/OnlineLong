#!/usr/bin/env bash
# Re-run the image_mix vs noprune caption eval under DIFFERENT prompts, to check
# whether the "image_mix writes shorter / lower-recall captions" result is
# prompt-specific or holds across instructions.
#
# Anchor (already run, work_dirs/image_mix_eval): the exact TRAINING prompt.
# Here: 3 alternative instructions, each its own invocation on its own GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}."

IMAGEMIX_DIR=work_dirs/compressor_pretrain_image_mix
NOPRUNE_DIR=work_dirs/compressor_prune_ablation/noprune
MANIFEST=eval_ablation/manifest_imagemix.json
BASE=work_dirs/image_mix_eval/prompts
mkdir -p "$BASE"

run () {
    local tag="$1" dev="$2" prompt="$3"
    echo "[launch $(date +%H:%M:%S)] $tag -> $dev"
    python eval_ablation/caption_eval.py \
        --models "noprune=${NOPRUNE_DIR}" "image_mix=${IMAGEMIX_DIR}" \
        --manifest "$MANIFEST" --num_videos 0 \
        --out "$BASE/$tag" --device "$dev" \
        --max_frames 64 --window_size 8 --force_image_size 448 \
        --repetition_penalty 1.1 --seed 42 \
        --prompt "$prompt" \
        > "$BASE/$tag.log" 2>&1 &
}

run temporal cuda:0 "Describe this video in detail. Narrate everything that happens in chronological order from the first frame to the last: every scene, every change of shot or camera angle, every action, and any text that appears on screen. Be thorough and write multiple paragraphs."
run vcg_style cuda:1 "Provide a detailed description of the given video."
run short cuda:2 "What is happening in this video?"

FAIL=0
wait || FAIL=1
echo "[done $(date +%H:%M:%S)] FAIL=$FAIL"
for tag in temporal vcg_style short; do
    echo "===== $tag ====="
    cat "$BASE/$tag/caption_eval.md" 2>/dev/null || echo "(missing)"
    echo
done
exit $FAIL
