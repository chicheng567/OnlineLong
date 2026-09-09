# Caption prompts

Drawn from by `dataset_util/recaption_videos.py` and `dataset_util/recaption_videos_vllm.py`
(`--prompt_pool`, default = every `.txt` here). With neither `--prompt` nor `--prompt_file`,
each sample draws one via `hash(video_id, seed)`, so a single pass over a dataset carries a mix
of caption styles instead of one voice repeated N times.

The `scene_shift` / `motion_scene` prompts were written for one goal: **carry the dynamics of
the video, and re-establish the scene whenever it changes.** `baseline_default` is the plain
fallback. They are deliberately different styles, so running a dataset through more than one buys
caption diversity rather than paraphrases of the same sentence.

## Measured on the frozen VideoLLaMA3 path (12 videos: 5 ActivityNet heavy scene change, 5 mixed, 2 single-scene Charades)

Frozen `videollama3_7b_local`, 16 frames at 1 FPS, greedy, `repetition_penalty 1.1`,
`max_new_tokens 420`. `action%` = share of sentences with a motion/change verb; `motion` = motion
verbs per 100 words; `scene` = distinct setting nouns per caption; `cam` = camera-motion mentions.

| prompt | words | action% | motion | temporal | scene | cam |
|---|---|---|---|---|---|---|
| baseline_default | 123 | 0.62 | 6.2 | 4.5 | 2.2 | 1.5 |
| **scene_shift** | 109 | 0.77 | 9.0 | 4.3 | 2.0 | 1.8 |
| **motion_scene** | 118 | 0.75 | 7.6 | 4.9 | **2.5** | 2.6 |

On the 5 heaviest scene-change videos, `motion_scene` names the most settings (3.0 vs 2.0 for
the baseline).

## Which to use

- `scene_shift` — best all-round. Shot-by-shot paragraph: establishes each shot, then narrates
  the action in it. Use this if you only run one.
- `motion_scene` — motion-first, surroundings described only at the opening and at each change.
  Best scene coverage on multi-shot videos. Softened 2026-09 so it says "the scene stays static"
  instead of forcing an action verb into every sentence (which made it invent motion on quiet
  footage).
- `baseline_default` — plain one-paragraph description; safe, a little generic.

## Qwen3-VL (recaption_videos_vllm.py) — verified 2026-09 on 4 internVid clips (5–15 min, 48–128 frames)

- `scene_shift`, `motion_scene`, `baseline_default` all degrade gracefully at this frame budget
  (≈1 frame / 5 s): they summarise honestly and rarely confabulate.
- **`timed_segments` was removed.** Asking Qwen3-VL for an exhaustive `<time>s, <where>, <what>`
  list makes it loop and fabricate on any low-information stretch — on a static clip it produced
  47 near-identical lines and a timeline running to 2700 s on a 324 s video. Raising `max_frames`
  to 128 did not fix it. If you need time-grounded output, do it on curated slow-paced footage
  with a much larger frame budget, not in the bulk pool.

## Two behaviours worth knowing before writing a new one

1. **VideoLLaMA3 overrides multi-line output formats** — a numbered per-shot template collapses to
   its native `"0.7 - 15.2seconds, A man is talking …"` dense-caption mode. Qwen3-VL will *follow*
   a multi-line template, but then pads/loops it to fill runtime it cannot see (see
   `timed_segments` above). Prefer a paragraph either way.
2. **Banning static appearance only half works.** "Do not describe clothing, hair or colours" cuts
   those passages down but never removes them; adding "every sentence must also carry an action"
   raises the action ratio but, on Qwen3-VL, also induces confabulation — pair it with an explicit
   "if nothing moves, say so".
