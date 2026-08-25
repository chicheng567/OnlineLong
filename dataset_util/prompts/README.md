# Caption prompts

Fed to `dataset_util/recaption_from_features.py --prompt_file <this>.txt`.

The four `scene_shift` / `motion_scene` / `state_change` / `timed_segments` prompts were written
and measured for one goal: **carry the dynamics of the video, and re-establish the scene whenever
it changes.** They are deliberately four different styles, so running the dataset through more
than one of them buys caption diversity rather than four paraphrases of the same sentence.

## Measured (12 videos: 5 ActivityNet with heavy scene change, 5 mixed, 2 single-scene Charades)

Frozen `videollama3_7b_local`, 16 frames at 1 FPS from the same feature cache, greedy,
`repetition_penalty 1.1`, `max_new_tokens 420`. `action%` = share of sentences carrying a motion
or change verb; `motion` = motion verbs per 100 words; `scene` = distinct setting nouns per
caption; `cam` = camera-motion mentions.

| prompt | words | action% | motion | temporal | scene | cam |
|---|---|---|---|---|---|---|
| baseline_default | 123 | 0.62 | 6.2 | 4.5 | 2.2 | 1.5 |
| dynamic_events | 91 | 0.80 | 7.1 | 2.6 | 1.5 | 1.0 |
| dynamic_narrative | 112 | 0.79 | 9.2 | 5.7 | 1.6 | 2.9 |
| motion_only | 95 | 0.82 | 9.0 | 3.7 | 1.4 | 2.4 |
| **scene_shift** | 109 | 0.77 | 9.0 | 4.3 | 2.0 | 1.8 |
| **motion_scene** | 118 | 0.75 | 7.6 | 4.9 | **2.5** | 2.6 |
| **state_change** | 129 | **0.81** | **10.1** | 5.0 | 1.5 | **3.7** |
| **timed_segments** | 53 | 0.87 | 8.1 | 1.6 | 0.6 | 1.0 |

On the 5 heaviest scene-change videos only, `motion_scene` names the most settings (3.0 vs 2.0
for the baseline) and `state_change` keeps the highest motion density (9.5).

## Which to use

- `scene_shift` — best all-round. Shot-by-shot paragraph: establishes each shot, then narrates the
  action in it. Use this if you only run one.
- `motion_scene` — motion-first, surroundings described only at the opening and at each change.
  Best scene coverage exactly where it matters (multi-shot videos).
- `state_change` — "At the start: … <change> … At the end: …". Highest motion and camera density;
  the 8-sentence cap is load-bearing (without it the model runs into `max_new_tokens`).
- `timed_segments` — the only prompt whose output is time-grounded: `<start>s - <end>s, <where>,
  <what happens>`, taken from the `Time XXs:` frame markers. Measured on 12 videos: every cited
  time inside the true duration, 9/12 monotonic, covering 98% of the runtime. Much terser than the
  others (≈53 words) — pair it with one of the paragraph prompts rather than using it alone, and
  only for feature caches with real timestamps (never `timestamps_synthetic`).

## Two behaviours worth knowing before writing a new one

1. **The model overrides multi-line output formats.** Asking for `1. Scene: … Action: …` lines, or
   any numbered per-shot template, collapses to its native dense-caption mode —
   `"0.7 - 15.2seconds, A man is talking …"` — three terse lines for the whole video. Ask for a
   paragraph, or ask in that native timed form (`timed_segments`), but do not fight it.
2. **Banning static appearance only half works.** "Do not describe clothing, hair or colours" cuts
   those passages down but never removes them; adding "every sentence must also carry an action"
   is what actually raises the action ratio.
