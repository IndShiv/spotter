# Spotter

A computer-vision rep counter for functional-fitness training — squats,
wall balls, burpees, sandbag lunges, box jumps, and the rest of a
functional-fitness-racing style workout. Spotter watches you the way a
training partner would: it counts honestly, it tells you when it can't see
you, and it never pretends to a certainty it doesn't have.

Phase 1 is a Python CLI you run against recorded video or a webcam. There is
no mobile app yet — the counting logic has to be proven on video first,
since that's where all the accuracy risk lives.

**Currently implemented: `squat`.** It's the reference exercise the rest of
the architecture is built and tested against; see `NOTES.md` for the state
of the other planned exercises.

## Install

Requires Python 3.11+. Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv
uv pip install -e ".[dev]"
```

Or plain `pip`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Linux note:** MediaPipe's Tasks API needs `libEGL`/`libGLES` even for
CPU-only inference. On a minimal/headless image you may need:

```bash
apt-get install -y libgl1 libegl1 libgles2
```

Pose model weights (`pose_landmarker_lite.task` / `_heavy.task`) aren't
checked into the repo — they're downloaded on first use into
`~/.cache/spotter/models/`.

## CLI

```bash
spotter video path/to/clip.mp4 --exercise squat
spotter live --exercise squat
spotter list-exercises
spotter eval
```

`video` — process a recorded clip:

```bash
spotter video clip.mp4 --exercise squat \
  --model heavy \            # lite | full | heavy — heavy for offline accuracy
  --counter fsm \             # fsm | peaks — see spotter/counting/
  --output annotated.mp4 \    # write the overlay to a file
  --dump-signals signals.csv \ # per-frame signal values, for threshold tuning
  --no-show                   # headless / no window
```

`live` — webcam, defaults to the fast `lite` model:

```bash
spotter live --exercise squat --camera 0
```

Press `q` or Esc to stop. Both commands print a final `Reps: N   No-rep: M`
line — "no-rep" is a rep that was attempted but rejected (bad form, too
fast, or a stuck FSM reset), which is intentionally tracked separately from
reps that just never happened.

`eval` — see [Evaluation](#evaluation) below.

## Camera setup

General guidance that applies to every exercise:

- **Full body in frame**, with headroom above the head and floor space
  below the feet through the exercise's full range of motion.
- **Steady camera.** A tripod or a fixed lean against something solid beats
  handheld — camera shake shows up as landmark jitter, which the One-Euro
  filter absorbs up to a point but which costs you tracking margin near
  threshold crossings.
- **Even, front-ish lighting.** Strong backlight (window behind the
  athlete) silhouettes the body and tanks MediaPipe's landmark confidence,
  which is exactly what trips the "Spotter can't see you" hold state.
- **One person per shot** ideally. Spotter locks onto the largest person in
  frame during the first ~30 frames and tracks them by bounding-box
  continuity after that (see `spotter/tracking/`), so it copes with someone
  briefly walking through the background, but a second athlete training
  right next to the camera's subject is asking for trouble.

Per-exercise guidance lives in the exercise's own YAML definition under
`camera:` (see `spotter/exercises/definitions/squat.yaml`) since that's
where it'll stay in sync as thresholds change — this README won't duplicate
it exercise by exercise.

**Squat**, today's reference exercise: front-on or up to ~45° off-front;
pure side-on works but occludes the far leg at depth. Hips through ankles
must stay unoccluded through the bottom of the rep.

## Evaluation

```bash
make eval
# or
spotter eval
```

Runs a fixed set of synthetic clips (`spotter/eval/synthetic.py` — clean,
noisy, dropout, and insufficient-depth scenarios, generated from a
parametric landmark model so they need no video files and run in CI) plus
any real annotated clips found in `data/videos/` (skipped if the video
files aren't present locally — see `data/videos/README.md`).

It reports MAE, OBO (off-by-one accuracy), false positives, and missed reps
per clip and in aggregate, and fails (`--no-gate` to disable) if aggregate
OBO drops below `eval/baseline.json`. Update the baseline deliberately with
`spotter eval --update-baseline` after a change you've verified improves
things — never as a way to silence a real regression.

## Architecture

```
spotter/pose/        backend-agnostic Frame + PoseBackend protocol; MediaPipeBackend
spotter/signals/      pure functions: joint angle, displacement, One-Euro smoothing
spotter/exercises/    declarative YAML exercise definitions + pydantic schema
spotter/counting/     FSMCounter (primary) and PeakCounter (benchmark), same interface
spotter/tracking/     single-subject lock (largest bbox -> IoU continuity)
spotter/overlay/      debug overlay: skeleton, live signal, FSM state, scrolling plot
spotter/eval/         metrics, synthetic clip generator, eval harness + regression gate
spotter/commands/     video/live frame-source loops shared by the CLI
```

Nothing outside `spotter/exercises/definitions/*.yaml` should ever contain
an exercise-specific number — see `NOTES.md` for why, and for the reasoning
behind the thresholds squat.yaml currently ships with.

## Development

```bash
make test    # pytest
make eval    # eval harness + regression gate
make lint    # ruff
```
