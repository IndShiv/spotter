# NOTES

Design rationale, threshold choices, and known failure modes. Read this
before changing thresholds or extending the FSM — the "why", not just the
"what", is what keeps future tuning from undoing past tuning.

## Status

`squat` is the only implemented exercise. It's the reference implementation
the rest of Phase 1 (FSM engine, calibration, form checks, eval harness) was
built and tested against. **The other seven exercises (wall ball, burpee
broad jump, sandbag lunge, box jump, push-up, thruster, pull-up) are not
implemented yet** — the schema in `spotter/exercises/schema.py` is meant to
be general enough for all of them (multi-state cycles, per-signal
bilateral averaging, structured form checks), but it's only been proven
against squat's two-state case, so it should get a second pair of eyes
before seven more YAML files get built on top of it. See "Open questions
for the schema" below.

## FSM design: forward-only ratchet, not a bidirectional Schmitt trigger

`spotter/counting/fsm.py` cycles through an exercise's states in a fixed
order — `states[0] -> states[1] -> ... -> states[-1] -> states[0]` — and
never transitions backward. Each state's `enter_when` + `hysteresis` gates
the *forward* transition into it.

This was a deliberate simplification over a classic two-threshold Schmitt
trigger (separate enter/exit thresholds per state). Reasons:

- It generalizes to N-state cycles (a burpee's chest-low -> flight ->
  standing chain) without special-casing a "last state" that needs
  different logic from the middle ones.
- Hysteresis still does real work: widening a state's entry threshold in
  the harder direction (`threshold + hysteresis` for "above", `threshold -
  hysteresis` for "below") means noise has to clearly cross the line, not
  brush it, to register.
- Debounce comes for free: because transitions never reverse, noise that
  bounces the signal back across a threshold *right after* a transition
  can't undo something that already happened. A bidirectional Schmitt
  trigger would need that guarantee proven separately.

The tradeoff: if the athlete does something the state graph didn't expect
(e.g. a half-rep that reverses direction mid-descent), the FSM can't back
out of a state it's already entered — it can only wait out
`rep_duration.max_seconds` and force-reset. That reset always logs a
no-rep with a reason string rather than failing silently; see "Confidence
gate and stuck-FSM reset" below.

## Threshold choices for squat

All in `spotter/exercises/definitions/squat.yaml`.

- **`knee_angle` as the primary signal**, hip-knee-ankle angle, bilateral
  average. Chosen over a vertical-displacement signal because it's roughly
  invariant to camera distance and to where in frame the athlete stands —
  a displacement-based signal would need per-video calibration for camera
  distance that an angle doesn't.
- **`standing: above 160°`, `bottom: below 100°`, both `hysteresis: 6°`.**
  160° is a generous "near full extension" bar — real full lockout is
  closer to 175-180°, but MediaPipe's knee landmark wobbles a few degrees
  even standing still, and 160° already excludes a half-rep. 100° is above
  true "thighs parallel" (~90°) on purpose: the goal here is *rep timing*
  (has the athlete clearly reached depth), not a strict depth judge — the
  `insufficient_depth` form check (115°) is the actual depth gate, kept
  separate so the timing threshold and the form threshold can be tuned
  independently. 6° of hysteresis was picked as roughly 2x the jitter band
  observed on `knee_angle` from a static standing pose during manual
  testing with MediaPipe Lite; this has not yet been re-validated against
  Heavy or real gym footage — **first thing to revisit once real clips are
  available.**
- **`beta: 0.4` on `knee_angle`'s One-Euro filter** (vs. `0.2` on
  `torso_lean_deg`). Squats turn around fast at the bottom, and lag on the
  timing-critical signal shifts when a threshold crossing is detected,
  which corrupts both rep duration and (at speed) can merge two reps into
  one or split one into two. `torso_lean_deg` only feeds a form check, not
  timing, so it's tuned toward smoothness instead.
- **`rep_duration: 0.4s - 6.0s`.** 0.4s lower bound assumes nobody performs
  a full stand-to-depth-to-stand squat cycle faster than that — it exists
  to reject landmark jitter that momentarily crosses both thresholds, not
  to cap how fast a real squat can go (an explosive squat is still well
  above 0.4s). 6.0s upper bound is a stuck-FSM safety net, not a real pacing
  limit — a slow/paused rep can still complete normally as long as it
  finishes within 6s of leaving standing.
- **`warmup_guard: 0.5s`.** Getting into the starting stance (walking into
  frame, adjusting footing) shouldn't register as leaving "standing." This
  only gates the *first* departure from standing for the whole set, not
  every rep — see the FSM design note above for why re-arming it every rep
  would be actively annoying.
- **Calibration (`calibration_reps: 4`, margins 5%/10%)** adapts
  `standing`'s and `bottom`'s thresholds to the athlete's own observed
  range of motion after 4 completed cycles, loosening from the observed
  extreme by the margin (see `FSMCounter._maybe_calibrate`) rather than
  tightening — the goal is fatigue tolerance (a rep that's a bit shallower
  than early-set form should still count late in a set), not a stricter
  depth judge. The 5%/10% split (looser on `bottom` than `standing`) is a
  guess that depth is what degrades with fatigue, not standing extension;
  **unvalidated against real fatigued-set footage.**

## Expected failure modes

- **Loose or draped clothing over the hip/knee crease** biases `knee_angle`
  — this is a known MediaPipe landmark-placement issue, not something
  Spotter's signal layer can correct for. Expect this to show up as
  systematically-off depth readings rather than random noise, so it's a
  "check camera setup / clothing" failure, not a threshold-tuning one.
- **Extreme camera angles (near-directly-overhead, near-ground)** are
  outside what `pose_landmarker` was trained on; visibility scores drop and
  the confidence gate should catch it, but a partially-visible pose with
  *plausible-looking but wrong* landmark positions is the harder case — the
  gate only looks at the visibility score MediaPipe reports, not at
  whether the geometry is physically sane.
- **PeakCounter (`spotter/counting/peaks.py`) recomputes PCA + find_peaks
  over the whole rolling window every frame** — O(window) per frame, fine
  at video frame rates for reasonable window sizes but not a true streaming
  algorithm. Its peaks near the live edge of the window are deliberately
  held back (`_EDGE_GUARD_SAMPLES`) since they can still change prominence
  as more frames arrive, which means it lags the FSM by a few frames on
  when a rep is confirmed. It's there for benchmarking, not as the default.
- **Single-subject lock is bbox-based, not identity-based.** If the locked
  athlete leaves frame entirely and someone else walks through roughly the
  same screen position, the IoU/centroid continuity logic in
  `spotter/tracking/single_subject.py` can hand the lock to the wrong
  person with no signal that the swap happened. Untested against real
  multi-person gym footage — synthetic tests only cover the lock-acquisition
  and simple-continuity cases (`tests/test_tracking.py`).
- **Calibration collects from every completed cycle within
  `rep_duration` bounds, including ones later marked no-rep by a form
  check.** A form-check failure is often informative about the athlete's
  real ROM (e.g. they genuinely didn't reach depth), so on reflection this
  might bias calibration toward being too lenient over a set with several
  early form breaks. Not yet revisited — flagged here rather than silently
  changed, since it changes counting behavior for every future exercise
  that turns calibration on.

## Open questions for the schema (before building the other 7 exercises)

Surfacing these now rather than baking in an answer no one signed off on:

1. **Multi-signal states.** Squat's states each gate on one signal
   (`knee_angle`). Wall ball needs squat depth *and* overhead arm
   extension to close a rep — does that mean two signals feeding one
   state's `enter_when` (schema doesn't support that yet — `EnterWhen` is
   single-signal), or a longer state chain (depth state -> extension
   state)? The latter fits the current schema with no changes but changes
   what "hysteresis" means when two states must both hold roughly
   simultaneously rather than in sequence.
2. **Object-detection-gated states** (wall ball's `ball_detected_above_target`
   stub, per the project brief). Nothing in the current schema represents
   a non-landmark-derived predicate — it'd need a new `SignalType` or a
   separate `external_predicates` section, not just a new signal.
3. **Occlusion-heavy exercises** (sandbag lunge: the bag blocks hip
   landmarks). `bilateral: average` already down-weights an occluded side
   by visibility, but a lunge's non-lead-leg tracking may need a signal
   that tolerates one whole side being unusable for a full rep, not just
   one frame — untested territory for the current `SignalComputer`.
4. **`world_landmarks`** (metric 3D, on `Frame` already) aren't used by any
   current signal — `spotter/signals/geometry.py`'s functions all take
   whatever points they're given and work in 2D or 3D transparently, but
   nothing in the squat schema exercises the 3D path. Worth deciding now
   whether exercises with heavy camera-angle variability (box jump?) should
   default to `world_landmarks` before seven YAML files pick inconsistent
   answers.

None of these block squat. They block confidently writing the next seven
exercises against the current schema without revisiting it.
