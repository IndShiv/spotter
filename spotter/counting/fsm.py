"""The primary rep counter: a hysteresis-gated finite state machine driven
entirely by an `ExerciseDef`.

Design notes (see NOTES.md for the full rationale):

- States form a forward-only ratchet, not a bidirectional Schmitt trigger:
  `states[0] -> states[1] -> ... -> states[-1] -> states[0]` (wrap =
  one rep). We never transition backward. This generalizes cleanly to
  exercises with more than two states (e.g. a burpee's chest-low -> flight
  -> standing chain) without special-casing, and it gets debounce "for
  free" — noise that bounces the signal back across a threshold right after
  a transition simply can't undo a transition that already happened.
- `hysteresis` widens each state's entry threshold in the direction that
  makes it *harder* to enter, so a transition only fires once the signal has
  clearly crossed the line, not just brushed it.
- Confidence gating, the warm-up guard, min/max rep duration, and
  calibration are all enforced here, uniformly, regardless of which
  exercise is loaded — none of that logic belongs in the YAML.
"""

from __future__ import annotations

from dataclasses import dataclass

from spotter.counting.base import CounterState, RepCounter
from spotter.counting.confidence import ConfidenceGate
from spotter.counting.signal_runtime import SignalComputer
from spotter.exercises.schema import ExerciseDef, FormCheckSpec, StateSpec
from spotter.pose.base import Frame


def _effective_threshold(state: StateSpec, threshold: float) -> float:
    if state.enter_when.condition == "above":
        return threshold + state.hysteresis
    return threshold - state.hysteresis


@dataclass
class _RepSample:
    timestamp: float
    values: dict[str, float]


@dataclass
class FSMCounter(RepCounter):
    exercise: ExerciseDef

    def __post_init__(self) -> None:
        self._signals = SignalComputer(self.exercise)
        self._gate = ConfidenceGate(self.exercise.landmarks)
        self.reset()

    def reset(self) -> None:
        self._signals.reset()
        self._gate.reset()
        self.rep_count = 0
        self.no_rep_count = 0
        self._state_idx: int | None = None  # None = seeking initial state
        self._state_entered_at: float = 0.0
        self._warmup_satisfied = False
        self._rep_start_time: float | None = None
        self._history: list[_RepSample] = []
        self._thresholds: dict[str, float] = {
            s.name: s.enter_when.threshold for s in self.exercise.fsm.states
        }
        self._calibration_samples: dict[str, list[float]] = {
            rule.state: [] for rule in self.exercise.calibration.adapt
        }
        self._calibration_reps_seen = 0
        self._calibrated = not self.exercise.calibration.enabled

    @property
    def _states(self) -> list[StateSpec]:
        return self.exercise.fsm.states

    def _state_name(self) -> str:
        if self._state_idx is None:
            return "seeking"
        return self._states[self._state_idx].name

    def _condition_met(self, state: StateSpec, values: dict[str, float]) -> bool:
        value = values[state.enter_when.signal]
        threshold = _effective_threshold(state, self._thresholds[state.name])
        if state.enter_when.condition == "above":
            return value > threshold
        return value < threshold

    def _enter_state(self, idx: int, timestamp: float) -> None:
        self._state_idx = idx
        self._state_entered_at = timestamp

    def update(self, frame: Frame) -> CounterState:
        should_hold, _mean_vis = self._gate.update(frame)
        values = self._signals.compute(frame)

        if should_hold:
            return CounterState(
                rep_count=self.rep_count,
                no_rep_count=self.no_rep_count,
                current_state=self._state_name(),
                signal_values=values,
                hold=True,
                hold_reason=self.exercise.confidence_gate.hold_message,
                calibrating=not self._calibrated,
            )

        if self._state_idx is None:
            if self._condition_met(self._states[0], values):
                self._enter_state(0, frame.timestamp)
            return CounterState(
                rep_count=self.rep_count,
                no_rep_count=self.no_rep_count,
                current_state=self._state_name(),
                signal_values=values,
                calibrating=not self._calibrated,
            )

        if self._state_idx == 0 and not self._warmup_satisfied:
            dwell = frame.timestamp - self._state_entered_at
            if dwell >= self.exercise.warmup_guard.seconds_in_initial_state:
                self._warmup_satisfied = True

        self._history.append(_RepSample(frame.timestamp, values))

        last_rep_duration: float | None = None
        last_no_rep_reason: str | None = None
        just_completed_rep = False

        if self._rep_start_time is not None:
            elapsed = frame.timestamp - self._rep_start_time
            if elapsed > self.exercise.rep_duration.max_seconds:
                self.no_rep_count += 1
                last_no_rep_reason = "FSM stuck: exceeded max rep duration, reset"
                just_completed_rep = True
                self._enter_state(0, frame.timestamp)
                self._rep_start_time = None
                self._history = [_RepSample(frame.timestamp, values)]
                return CounterState(
                    rep_count=self.rep_count,
                    no_rep_count=self.no_rep_count,
                    current_state=self._state_name(),
                    signal_values=values,
                    calibrating=not self._calibrated,
                    last_no_rep_reason=last_no_rep_reason,
                    just_completed_rep=just_completed_rep,
                )

        next_idx = (self._state_idx + 1) % len(self._states)
        next_state = self._states[next_idx]
        leaving_initial = self._state_idx == 0 and next_idx != 0

        if self._condition_met(next_state, values) and not (
            leaving_initial and not self._warmup_satisfied
        ):
            self._enter_state(next_idx, frame.timestamp)
            if leaving_initial:
                self._rep_start_time = frame.timestamp
                self._history = [_RepSample(frame.timestamp, values)]

            if next_idx == 0:  # wrapped back to initial state: one full cycle
                just_completed_rep = True
                last_rep_duration = frame.timestamp - (self._rep_start_time or frame.timestamp)
                counted, reason = self._finish_rep(last_rep_duration)
                last_no_rep_reason = reason
                self._rep_start_time = None
                self._history = [_RepSample(frame.timestamp, values)]
                if not counted:
                    pass  # no_rep_count already incremented in _finish_rep

        return CounterState(
            rep_count=self.rep_count,
            no_rep_count=self.no_rep_count,
            current_state=self._state_name(),
            signal_values=values,
            calibrating=not self._calibrated,
            last_rep_duration=last_rep_duration,
            last_no_rep_reason=last_no_rep_reason,
            just_completed_rep=just_completed_rep,
        )

    def _finish_rep(self, duration: float) -> tuple[bool, str | None]:
        if duration < self.exercise.rep_duration.min_seconds:
            self.no_rep_count += 1
            return False, "rejected: rep shorter than min_rep_duration (jitter)"

        failure = self._evaluate_form_checks()
        if failure is not None:
            self.no_rep_count += 1
            return False, failure.message

        self.rep_count += 1
        self._maybe_calibrate()
        return True, None

    def _evaluate_form_checks(self) -> FormCheckSpec | None:
        for check in self.exercise.form_checks:
            if not check.enabled:
                continue
            series = [s.values[check.signal] for s in self._history if check.signal in s.values]
            if not series:
                continue
            if check.aggregate == "min":
                agg = min(series)
            elif check.aggregate == "max":
                agg = max(series)
            else:
                agg = sum(series) / len(series)
            failed = agg > check.threshold if check.condition == "above" else agg < check.threshold
            if failed:
                return check
        return None

    def _maybe_calibrate(self) -> None:
        if self._calibrated:
            return
        primary = self.exercise.primary_signal
        series = [s.values[primary] for s in self._history if primary in s.values]
        if not series:
            return
        self._calibration_reps_seen += 1
        for rule in self.exercise.calibration.adapt:
            bound_value = min(series) if rule.bound == "min" else max(series)
            self._calibration_samples[rule.state].append(bound_value)

        if self._calibration_reps_seen >= self.exercise.calibration.calibration_reps:
            for rule in self.exercise.calibration.adapt:
                samples = self._calibration_samples[rule.state]
                if not samples:
                    continue
                observed = min(samples) if rule.bound == "min" else max(samples)
                margin = rule.margin_pct / 100.0
                if rule.bound == "max":
                    self._thresholds[rule.state] = observed * (1 - margin)
                else:
                    self._thresholds[rule.state] = observed * (1 + margin)
            self._calibrated = True
