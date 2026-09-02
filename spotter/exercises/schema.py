"""Pydantic schema for exercise definitions (spotter/exercises/definitions/*.yaml).

This is the contract between "someone tuning an exercise" and "the FSM
engine that runs it" — see spotter/counting/fsm.py for how each field is
consumed. Deliberately no Python thresholds live outside this schema: a new
exercise, or a re-tune of an existing one, should never require touching
counting/fsm.py.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Condition = Literal["above", "below"]
Aggregate = Literal["min", "max", "mean"]
SignalType = Literal["joint_angle", "vertical_displacement", "height_ratio", "torso_lean"]
Bilateral = Literal["average", "left", "right", "none"]


class CameraGuidance(BaseModel):
    recommended_angle: str
    distance: str = ""
    notes: str = ""


class LandmarksSpec(BaseModel):
    required: list[str]
    min_visibility: float = Field(0.6, ge=0.0, le=1.0)
    frames_before_hold: int = Field(15, ge=1)


class SmoothingSpec(BaseModel):
    filter: Literal["one_euro", "none"] = "one_euro"
    min_cutoff: float = Field(1.0, gt=0)
    beta: float = Field(0.3, ge=0)
    d_cutoff: float = Field(1.0, gt=0)


class SignalSpec(BaseModel):
    name: str
    type: SignalType
    # Landmark names the signal is computed from. Meaning depends on `type`:
    #   joint_angle: [point_a, vertex, point_c] (or left_/right_-prefixed
    #     variants supplied via `bilateral`)
    #   vertical_displacement: [point, reference]
    #   height_ratio: [point_a, point_b]
    #   torso_lean: [shoulder, hip] (mid-points; bilateral is ignored)
    points: list[str]
    bilateral: Bilateral = "none"
    scale_points: list[str] = Field(
        default_factory=list,
        description="For vertical_displacement: two landmarks whose distance normalizes the signal.",
    )
    smoothing: SmoothingSpec = Field(default_factory=SmoothingSpec)


class AdaptRule(BaseModel):
    state: str
    bound: Literal["min", "max"]
    margin_pct: float = 0.0


class CalibrationSpec(BaseModel):
    enabled: bool = False
    calibration_reps: int = Field(3, ge=1, le=20)
    adapt: list[AdaptRule] = Field(default_factory=list)


class EnterWhen(BaseModel):
    signal: str
    condition: Condition
    threshold: float


class StateSpec(BaseModel):
    name: str
    description: str = ""
    # Condition that ratchets the FSM forward into this state from whichever
    # state precedes it in the cycle (see spotter/counting/fsm.py). Every
    # state needs one, including the initial state: its enter_when is what
    # confirms the athlete has returned to the start position, which is both
    # what the warm-up guard watches for and what closes the rep cycle.
    enter_when: EnterWhen
    hysteresis: float = Field(0.0, ge=0.0, description="Extra margin past threshold before the transition fires, to reject noise.")


class FSMSpec(BaseModel):
    initial_state: str
    # Cycle order matches list order: states[0] (== initial_state) ->
    # states[1] -> ... -> states[-1] -> back to states[0], which completes
    # one rep. The FSM only ever moves forward through this list.
    states: list[StateSpec]

    @model_validator(mode="after")
    def _check_states(self) -> FSMSpec:
        names = [s.name for s in self.states]
        if len(names) != len(set(names)):
            raise ValueError("FSM state names must be unique")
        if not names or names[0] != self.initial_state:
            raise ValueError("initial_state must be states[0]")
        if len(self.states) < 2:
            raise ValueError("FSM needs at least 2 states to form a rep cycle")
        return self


class RepDurationSpec(BaseModel):
    min_seconds: float = Field(0.3, gt=0)
    max_seconds: float = Field(10.0, gt=0)

    @model_validator(mode="after")
    def _check_bounds(self) -> RepDurationSpec:
        if self.min_seconds >= self.max_seconds:
            raise ValueError("min_seconds must be < max_seconds")
        return self


class WarmupGuardSpec(BaseModel):
    seconds_in_initial_state: float = Field(0.5, ge=0)


class FormCheckSpec(BaseModel):
    name: str
    signal: str
    aggregate: Aggregate
    condition: Condition
    threshold: float
    message: str
    enabled: bool = True


class ConfidenceGateSpec(BaseModel):
    hold_message: str = "Spotter can't see you"


class ExerciseDef(BaseModel):
    name: str
    display_name: str
    description: str = ""
    camera: CameraGuidance
    landmarks: LandmarksSpec
    signals: list[SignalSpec]
    primary_signal: str
    calibration: CalibrationSpec = Field(default_factory=CalibrationSpec)
    fsm: FSMSpec
    rep_duration: RepDurationSpec = Field(default_factory=RepDurationSpec)
    warmup_guard: WarmupGuardSpec = Field(default_factory=WarmupGuardSpec)
    form_checks: list[FormCheckSpec] = Field(default_factory=list)
    confidence_gate: ConfidenceGateSpec = Field(default_factory=ConfidenceGateSpec)

    @field_validator("signals")
    @classmethod
    def _unique_signal_names(cls, signals: list[SignalSpec]) -> list[SignalSpec]:
        names = [s.name for s in signals]
        if len(names) != len(set(names)):
            raise ValueError("signal names must be unique")
        return signals

    @model_validator(mode="after")
    def _check_signal_references(self) -> ExerciseDef:
        signal_names = {s.name for s in self.signals}

        def require(name: str, where: str) -> None:
            if name not in signal_names:
                raise ValueError(f"{where} references unknown signal {name!r}")

        require(self.primary_signal, "primary_signal")
        for state in self.fsm.states:
            if state.enter_when is not None:
                require(state.enter_when.signal, f"fsm.states[{state.name}].enter_when")
        for check in self.form_checks:
            require(check.signal, f"form_checks[{check.name}]")

        state_names = {s.name for s in self.fsm.states}
        for rule in self.calibration.adapt:
            if rule.state not in state_names:
                raise ValueError(f"calibration.adapt references unknown state {rule.state!r}")
        return self
