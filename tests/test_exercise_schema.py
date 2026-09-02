import pytest
from pydantic import ValidationError

from spotter.exercises.loader import available_exercises, load_exercise
from spotter.exercises.schema import ExerciseDef


def test_squat_yaml_loads_and_validates():
    ex = load_exercise("squat")
    assert ex.name == "squat"
    assert ex.fsm.states[0].name == ex.fsm.initial_state
    assert "squat" in available_exercises()


def test_unknown_exercise_raises_with_helpful_message():
    with pytest.raises(FileNotFoundError, match="squat"):
        load_exercise("not_a_real_exercise")


def _minimal_exercise_dict(**overrides):
    base = {
        "name": "test_ex",
        "display_name": "Test",
        "camera": {"recommended_angle": "front"},
        "landmarks": {"required": ["left_hip"]},
        "signals": [
            {"name": "knee_angle", "type": "joint_angle", "points": ["hip", "knee", "ankle"], "bilateral": "average"}
        ],
        "primary_signal": "knee_angle",
        "fsm": {
            "initial_state": "top",
            "states": [
                {"name": "top", "enter_when": {"signal": "knee_angle", "condition": "above", "threshold": 160}},
                {"name": "bottom", "enter_when": {"signal": "knee_angle", "condition": "below", "threshold": 100}},
            ],
        },
    }
    base.update(overrides)
    return base


def test_signal_reference_validation_catches_typo():
    bad = _minimal_exercise_dict(primary_signal="not_a_signal")
    with pytest.raises(ValidationError, match="unknown signal"):
        ExerciseDef.model_validate(bad)


def test_initial_state_must_be_first_in_list():
    bad = _minimal_exercise_dict()
    bad["fsm"]["initial_state"] = "bottom"
    with pytest.raises(ValidationError, match="initial_state"):
        ExerciseDef.model_validate(bad)


def test_needs_at_least_two_states():
    bad = _minimal_exercise_dict()
    bad["fsm"]["states"] = bad["fsm"]["states"][:1]
    with pytest.raises(ValidationError, match="at least 2 states"):
        ExerciseDef.model_validate(bad)
