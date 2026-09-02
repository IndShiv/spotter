from __future__ import annotations

from pathlib import Path

import yaml

from spotter.exercises.schema import ExerciseDef

DEFINITIONS_DIR = Path(__file__).parent / "definitions"


def available_exercises() -> list[str]:
    return sorted(p.stem for p in DEFINITIONS_DIR.glob("*.yaml"))


def load_exercise(name_or_path: str) -> ExerciseDef:
    """Load an exercise by name (looked up in definitions/) or by explicit path."""
    candidate = Path(name_or_path)
    if candidate.suffix in (".yaml", ".yml") and candidate.exists():
        path = candidate
    else:
        path = DEFINITIONS_DIR / f"{name_or_path}.yaml"
        if not path.exists():
            known = ", ".join(available_exercises()) or "none defined yet"
            raise FileNotFoundError(
                f"No exercise definition for {name_or_path!r}. Available: {known}"
            )

    with path.open() as f:
        raw = yaml.safe_load(f)
    return ExerciseDef.model_validate(raw)
