"""Shared interface for the two counting strategies (FSM and PeakCounter).

Keeping both behind one `RepCounter` protocol is what lets `spotter/eval/run.py`
benchmark them head to head on the same clips without any exercise- or
counter-specific branching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from spotter.pose.base import Frame


@dataclass
class CounterState:
    """Snapshot returned after every `RepCounter.update()` call."""

    rep_count: int = 0
    no_rep_count: int = 0
    current_state: str = ""
    signal_values: dict[str, float] = field(default_factory=dict)
    hold: bool = False
    hold_reason: str = ""
    calibrating: bool = False
    last_rep_duration: float | None = None
    last_no_rep_reason: str | None = None
    just_completed_rep: bool = False


class RepCounter(Protocol):
    def update(self, frame: Frame) -> CounterState:
        """Feed one frame for the locked subject; returns the current state."""
        ...

    def reset(self) -> None:
        """Zero counts and clear all internal state (new set / new athlete)."""
        ...
