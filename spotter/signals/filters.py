"""One-Euro filter for low-latency, jitter-resistant scalar smoothing.

A plain moving average trades latency for smoothness on a fixed window,
which is wrong for rep counting: at the top/bottom of a rep the signal is
nearly static (we want it dead stable, no threshold-crossing jitter) but
during the fast middle of the movement we want to track it with almost no
lag (a laggy signal shifts the FSM's transition timing, which corrupts rep
duration and, at speed, can merge or split reps). The One-Euro filter
(Casiez, Roussel & Vogel, 2012) adapts its cutoff frequency to the signal's
own speed, giving both properties from one filter.
"""

from __future__ import annotations

import math


def _low_pass(x: float, prev: float | None, alpha: float) -> float:
    if prev is None:
        return x
    return alpha * x + (1.0 - alpha) * prev


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Stateful filter for one scalar signal, fed one sample at a time.

    min_cutoff: base cutoff frequency (Hz). Lower = smoother at rest,
        more lag. Raise this if the signal looks jittery when the athlete
        is holding still.
    beta: speed coefficient. Higher = cutoff opens up more aggressively
        as the signal moves faster, cutting lag during fast reps at the
        cost of a bit more noise passing through mid-rep.
    d_cutoff: cutoff frequency for the derivative estimate itself; 1.0 Hz
        is the value used in the reference implementation and rarely
        needs changing.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.3, d_cutoff: float = 1.0):
        if min_cutoff <= 0 or d_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def __call__(self, x: float, timestamp: float) -> float:
        """Filter one sample. `timestamp` is in seconds, monotonically increasing."""
        if self._t_prev is None:
            self._t_prev = timestamp
            self._x_prev = x
            return x

        dt = timestamp - self._t_prev
        if dt <= 0:
            # Duplicate or out-of-order timestamp; hold last output rather
            # than dividing by zero.
            return self._x_prev if self._x_prev is not None else x
        self._t_prev = timestamp

        dx = (x - self._x_prev) / dt
        dx_smoothed = _low_pass(dx, self._dx_prev, _alpha(self.d_cutoff, dt))
        self._dx_prev = dx_smoothed

        cutoff = self.min_cutoff + self.beta * abs(dx_smoothed)
        x_smoothed = _low_pass(x, self._x_prev, _alpha(cutoff, dt))
        self._x_prev = x_smoothed
        return x_smoothed
