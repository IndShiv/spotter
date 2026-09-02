import math

from spotter.signals.filters import OneEuroFilter


def test_first_sample_passes_through_unfiltered():
    f = OneEuroFilter()
    assert f(42.0, 0.0) == 42.0


def test_smooths_constant_signal_to_itself():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.3)
    t = 0.0
    dt = 1 / 30
    value = 10.0
    for _ in range(30):
        out = f(value, t)
        t += dt
    assert math.isclose(out, value, abs_tol=1e-6)


def test_rejects_jitter_at_rest_more_than_it_lags_fast_motion():
    # At rest: constant value + tiny noise should barely move the output.
    f_rest = OneEuroFilter(min_cutoff=1.0, beta=0.3)
    t = 0.0
    dt = 1 / 30
    outputs = []
    for i in range(60):
        noisy = 100.0 + (0.5 if i % 2 == 0 else -0.5)
        outputs.append(f_rest(noisy, t))
        t += dt
    assert max(outputs[-10:]) - min(outputs[-10:]) < 0.5

    # Fast, real motion should be tracked with low lag (final value close to target).
    f_fast = OneEuroFilter(min_cutoff=1.0, beta=0.3)
    t = 0.0
    out = 0.0
    for i in range(15):
        out = f_fast(0.0 if i < 5 else 100.0, t)
        t += dt
    assert out > 60.0  # caught up most of the way within 10 frames of the step


def test_reset_clears_state():
    f = OneEuroFilter()
    f(10.0, 0.0)
    f(20.0, 1 / 30)
    f.reset()
    assert f(5.0, 100.0) == 5.0
