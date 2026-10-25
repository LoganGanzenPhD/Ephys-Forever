"""Min/max envelope used to draw long traces."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ephys_forever.gui.views.envelope import BASE_BIN, FACTOR, Envelope


def test_zoomed_in_returns_raw_samples() -> None:
    y = np.arange(100_000, dtype=np.float64)
    x, v = Envelope(y).view(1000, 1500, max_points=2000)
    np.testing.assert_array_equal(x, np.arange(999, 1501))
    np.testing.assert_array_equal(v, y[999:1501])


@pytest.mark.parametrize("n", [2_000_000, BASE_BIN * FACTOR**3 + 17])
def test_zoomed_out_stays_under_budget_and_keeps_extremes(n: int) -> None:
    y = np.random.default_rng(1).normal(size=n)
    y[n // 2 + 3] = 50.0  # single-sample spikes must survive
    y[7] = -50.0
    env = Envelope(y)
    x, v = env.view(0, n, max_points=4000)
    assert x.size <= 4000
    assert (v.max(), v.min()) == (50.0, -50.0)
    assert env.extent() == (-50.0, 50.0)


def test_short_signal() -> None:
    env = Envelope(np.array([3.0, 1.0, 2.0]))
    _, v = env.view(0, 3, max_points=8)
    np.testing.assert_array_equal(v, [3.0, 1.0, 2.0])
    assert env.extent() == (1.0, 3.0)


def test_empty_and_out_of_range_views() -> None:
    x, _ = Envelope(np.arange(10.0)).view(20, 30, max_points=100)
    assert x.size == 0
    assert np.isnan(Envelope(np.empty(0)).extent()[0])


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        Envelope(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="max_points"):
        Envelope(np.zeros(4)).view(0, 4, max_points=2)


@settings(max_examples=80, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=3 * BASE_BIN * FACTOR),
    a=st.floats(min_value=0, max_value=1),
    b=st.floats(min_value=0, max_value=1),
    budget=st.integers(min_value=8, max_value=3000),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_view_covers_every_visible_extreme(
    n: int, a: float, b: float, budget: int, seed: int
) -> None:
    y = np.random.default_rng(seed).normal(size=n)
    start, stop = sorted((a * n, b * n))
    x, v = Envelope(y).view(start, stop, max_points=budget)
    lo, hi = max(int(np.floor(start)), 0), min(int(np.ceil(stop)), n)
    if hi <= lo:
        return
    assert x.size <= budget
    assert v.max() >= y[lo:hi].max()
    assert v.min() <= y[lo:hi].min()
    assert np.isin(v, y).all()
    assert np.all(np.diff(x) >= 0)
    assert x.min() >= 0 and x.max() <= n - 1
