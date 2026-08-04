"""The cost arithmetic, tested to 6 decimal places.

This is the number that goes on a résumé and into F8's headline table, so it is
tested against values computed by hand rather than against whatever the code
happens to return today. `cost_per_1k` is pure arithmetic and needs no GPU, which
is exactly why F7 requires it to be laptop-testable (F7 §Scope 3).
"""

from __future__ import annotations

import pytest

from sxl.config import T4_HOURLY_USD
from sxl.gpu.bench import BenchError, cost_per_1k


def test_ten_docs_per_second_on_a_t4_costs_the_hand_computed_amount():
    # (1000 / 10) / 3600 * 0.35 = 100 s of GPU time = 0.0277... h * $0.35
    assert cost_per_1k(10.0, 0.35) == pytest.approx(0.009722222, abs=1e-6)


def test_one_doc_per_second_costs_ten_times_as_much():
    assert cost_per_1k(1.0, 0.35) == pytest.approx(0.097222222, abs=1e-6)


def test_doubling_throughput_exactly_halves_cost():
    # The relationship is the whole point of the metric: if it ever stops holding,
    # someone has introduced a fixed overhead term that does not belong here.
    assert cost_per_1k(2.0, 0.35) == pytest.approx(cost_per_1k(1.0, 0.35) / 2)
    assert cost_per_1k(37.5, 0.35) == pytest.approx(cost_per_1k(18.75, 0.35) / 2)


def test_cost_scales_linearly_with_the_hourly_rate():
    # A reader on AWS g4dn.xlarge (~$0.53/h) must be able to rescale in one
    # multiplication, which is only true if the rate is a clean linear factor.
    assert cost_per_1k(10.0, 0.70) == pytest.approx(cost_per_1k(10.0, 0.35) * 2)


def test_the_default_rate_is_the_documented_assumption():
    # Not a measurement. If this constant moves, every committed bench file's
    # cost figure is stale and must be regenerated, not edited.
    assert T4_HOURLY_USD == 0.35


@pytest.mark.parametrize("throughput", [0.0, -1.0, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_throughput_raises_instead_of_returning_inf(throughput):
    # `inf` would serialize as the literal `Infinity`, which is not valid JSON, and
    # would reach F8's table as a silent placeholder rather than a crash.
    with pytest.raises(BenchError):
        cost_per_1k(throughput, 0.35)


def test_a_negative_hourly_rate_raises():
    with pytest.raises(BenchError):
        cost_per_1k(10.0, -0.35)


def test_a_zero_hourly_rate_is_allowed_and_costs_nothing():
    # Someone benchmarking on hardware they already own is not an error.
    assert cost_per_1k(10.0, 0.0) == 0.0
