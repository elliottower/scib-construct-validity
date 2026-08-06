"""Tests for Module 9: Ecological bias (Paper 9)."""
import math

import pytest

from preflight.core.modules.module7_ecological_bias.s4_ecological_bounds import (
    duncan_davis_bounds,
)
from preflight.core.modules.module7_ecological_bias.s7_evalue_sensitivity import (
    compute_evalue_from_beta,
    compute_evalue_rr,
)


class TestDuncanDavisBounds:
    def test_known_case(self):
        lower, upper = duncan_davis_bounds(total_mortality=0.3, elderly_prop=0.2)
        assert lower == pytest.approx(max(0.0, (0.3 - 0.8) / 0.2), abs=1e-10)
        assert upper == pytest.approx(min(1.0, 0.3 / 0.2), abs=1e-10)

    def test_bounds_contain_unit_interval(self):
        lower, upper = duncan_davis_bounds(0.5, 0.5)
        assert 0 <= lower <= upper <= 1

    def test_narrow_when_large_elderly_fraction(self):
        lower, upper = duncan_davis_bounds(0.4, 0.9)
        width = upper - lower
        lower2, upper2 = duncan_davis_bounds(0.4, 0.1)
        width2 = upper2 - lower2
        assert width < width2

    def test_extreme_mortality_zero(self):
        lower, upper = duncan_davis_bounds(0.0, 0.5)
        assert lower == pytest.approx(0.0, abs=1e-10)
        assert upper == pytest.approx(0.0, abs=1e-10)

    def test_extreme_mortality_one(self):
        lower, upper = duncan_davis_bounds(1.0, 0.5)
        assert lower == pytest.approx(1.0, abs=1e-10)
        assert upper == pytest.approx(1.0, abs=1e-10)

    def test_returns_none_when_p_zero(self):
        result = duncan_davis_bounds(0.3, 0.0)
        assert result == (None, None)

    def test_returns_none_when_p_one(self):
        result = duncan_davis_bounds(0.3, 1.0)
        assert result == (None, None)

    def test_lower_never_exceeds_upper(self):
        for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
                lower, upper = duncan_davis_bounds(m, p)
                assert lower <= upper + 1e-10


class TestComputeEvalueRR:
    def test_rr_of_one_gives_one(self):
        assert compute_evalue_rr(1.0) == pytest.approx(1.0, abs=1e-10)

    def test_rr_gt_one_gives_evalue_gt_one(self):
        assert compute_evalue_rr(2.0) > 1.0

    def test_rr_lt_one_inverts_first(self):
        assert compute_evalue_rr(0.5) == pytest.approx(compute_evalue_rr(2.0), abs=1e-10)

    def test_known_value_rr_2(self):
        expected = 2.0 + math.sqrt(2.0 * 1.0)
        assert compute_evalue_rr(2.0) == pytest.approx(expected, abs=1e-10)

    def test_monotonically_increasing(self):
        evalues = [compute_evalue_rr(rr) for rr in [1.5, 2.0, 3.0, 5.0, 10.0]]
        for i in range(len(evalues) - 1):
            assert evalues[i] < evalues[i + 1]


class TestComputeEvalueFromBeta:
    def test_positive_beta_gives_rr_gt_one(self):
        result = compute_evalue_from_beta(0.5, 0.1)
        assert result["rr_point"] > 1.0
        assert result["evalue_point"] > 1.0

    def test_large_beta_gives_large_evalue(self):
        small = compute_evalue_from_beta(0.5, 0.1)
        large = compute_evalue_from_beta(5.0, 0.1)
        assert large["evalue_point"] > small["evalue_point"]

    def test_ci_lower_evalue_less_than_point(self):
        result = compute_evalue_from_beta(1.0, 0.2)
        assert result["evalue_ci_lower"] < result["evalue_point"]

    def test_rr_equals_exp_beta(self):
        result = compute_evalue_from_beta(2.0, 0.5)
        assert result["rr_point"] == pytest.approx(math.exp(2.0), abs=1e-10)

    def test_rr_ci_lower_equals_exp_beta_minus_196se(self):
        beta, se = 1.5, 0.3
        result = compute_evalue_from_beta(beta, se)
        assert result["rr_ci_lower"] == pytest.approx(math.exp(beta - 1.96 * se), abs=1e-10)
