"""Tests for Module 7: Domain-of-validity (Paper 7)."""
import numpy as np
import pytest

from preflight.core.modules.module4_domain_validity.classify_v34 import (
    bootstrap_ba,
    compute_metrics,
    compute_wald_ratio,
)


class TestComputeWaldRatio:
    def test_basic_causal_estimate(self):
        result = compute_wald_ratio(
            beta_exposure=0.5, se_exposure=0.1,
            beta_outcome=0.25, se_outcome=0.05,
        )
        assert result["beta"] == pytest.approx(0.5, abs=1e-10)
        assert result["se"] == pytest.approx(0.1, abs=1e-10)
        assert result["ci_lower"] < result["beta"] < result["ci_upper"]

    def test_significant_result_marked_causal(self):
        result = compute_wald_ratio(
            beta_exposure=1.0, se_exposure=0.1,
            beta_outcome=0.5, se_outcome=0.01,
        )
        assert result["mr_supports_causal"] is True
        assert result["p"] < 0.05

    def test_null_result_not_causal(self):
        result = compute_wald_ratio(
            beta_exposure=1.0, se_exposure=0.1,
            beta_outcome=0.001, se_outcome=0.5,
        )
        assert result["mr_supports_causal"] is False
        assert result["p"] > 0.05

    def test_ci_width_proportional_to_se(self):
        narrow = compute_wald_ratio(1.0, 0.1, 0.5, 0.01)
        wide = compute_wald_ratio(1.0, 0.1, 0.5, 0.5)
        narrow_width = narrow["ci_upper"] - narrow["ci_lower"]
        wide_width = wide["ci_upper"] - wide["ci_lower"]
        assert wide_width > narrow_width

    def test_negative_exposure_flips_direction(self):
        result = compute_wald_ratio(
            beta_exposure=-1.0, se_exposure=0.1,
            beta_outcome=0.5, se_outcome=0.05,
        )
        assert result["beta"] == pytest.approx(-0.5, abs=1e-10)


class TestComputeMetrics:
    def test_perfect_classifier(self):
        preds = ["SUCCESS"] * 10 + ["FAILURE"] * 10
        labels = ["SUCCESS"] * 10 + ["FAILURE"] * 10
        m = compute_metrics(preds, labels)
        assert m["balanced_accuracy"] == pytest.approx(1.0)
        assert m["mcc"] == pytest.approx(1.0)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(1.0)
        assert m["tp"] == 10
        assert m["tn"] == 10
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_all_wrong_classifier(self):
        preds = ["FAILURE"] * 10 + ["SUCCESS"] * 10
        labels = ["SUCCESS"] * 10 + ["FAILURE"] * 10
        m = compute_metrics(preds, labels)
        assert m["balanced_accuracy"] == pytest.approx(0.0)
        assert m["mcc"] == pytest.approx(-1.0)

    def test_random_classifier_near_half(self):
        rng = np.random.default_rng(42)
        n = 1000
        labels = rng.choice(["SUCCESS", "FAILURE"], n).tolist()
        preds = rng.choice(["SUCCESS", "FAILURE"], n).tolist()
        m = compute_metrics(preds, labels)
        assert 0.3 < m["balanced_accuracy"] < 0.7

    def test_all_positive_predictions(self):
        preds = ["SUCCESS"] * 20
        labels = ["SUCCESS"] * 10 + ["FAILURE"] * 10
        m = compute_metrics(preds, labels)
        assert m["sensitivity"] == pytest.approx(1.0)
        assert m["specificity"] == pytest.approx(0.0)
        assert m["balanced_accuracy"] == pytest.approx(0.5)

    def test_counts_sum_to_n(self):
        preds = ["SUCCESS", "FAILURE", "SUCCESS", "FAILURE", "SUCCESS"]
        labels = ["SUCCESS", "SUCCESS", "FAILURE", "FAILURE", "SUCCESS"]
        m = compute_metrics(preds, labels)
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == m["n"]


class TestBootstrapBA:
    def test_perfect_classifier_bootstrap(self):
        preds = ["SUCCESS"] * 20 + ["FAILURE"] * 20
        labels = ["SUCCESS"] * 20 + ["FAILURE"] * 20
        result = bootstrap_ba(preds, labels, n_boot=500)
        assert result["ba_mean"] == pytest.approx(1.0, abs=0.02)
        assert result["ba_ci_lo"] > 0.9

    def test_random_classifier_bootstrap_near_half(self):
        rng = np.random.default_rng(55)
        n = 200
        labels = rng.choice(["SUCCESS", "FAILURE"], n).tolist()
        preds = rng.choice(["SUCCESS", "FAILURE"], n).tolist()
        result = bootstrap_ba(preds, labels, n_boot=500)
        assert 0.3 < result["ba_mean"] < 0.7
        assert result["ba_ci_lo"] < result["ba_mean"] < result["ba_ci_hi"]

    def test_ci_contains_point_estimate(self):
        preds = ["SUCCESS"] * 15 + ["FAILURE"] * 5 + ["SUCCESS"] * 3 + ["FAILURE"] * 7
        labels = ["SUCCESS"] * 15 + ["SUCCESS"] * 5 + ["FAILURE"] * 3 + ["FAILURE"] * 7
        result = bootstrap_ba(preds, labels, n_boot=1000)
        point = compute_metrics(preds, labels)["balanced_accuracy"]
        assert result["ba_ci_lo"] <= point <= result["ba_ci_hi"]
