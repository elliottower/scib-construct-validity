"""Tests for Module 5: Direction instability (Paper 5)."""
import numpy as np
import pytest

from preflight.core.modules.module3_direction_stability.drug_transport import (
    compute_all_transport_metrics,
    direction_stability,
    frechet_variance,
    gene_level_consistency,
    magnitude_stability,
    subspace_transport,
)
from preflight.core.modules.module3_direction_stability.distances import (
    grassmannian_distance,
    principal_angles,
    subspace_overlap,
)


class TestDirectionStability:
    def test_identical_signatures_have_zero_instability(self):
        sig = np.array([1.0, 0.0, 0.0, 0.5, -0.3])
        signatures = {"A": sig, "B": sig, "C": sig}
        result = direction_stability(signatures)
        assert result["direction_instability"] == pytest.approx(0.0, abs=1e-10)
        assert result["mean_pairwise_cosine"] == pytest.approx(1.0, abs=1e-10)

    def test_orthogonal_signatures_have_high_instability(self):
        signatures = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        result = direction_stability(signatures)
        assert result["direction_instability"] == pytest.approx(1.0, abs=1e-10)

    def test_single_cell_line_returns_nan(self):
        result = direction_stability({"A": np.array([1.0, 2.0])})
        assert np.isnan(result["direction_instability"])

    def test_instability_between_zero_and_two(self):
        rng = np.random.default_rng(42)
        sigs = {f"cell_{i}": rng.standard_normal(100) for i in range(5)}
        result = direction_stability(sigs)
        assert 0 <= result["direction_instability"] <= 2.0


class TestMagnitudeStability:
    def test_identical_norms_have_zero_cv(self):
        signatures = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
        }
        result = magnitude_stability(signatures)
        assert result["magnitude_cv"] == pytest.approx(0.0, abs=1e-10)

    def test_varied_norms_have_positive_cv(self):
        signatures = {
            "A": np.array([1.0, 0.0]),
            "B": np.array([0.0, 10.0]),
        }
        result = magnitude_stability(signatures)
        assert result["magnitude_cv"] > 0


class TestFrechetVariance:
    def test_identical_directions_have_zero_variance(self):
        sigs = {"A": np.array([1.0, 0.0]), "B": np.array([2.0, 0.0])}
        assert frechet_variance(sigs) == pytest.approx(0.0, abs=1e-6)

    def test_dispersed_directions_have_positive_variance(self):
        sigs = {
            "A": np.array([1.0, 0.0, 0.0]),
            "B": np.array([0.0, 1.0, 0.0]),
            "C": np.array([0.0, 0.0, 1.0]),
        }
        assert frechet_variance(sigs) > 0


class TestGeneLevelConsistency:
    def test_identical_signatures_have_perfect_jaccard(self):
        sig = np.arange(100, dtype=float)
        sigs = {"A": sig, "B": sig}
        result = gene_level_consistency(sigs, top_k=10)
        assert result["mean_top_gene_jaccard"] == pytest.approx(1.0, abs=1e-10)

    def test_random_signatures_have_low_jaccard(self):
        rng = np.random.default_rng(55)
        sigs = {f"cell_{i}": rng.standard_normal(1000) for i in range(5)}
        result = gene_level_consistency(sigs, top_k=50)
        assert result["mean_top_gene_jaccard"] < 0.5


class TestComputeAllTransportMetrics:
    def test_returns_all_keys(self):
        rng = np.random.default_rng(77)
        sigs = {f"cell_{i}": rng.standard_normal(50) for i in range(3)}
        result = compute_all_transport_metrics(sigs)
        assert "direction_instability" in result
        assert "magnitude_cv" in result
        assert "frechet_variance" in result
        assert "mean_top_gene_jaccard" in result
