"""Tests for Module 6: Embedding collapse (Paper 6)."""
import numpy as np
import pytest

from preflight.core.modules.module2_domain_shift.transportability import (
    all_distances,
    bootstrap_correlation,
    centroid_distance,
    geodesic_distance,
    mmd_rbf,
    permutation_null,
    principal_angles,
    sheaf_h1_multi_cohort,
    sheaf_h1_two_cohort,
    sliced_wasserstein,
    spectral_matching_degradation,
    top_k_subspace,
)


def _random_subspace(d, k, rng):
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q


class TestMMDRBF:
    def test_zero_for_identical_samples(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        assert mmd_rbf(X, X) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_shifted_distributions(self):
        rng = np.random.default_rng(43)
        X1 = rng.standard_normal((100, 10))
        X2 = rng.standard_normal((100, 10)) + 3.0
        assert mmd_rbf(X1, X2) > 0

    def test_increases_with_larger_shift(self):
        rng = np.random.default_rng(44)
        X1 = rng.standard_normal((200, 10))
        X2_small = X1 + 1.0
        X2_large = X1 + 5.0
        assert mmd_rbf(X1, X2_large) > mmd_rbf(X1, X2_small)


class TestSlicedWasserstein:
    def test_zero_for_identical_samples(self):
        rng = np.random.default_rng(50)
        X = rng.standard_normal((100, 10))
        assert sliced_wasserstein(X, X) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_shifted_distributions(self):
        rng = np.random.default_rng(51)
        X1 = rng.standard_normal((100, 10))
        X2 = rng.standard_normal((100, 10)) + 3.0
        assert sliced_wasserstein(X1, X2) > 0

    def test_increases_with_larger_shift(self):
        rng = np.random.default_rng(52)
        X1 = rng.standard_normal((200, 10))
        X2_small = X1 + 0.5
        X2_large = X1 + 5.0
        assert sliced_wasserstein(X1, X2_large) > sliced_wasserstein(X1, X2_small)


class TestGeodesicDistance:
    def test_zero_for_identical_subspaces(self):
        rng = np.random.default_rng(60)
        U = _random_subspace(15, 3, rng)
        assert geodesic_distance(U, U) == pytest.approx(0.0, abs=1e-6)

    def test_maximal_for_orthogonal_subspaces(self):
        U1 = np.eye(6, 3)
        U2 = np.eye(6, 3, k=3)
        d = geodesic_distance(U1, U2)
        expected = np.sqrt(3) * np.pi / 2
        assert d == pytest.approx(expected, abs=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(61)
        U1 = _random_subspace(20, 3, rng)
        U2 = _random_subspace(20, 3, rng)
        assert geodesic_distance(U1, U2) == pytest.approx(
            geodesic_distance(U2, U1), abs=1e-10
        )


class TestSheafH1TwoCohort:
    def test_identical_data_has_zero_geodesic(self):
        rng = np.random.default_rng(70)
        X = rng.standard_normal((100, 10))
        result = sheaf_h1_two_cohort(X, X, k=3)
        assert result["geodesic_dist"] == pytest.approx(0.0, abs=1e-6)

    def test_different_data_has_positive_geodesic(self):
        rng = np.random.default_rng(71)
        X1 = rng.standard_normal((100, 10))
        X2 = rng.standard_normal((100, 10)) + 2.0
        result = sheaf_h1_two_cohort(X1, X2, k=3)
        assert result["geodesic_dist"] > 0

    def test_returns_explained_variance(self):
        rng = np.random.default_rng(72)
        X1 = rng.standard_normal((100, 10))
        X2 = rng.standard_normal((100, 10))
        result = sheaf_h1_two_cohort(X1, X2, k=3)
        assert len(result["explained_var_c1"]) == 3
        assert len(result["explained_var_c2"]) == 3
        assert all(e > 0 for e in result["explained_var_c1"])


class TestSheafH1MultiCohort:
    def test_identical_cohorts_have_zero_h1(self):
        rng = np.random.default_rng(80)
        X = rng.standard_normal((100, 10))
        result = sheaf_h1_multi_cohort([X, X, X], k=3)
        assert result["h1_norm"] == pytest.approx(0.0, abs=1e-6)

    def test_different_cohorts_have_positive_h1(self):
        rng = np.random.default_rng(81)
        cohorts = [rng.standard_normal((50, 10)) + i for i in range(4)]
        result = sheaf_h1_multi_cohort(cohorts, k=3)
        assert result["h1_norm"] > 0
        assert result["n_cohorts"] == 4
        assert result["n_pairs"] == 6


class TestPermutationNull:
    def test_null_distribution_has_correct_length(self):
        rng = np.random.default_rng(90)
        X1 = rng.standard_normal((50, 10))
        X2 = rng.standard_normal((50, 10))
        null = permutation_null(X1, X2, k=3, n_perms=20)
        assert len(null) == 20

    def test_null_values_are_nonnegative(self):
        rng = np.random.default_rng(91)
        X1 = rng.standard_normal((50, 10))
        X2 = rng.standard_normal((50, 10))
        null = permutation_null(X1, X2, k=3, n_perms=20)
        assert all(d >= 0 for d in null)

    def test_observed_exceeds_most_null_for_shifted_data(self):
        rng = np.random.default_rng(92)
        X1 = rng.standard_normal((80, 10))
        X2 = rng.standard_normal((80, 10)) + 5.0
        U1, _ = top_k_subspace(X1, 3)
        U2, _ = top_k_subspace(X2, 3)
        observed = geodesic_distance(U1, U2)
        null = permutation_null(X1, X2, k=3, n_perms=100)
        p_value = np.mean(null >= observed)
        assert p_value <= 0.1


class TestBootstrapCorrelation:
    def test_perfect_positive_correlation(self):
        xs = np.arange(20, dtype=float)
        ys = xs * 2.0 + 1.0
        result = bootstrap_correlation(xs, ys, n_boot=500)
        assert result["r"] == pytest.approx(1.0, abs=1e-10)
        assert result["p"] < 0.001

    def test_no_correlation_has_wide_ci(self):
        rng = np.random.default_rng(100)
        xs = rng.standard_normal(50)
        ys = rng.standard_normal(50)
        result = bootstrap_correlation(xs, ys, n_boot=500)
        assert result["ci_95"][0] < result["r"] < result["ci_95"][1]


class TestSpectralMatchingDegradation:
    def test_identical_embeddings_cross_hit_is_perfect(self):
        rng = np.random.default_rng(110)
        n = 60
        X = rng.standard_normal((n, 10))
        labels = np.array([f"c_{i % 10}" for i in range(n)])
        within, cross, degradation = spectral_matching_degradation(X, X, labels, labels)
        assert cross == pytest.approx(1.0, abs=1e-10)
        assert degradation == pytest.approx(within - 1.0, abs=1e-10)

    def test_returns_three_values(self):
        rng = np.random.default_rng(111)
        n = 30
        X1 = rng.standard_normal((n, 10))
        X2 = X1 + rng.standard_normal((n, 10)) * 2.0
        labels = np.array([f"c_{i % 10}" for i in range(n)])
        within, cross, degradation = spectral_matching_degradation(X1, X2, labels, labels)
        assert degradation == pytest.approx(within - cross, abs=1e-10)
        assert 0 <= within <= 1
        assert 0 <= cross <= 1


class TestAllDistances:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(120)
        X1 = rng.standard_normal((50, 10))
        X2 = rng.standard_normal((50, 10))
        result = all_distances(X1, X2, k=3, compute_expensive=True)
        assert "geodesic_dist" in result
        assert "centroid_dist" in result
        assert "domain_auc" in result
        assert "mmd" in result
        assert "sliced_wasserstein" in result
        assert "proxy_a_distance" in result

    def test_cheap_mode_skips_expensive_metrics(self):
        rng = np.random.default_rng(121)
        X1 = rng.standard_normal((50, 10))
        X2 = rng.standard_normal((50, 10))
        result = all_distances(X1, X2, k=3, compute_expensive=False)
        assert "geodesic_dist" in result
        assert "mmd" not in result

    def test_identical_data_has_zero_geodesic(self):
        rng = np.random.default_rng(122)
        X = rng.standard_normal((50, 10))
        result = all_distances(X, X, k=3, compute_expensive=False)
        assert result["geodesic_dist"] == pytest.approx(0.0, abs=1e-6)
        assert result["centroid_dist"] == pytest.approx(0.0, abs=1e-10)
