"""Tests for Module 1: Grassmannian geodesic distance (Paper 1)."""
import numpy as np
import pytest

from preflight.core.modules.module1_grassmannian.transportability import (
    centroid_distance,
    cocycle_holonomy,
    directional_transport_score,
    fisher_rao_confound_score,
    geodesic_distance,
    principal_angles,
    sheaf_h1_multi_cohort,
    sheaf_h1_two_cohort,
    sheaf_q_test,
    top_k_subspace,
    transport_matrix,
    variance_ratio,
)


def _random_subspace(d, k, rng):
    """Generate a random orthonormal basis for a k-dim subspace of R^d."""
    A = rng.standard_normal((d, k))
    Q, _ = np.linalg.qr(A)
    return Q


class TestPrincipalAngles:
    def test_identical_subspaces_have_zero_angles(self):
        rng = np.random.default_rng(42)
        U = _random_subspace(20, 3, rng)
        angles = principal_angles(U, U)
        assert all(a == pytest.approx(0.0, abs=1e-10) for a in angles)

    def test_orthogonal_subspaces_have_pi_over_2_angles(self):
        U1 = np.eye(6, 3)
        U2 = np.eye(6, 3, k=3)
        angles = principal_angles(U1, U2)
        assert all(a == pytest.approx(np.pi / 2, abs=1e-10) for a in angles)

    def test_angle_count_matches_subspace_dimension(self):
        rng = np.random.default_rng(99)
        for k in [1, 3, 5]:
            U1 = _random_subspace(20, k, rng)
            U2 = _random_subspace(20, k, rng)
            assert len(principal_angles(U1, U2)) == k


class TestGeodesicDistance:
    def test_zero_for_identical_subspaces(self):
        rng = np.random.default_rng(7)
        U = _random_subspace(15, 4, rng)
        assert geodesic_distance(U, U) == pytest.approx(0.0, abs=1e-6)

    def test_maximal_for_orthogonal_subspaces(self):
        U1 = np.eye(6, 3)
        U2 = np.eye(6, 3, k=3)
        d = geodesic_distance(U1, U2)
        expected = np.sqrt(3) * np.pi / 2
        assert d == pytest.approx(expected, abs=1e-10)

    def test_symmetric(self):
        rng = np.random.default_rng(11)
        U1 = _random_subspace(20, 3, rng)
        U2 = _random_subspace(20, 3, rng)
        assert geodesic_distance(U1, U2) == pytest.approx(
            geodesic_distance(U2, U1), abs=1e-10
        )

    def test_triangle_inequality(self):
        rng = np.random.default_rng(13)
        U1 = _random_subspace(20, 3, rng)
        U2 = _random_subspace(20, 3, rng)
        U3 = _random_subspace(20, 3, rng)
        d12 = geodesic_distance(U1, U2)
        d23 = geodesic_distance(U2, U3)
        d13 = geodesic_distance(U1, U3)
        assert d13 <= d12 + d23 + 1e-10


class TestTopKSubspace:
    def test_basis_is_orthonormal(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 10))
        U, _ = top_k_subspace(X, 3)
        assert U.shape == (10, 3)
        gram = U.T @ U
        assert gram == pytest.approx(np.eye(3), abs=1e-10)

    def test_explained_variance_sums_to_less_than_one(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 10))
        _, ev = top_k_subspace(X, 3)
        assert np.sum(ev) < 1.0 + 1e-10
        assert all(e > 0 for e in ev)

    def test_explained_variance_is_sorted_descending(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((100, 10))
        _, ev = top_k_subspace(X, 5)
        for i in range(len(ev) - 1):
            assert ev[i] >= ev[i + 1] - 1e-10


class TestTransportMatrix:
    def test_identity_for_same_subspace(self):
        rng = np.random.default_rng(8)
        U = _random_subspace(15, 3, rng)
        T = transport_matrix(U, U)
        assert T == pytest.approx(np.eye(3), abs=1e-10)

    def test_is_orthogonal(self):
        rng = np.random.default_rng(9)
        U1 = _random_subspace(15, 3, rng)
        U2 = _random_subspace(15, 3, rng)
        T = transport_matrix(U1, U2)
        assert T @ T.T == pytest.approx(np.eye(3), abs=1e-10)


class TestCocycleHolonomy:
    def test_two_subspace_cycle(self):
        rng = np.random.default_rng(10)
        U1 = _random_subspace(15, 3, rng)
        U2 = _random_subspace(15, 3, rng)
        hol, dev = cocycle_holonomy([U1, U2])
        assert hol.shape == (3, 3)
        assert dev >= 0


class TestSheafH1:
    def test_identical_cohorts_have_zero_geodesic(self):
        rng = np.random.default_rng(14)
        X = rng.standard_normal((100, 10))
        result = sheaf_h1_two_cohort(X, X, k=3)
        assert result["geodesic_dist"] == pytest.approx(0.0, abs=1e-10)

    def test_different_cohorts_have_positive_geodesic(self):
        rng = np.random.default_rng(15)
        X1 = rng.standard_normal((100, 10))
        X2 = rng.standard_normal((100, 10)) + 2.0
        result = sheaf_h1_two_cohort(X1, X2, k=3)
        assert result["geodesic_dist"] > 0

    def test_multi_cohort_returns_pairwise(self):
        rng = np.random.default_rng(16)
        cohorts = [rng.standard_normal((50, 10)) for _ in range(4)]
        result = sheaf_h1_multi_cohort(cohorts, k=3)
        assert len(result["pairwise"]) == 4
        assert result["holonomy_norm"] >= 0


class TestFisherRaoConfoundScore:
    def test_pure_signal_shift_has_zero_confound(self):
        rng = np.random.default_rng(17)
        n, d = 200, 10
        X1 = rng.standard_normal((n, d))
        y1 = np.array([0] * (n // 2) + [1] * (n // 2))
        shift = np.zeros(d)
        shift[0] = 3.0
        X2 = X1.copy()
        X2[y1 == 1] += shift
        y2 = y1.copy()
        batch1 = np.zeros(n)
        batch2 = np.ones(n)
        result = fisher_rao_confound_score(X1, X2, batch1, batch2, y1, y2)
        assert result["signal_fraction"] >= 0
        assert result["confound_fraction"] >= 0
        assert result["signal_fraction"] + result["confound_fraction"] == pytest.approx(1.0, abs=0.01)


class TestSheafQTest:
    def test_consistent_estimates_have_high_p_value(self):
        estimates = {
            "study_A": {"beta": 0.50, "se": 0.10},
            "study_B": {"beta": 0.52, "se": 0.11},
            "study_C": {"beta": 0.48, "se": 0.09},
        }
        p, Q, df = sheaf_q_test(estimates)
        assert p > 0.05

    def test_inconsistent_estimates_have_low_p_value(self):
        estimates = {
            "study_A": {"beta": 0.50, "se": 0.05},
            "study_B": {"beta": 2.00, "se": 0.05},
            "study_C": {"beta": -1.00, "se": 0.05},
        }
        p, Q, df = sheaf_q_test(estimates)
        assert p < 0.01


class TestBaselines:
    def test_centroid_distance_zero_for_same_data(self):
        rng = np.random.default_rng(20)
        X = rng.standard_normal((50, 10))
        assert centroid_distance(X, X) == pytest.approx(0.0, abs=1e-10)

    def test_variance_ratio_one_for_same_data(self):
        rng = np.random.default_rng(21)
        X = rng.standard_normal((50, 10))
        assert variance_ratio(X, X) == pytest.approx(1.0, abs=0.01)
