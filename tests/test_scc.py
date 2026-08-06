import numpy as np
import pytest
from scib_validity import scc, scc_multi, cka_null, cka_certifiable


def _make_separable_data(n_per_class=50, d=20, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 5
    X = np.vstack([centers[i] + rng.standard_normal((n_per_class, d)) for i in range(n_classes)])
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return X, labels


class TestSCC:
    def test_perfect_transfer_scores_high(self):
        X, labels = _make_separable_data(n_per_class=100, d=20, seed=0)
        score = scc(X, X, labels, labels)
        assert score > 0.9

    def test_random_target_scores_lower(self):
        X_src, labels_src = _make_separable_data(n_per_class=100, d=20, seed=0)
        rng = np.random.default_rng(99)
        X_tgt = rng.standard_normal(X_src.shape)
        score = scc(X_src, X_tgt, labels_src)
        assert score < scc(X_src, X_src, labels_src, labels_src)

    def test_all_classifiers_return_scores(self):
        X, labels = _make_separable_data(seed=1)
        result = scc_multi(X, X, labels, labels)
        assert set(result.keys()) == {"logreg", "knn", "rf", "svm"}
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_shared_types_filtering(self):
        X, labels = _make_separable_data(n_per_class=30, d=5, n_classes=5, seed=2)
        rng = np.random.default_rng(42)
        X_noisy = X + rng.standard_normal(X.shape) * 3
        score_all = scc(X, X_noisy, labels, labels)
        score_subset = scc(X, X_noisy, labels, labels, shared_types=[0, 1])
        assert score_all != pytest.approx(score_subset, abs=0.01)

    def test_unknown_classifier_raises(self):
        X, labels = _make_separable_data(seed=3)
        with pytest.raises(ValueError, match="Unknown classifier"):
            scc(X, X, labels, classifier="xgboost")


class TestCKANull:
    def test_null_increases_with_d(self):
        assert cka_null(10, 512) > cka_null(10, 64)

    def test_null_decreases_with_k(self):
        assert cka_null(5, 512) > cka_null(50, 512)

    def test_high_d_low_k_near_one(self):
        assert cka_null(5, 1024) > 0.99

    def test_certifiable_below_null(self):
        floor = cka_null(10, 512)
        assert not cka_certifiable(floor - 0.01, 10, 512)

    def test_certifiable_above_null(self):
        floor = cka_null(10, 512)
        assert cka_certifiable(floor + 0.01, 10, 512)

    def test_k_less_than_2_raises(self):
        with pytest.raises(ValueError):
            cka_null(1, 512)

    def test_formula_matches_empirical(self):
        from scib_validity.metrics.cka_null import cka_null_empirical
        for k in [5, 15, 30]:
            emp = cka_null_empirical(k, 256, n_trials=3000, seed=0)
            formula = cka_null(k, 256)
            assert formula == pytest.approx(emp["mean"], abs=0.02)
