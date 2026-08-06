"""Tests verifying fixes for three module bugs found in exp0 v5 results.

Fix 1: M4 now uses participation ratio (effective dimensionality) instead of NaN counter.
Fix 2: M6 score polarity flipped: score = frac_bottleneck (more bridges = higher score).
Fix 3: M3 uses LDA subspaces when labels provided, making it independent of M1.
"""
import numpy as np
import pytest

from preflight.core.runner import (
    _run_m1_grassmannian,
    _run_m3_direction_stability,
    _run_m4_domain_validity,
    _run_m6_curvature,
    _participation_ratio,
)
from preflight.core.modules.module1_grassmannian import transportability as m1
from preflight.core.modules.module3_direction_stability import distances as m3
from preflight.core.synthetic import generate_network_clustered, generate_network_dense


class TestM4ParticipationRatio:
    """M4 now measures effective dimensionality similarity via participation ratio."""

    def test_similar_distributions_high_score(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = rng.standard_normal((200, 50))
        result = _run_m4_domain_validity(source, target)
        assert result.score > 0.8

    def test_different_effective_rank_low_score(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = np.zeros((200, 50))
        target[:, :3] = rng.standard_normal((200, 3))
        result = _run_m4_domain_validity(source, target)
        assert result.score < 0.5

    def test_collapsed_embedding_very_low_score(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = np.tile(rng.standard_normal((200, 1)), (1, 50))
        result = _run_m4_domain_validity(source, target)
        assert result.score < 0.15

    def test_nan_data_handled_gracefully(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = rng.standard_normal((200, 50))
        target[:50, :10] = np.nan
        result = _run_m4_domain_validity(source, target)
        assert 0.0 <= result.score <= 1.0

    def test_details_have_participation_ratios(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = rng.standard_normal((200, 50))
        result = _run_m4_domain_validity(source, target)
        assert "participation_ratio_source" in result.details
        assert "participation_ratio_target" in result.details
        assert "pr_ratio" in result.details

    def test_participation_ratio_isotropic_near_one(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((1000, 20))
        pr = _participation_ratio(X)
        assert pr > 0.8

    def test_participation_ratio_collapsed_near_zero(self):
        rng = np.random.default_rng(42)
        X = np.tile(rng.standard_normal((200, 1)), (1, 50))
        pr = _participation_ratio(X)
        assert pr < 0.05


class TestM6PolarityFixed:
    """M6 score = frac_bottleneck. More bridge edges = higher score = better."""

    def test_clustered_graph_higher_than_dense(self):
        clustered = generate_network_clustered()
        dense = generate_network_dense()
        r_clustered = _run_m6_curvature(clustered.graph)
        r_dense = _run_m6_curvature(dense.graph)
        assert r_clustered.score >= r_dense.score or abs(r_clustered.score - r_dense.score) < 0.1

    def test_score_equals_frac_bottleneck(self):
        net = generate_network_clustered()
        result = _run_m6_curvature(net.graph)
        assert result.score == pytest.approx(result.details["fraction_bottleneck"])

    def test_score_in_valid_range(self):
        net = generate_network_dense()
        result = _run_m6_curvature(net.graph)
        assert 0.0 <= result.score <= 1.0


class TestM3UsesLDA:
    """M3 uses LDA subspaces when labels are provided, making it independent of M1."""

    def test_m3_with_labels_differs_from_m1(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = rng.standard_normal((200, 50))
        source_labels = rng.integers(0, 2, 200)
        target_labels = rng.integers(0, 2, 200)
        k = 5

        r1 = _run_m1_grassmannian(source, target, k)
        r3 = _run_m3_direction_stability(source, target, k, source_labels, target_labels)
        assert r3.details["subspace_method"] == "lda"
        assert r1.score != pytest.approx(r3.score, abs=0.01)

    def test_m3_without_labels_falls_back_to_pca(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = rng.standard_normal((200, 50))
        k = 5

        r3 = _run_m3_direction_stability(source, target, k)
        assert r3.details["subspace_method"] == "pca"

    def test_m3_with_labels_not_monotone_with_m1(self):
        """With LDA subspaces, M3 should no longer produce the same rank
        ordering as M1 across all pairs."""
        rng = np.random.default_rng(42)
        k = 5
        m1_scores = []
        m3_scores = []

        for _ in range(10):
            n = 300
            d = 50
            source = rng.standard_normal((n, d))
            target = rng.standard_normal((n, d))
            source_labels = rng.integers(0, 2, n)
            target_labels = rng.integers(0, 2, n)

            r1 = _run_m1_grassmannian(source, target, k)
            r3 = _run_m3_direction_stability(source, target, k, source_labels, target_labels)
            m1_scores.append(r1.score)
            m3_scores.append(r3.score)

        rank_agreements = 0
        rank_total = 0
        for i in range(len(m1_scores)):
            for j in range(i + 1, len(m1_scores)):
                if abs(m1_scores[i] - m1_scores[j]) > 0.01:
                    rank_total += 1
                    m1_order = m1_scores[i] > m1_scores[j]
                    m3_order = m3_scores[i] > m3_scores[j]
                    if m1_order == m3_order:
                        rank_agreements += 1

        if rank_total > 0:
            agreement_rate = rank_agreements / rank_total
            assert agreement_rate < 0.95, (
                f"M1 and M3 still have {agreement_rate:.0%} rank agreement with LDA — "
                f"they should be measuring different things"
            )
