"""Tests for preflight.run() — the composite pipeline runner."""
import numpy as np
import pytest

from preflight import run, PreflightReport, ModuleResult
from preflight.core.preregister import DatasetSpec, load_preregistration, verify_preregistration
from preflight.core.runner import (
    DEFAULT_WEIGHTS,
    _composite_score,
    _score_to_tier,
    ALL_MODULE_SOURCES,
)
from preflight.core.synthetic import (
    generate_concordant_estimates,
    generate_discordant_estimates,
    generate_ecological_biased,
    generate_ecological_unbiased,
    generate_missingness_mcar,
    generate_missingness_mnar,
    generate_network_clustered,
    generate_network_dense,
    generate_transfer_positive,
    generate_transfer_negative,
)


class TestRunBasic:
    def test_returns_preflight_report(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        assert isinstance(report, PreflightReport)

    def test_report_has_embedding_modules_by_default(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        assert "m1_grassmannian" in report.module_results
        assert "m3_direction_stability" in report.module_results
        assert "m2_domain_shift" in report.module_results
        assert "m4_domain_validity" in report.module_results

    def test_module_results_are_module_result(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        for name, r in report.module_results.items():
            assert isinstance(r, ModuleResult)
            assert 0.0 <= r.score <= 1.0
            assert 1 <= r.tier <= 7

    def test_overall_tier_matches_score(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        assert 1 <= report.overall_tier <= 7

    def test_timestamp_is_set(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        assert report.timestamp is not None
        assert len(report.timestamp) > 10

    def test_weights_in_hyperparameters(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        assert "weights" in report.hyperparameters


class TestRunModuleSelection:
    def test_run_single_module(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m1_grassmannian"])
        assert len(report.module_results) == 1
        assert "m1_grassmannian" in report.module_results

    def test_run_two_modules(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(
            pair.source, pair.target,
            modules=["m1_grassmannian", "m2_domain_shift"],
        )
        assert len(report.module_results) == 2

    def test_unknown_module_ignored(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m1_grassmannian", "nonexistent"])
        assert len(report.module_results) == 1


class TestRunTransferability:
    def test_positive_pair_scores_higher(self):
        pos = generate_transfer_positive(n_source=300, n_target=300, d=50, k=5)
        neg = generate_transfer_negative(n_source=300, n_target=300, d=50, k=5)
        report_pos = run(pos.source, pos.target, hyperparameters={"k": 5})
        report_neg = run(neg.source, neg.target, hyperparameters={"k": 5})
        assert report_pos.overall_score > report_neg.overall_score

    def test_positive_pair_higher_tier(self):
        pos = generate_transfer_positive(n_source=300, n_target=300, d=50, k=5)
        neg = generate_transfer_negative(n_source=300, n_target=300, d=50, k=5)
        report_pos = run(pos.source, pos.target, hyperparameters={"k": 5})
        report_neg = run(neg.source, neg.target, hyperparameters={"k": 5})
        assert report_pos.overall_tier >= report_neg.overall_tier


# ======================================================================
# M5 cross-design via metadata
# ======================================================================

class TestRunM5CrossDesign:
    def test_concordant_estimates_high_score(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        estimates = generate_concordant_estimates()
        report = run(
            pair.source, pair.target,
            metadata={"estimates": estimates.estimates},
            modules=["m5_cross_design"],
        )
        assert "m5_cross_design" in report.module_results
        r = report.module_results["m5_cross_design"]
        assert r.score > 0.5

    def test_discordant_estimates_low_score(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        estimates = generate_discordant_estimates()
        report = run(
            pair.source, pair.target,
            metadata={"estimates": estimates.estimates},
            modules=["m5_cross_design"],
        )
        r = report.module_results["m5_cross_design"]
        assert r.score < 0.5

    def test_m2_skipped_without_estimates(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m5_cross_design"])
        assert "m5_cross_design" not in report.module_results

    def test_m2_auto_included_with_estimates(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        estimates = generate_concordant_estimates()
        report = run(pair.source, pair.target, metadata={"estimates": estimates.estimates})
        assert "m5_cross_design" in report.module_results


# ======================================================================
# M4 domain validity
# ======================================================================

class TestRunM4DomainValidity:
    def test_similar_distributions_high_score(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m4_domain_validity"])
        r = report.module_results["m4_domain_validity"]
        assert r.score > 0.8

    def test_different_effective_rank_lower_score(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = np.zeros((200, 50))
        target[:, :3] = rng.standard_normal((200, 3))
        report = run(source, target, modules=["m4_domain_validity"])
        r = report.module_results["m4_domain_validity"]
        assert r.score < 0.5

    def test_collapsed_embedding_low_score(self):
        rng = np.random.default_rng(42)
        source = rng.standard_normal((200, 50))
        target = np.tile(rng.standard_normal((200, 1)), (1, 50))
        report = run(source, target, modules=["m4_domain_validity"])
        r = report.module_results["m4_domain_validity"]
        assert r.score < 0.3

    def test_details_contain_participation_ratio(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m4_domain_validity"])
        d = report.module_results["m4_domain_validity"].details
        assert "participation_ratio_source" in d
        assert "participation_ratio_target" in d
        assert "pr_ratio" in d


# ======================================================================
# M6 curvature via metadata
# ======================================================================

class TestRunM6Curvature:
    def test_clustered_network_produces_result(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        net = generate_network_clustered()
        report = run(
            pair.source, pair.target,
            metadata={"graph": net.graph},
            modules=["m6_curvature"],
        )
        assert "m6_curvature" in report.module_results
        r = report.module_results["m6_curvature"]
        assert 0.0 <= r.score <= 1.0

    def test_m6_skipped_without_graph(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m6_curvature"])
        assert "m6_curvature" not in report.module_results

    def test_m6_auto_included_with_graph(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        net = generate_network_clustered()
        report = run(pair.source, pair.target, metadata={"graph": net.graph})
        assert "m6_curvature" in report.module_results

    def test_m6_details_contain_orc(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        net = generate_network_clustered()
        report = run(
            pair.source, pair.target,
            metadata={"graph": net.graph},
            modules=["m6_curvature"],
        )
        d = report.module_results["m6_curvature"].details
        assert "mean_orc" in d
        assert "fraction_bottleneck" in d


# ======================================================================
# M7 ecological bias via metadata
# ======================================================================

class TestRunM7EcologicalBias:
    def test_biased_data_lower_score(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        eco = generate_ecological_biased()
        report = run(
            pair.source, pair.target,
            metadata={"records": eco.records},
            modules=["m7_ecological_bias"],
        )
        assert "m7_ecological_bias" in report.module_results
        r = report.module_results["m7_ecological_bias"]
        assert 0.0 <= r.score <= 1.0

    def test_unbiased_scores_higher_than_biased(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        biased = generate_ecological_biased()
        unbiased = generate_ecological_unbiased()
        r_biased = run(
            pair.source, pair.target,
            metadata={"records": biased.records},
            modules=["m7_ecological_bias"],
        )
        r_unbiased = run(
            pair.source, pair.target,
            metadata={"records": unbiased.records},
            modules=["m7_ecological_bias"],
        )
        assert (r_unbiased.module_results["m7_ecological_bias"].score >=
                r_biased.module_results["m7_ecological_bias"].score)

    def test_m7_skipped_without_records(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target, modules=["m7_ecological_bias"])
        assert "m7_ecological_bias" not in report.module_results

    def test_m7_details_contain_evalue(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        eco = generate_ecological_biased()
        report = run(
            pair.source, pair.target,
            metadata={"records": eco.records},
            modules=["m7_ecological_bias"],
        )
        d = report.module_results["m7_ecological_bias"].details
        assert "evalue" in d
        assert "site_rate_cv" in d


# ======================================================================
# Weighted composite scoring
# ======================================================================

class TestWeightedComposite:
    def test_weighted_score_differs_from_unweighted_mean(self):
        r1 = ModuleResult("m1", 0.9, 7, {}, "")
        r2 = ModuleResult("m6", 0.1, 1, {}, "")
        results = {"m1_grassmannian": r1, "m6_curvature": r2}
        weighted = _composite_score(results, DEFAULT_WEIGHTS)
        unweighted = (0.9 + 0.1) / 2
        assert weighted != pytest.approx(unweighted, abs=0.01)

    def test_higher_weight_module_dominates(self):
        r_high = ModuleResult("m1", 0.9, 7, {}, "")
        r_low = ModuleResult("m6", 0.1, 1, {}, "")
        results = {"m1_grassmannian": r_high, "m6_curvature": r_low}
        score = _composite_score(results, DEFAULT_WEIGHTS)
        assert score > 0.5

    def test_custom_weights_via_hyperparameters(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        custom_weights = {k: 1.0 for k in DEFAULT_WEIGHTS}
        report = run(
            pair.source, pair.target,
            hyperparameters={"k": 5, "weights": custom_weights},
        )
        assert report.hyperparameters["weights"] == custom_weights


# ======================================================================
# All 7 modules together
# ======================================================================

class TestRunAll7Modules:
    def test_all_modules_when_metadata_provided(self):
        pair = generate_transfer_positive(n_source=200, n_target=200, d=50, k=5)
        estimates = generate_concordant_estimates()
        net = generate_network_clustered()
        eco = generate_ecological_unbiased()
        report = run(
            pair.source, pair.target,
            metadata={
                "source_labels": pair.source_labels,
                "target_labels": pair.target_labels,
                "estimates": estimates.estimates,
                "graph": net.graph,
                "records": eco.records,
            },
        )
        expected = {"m1_grassmannian", "m5_cross_design", "m3_direction_stability",
                    "m2_domain_shift", "m4_domain_validity",
                    "m6_curvature", "m7_ecological_bias"}
        assert set(report.module_results.keys()) == expected

    def test_all_7_modules_produce_valid_scores(self):
        pair = generate_transfer_positive(n_source=200, n_target=200, d=50, k=5)
        estimates = generate_concordant_estimates()
        net = generate_network_clustered()
        eco = generate_ecological_unbiased()
        report = run(
            pair.source, pair.target,
            metadata={
                "estimates": estimates.estimates,
                "graph": net.graph,
                "records": eco.records,
            },
        )
        for name, r in report.module_results.items():
            assert 0.0 <= r.score <= 1.0, f"{name} score {r.score} out of range"
            assert 1 <= r.tier <= 7, f"{name} tier {r.tier} out of range"


# ======================================================================
# Preregistration
# ======================================================================

class TestRunPreregistration:
    def test_preregister_saves_sha(self, tmp_path):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        spec = DatasetSpec(
            name="test_prereg", source_description="s", target_description="t",
        )
        prereg_path = tmp_path / "prereg.json"
        report = run(
            pair.source, pair.target,
            preregister=True,
            preregister_path=str(prereg_path),
            dataset_spec=spec,
        )
        assert report.preregistration_sha is not None
        assert len(report.preregistration_sha) == 64
        assert prereg_path.exists()

    def test_preregister_sha_includes_weights(self, tmp_path):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        spec = DatasetSpec(name="t", source_description="s", target_description="t")
        path1 = tmp_path / "p1.json"
        path2 = tmp_path / "p2.json"

        r1 = run(pair.source, pair.target, preregister=True,
                 preregister_path=str(path1), dataset_spec=spec)
        r2 = run(pair.source, pair.target, preregister=True,
                 preregister_path=str(path2), dataset_spec=spec,
                 hyperparameters={"weights": {"m1_grassmannian": 999.0}})
        assert r1.preregistration_sha != r2.preregistration_sha

    def test_no_preregister_by_default(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        report = run(pair.source, pair.target)
        assert report.preregistration_sha is None

    def test_preregister_requires_spec(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        with pytest.raises(ValueError, match="dataset_spec"):
            run(pair.source, pair.target, preregister=True, preregister_path="/tmp/x.json")

    def test_preregister_requires_path(self):
        pair = generate_transfer_positive(n_source=100, n_target=100, d=50, k=5)
        spec = DatasetSpec(name="t", source_description="s", target_description="t")
        with pytest.raises(ValueError, match="preregister_path"):
            run(pair.source, pair.target, preregister=True, dataset_spec=spec)


class TestRunRecommendations:
    def test_good_pair_has_recommendations(self):
        pair = generate_transfer_positive(n_source=200, n_target=200, d=50, k=5)
        report = run(pair.source, pair.target, hyperparameters={"k": 5})
        assert len(report.recommendations) > 0

    def test_bad_pair_flags_low_tiers(self):
        pair = generate_transfer_negative(n_source=200, n_target=200, d=50, k=5)
        report = run(pair.source, pair.target, hyperparameters={"k": 5})
        low_tier_recs = [r for r in report.recommendations if "Tier" in r]
        assert len(low_tier_recs) > 0


class TestScoreToTier:
    def test_perfect_score_is_tier_7(self):
        assert _score_to_tier(1.0) == 7

    def test_zero_score_is_tier_1(self):
        assert _score_to_tier(0.0) == 1

    def test_monotonic(self):
        scores = [0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
        tiers = [_score_to_tier(s) for s in scores]
        for i in range(len(tiers) - 1):
            assert tiers[i] <= tiers[i + 1]
