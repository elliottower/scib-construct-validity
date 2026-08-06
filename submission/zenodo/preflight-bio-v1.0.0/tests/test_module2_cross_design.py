"""Tests for Module 3: Cross-design evidence discordance (Paper 3)."""
import math

import pytest

from preflight.core.modules.module5_cross_design.classify_families import (
    CARDIO_FAMILIES,
    NEURO_FAMILIES,
    chinn_d,
    ci_excludes_null,
    classify_family,
    classify_mr,
    classify_obs,
    rescale_per_allele_to_per_sd,
    run_classification,
)


class TestChinnD:
    def test_or_of_1_gives_zero(self):
        assert chinn_d(1.0) == pytest.approx(0.0, abs=1e-10)

    def test_or_gt_1_gives_positive(self):
        assert chinn_d(2.0) > 0

    def test_symmetric_around_1(self):
        assert chinn_d(2.0) == pytest.approx(chinn_d(0.5), abs=1e-10)

    def test_known_value(self):
        expected = abs(math.log(2.0)) * math.sqrt(3) / math.pi
        assert chinn_d(2.0) == pytest.approx(expected, abs=1e-10)

    def test_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            chinn_d(0.0)
        with pytest.raises(ValueError):
            chinn_d(-1.0)


class TestCIExcludesNull:
    def test_excludes_when_both_above_one(self):
        assert ci_excludes_null(1.1, 2.0) is True

    def test_excludes_when_both_below_one(self):
        assert ci_excludes_null(0.5, 0.9) is True

    def test_includes_when_spanning_one(self):
        assert ci_excludes_null(0.8, 1.2) is False

    def test_boundary_includes(self):
        assert ci_excludes_null(1.0, 2.0) is False
        assert ci_excludes_null(0.5, 1.0) is False


class TestClassifyMR:
    def test_causal_when_sig_and_above_floor(self):
        assert classify_mr(1.1, 2.0, 0.20, threshold=0.10) == "causal"

    def test_null_when_not_sig(self):
        assert classify_mr(0.8, 1.2, 0.20, threshold=0.10) == "null"

    def test_null_when_below_floor(self):
        assert classify_mr(1.01, 1.02, 0.01, threshold=0.10) == "null"


class TestClassifyObs:
    def test_nontrivial_above_threshold(self):
        assert classify_obs(0.20, threshold=0.10) == "non-trivial"

    def test_trivial_below_threshold(self):
        assert classify_obs(0.05, threshold=0.10) == "trivial"


class TestClassifyFamily:
    def test_discordance_predicts_failure(self):
        cls, pred = classify_family("non-trivial", "null")
        assert cls == "qualitative discordance"
        assert pred == "failure"

    def test_concordance_predicts_success(self):
        cls, pred = classify_family("non-trivial", "causal")
        assert cls == "concordance"
        assert pred == "success"

    def test_null_concordance_is_ambiguous(self):
        cls, pred = classify_family("trivial", "null")
        assert pred == "ambiguous"

    def test_genetic_only_predicts_success(self):
        cls, pred = classify_family("trivial", "causal")
        assert pred == "success"


class TestRunClassification:
    def test_neuro_families_produce_results(self):
        results = run_classification(NEURO_FAMILIES)
        assert len(results) == len(NEURO_FAMILIES)
        for r in results:
            assert "classification" in r
            assert "prediction" in r
            assert r["prediction"] in ("failure", "success", "ambiguous")

    def test_cardio_families_produce_results(self):
        results = run_classification(CARDIO_FAMILIES)
        assert len(results) == len(CARDIO_FAMILIES)

    def test_ldl_pcsk9_classified_as_concordance(self):
        ldl = [f for f in CARDIO_FAMILIES if f["family"] == "LDL/PCSK9"]
        results = run_classification(ldl)
        assert results[0]["classification"] == "concordance"
        assert results[0]["prediction"] == "success"

    def test_hdl_cetp_classified_as_discordance(self):
        hdl = [f for f in CARDIO_FAMILIES if f["family"] == "HDL/CETP"]
        results = run_classification(hdl)
        assert results[0]["classification"] == "qualitative discordance"
        assert results[0]["prediction"] == "failure"


class TestRescalePerAllele:
    def test_identity_when_sd_is_one(self):
        assert rescale_per_allele_to_per_sd(2.0, 1.0) == pytest.approx(2.0)

    def test_amplifies_when_sd_lt_one(self):
        result = rescale_per_allele_to_per_sd(1.1, 0.5)
        assert result > 1.1

    def test_rejects_nonpositive_sd(self):
        with pytest.raises(ValueError):
            rescale_per_allele_to_per_sd(1.5, 0.0)
