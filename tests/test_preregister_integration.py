"""Integration test: full preregistration dress rehearsal on synthetic data.

Validates the freeze-then-run loop:
  1. Generate synthetic data with a fixed spec
  2. Freeze module code + hyperparameters + dataset spec into a SHA
  3. Run the modules on that data
  4. Verify the SHA still matches (no drift between freeze and execution)
"""
import numpy as np
import pytest

from preflight.core.preregister import (
    DatasetSpec,
    compute_preregistration_sha,
    load_preregistration,
    save_preregistration,
    verify_preregistration,
)
from preflight.core.synthetic import (
    generate_concordant_estimates,
    generate_discordant_estimates,
    generate_ecological_biased,
    generate_embedding_pair,
    generate_missingness_mcar,
    generate_network_clustered,
    generate_transfer_negative,
    generate_transfer_positive,
)
from preflight.core.modules.module1_grassmannian import transportability as m1
from preflight.core.modules.module3_direction_instability import distances as m3
from preflight.core.modules.module4_embedding_collapse import transportability as m4
from preflight.core.modules.module6_curvature import weighted_orc as m6
from preflight.core.modules.module7_ecological_bias import s7_evalue_sensitivity as m7


EMBEDDING_HYPERPARAMS = {"k": 5, "metric": "geodesic"}
EMBEDDING_SPEC = DatasetSpec(
    name="synthetic_transfer_negative_seed200",
    source_description="Gaussian R^100, k=10 subspace, class_sep=2.0",
    target_description="Rotated by 1.200 rad, confound=5.0",
    n_source=200,
    n_target=200,
    d=100,
    extra={"seed": 200, "level": "embedding"},
)


class TestEmbeddingDressRehearsal:
    """Full freeze-run-verify on M1+M3+M4 with embedding-level synthetic data."""

    def test_freeze_then_run_then_verify(self, tmp_path):
        modules = [m1, m3, m4]

        record = compute_preregistration_sha(modules, EMBEDDING_HYPERPARAMS, EMBEDDING_SPEC)
        path = save_preregistration(record, tmp_path / "prereg_embedding.json")

        pair = generate_transfer_negative(seed=200)
        k = EMBEDDING_HYPERPARAMS["k"]

        U_s, _ = m1.top_k_subspace(pair.source, k)
        U_t, _ = m1.top_k_subspace(pair.target, k)
        geodesic = m1.geodesic_distance(U_s, U_t)
        assert geodesic > 0.5

        dists = m3.all_subspace_distances(U_s, U_t)
        assert "grassmannian" in dists
        assert dists["grassmannian"] > 0

        centroid_d = m4.centroid_distance(pair.source, pair.target)
        assert centroid_d > 0

        loaded = load_preregistration(path)
        matches, msg = verify_preregistration(loaded, modules, EMBEDDING_HYPERPARAMS, EMBEDDING_SPEC)
        assert matches, f"SHA mismatch after run: {msg}"

    def test_sha_changes_if_hyperparams_drift(self, tmp_path):
        modules = [m1, m3, m4]
        record = compute_preregistration_sha(modules, EMBEDDING_HYPERPARAMS, EMBEDDING_SPEC)
        save_preregistration(record, tmp_path / "prereg.json")

        drifted_hp = {"k": 10, "metric": "geodesic"}
        matches, msg = verify_preregistration(record, modules, drifted_hp, EMBEDDING_SPEC)
        assert not matches
        assert "hyperparameters" in msg.lower()

    def test_sha_changes_if_dataset_drifts(self, tmp_path):
        modules = [m1, m3, m4]
        record = compute_preregistration_sha(modules, EMBEDDING_HYPERPARAMS, EMBEDDING_SPEC)

        wrong_spec = DatasetSpec(
            name="synthetic_transfer_positive_seed100",
            source_description="different",
            target_description="different",
        )
        matches, msg = verify_preregistration(record, modules, EMBEDDING_HYPERPARAMS, wrong_spec)
        assert not matches
        assert "dataset" in msg.lower()


class TestNetworkDressRehearsal:
    """Freeze-run-verify on M6 with network-level synthetic data."""

    def test_freeze_then_run_orc(self, tmp_path):
        hp = {"alpha": 0.5, "n_clusters": 4}
        spec = DatasetSpec(
            name="synthetic_network_clustered_seed600",
            source_description="SBM: 4 clusters x 12 nodes",
            target_description="p_within=0.7, p_between=0.03",
            extra={"seed": 600, "level": "network"},
        )

        record = compute_preregistration_sha([m6], hp, spec)
        path = save_preregistration(record, tmp_path / "prereg_network.json")

        data = generate_network_clustered()
        curvatures = m6.ollivier_ricci_curvature(data.graph, alpha=hp["alpha"])
        assert len(curvatures) > 0

        hub_curv = []
        for h in data.hub_nodes:
            for nb in data.graph.neighbors(h):
                e = (h, nb)
                e_rev = (nb, h)
                if e in curvatures:
                    hub_curv.append(curvatures[e])
                elif e_rev in curvatures:
                    hub_curv.append(curvatures[e_rev])
        assert len(hub_curv) > 0

        loaded = load_preregistration(path)
        matches, msg = verify_preregistration(loaded, [m6], hp, spec)
        assert matches, f"SHA mismatch: {msg}"


class TestEcologicalDressRehearsal:
    """Freeze-run-verify on M7 with ecological-level synthetic data."""

    def test_freeze_then_run_evalue(self, tmp_path):
        hp = {"sensitivity_method": "evalue"}
        spec = DatasetSpec(
            name="synthetic_ecological_biased_seed700",
            source_description="25 sites x 300 individuals",
            target_description="within=0.3, between=2.5",
            extra={"seed": 700, "level": "ecological"},
        )

        record = compute_preregistration_sha([m7], hp, spec)
        path = save_preregistration(record, tmp_path / "prereg_ecological.json")

        data = generate_ecological_biased()
        result = m7.compute_evalue_from_beta(
            data.ground_truth.extra["between_effect"], se=0.3,
        )
        assert result["evalue_point"] > 1.0

        loaded = load_preregistration(path)
        matches, msg = verify_preregistration(loaded, [m7], hp, spec)
        assert matches, f"SHA mismatch: {msg}"


class TestCrossDesignDressRehearsal:
    """Freeze-run-verify on M1 sheaf Q-test with cross-design synthetic data."""

    def test_freeze_then_run_sheaf_q(self, tmp_path):
        hp = {"test": "sheaf_q", "n_designs": 5}
        spec = DatasetSpec(
            name="synthetic_discordant_seed850",
            source_description="5 designs, true_effect=0.50",
            target_description="discordance=40%, bias=2.0",
            extra={"seed": 850, "level": "cross_design"},
        )

        record = compute_preregistration_sha([m1], hp, spec)
        path = save_preregistration(record, tmp_path / "prereg_crossdesign.json")

        data = generate_discordant_estimates()
        p, Q, df = m1.sheaf_q_test(data.estimates)
        assert p < 0.05

        loaded = load_preregistration(path)
        matches, msg = verify_preregistration(loaded, [m1], hp, spec)
        assert matches, f"SHA mismatch: {msg}"


class TestMultiModuleDressRehearsal:
    """Full pipeline dress rehearsal: freeze all testable modules at once."""

    def test_all_modules_frozen_together(self, tmp_path):
        all_modules = [m1, m3, m4, m6, m7]
        hp = {
            "k": 5,
            "metric": "geodesic",
            "orc_alpha": 0.5,
            "sensitivity_method": "evalue",
        }
        spec = DatasetSpec(
            name="synthetic_full_pipeline_dress_rehearsal",
            source_description="Multi-level synthetic data",
            target_description="Embedding + network + ecological + cross-design",
            extra={"levels": [1, 2, 3, 4, 5, 6]},
        )

        record = compute_preregistration_sha(all_modules, hp, spec)
        path = save_preregistration(record, tmp_path / "prereg_full.json")

        assert len(record.modules_used) == 5
        assert all(v > 0 for v in record.module_code_lengths.values())

        loaded = load_preregistration(path)
        matches, msg = verify_preregistration(loaded, all_modules, hp, spec)
        assert matches, f"SHA mismatch: {msg}"

    def test_removing_a_module_breaks_sha(self, tmp_path):
        all_modules = [m1, m3, m4, m6, m7]
        hp = {"k": 5}
        spec = DatasetSpec(name="test", source_description="s", target_description="t")

        record = compute_preregistration_sha(all_modules, hp, spec)

        fewer_modules = [m1, m3, m4]
        matches, msg = verify_preregistration(record, fewer_modules, hp, spec)
        assert not matches
