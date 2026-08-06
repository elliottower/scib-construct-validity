"""Tests for Module 8: Curvature (Paper 8)."""
import numpy as np
import pandas as pd
import pytest

networkx = pytest.importorskip("networkx")
ot = pytest.importorskip("ot")

from preflight.core.modules.module6_curvature.curvature_core import (
    build_comorbidity_graph,
    curvature_profile,
    find_core_subgraph,
    forman_ricci_curvature,
    intervention_targets,
)
from preflight.core.modules.module6_curvature.weighted_orc import (
    _lazy_distribution,
    degree_preserving_rewire,
    ollivier_ricci_curvature,
)


def _triangle_graph():
    G = networkx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("A", "C", weight=1.0)
    return G


def _star_graph():
    G = networkx.Graph()
    for spoke in ["B", "C", "D", "E"]:
        G.add_edge("A", spoke, weight=1.0)
    return G


def _line_graph():
    G = networkx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "D", weight=1.0)
    return G


class TestLazyDistribution:
    def test_isolated_node_concentrates_mass(self):
        G = networkx.Graph()
        G.add_node("X")
        support, probs = _lazy_distribution(G, "X", alpha=0.5)
        assert support == ["X"]
        assert probs == pytest.approx([1.0])

    def test_probabilities_sum_to_one(self):
        G = _triangle_graph()
        support, probs = _lazy_distribution(G, "A", alpha=0.5)
        assert sum(probs) == pytest.approx(1.0, abs=1e-10)

    def test_idle_probability_equals_alpha(self):
        G = _triangle_graph()
        support, probs = _lazy_distribution(G, "A", alpha=0.3)
        assert probs[0] == pytest.approx(0.3, abs=1e-10)

    def test_weighted_distribution_respects_weights(self):
        G = networkx.Graph()
        G.add_edge("A", "B", weight=3.0)
        G.add_edge("A", "C", weight=1.0)
        support, probs = _lazy_distribution(G, "A", alpha=0.0)
        b_idx = support.index("B")
        c_idx = support.index("C")
        assert probs[b_idx] == pytest.approx(0.75, abs=1e-10)
        assert probs[c_idx] == pytest.approx(0.25, abs=1e-10)


class TestOllivierRicciCurvature:
    def test_triangle_has_positive_curvature(self):
        G = _triangle_graph()
        curvatures = ollivier_ricci_curvature(G, alpha=0.5)
        for kappa in curvatures.values():
            assert kappa > 0

    def test_star_has_lower_curvature_than_triangle(self):
        G_star = _star_graph()
        G_tri = _triangle_graph()
        curv_star = ollivier_ricci_curvature(G_star, alpha=0.5)
        curv_tri = ollivier_ricci_curvature(G_tri, alpha=0.5)
        assert np.mean(list(curv_star.values())) < np.mean(list(curv_tri.values()))

    def test_line_graph_interior_edge_has_lower_curvature_than_triangle(self):
        tri = _triangle_graph()
        line = _line_graph()
        tri_curv = ollivier_ricci_curvature(tri, alpha=0.5)
        line_curv = ollivier_ricci_curvature(line, alpha=0.5)
        tri_mean = np.mean(list(tri_curv.values()))
        line_mean = np.mean(list(line_curv.values()))
        assert tri_mean > line_mean


class TestDegreePreservingRewire:
    def test_preserves_degree_sequence(self):
        G = networkx.gnm_random_graph(20, 40, seed=42)
        degrees_before = sorted(dict(G.degree()).values())
        H = degree_preserving_rewire(G, n_swaps=50, seed=7)
        degrees_after = sorted(dict(H.degree()).values())
        assert degrees_before == degrees_after

    def test_preserves_node_count(self):
        G = networkx.gnm_random_graph(15, 30, seed=43)
        H = degree_preserving_rewire(G, seed=8)
        assert H.number_of_nodes() == G.number_of_nodes()

    def test_preserves_edge_count(self):
        G = networkx.gnm_random_graph(15, 30, seed=44)
        H = degree_preserving_rewire(G, seed=9)
        assert H.number_of_edges() == G.number_of_edges()


class TestBuildComorbidityGraph:
    def test_from_gwas_correlation(self):
        corr = np.array([[1.0, 0.5, 0.05], [0.5, 1.0, 0.3], [0.05, 0.3, 1.0]])
        traits = ["A", "B", "C"]
        G = build_comorbidity_graph(gwas_correlation=corr, trait_names=traits, threshold=0.1)
        assert G.number_of_nodes() == 3
        assert G.has_edge("A", "B")
        assert not G.has_edge("A", "C")

    def test_from_co_occurrence_dataframe(self):
        df = pd.DataFrame({
            "disease_a": ["dep", "dep", "anx"],
            "disease_b": ["anx", "ptsd", "ptsd"],
            "weight": [1.5, 0.05, 1.2],
        })
        G = build_comorbidity_graph(co_occurrence=df, threshold=0.1)
        assert G.has_edge("dep", "anx")
        assert not G.has_edge("dep", "ptsd")


class TestFormanRicciCurvature:
    def test_returns_curvature_for_all_edges(self):
        G = networkx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("A", "C", weight=1.0)
        curvatures = forman_ricci_curvature(G)
        assert len(curvatures) == 3

    def test_dense_graph_has_higher_curvature(self):
        dense = networkx.complete_graph(5, create_using=networkx.DiGraph)
        sparse = networkx.path_graph(5, create_using=networkx.DiGraph)
        for G in [dense, sparse]:
            for u, v in G.edges():
                G[u][v]["weight"] = 1.0
        curv_dense = forman_ricci_curvature(dense)
        curv_sparse = forman_ricci_curvature(sparse)
        mean_dense = np.mean(list(curv_dense.values()))
        mean_sparse = np.mean(list(curv_sparse.values()))
        assert mean_dense > mean_sparse


class TestFindCoreSubgraph:
    def test_positive_curvature_method(self):
        G = networkx.DiGraph()
        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")]
        for u, v in edges:
            G.add_edge(u, v, weight=1.0)
        curvatures = {("A", "B"): 2.0, ("B", "C"): 1.5, ("C", "A"): 0.5, ("A", "D"): -1.0}
        core = find_core_subgraph(G, curvatures, method="positive_curvature")
        assert ("A", "D") not in core.edges()
        assert ("A", "B") in core.edges()

    def test_bridges_method(self):
        G = networkx.DiGraph()
        edges = [("A", "B"), ("B", "C"), ("C", "A"), ("A", "D")]
        for u, v in edges:
            G.add_edge(u, v, weight=1.0)
        curvatures = {("A", "B"): 2.0, ("B", "C"): 1.5, ("C", "A"): 0.5, ("A", "D"): -1.0}
        bridges = find_core_subgraph(G, curvatures, method="bridges", percentile=75)
        assert ("A", "D") in bridges.edges()

    def test_unknown_method_raises(self):
        G = networkx.DiGraph()
        G.add_edge("A", "B")
        with pytest.raises(ValueError, match="Unknown method"):
            find_core_subgraph(G, {("A", "B"): 1.0}, method="bogus")


class TestCurvatureProfile:
    def test_profile_keys(self):
        curvatures = {("A", "B"): 1.0, ("B", "C"): -0.5, ("C", "D"): 0.3}
        profile = curvature_profile(curvatures)
        assert profile["n_edges"] == 3
        assert profile["mean"] == pytest.approx(np.mean([1.0, -0.5, 0.3]))
        assert 0 < profile["frac_positive"] < 1
        assert 0 < profile["frac_negative"] < 1
        assert profile["frac_positive"] + profile["frac_negative"] <= 1.0


class TestInterventionTargets:
    def test_returns_dataframe_with_expected_columns(self):
        G = networkx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("A", "C", weight=1.0)
        curvatures = forman_ricci_curvature(G)
        df = intervention_targets(G, curvatures, top_k=2)
        assert isinstance(df, pd.DataFrame)
        assert "intervention_score" in df.columns
        assert "ricci_curvature" in df.columns
        assert len(df) <= 2

    def test_bridge_edges_rank_higher(self):
        G = networkx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("B", "C", weight=1.0)
        G.add_edge("A", "C", weight=1.0)
        curvatures = {("A", "B"): 2.0, ("B", "C"): -3.0, ("A", "C"): 0.5}
        df = intervention_targets(G, curvatures, top_k=3)
        assert df.iloc[0]["source"] == "B" or df.iloc[0]["ricci_curvature"] < 0
