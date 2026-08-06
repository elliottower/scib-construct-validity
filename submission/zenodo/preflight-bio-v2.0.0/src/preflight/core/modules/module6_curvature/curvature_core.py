"""Curvature-based core-subgraph discovery in disease comorbidity networks.

Edges with high positive curvature are in redundantly-connected regions
(robust pathways), while edges with negative curvature are bridges
(fragile but potentially high-impact intervention targets).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_comorbidity_graph(
    co_occurrence: pd.DataFrame | None = None,
    gwas_correlation: np.ndarray | None = None,
    trait_names: list[str] | None = None,
    threshold: float = 0.1,
):
    """Build a disease comorbidity network.

    Accepts either a co-occurrence DataFrame or a genetic correlation
    matrix. Returns a networkx DiGraph with edge weights.
    """
    import networkx as nx

    G = nx.DiGraph()

    if gwas_correlation is not None and trait_names is not None:
        n = len(trait_names)
        G.add_nodes_from(trait_names)
        for i in range(n):
            for j in range(n):
                if i != j and abs(gwas_correlation[i, j]) > threshold:
                    G.add_edge(
                        trait_names[i], trait_names[j],
                        weight=float(gwas_correlation[i, j]),
                    )
    elif co_occurrence is not None:
        for _, row in co_occurrence.iterrows():
            if abs(row.get("weight", row.get("rr", 1.0))) > threshold:
                G.add_edge(
                    row["disease_a"], row["disease_b"],
                    weight=float(row.get("weight", row.get("rr", 1.0))),
                )

    return G


def forman_ricci_curvature(G) -> dict[tuple, float]:
    """Compute Forman-Ricci curvature for all edges.

    For a directed edge (u, v), Forman-Ricci curvature is:
      F(u,v) = w(u,v) * (
          w(u)/w(u,v) + w(v)/w(u,v)
          - sum_{e parallel to (u,v)} w(u,v)/w(e)
      )

    Simplified for unweighted: F(u,v) = 4 - deg(u) - deg(v)
    For weighted graphs, uses the edge weight formulation.
    """
    import networkx as nx

    curvatures = {}
    for u, v, data in G.edges(data=True):
        w_uv = data.get("weight", 1.0)

        deg_u = G.degree(u)
        deg_v = G.degree(v)

        parallel_sum = 0
        for pred in G.predecessors(u):
            if pred != v:
                w_e = G[pred][u].get("weight", 1.0)
                parallel_sum += abs(w_uv) / (abs(w_e) + 1e-10)
        for succ in G.successors(v):
            if succ != u:
                w_e = G[v][succ].get("weight", 1.0)
                parallel_sum += abs(w_uv) / (abs(w_e) + 1e-10)

        node_weight_u = sum(abs(G[u][s].get("weight", 1.0)) for s in G.successors(u))
        node_weight_u += sum(abs(G[p][u].get("weight", 1.0)) for p in G.predecessors(u))
        node_weight_v = sum(abs(G[v][s].get("weight", 1.0)) for s in G.successors(v))
        node_weight_v += sum(abs(G[p][v].get("weight", 1.0)) for p in G.predecessors(v))

        curvatures[(u, v)] = (
            abs(w_uv) * (
                node_weight_u / (abs(w_uv) + 1e-10)
                + node_weight_v / (abs(w_uv) + 1e-10)
            )
            - parallel_sum
        )

    return curvatures


def find_core_subgraph(
    G,
    curvatures: dict[tuple, float],
    method: str = "positive_curvature",
    percentile: float = 75,
):
    """Extract the core subgraph using curvature-based criteria.

    Methods:
      - "positive_curvature": keep edges with curvature > 0
      - "top_percentile": keep top-k% by curvature
      - "bridges": keep edges with most negative curvature (bridges)
    """
    import networkx as nx

    values = list(curvatures.values())

    if method == "positive_curvature":
        core_edges = [(u, v) for (u, v), k in curvatures.items() if k > 0]
    elif method == "top_percentile":
        threshold = np.percentile(values, percentile)
        core_edges = [(u, v) for (u, v), k in curvatures.items() if k >= threshold]
    elif method == "bridges":
        threshold = np.percentile(values, 100 - percentile)
        core_edges = [(u, v) for (u, v), k in curvatures.items() if k <= threshold]
    else:
        raise ValueError(f"Unknown method: {method}")

    core = G.edge_subgraph(core_edges).copy()
    return core


def curvature_profile(curvatures: dict[tuple, float]) -> dict:
    """Summary statistics of the curvature distribution."""
    values = np.array(list(curvatures.values()))
    return {
        "n_edges": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "frac_positive": float(np.mean(values > 0)),
        "frac_negative": float(np.mean(values < 0)),
    }


def intervention_targets(
    G,
    curvatures: dict[tuple, float],
    top_k: int = 5,
) -> pd.DataFrame:
    """Rank edges as intervention targets.

    Bridges (negative curvature, high betweenness) are high-leverage
    intervention points — removing them disconnects clusters.
    """
    import networkx as nx

    try:
        betweenness = nx.edge_betweenness_centrality(G)
    except Exception:
        betweenness = {e: 0 for e in G.edges()}

    rows = []
    for (u, v), kappa in curvatures.items():
        rows.append({
            "source": u,
            "target": v,
            "ricci_curvature": kappa,
            "betweenness": betweenness.get((u, v), 0),
            "weight": G[u][v].get("weight", 1.0),
            "intervention_score": -kappa * (1 + betweenness.get((u, v), 0)),
        })

    df = pd.DataFrame(rows)
    return df.sort_values("intervention_score", ascending=False).head(top_k)


def run_demo():
    """Demo with a synthetic psychiatric comorbidity network."""
    import networkx as nx

    print("=== Curvature Core-Subgraph Discovery Demo ===\n")

    G = nx.DiGraph()
    edges = [
        ("depression", "anxiety", 2.1),
        ("anxiety", "depression", 1.8),
        ("depression", "insomnia", 1.5),
        ("insomnia", "depression", 1.3),
        ("anxiety", "panic", 1.9),
        ("panic", "agoraphobia", 1.6),
        ("depression", "substance_use", 1.4),
        ("substance_use", "depression", 1.2),
        ("ptsd", "depression", 2.3),
        ("ptsd", "substance_use", 1.7),
        ("ptsd", "anxiety", 2.0),
        ("schizophrenia", "depression", 1.1),
        ("bipolar", "depression", 1.8),
        ("bipolar", "substance_use", 1.5),
        ("adhd", "substance_use", 1.3),
        ("adhd", "anxiety", 1.1),
        ("autism", "anxiety", 1.2),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    curvatures = forman_ricci_curvature(G)
    profile = curvature_profile(curvatures)
    print(f"\nCurvature profile:")
    for k, v in profile.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nTop intervention targets (bridge edges):")
    targets = intervention_targets(G, curvatures, top_k=5)
    print(targets.to_string(index=False))

    core = find_core_subgraph(G, curvatures, method="top_percentile", percentile=50)
    print(f"\nCore subgraph (top 50% curvature): {core.number_of_nodes()} nodes, {core.number_of_edges()} edges")
    print(f"  Edges: {list(core.edges())}")

    bridges = find_core_subgraph(G, curvatures, method="bridges", percentile=75)
    print(f"\nBridge subgraph (bottom 25% curvature): {bridges.number_of_nodes()} nodes, {bridges.number_of_edges()} edges")
    print(f"  Edges: {list(bridges.edges())}")

    return curvatures, profile, targets


if __name__ == "__main__":
    run_demo()
