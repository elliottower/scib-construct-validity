"""Weighted Ollivier-Ricci curvature + comparison methods pipeline.

Provides:
  - Weighted ORC via POT (Wasserstein-1 on weighted shortest paths)
  - Degree-preserving null with weight reshuffling
  - Comparison methods: Jaccard overlap, spectral gap, edge betweenness,
    clustering coefficient, graphlet degree vectors
  - Verdict prediction: cross-validated classification from graph features
  - Full pipeline: build graph → compute features → null model → predict

All methods accept nx.Graph with optional edge attribute 'weight'.
Unweighted graphs treated as weight=1 everywhere.
"""

import json
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from tqdm import tqdm

try:
    import ot
except ImportError:
    raise ImportError("pip install POT")


# ---------------------------------------------------------------------------
# 1. Weighted Ollivier-Ricci curvature
# ---------------------------------------------------------------------------

def _lazy_distribution(G, node, alpha=0.5):
    """Lazy random walk distribution at `node` with idleness `alpha`.

    On a weighted graph, transition probabilities are proportional to edge
    weights.  Returns (support_nodes, probabilities).
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        return [node], np.array([1.0])

    weights = np.array([G[node][n].get("weight", 1.0) for n in neighbors],
                       dtype=np.float64)
    weights /= weights.sum()

    support = [node] + neighbors
    probs = np.zeros(len(support))
    probs[0] = alpha
    probs[1:] = (1 - alpha) * weights
    return support, probs


def _shortest_path_matrix(G, nodes_a, nodes_b):
    """Shortest-path distance matrix between two node lists.

    Uses weighted shortest paths when edge weights are present.
    """
    all_nodes = list(set(nodes_a) | set(nodes_b))
    sp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    D = np.full((len(nodes_a), len(nodes_b)), 1e12)
    for i, u in enumerate(nodes_a):
        for j, v in enumerate(nodes_b):
            if v in sp.get(u, {}):
                D[i, j] = sp[u][v]
    return D


def ollivier_ricci_curvature(G, alpha=0.5, edge_list=None):
    """Compute ORC for every edge (or a subset) of G.

    Returns dict {(u,v): kappa} where kappa = 1 - W1/d(u,v).
    Supports weighted graphs via weighted shortest paths and
    weight-proportional lazy random walks.
    """
    if edge_list is None:
        edge_list = list(G.edges())

    sp_dict = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))

    curvatures = {}
    for u, v in tqdm(edge_list, desc="ORC", leave=False):
        supp_u, prob_u = _lazy_distribution(G, u, alpha)
        supp_v, prob_v = _lazy_distribution(G, v, alpha)

        all_supp = list(set(supp_u) | set(supp_v))
        n_u, n_v = len(supp_u), len(supp_v)

        D = np.zeros((n_u, n_v))
        for i, s in enumerate(supp_u):
            for j, t in enumerate(supp_v):
                D[i, j] = sp_dict.get(s, {}).get(t, 1e12)

        W1 = ot.emd2(prob_u, prob_v, D)
        d_uv = sp_dict.get(u, {}).get(v, 1.0)
        kappa = 1.0 - W1 / d_uv if d_uv > 0 else 0.0
        curvatures[(u, v)] = kappa

    return curvatures


# ---------------------------------------------------------------------------
# 2. Null models
# ---------------------------------------------------------------------------

def degree_preserving_rewire(G, n_swaps=None, seed=None):
    """Degree-preserving double-edge swap (Maslov-Sneppen).

    For weighted graphs, edge weights are reshuffled after rewiring
    to break weight-topology correlations while preserving the weight
    distribution.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()
    if n_swaps is None:
        n_swaps = max(100, 10 * H.number_of_edges())

    edges = list(H.edges())
    weights = [H[u][v].get("weight", 1.0) for u, v in edges]

    nx.double_edge_swap(H, nswap=n_swaps, max_tries=n_swaps * 10, seed=int(rng.integers(2**31)))

    new_edges = list(H.edges())
    shuffled_weights = list(rng.permutation(weights))
    for (u, v), w in zip(new_edges, shuffled_weights):
        H[u][v]["weight"] = w

    return H


def run_edge_null(G, n_perms=200, alpha=0.5, min_null_samples=10):
    """Edge-level permutation null.

    Returns dict {(u,v): {"kappa_obs", "z", "mean_null", "std_null", "n_samples"}}.
    """
    obs_curvatures = ollivier_ricci_curvature(G, alpha=alpha)

    null_curvatures = defaultdict(list)
    for i in tqdm(range(n_perms), desc="Null permutations"):
        H = degree_preserving_rewire(G, seed=i)
        kappas = ollivier_ricci_curvature(H, alpha=alpha)
        for e, k in kappas.items():
            e_key = tuple(sorted(e))
            null_curvatures[e_key].append(k)

    results = {}
    for e, kappa in obs_curvatures.items():
        e_key = tuple(sorted(e))
        nulls = null_curvatures.get(e_key, [])
        if len(nulls) < min_null_samples:
            continue
        nulls = np.array(nulls)
        mu, sigma = nulls.mean(), nulls.std()
        z = (kappa - mu) / sigma if sigma > 1e-10 else 0.0
        results[e_key] = {
            "kappa_obs": kappa,
            "z": z,
            "mean_null": mu,
            "std_null": sigma,
            "n_samples": len(nulls),
        }

    return results


# ---------------------------------------------------------------------------
# 3. Comparison methods
# ---------------------------------------------------------------------------

def jaccard_edge_overlap(G):
    """Jaccard coefficient of neighborhoods for each edge."""
    result = {}
    for u, v in G.edges():
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        intersection = len(nu & nv)
        union = len(nu | nv)
        result[(u, v)] = intersection / union if union > 0 else 0.0
    return result


def edge_betweenness(G):
    """Edge betweenness centrality (weighted)."""
    return nx.edge_betweenness_centrality(G, weight="weight")


def edge_clustering_coefficient(G):
    """Edge clustering coefficient: fraction of triangles through each edge."""
    result = {}
    for u, v in G.edges():
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        common = nu & nv
        max_triangles = min(len(nu) - 1, len(nv) - 1)
        result[(u, v)] = len(common) / max_triangles if max_triangles > 0 else 0.0
    return result


def spectral_edge_gap(G):
    """Per-edge spectral gap contribution.

    Uses the Fiedler vector (second-smallest eigenvector of the Laplacian).
    Edges with large |fiedler[u] - fiedler[v]| are spectral bottlenecks.
    """
    L = nx.laplacian_matrix(G, weight="weight").toarray().astype(float)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    result = {}
    for u, v in G.edges():
        gap = abs(fiedler[node_idx[u]] - fiedler[node_idx[v]])
        result[(u, v)] = gap
    return result


def compute_all_edge_features(G, alpha=0.5):
    """Compute all edge-level features for method comparison.

    Returns dict {(u,v): {feature_name: value}}.
    """
    orc = ollivier_ricci_curvature(G, alpha=alpha)
    jacc = jaccard_edge_overlap(G)
    betw = edge_betweenness(G)
    clust = edge_clustering_coefficient(G)

    try:
        spec = spectral_edge_gap(G)
    except Exception:
        spec = {e: np.nan for e in G.edges()}

    features = {}
    for e in G.edges():
        e_key = tuple(sorted(e))
        features[e_key] = {
            "orc": orc.get(e, orc.get((e[1], e[0]), np.nan)),
            "jaccard": jacc.get(e, jacc.get((e[1], e[0]), np.nan)),
            "betweenness": betw.get(e, betw.get((e[1], e[0]), np.nan)),
            "clustering": clust.get(e, clust.get((e[1], e[0]), np.nan)),
            "spectral_gap": spec.get(e, spec.get((e[1], e[0]), np.nan)),
        }

    return features


# ---------------------------------------------------------------------------
# 4. Verdict prediction
# ---------------------------------------------------------------------------

def extract_node_features(G, curvatures, node):
    """Extract graph features for a single node.

    Features: mean/min/max ORC of incident edges, degree, betweenness,
    clustering coefficient, mean neighbor degree.
    """
    incident_edges = [(node, n) for n in G.neighbors(node)]
    incident_curvatures = []
    for e in incident_edges:
        k = tuple(sorted(e))
        if k in curvatures:
            incident_curvatures.append(curvatures[k])

    if not incident_curvatures:
        incident_curvatures = [0.0]

    ic = np.array(incident_curvatures)
    deg = G.degree(node)
    neighbors = list(G.neighbors(node))

    return {
        "mean_orc": ic.mean(),
        "min_orc": ic.min(),
        "max_orc": ic.max(),
        "std_orc": ic.std() if len(ic) > 1 else 0.0,
        "degree": deg,
        "betweenness": nx.betweenness_centrality(G, weight="weight").get(node, 0),
        "clustering": nx.clustering(G, node, weight="weight"),
        "mean_neighbor_degree": np.mean([G.degree(n) for n in neighbors]) if neighbors else 0,
    }


def verdict_prediction_loocv(G, curvatures, claim_nodes, verdicts, threshold="Disconfirmed"):
    """Leave-one-out cross-validated prediction of verdict from graph features.

    Binary classification: threshold tier vs rest.
    Returns AUC, accuracy, per-claim predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    feature_names = ["mean_orc", "min_orc", "max_orc", "std_orc",
                     "degree", "clustering", "mean_neighbor_degree"]

    X = []
    y = []
    node_ids = []
    for node in claim_nodes:
        feats = extract_node_features(G, curvatures, node)
        X.append([feats[f] for f in feature_names])
        y.append(1 if verdicts[node] == threshold else 0)
        node_ids.append(node)

    X = np.array(X)
    y = np.array(y)

    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        warnings.warn(f"Too few positive ({y.sum()}) or negative ({len(y) - y.sum()}) samples")
        return {"auc": np.nan, "accuracy": np.nan, "n_positive": int(y.sum()),
                "n_total": len(y), "predictions": []}

    predictions = np.zeros(len(y))
    for i in range(len(y)):
        train_idx = [j for j in range(len(y)) if j != i]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[i:i+1]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, solver="lbfgs")
        clf.fit(X_train_s, y_train)
        predictions[i] = clf.predict_proba(X_test_s)[0, 1]

    auc = roc_auc_score(y, predictions)
    acc = accuracy_score(y, (predictions > 0.5).astype(int))

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "n_positive": int(y.sum()),
        "n_total": len(y),
        "feature_names": feature_names,
        "predictions": [
            {"node": node_ids[i], "true": int(y[i]), "pred_prob": float(predictions[i])}
            for i in range(len(y))
        ],
    }


# ---------------------------------------------------------------------------
# 5. Method comparison
# ---------------------------------------------------------------------------

def method_comparison(G, curvatures, claim_nodes, verdicts, threshold="Disconfirmed"):
    """Compare curvature to alternative methods for verdict prediction.

    For each method, compute per-claim feature and report Mann-Whitney U
    and Cohen's d for threshold vs rest.
    """
    methods = {
        "mean_orc": lambda n: extract_node_features(G, curvatures, n)["mean_orc"],
        "degree": lambda n: G.degree(n),
        "betweenness": lambda n: nx.betweenness_centrality(G, weight="weight").get(n, 0),
        "clustering": lambda n: nx.clustering(G, n, weight="weight"),
    }

    jacc = jaccard_edge_overlap(G)
    def mean_jaccard(n):
        vals = []
        for nb in G.neighbors(n):
            e = tuple(sorted((n, nb)))
            if e in jacc:
                vals.append(jacc[e])
        return np.mean(vals) if vals else 0.0
    methods["mean_jaccard"] = mean_jaccard

    try:
        spec = spectral_edge_gap(G)
        def mean_spectral(n):
            vals = []
            for nb in G.neighbors(n):
                e = tuple(sorted((n, nb)))
                if e in spec:
                    vals.append(spec[e])
            return np.mean(vals) if vals else 0.0
        methods["mean_spectral_gap"] = mean_spectral
    except Exception:
        pass

    results = {}
    for method_name, feat_fn in methods.items():
        group_threshold = []
        group_rest = []
        for node in claim_nodes:
            val = feat_fn(node)
            if verdicts[node] == threshold:
                group_threshold.append(val)
            else:
                group_rest.append(val)

        group_threshold = np.array(group_threshold)
        group_rest = np.array(group_rest)

        if len(group_threshold) < 2 or len(group_rest) < 2:
            continue

        U, p = stats.mannwhitneyu(group_threshold, group_rest, alternative="two-sided")
        pooled_std = np.sqrt(
            ((len(group_threshold) - 1) * group_threshold.std()**2 +
             (len(group_rest) - 1) * group_rest.std()**2) /
            (len(group_threshold) + len(group_rest) - 2)
        )
        d = (group_rest.mean() - group_threshold.mean()) / pooled_std if pooled_std > 1e-10 else 0.0

        results[method_name] = {
            "U": float(U),
            "p": float(p),
            "cohens_d": float(d),
            "mean_threshold": float(group_threshold.mean()),
            "mean_rest": float(group_rest.mean()),
            "n_threshold": len(group_threshold),
            "n_rest": len(group_rest),
        }

    return results


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(G, claim_nodes=None, verdicts=None,
                      alpha=0.5, n_perms=200, output_dir=None,
                      graph_name="unnamed"):
    """Run the complete analysis pipeline.

    Args:
        G: nx.Graph (optionally weighted)
        claim_nodes: list of nodes that have verdict labels
        verdicts: dict {node: verdict_string}
        alpha: ORC laziness parameter
        n_perms: number of permutations for null model
        output_dir: where to save results JSON
        graph_name: identifier for this graph

    Returns:
        dict with all results
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[{timestamp}] Starting pipeline for {graph_name}")
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    weighted = any("weight" in G[u][v] for u, v in G.edges())
    print(f"  Weighted: {weighted}")

    results = {
        "metadata": {
            "graph_name": graph_name,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "weighted": weighted,
            "alpha": alpha,
            "n_perms": n_perms,
            "timestamp": timestamp,
        }
    }

    # Edge features (all methods)
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Computing edge features...")
    edge_features = compute_all_edge_features(G, alpha=alpha)
    results["edge_features_summary"] = {
        method: {
            "mean": float(np.nanmean([f[method] for f in edge_features.values()])),
            "std": float(np.nanstd([f[method] for f in edge_features.values()])),
            "min": float(np.nanmin([f[method] for f in edge_features.values()])),
            "max": float(np.nanmax([f[method] for f in edge_features.values()])),
        }
        for method in ["orc", "jaccard", "betweenness", "clustering", "spectral_gap"]
    }

    # Degree-curvature correlation
    orc_vals = {e: f["orc"] for e, f in edge_features.items()}
    mechanism_nodes = [n for n in G.nodes() if G.degree(n) >= 3]
    if mechanism_nodes:
        degrees = [G.degree(n) for n in mechanism_nodes]
        mean_curv = []
        for n in mechanism_nodes:
            edges = [tuple(sorted((n, nb))) for nb in G.neighbors(n)]
            curv = [orc_vals.get(e, 0) for e in edges]
            mean_curv.append(np.mean(curv) if curv else 0)
        r, p = stats.pearsonr(degrees, mean_curv)
        results["degree_curvature"] = {
            "r": float(r), "p": float(p),
            "n_nodes": len(mechanism_nodes),
        }
        print(f"  Degree-curvature correlation: r={r:.3f}, p={p:.4f}")

    # Edge-level null model
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Running edge null ({n_perms} permutations)...")
    edge_null = run_edge_null(G, n_perms=n_perms, alpha=alpha)

    bottleneck = {k: v for k, v in edge_null.items() if v["z"] < -1.96}
    redundant = {k: v for k, v in edge_null.items() if v["z"] > 1.96}
    results["edge_null"] = {
        "n_tested": len(edge_null),
        "n_bottleneck": len(bottleneck),
        "n_redundant": len(redundant),
        "bottleneck_edges": {str(k): v for k, v in
                            sorted(bottleneck.items(), key=lambda x: x[1]["z"])},
        "redundant_edges": {str(k): v for k, v in
                           sorted(redundant.items(), key=lambda x: x[1]["z"], reverse=True)[:20]},
    }
    print(f"  Bottleneck: {len(bottleneck)}, Redundant: {len(redundant)}")

    # Verdict analysis
    if claim_nodes and verdicts:
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Verdict analysis...")

        # Method comparison
        results["method_comparison"] = method_comparison(
            G, orc_vals, claim_nodes, verdicts)

        # LOOCV prediction
        try:
            results["verdict_prediction"] = verdict_prediction_loocv(
                G, orc_vals, claim_nodes, verdicts)
            print(f"  LOOCV AUC: {results['verdict_prediction']['auc']:.3f}")
        except Exception as e:
            results["verdict_prediction"] = {"error": str(e)}
            print(f"  LOOCV failed: {e}")

    # Save
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{graph_name}_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Saved to {out_path}")

    return results
