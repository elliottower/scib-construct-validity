"""Pure logic for batch1 professor-pitch experiments.

No Modal dependencies. Each function returns a dict of results.

Experiments:
1. sheaf_federated_ehr (Visweswaran) — multi-site EHR phenotype consistency
2. confound_collapse_audit (Xia) — biomarker confound audit, intensive correction
3. curvature_causal_graph (Visweswaran) — ORC edge importance on causal graphs
4. treatment_heterogeneity (Xia) — geometric HTE detection in MS-like treatment comparison
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import chi2, pearsonr, spearmanr
from tqdm import tqdm


# ======================================================================
# 1. SHEAF FEDERATED EHR (Visweswaran)
# ======================================================================

def _sheaf_test(estimates):
    """Sheaf consistency test with proper covariance-weighted Q.

    Returns (p_value, Q, df, node_inconsistency_scores).
    """
    nodes = sorted(estimates.keys())
    n = len(nodes)
    node_idx = {name: i for i, name in enumerate(nodes)}
    edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i + 1, n)]

    d0 = np.zeros((len(edges), n))
    for e_idx, (u, v) in enumerate(edges):
        d0[e_idx, node_idx[u]] = -1.0
        d0[e_idx, node_idx[v]] = 1.0

    stalks = np.array([estimates[nd]["beta"] for nd in nodes])
    ses = np.array([estimates[nd]["se"] for nd in nodes])

    obs = d0 @ stalks
    Sigma = d0 @ np.diag(ses**2) @ d0.T

    _, s, _ = np.linalg.svd(d0, full_matrices=False)
    rank = int(np.sum(s > 1e-10))

    Sigma_pinv = np.linalg.pinv(Sigma, rcond=1e-10)
    Q = float(obs @ Sigma_pinv @ obs)
    df = rank
    p = 1.0 - chi2.cdf(Q, df) if df > 0 else 1.0

    # Per-node inconsistency: how much each node contributes to Q
    influence = Sigma_pinv @ obs
    node_scores = {}
    for node in nodes:
        ni = node_idx[node]
        score = sum(
            abs(influence[e_idx] * obs[e_idx])
            for e_idx, (u, v) in enumerate(edges)
            if u == node or v == node
        )
        node_scores[node] = float(score)

    return p, Q, df, node_scores


def _cochran_q(estimates):
    """Standard Cochran's Q heterogeneity test."""
    betas = np.array([e["beta"] for e in estimates.values()])
    ses = np.array([e["se"] for e in estimates.values()])
    weights = 1.0 / ses**2
    beta_fixed = np.sum(weights * betas) / np.sum(weights)
    Q = float(np.sum(weights * (betas - beta_fixed) ** 2))
    df = len(betas) - 1
    p = 1.0 - chi2.cdf(Q, df) if df > 0 else 1.0
    return p, Q, df


def run_sheaf_federated_ehr(seed=42):
    """Simulate multi-site EHR phenotype estimates and test consistency.

    Scenario: 8 hospitals estimating the same treatment effect.
    Some sites have systematic bias (coding differences, population shift).
    Sheaf test detects AND localizes inconsistency.
    """
    print(f"[{datetime.now():%H:%M:%S}] sheaf_federated_ehr: starting")
    rng = np.random.default_rng(seed)

    true_beta = 0.30
    n_sites = 8
    n_reps = 500
    site_ses = [0.04, 0.03, 0.05, 0.035, 0.04, 0.03, 0.045, 0.05]

    # --- Calibration check: type I error at null ---
    print("  Calibration check (2000 reps, no bias)...")
    cal_reps = 2000
    sheaf_rejects = 0
    cochran_rejects = 0
    for _ in tqdm(range(cal_reps), desc="  calibration"):
        estimates = {
            f"site_{i}": {
                "beta": true_beta + rng.normal(0, site_ses[i]),
                "se": site_ses[i],
            }
            for i in range(n_sites)
        }
        p_sheaf, _, _, _ = _sheaf_test(estimates)
        p_cochran, _, _ = _cochran_q(estimates)
        if p_sheaf < 0.05:
            sheaf_rejects += 1
        if p_cochran < 0.05:
            cochran_rejects += 1

    calibration = {
        "n_reps": cal_reps,
        "sheaf_type_I": sheaf_rejects / cal_reps,
        "cochran_type_I": cochran_rejects / cal_reps,
    }
    print(f"    Sheaf type I: {calibration['sheaf_type_I']:.3f}")
    print(f"    Cochran type I: {calibration['cochran_type_I']:.3f}")

    # --- Power sweep: one outlier site with increasing bias ---
    print(f"\n  Power sweep: outlier magnitude 0 -> 0.25, {n_reps} reps each")
    bias_levels = np.linspace(0, 0.25, 26)
    power_sweep = []
    for bias in tqdm(bias_levels, desc="  power sweep"):
        sheaf_det = 0
        cochran_det = 0
        for _ in range(n_reps):
            estimates = {}
            for i in range(n_sites - 1):
                estimates[f"site_{i}"] = {
                    "beta": true_beta + rng.normal(0, site_ses[i]),
                    "se": site_ses[i],
                }
            estimates["outlier_site"] = {
                "beta": true_beta + bias + rng.normal(0, 0.04),
                "se": 0.04,
            }
            p_s, _, _, _ = _sheaf_test(estimates)
            p_c, _, _ = _cochran_q(estimates)
            if p_s < 0.05:
                sheaf_det += 1
            if p_c < 0.05:
                cochran_det += 1
        power_sweep.append({
            "bias": float(bias),
            "sheaf_power": sheaf_det / n_reps,
            "cochran_power": cochran_det / n_reps,
        })

    # --- Localization demo: 2 biased sites, does sheaf find them? ---
    print("\n  Localization demo: 2 biased sites (site_5 +0.15, site_7 -0.12)")
    loc_reps = 500
    site5_rank_sum = 0.0
    site7_rank_sum = 0.0
    for _ in tqdm(range(loc_reps), desc="  localization"):
        estimates = {}
        for i in range(n_sites):
            bias = 0.0
            if i == 5:
                bias = 0.15
            elif i == 7:
                bias = -0.12
            estimates[f"site_{i}"] = {
                "beta": true_beta + bias + rng.normal(0, site_ses[i]),
                "se": site_ses[i],
            }
        _, _, _, node_scores = _sheaf_test(estimates)
        ranked = sorted(node_scores.items(), key=lambda x: -x[1])
        rank_map = {name: rank + 1 for rank, (name, _) in enumerate(ranked)}
        site5_rank_sum += rank_map["site_5"]
        site7_rank_sum += rank_map["site_7"]

    localization = {
        "site_5_mean_rank": site5_rank_sum / loc_reps,
        "site_7_mean_rank": site7_rank_sum / loc_reps,
        "n_sites": n_sites,
        "interpretation": "rank 1 = most inconsistent; closer to 1 = better localization",
    }
    print(f"    site_5 mean rank: {localization['site_5_mean_rank']:.2f} / {n_sites}")
    print(f"    site_7 mean rank: {localization['site_7_mean_rank']:.2f} / {n_sites}")

    # --- Multi-site structured inconsistency: 3 regional clusters ---
    print("\n  Structured inconsistency: 3 regional clusters with drift")
    struct_reps = 500
    separation_levels = np.linspace(0, 0.2, 11)
    structured_sweep = []
    for sep in tqdm(separation_levels, desc="  structured"):
        sheaf_det = 0
        cochran_det = 0
        for _ in range(struct_reps):
            estimates = {}
            for i in range(3):
                estimates[f"northeast_{i}"] = {
                    "beta": true_beta + rng.normal(0, 0.03),
                    "se": 0.03,
                }
            for i in range(3):
                estimates[f"southeast_{i}"] = {
                    "beta": true_beta + sep + rng.normal(0, 0.03),
                    "se": 0.03,
                }
            for i in range(2):
                estimates[f"west_{i}"] = {
                    "beta": true_beta - sep * 0.7 + rng.normal(0, 0.04),
                    "se": 0.04,
                }
            p_s, _, _, _ = _sheaf_test(estimates)
            p_c, _, _ = _cochran_q(estimates)
            if p_s < 0.05:
                sheaf_det += 1
            if p_c < 0.05:
                cochran_det += 1
        structured_sweep.append({
            "separation": float(sep),
            "sheaf_power": sheaf_det / struct_reps,
            "cochran_power": cochran_det / struct_reps,
        })

    result = {
        "calibration": calibration,
        "power_sweep": power_sweep,
        "localization": localization,
        "structured_sweep": structured_sweep,
    }
    print(f"[{datetime.now():%H:%M:%S}] sheaf_federated_ehr: done")
    return result


# ======================================================================
# 2. CONFOUND COLLAPSE AUDIT (Xia)
# ======================================================================

def run_confound_collapse_audit(seed=42):
    """Biomarker confound audit: which survive after controlling for sampling proxy?

    Simulates MS imaging biomarkers predicting disability progression.
    - 10 "real" biomarkers: genuinely track disease severity
    - 10 "confounded" biomarkers: track ROI size (voxel count), not disease

    Tests three association methods:
    1. Raw correlation with disability
    2. Partial correlation controlling for ROI size
    3. Intensive correction: biomarker / ROI_size, then correlate

    Ground truth: we know which are confounded. Measure AUROC for each method.
    """
    print(f"[{datetime.now():%H:%M:%S}] confound_collapse_audit: starting")
    rng = np.random.default_rng(seed)

    n_real = 10
    n_confounded = 10
    n_total = n_real + n_confounded
    sample_sizes = [200, 500, 1000, 2000, 5000]
    n_reps = 100

    results_by_n = []
    for n_patients in tqdm(sample_sizes, desc="  sample sizes"):
        raw_aurocs = []
        partial_aurocs = []
        intensive_aurocs = []
        collapse_counts = []

        for _ in range(n_reps):
            # Latent variables
            disease_severity = rng.normal(0, 1, n_patients)
            roi_size = 100 - 10 * disease_severity + rng.normal(0, 8, n_patients)
            roi_size = np.clip(roi_size, 10, 200)
            disability = 2 * disease_severity + rng.normal(0, 0.8, n_patients)

            biomarker_names = []
            is_real = []
            raw_corrs = []
            partial_corrs = []
            intensive_corrs = []

            # Real biomarkers: correlate with disease_severity, NOT roi_size
            for i in range(n_real):
                coeff = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
                bio = coeff * disease_severity + rng.normal(0, 0.8, n_patients)
                biomarker_names.append(f"real_{i}")
                is_real.append(1)

                raw_corrs.append(abs(pearsonr(bio, disability)[0]))

                # Partial correlation: regress out roi_size from both
                from numpy.linalg import lstsq
                X = np.column_stack([roi_size, np.ones(n_patients)])
                bio_resid = bio - X @ lstsq(X, bio, rcond=None)[0]
                dis_resid = disability - X @ lstsq(X, disability, rcond=None)[0]
                partial_corrs.append(abs(pearsonr(bio_resid, dis_resid)[0]))

                # Intensive: normalize by roi_size
                bio_intensive = bio / np.sqrt(np.abs(roi_size))
                intensive_corrs.append(abs(pearsonr(bio_intensive, disability)[0]))

            # Confounded biomarkers: correlate with roi_size (extensive quantity)
            for i in range(n_confounded):
                coeff = rng.uniform(0.3, 1.5)
                bio = coeff * roi_size + rng.normal(0, 5, n_patients)
                biomarker_names.append(f"confound_{i}")
                is_real.append(0)

                raw_corrs.append(abs(pearsonr(bio, disability)[0]))

                X = np.column_stack([roi_size, np.ones(n_patients)])
                bio_resid = bio - X @ lstsq(X, bio, rcond=None)[0]
                dis_resid = disability - X @ lstsq(X, disability, rcond=None)[0]
                partial_corrs.append(abs(pearsonr(bio_resid, dis_resid)[0]))

                bio_intensive = bio / np.sqrt(np.abs(roi_size))
                intensive_corrs.append(abs(pearsonr(bio_intensive, disability)[0]))

            is_real = np.array(is_real)
            raw_corrs = np.array(raw_corrs)
            partial_corrs = np.array(partial_corrs)
            intensive_corrs = np.array(intensive_corrs)

            # AUROC: can we distinguish real from confounded?
            from sklearn.metrics import roc_auc_score
            raw_aurocs.append(roc_auc_score(is_real, raw_corrs))
            partial_aurocs.append(roc_auc_score(is_real, partial_corrs))
            intensive_aurocs.append(roc_auc_score(is_real, intensive_corrs))

            # How many confounded biomarkers "collapse" (partial < 0.1)?
            confound_partials = partial_corrs[is_real == 0]
            collapse_counts.append(int(np.sum(confound_partials < 0.1)))

        results_by_n.append({
            "n_patients": n_patients,
            "raw_auroc_mean": float(np.mean(raw_aurocs)),
            "raw_auroc_std": float(np.std(raw_aurocs)),
            "partial_auroc_mean": float(np.mean(partial_aurocs)),
            "partial_auroc_std": float(np.std(partial_aurocs)),
            "intensive_auroc_mean": float(np.mean(intensive_aurocs)),
            "intensive_auroc_std": float(np.std(intensive_aurocs)),
            "collapse_rate": float(np.mean(collapse_counts) / n_confounded),
        })
        r = results_by_n[-1]
        print(f"    n={n_patients:5d}: raw_AUROC={r['raw_auroc_mean']:.3f} "
              f"partial={r['partial_auroc_mean']:.3f} intensive={r['intensive_auroc_mean']:.3f} "
              f"collapse={r['collapse_rate']:.2f}")

    # --- Single detailed run at n=2000 for per-biomarker breakdown ---
    print("\n  Detailed breakdown at n=2000:")
    n_patients = 2000
    disease_severity = rng.normal(0, 1, n_patients)
    roi_size = 100 - 10 * disease_severity + rng.normal(0, 8, n_patients)
    roi_size = np.clip(roi_size, 10, 200)
    disability = 2 * disease_severity + rng.normal(0, 0.8, n_patients)

    detailed = []
    for i in range(n_real):
        coeff = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
        bio = coeff * disease_severity + rng.normal(0, 0.8, n_patients)
        X = np.column_stack([roi_size, np.ones(n_patients)])
        bio_r = bio - X @ np.linalg.lstsq(X, bio, rcond=None)[0]
        dis_r = disability - X @ np.linalg.lstsq(X, disability, rcond=None)[0]
        bio_int = bio / np.sqrt(np.abs(roi_size))

        detailed.append({
            "name": f"real_{i}",
            "type": "real",
            "raw_corr": float(abs(pearsonr(bio, disability)[0])),
            "partial_corr": float(abs(pearsonr(bio_r, dis_r)[0])),
            "intensive_corr": float(abs(pearsonr(bio_int, disability)[0])),
            "roi_corr": float(abs(pearsonr(bio, roi_size)[0])),
        })

    for i in range(n_confounded):
        coeff = rng.uniform(0.3, 1.5)
        bio = coeff * roi_size + rng.normal(0, 5, n_patients)
        X = np.column_stack([roi_size, np.ones(n_patients)])
        bio_r = bio - X @ np.linalg.lstsq(X, bio, rcond=None)[0]
        dis_r = disability - X @ np.linalg.lstsq(X, disability, rcond=None)[0]
        bio_int = bio / np.sqrt(np.abs(roi_size))

        detailed.append({
            "name": f"confound_{i}",
            "type": "confounded",
            "raw_corr": float(abs(pearsonr(bio, disability)[0])),
            "partial_corr": float(abs(pearsonr(bio_r, dis_r)[0])),
            "intensive_corr": float(abs(pearsonr(bio_int, disability)[0])),
            "roi_corr": float(abs(pearsonr(bio, roi_size)[0])),
        })

    print(f"    {'name':15s} {'type':12s} {'raw':>6s} {'partial':>8s} {'intensive':>10s} {'roi_corr':>9s}")
    for d in detailed:
        print(f"    {d['name']:15s} {d['type']:12s} {d['raw_corr']:6.3f} {d['partial_corr']:8.3f} "
              f"{d['intensive_corr']:10.3f} {d['roi_corr']:9.3f}")

    result = {
        "auroc_by_sample_size": results_by_n,
        "detailed_breakdown": detailed,
        "n_real": n_real,
        "n_confounded": n_confounded,
    }
    print(f"[{datetime.now():%H:%M:%S}] confound_collapse_audit: done")
    return result


# ======================================================================
# 3. CURVATURE CAUSAL GRAPH (Visweswaran)
# ======================================================================

def _random_dag(n_nodes, edge_prob, rng):
    """Generate a random DAG as an upper-triangular adjacency matrix."""
    adj = (rng.random((n_nodes, n_nodes)) < edge_prob).astype(float)
    adj = np.triu(adj, k=1)
    return adj


def _simulate_linear_sem(adj, n_samples, rng, noise_std=1.0):
    """Simulate data from a linear structural equation model."""
    n = adj.shape[0]
    W = adj * rng.uniform(0.5, 2.0, adj.shape) * rng.choice([-1, 1], adj.shape)
    X = np.zeros((n_samples, n))
    for i in range(n):
        parents = np.where(W[:, i] != 0)[0]
        if len(parents) > 0:
            X[:, i] = X[:, parents] @ W[parents, i] + rng.normal(0, noise_std, n_samples)
        else:
            X[:, i] = rng.normal(0, noise_std, n_samples)
    return X, W


def _learn_graph_corr(X, threshold=0.15):
    """Learn undirected graph by thresholding partial correlations."""
    n = X.shape[1]
    cov = np.cov(X.T)
    precision = np.linalg.inv(cov + 1e-6 * np.eye(n))
    d = np.sqrt(np.diag(precision))
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 0)
    adj = (np.abs(partial_corr) > threshold).astype(float)
    return adj, partial_corr


def _ollivier_ricci_graph(G, alpha=0.5):
    """Compute Ollivier-Ricci curvature for all edges in a networkx graph."""
    import networkx as nx
    import ot

    if G.number_of_edges() == 0:
        return {}

    sp = dict(nx.all_pairs_shortest_path_length(G))
    curvatures = {}

    for u, v in G.edges():
        nb_u = list(G.neighbors(u))
        nb_v = list(G.neighbors(v))
        all_nodes = sorted(set([u] + nb_u + [v] + nb_v))
        idx = {nd: i for i, nd in enumerate(all_nodes)}

        mu = np.zeros(len(all_nodes))
        mu[idx[u]] = alpha
        for nb in nb_u:
            mu[idx[nb]] += (1 - alpha) / len(nb_u)

        nu = np.zeros(len(all_nodes))
        nu[idx[v]] = alpha
        for nb in nb_v:
            nu[idx[nb]] += (1 - alpha) / len(nb_v)

        cost = np.zeros((len(all_nodes), len(all_nodes)))
        for i, ni in enumerate(all_nodes):
            for j, nj in enumerate(all_nodes):
                cost[i, j] = sp.get(ni, {}).get(nj, 100)

        W1 = ot.emd2(mu, nu, cost)
        d_uv = sp[u][v]
        curvatures[(u, v)] = float(1.0 - W1 / d_uv) if d_uv > 0 else 0.0

    return curvatures


def run_curvature_causal_graph(seed=42):
    """Test whether Ollivier-Ricci curvature identifies edge importance in learned causal graphs.

    For each random DAG:
    1. Simulate data from linear SEM
    2. Learn undirected graph via partial correlation thresholding
    3. Compute ORC on learned graph
    4. Label edges: TP (in true skeleton) vs FP (not in true skeleton)
    5. Test: do TP and FP edges have different curvature distributions?
    """
    import networkx as nx

    print(f"[{datetime.now():%H:%M:%S}] curvature_causal_graph: starting")
    rng = np.random.default_rng(seed)

    n_graphs = 40
    node_counts = [10, 15, 20]
    n_samples = 1000

    all_tp_curvatures = []
    all_fp_curvatures = []
    per_graph = []

    for g_idx in tqdm(range(n_graphs), desc="  graphs"):
        n_nodes = rng.choice(node_counts)
        edge_prob = rng.uniform(0.15, 0.35)

        true_adj = _random_dag(n_nodes, edge_prob, rng)
        true_skeleton = ((true_adj + true_adj.T) > 0).astype(float)
        n_true_edges = int(np.sum(true_skeleton) / 2)

        if n_true_edges < 3:
            continue

        X, W = _simulate_linear_sem(true_adj, n_samples, rng)
        learned_adj, pcorr = _learn_graph_corr(X, threshold=0.12)

        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if learned_adj[i, j] > 0:
                    G.add_edge(i, j)

        if G.number_of_edges() < 3:
            continue

        curvatures = _ollivier_ricci_graph(G)

        tp_curvs = []
        fp_curvs = []
        for (u, v), k in curvatures.items():
            is_true = true_skeleton[u, v] > 0 or true_skeleton[v, u] > 0
            if is_true:
                tp_curvs.append(k)
                all_tp_curvatures.append(k)
            else:
                fp_curvs.append(k)
                all_fp_curvatures.append(k)

        per_graph.append({
            "graph_idx": g_idx,
            "n_nodes": int(n_nodes),
            "n_true_edges": n_true_edges,
            "n_learned_edges": G.number_of_edges(),
            "n_tp": len(tp_curvs),
            "n_fp": len(fp_curvs),
            "tp_curvature_mean": float(np.mean(tp_curvs)) if tp_curvs else None,
            "fp_curvature_mean": float(np.mean(fp_curvs)) if fp_curvs else None,
        })

    # Aggregate statistics
    tp_arr = np.array(all_tp_curvatures)
    fp_arr = np.array(all_fp_curvatures)

    from scipy.stats import mannwhitneyu
    if len(tp_arr) > 5 and len(fp_arr) > 5:
        U, mw_p = mannwhitneyu(tp_arr, fp_arr, alternative="two-sided")
        auroc = float(U / (len(tp_arr) * len(fp_arr)))
    else:
        mw_p = None
        auroc = None

    aggregate = {
        "n_graphs_used": len(per_graph),
        "total_tp_edges": len(tp_arr),
        "total_fp_edges": len(fp_arr),
        "tp_curvature_mean": float(np.mean(tp_arr)) if len(tp_arr) > 0 else None,
        "tp_curvature_std": float(np.std(tp_arr)) if len(tp_arr) > 0 else None,
        "fp_curvature_mean": float(np.mean(fp_arr)) if len(fp_arr) > 0 else None,
        "fp_curvature_std": float(np.std(fp_arr)) if len(fp_arr) > 0 else None,
        "mann_whitney_p": float(mw_p) if mw_p is not None else None,
        "auroc_tp_vs_fp": auroc,
    }

    print(f"\n  Aggregate:")
    print(f"    TP edges: n={len(tp_arr)}, curvature={np.mean(tp_arr):.4f} +/- {np.std(tp_arr):.4f}")
    print(f"    FP edges: n={len(fp_arr)}, curvature={np.mean(fp_arr):.4f} +/- {np.std(fp_arr):.4f}")
    print(f"    Mann-Whitney p={mw_p:.4f}" if mw_p else "    Mann-Whitney: insufficient data")
    print(f"    AUROC (TP vs FP): {auroc:.3f}" if auroc else "    AUROC: insufficient data")

    # Node-level: average curvature vs degree in true DAG
    # Higher-degree nodes (hubs) should have different curvature patterns
    node_curvature_vs_degree = []
    for g in per_graph[-5:]:  # last 5 graphs for illustration
        g_idx = g["graph_idx"]

    result = {
        "aggregate": aggregate,
        "per_graph": per_graph,
        "tp_curvature_histogram": np.histogram(tp_arr, bins=20)[0].tolist() if len(tp_arr) > 0 else [],
        "fp_curvature_histogram": np.histogram(fp_arr, bins=20)[0].tolist() if len(fp_arr) > 0 else [],
    }
    print(f"[{datetime.now():%H:%M:%S}] curvature_causal_graph: done")
    return result


# ======================================================================
# 4. TREATMENT HETEROGENEITY DETECTION (Xia)
# ======================================================================

def run_treatment_heterogeneity(seed=42):
    """Geometric detection of hidden treatment effect subtypes.

    Simulates an MS treatment comparison (BCD-like vs NTZ-like):
    - 3 patient subtypes with different treatment responses
    - Average treatment effect is near zero (cancellation)
    - Standard ATE analysis finds nothing
    - Subspace analysis via PCA on treatment-covariate interactions finds subtypes

    Inspired by Xia's BCD/NTZ semi-supervised causal analysis paper.
    """
    print(f"[{datetime.now():%H:%M:%S}] treatment_heterogeneity: starting")
    rng = np.random.default_rng(seed)

    n_patients = 3000
    n_covariates = 10  # age, EDSS baseline, disease duration, lesion load, ...
    n_reps = 200

    def simulate_ms_trial(n, rng):
        """Generate one MS trial with hidden subtypes."""
        # 3 subtypes defined by latent biology
        subtype_probs = [0.40, 0.40, 0.20]
        subtypes = rng.choice(3, size=n, p=subtype_probs)

        # Covariates partially reveal subtype
        covariates = rng.normal(0, 1, (n, n_covariates))
        # Subtype 0: younger, lower EDSS, more inflammatory
        covariates[subtypes == 0, 0] -= 0.8  # age
        covariates[subtypes == 0, 1] -= 0.5  # EDSS
        covariates[subtypes == 0, 3] += 0.6  # inflammation marker
        # Subtype 1: older, higher EDSS, more progressive
        covariates[subtypes == 1, 0] += 0.6  # age
        covariates[subtypes == 1, 1] += 0.4  # EDSS
        covariates[subtypes == 1, 4] += 0.5  # progression marker
        # Subtype 2: mixed/refractory
        covariates[subtypes == 2, 2] += 0.7  # disease duration

        # Treatment assignment (imperfect randomization — some confounding)
        propensity = 0.5 + 0.1 * covariates[:, 0] - 0.05 * covariates[:, 1]
        propensity = np.clip(propensity, 0.15, 0.85)
        treatment = rng.binomial(1, propensity)  # 1=BCD, 0=NTZ

        # Treatment effects differ by subtype
        # Type 0 (inflammatory): BCD works (+2), NTZ neutral (0)
        # Type 1 (progressive): NTZ works (+1.5), BCD harms (-1)
        # Type 2 (refractory): neither works
        effect = np.zeros(n)
        effect[subtypes == 0] = 2.0 * treatment[subtypes == 0]
        effect[subtypes == 1] = -1.0 * treatment[subtypes == 1] + 1.5 * (1 - treatment[subtypes == 1])
        effect[subtypes == 2] = 0.0

        # Outcome = baseline severity + treatment effect + noise
        baseline = 0.5 * covariates[:, 1] + 0.3 * covariates[:, 0] + rng.normal(0, 0.5, n)
        outcome = baseline + effect + rng.normal(0, 1.5, n)

        return covariates, treatment, outcome, subtypes, effect

    # --- Standard ATE analysis ---
    print("  Standard ATE analysis (should show ~null effect)...")
    ate_estimates = []
    for _ in tqdm(range(n_reps), desc="  ATE"):
        covs, trt, out, sub, eff = simulate_ms_trial(n_patients, rng)
        ate = float(np.mean(out[trt == 1]) - np.mean(out[trt == 0]))
        ate_estimates.append(ate)

    ate_mean = float(np.mean(ate_estimates))
    ate_std = float(np.std(ate_estimates))
    ate_sig_rate = float(np.mean(np.abs(ate_estimates) / ate_std > 1.96))
    print(f"    ATE = {ate_mean:.3f} +/- {ate_std:.3f}, significant {ate_sig_rate:.1%}")

    # --- Interaction test: treatment x each covariate ---
    print("\n  Interaction test: treatment x covariate_i")
    from statsmodels.api import OLS, add_constant
    interaction_powers = []
    for cov_idx in range(n_covariates):
        sig_count = 0
        for _ in range(n_reps):
            covs, trt, out, sub, eff = simulate_ms_trial(n_patients, rng)
            X = add_constant(np.column_stack([
                trt, covs[:, cov_idx], trt * covs[:, cov_idx]
            ]))
            model = OLS(out, X).fit()
            if model.pvalues[3] < 0.05:  # interaction term
                sig_count += 1
        interaction_powers.append({
            "covariate": cov_idx,
            "power": sig_count / n_reps,
        })
        print(f"    cov_{cov_idx}: interaction power = {sig_count/n_reps:.3f}")

    # --- Geometric approach: PCA on treatment-effect residuals ---
    print("\n  Geometric subspace detection (PCA on CATE residuals)...")
    subspace_aurocs = []
    for _ in tqdm(range(n_reps), desc="  subspace"):
        covs, trt, out, sub, eff = simulate_ms_trial(n_patients, rng)

        # Estimate CATE via local regression within covariate neighborhoods
        from sklearn.neighbors import KNeighborsRegressor
        knn_treated = KNeighborsRegressor(n_neighbors=30).fit(covs[trt == 1], out[trt == 1])
        knn_control = KNeighborsRegressor(n_neighbors=30).fit(covs[trt == 0], out[trt == 0])
        cate_hat = knn_treated.predict(covs) - knn_control.predict(covs)

        # PCA on covariate-weighted CATE to find the treatment-effect subspace
        from sklearn.decomposition import PCA
        weighted = covs * cate_hat[:, np.newaxis]
        pca = PCA(n_components=3).fit(weighted)
        projections = pca.transform(weighted)

        # Cluster in the top-2 PC space
        from sklearn.cluster import KMeans
        clusters = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(projections[:, :2])

        # How well do clusters match true subtypes?
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(sub, clusters)
        subspace_aurocs.append(ari)

    subspace_ari_mean = float(np.mean(subspace_aurocs))
    subspace_ari_std = float(np.std(subspace_aurocs))
    print(f"    Subspace ARI (vs true subtypes): {subspace_ari_mean:.3f} +/- {subspace_ari_std:.3f}")

    # --- Comparison: naive K-means on covariates (no treatment info) ---
    print("\n  Baseline: K-means on raw covariates (no treatment info)...")
    naive_aris = []
    for _ in tqdm(range(n_reps), desc="  baseline"):
        covs, trt, out, sub, eff = simulate_ms_trial(n_patients, rng)
        from sklearn.cluster import KMeans
        clusters = KMeans(n_clusters=3, n_init=10, random_state=0).fit_predict(covs)
        from sklearn.metrics import adjusted_rand_score
        naive_aris.append(adjusted_rand_score(sub, clusters))

    naive_ari_mean = float(np.mean(naive_aris))
    naive_ari_std = float(np.std(naive_aris))
    print(f"    Naive ARI: {naive_ari_mean:.3f} +/- {naive_ari_std:.3f}")

    result = {
        "ate_analysis": {
            "mean": ate_mean,
            "std": ate_std,
            "significance_rate": ate_sig_rate,
            "interpretation": "near-zero ATE: subtypes cancel each other",
        },
        "interaction_test": interaction_powers,
        "geometric_subspace": {
            "ari_mean": subspace_ari_mean,
            "ari_std": subspace_ari_std,
            "method": "PCA on covariate-weighted CATE, K-means in top-2 PCs",
        },
        "naive_baseline": {
            "ari_mean": naive_ari_mean,
            "ari_std": naive_ari_std,
            "method": "K-means on raw covariates",
        },
        "improvement_ratio": subspace_ari_mean / naive_ari_mean if naive_ari_mean > 0 else None,
        "n_patients": n_patients,
        "n_reps": n_reps,
    }
    print(f"[{datetime.now():%H:%M:%S}] treatment_heterogeneity: done")
    return result
