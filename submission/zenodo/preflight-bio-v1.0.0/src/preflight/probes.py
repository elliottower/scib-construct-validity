"""Probing suite: cell-type classification + regulatory structure probes.

Model-agnostic: takes embedding matrices + labels + batch IDs.
Reimplements and extends the methodology of Morpheus (Cole, 2026) with
a multi-model comparison interface.

Probe types:
    cell_type    — donor-stratified linear probe, macro F1 (Layer 1)
    tf_target    — per-TF binary classification, AUC (Layer 2)
    hub_tf       — hub vs non-hub TF classification, AUC (Layer 3)
    rsa          — embedding similarity vs regulatory adjacency, Spearman rho

Usage:
    from preflight.probes import run_probes

    # Cell-level only (minimal)
    results = run_probes(
        embeddings={"geneformer": X_gf, "scvi": X_scvi},
        labels=cell_types,
        batch_ids=donor_ids,
    )

    # With gene-level regulatory probes
    results = run_probes(
        embeddings={"geneformer": X_cells_gf},
        labels=cell_types,
        batch_ids=donor_ids,
        gene_embeddings={"geneformer": X_genes_gf},
        gene_ids=ensembl_ids,
        regulatory_edges=edges,
    )
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ProbeResult:
    model_name: str
    probe_name: str
    scores: list[float]
    mean: float
    ci_lo: float
    ci_hi: float
    n_items: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PairedComparison:
    model_a: str
    model_b: str
    probe_name: str
    n_pairs: int
    deltas: list[float]
    mean_delta: float
    ci_lo: float
    ci_hi: float
    wilcoxon_stat: float | None
    p_value: float | None


# ======================================================================
# Utilities
# ======================================================================

def donor_stratified_splits(batch_ids: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Assign cells to k folds keeping all cells from the same batch together.

    Uses greedy bin-packing: largest batches assigned first to the
    smallest fold, producing approximately balanced fold sizes.

    Returns (n_cells,) int array of fold assignments.
    """
    rng = np.random.default_rng(seed)
    batch_ids = np.asarray(batch_ids)

    unique_batches, inverse = np.unique(batch_ids, return_inverse=True)
    batch_sizes = np.bincount(inverse)

    order = rng.permutation(len(unique_batches))
    size_order = np.argsort(-batch_sizes[order])
    sorted_batches = order[size_order]

    fold_counts = np.zeros(k, dtype=np.int64)
    batch_to_fold = np.empty(len(unique_batches), dtype=np.int64)

    for b_idx in sorted_batches:
        fold = int(np.argmin(fold_counts))
        batch_to_fold[b_idx] = fold
        fold_counts[fold] += batch_sizes[b_idx]

    folds = batch_to_fold[inverse]

    for b_idx in range(len(unique_batches)):
        mask = inverse == b_idx
        assert len(np.unique(folds[mask])) == 1

    return folds


def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float(np.nan), float(np.nan))
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    return (float(np.quantile(boot_means, alpha / 2)),
            float(np.quantile(boot_means, 1 - alpha / 2)))


def load_regulatory_edges(
    confidence: str = "ABC",
    organism: int = 9606,
) -> list[tuple[str, str]]:
    """Pull TF-target edges from OmniPath (DoRothEA + CollecTRI).

    Args:
        confidence: DoRothEA confidence levels to include (e.g. "ABC" for A+B+C).
        organism: NCBI taxonomy ID (9606 = human).

    Returns:
        List of (tf_gene_symbol, target_gene_symbol) tuples.
    """
    import json
    import urllib.request

    url = (
        f"https://omnipathdb.org/interactions?"
        f"datasets=dorothea,collectri&organisms={organism}"
        f"&dorothea_levels={','.join(confidence)}"
        f"&fields=dorothea_level&format=json"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.loads(resp.read().decode())

    edges = []
    for row in data:
        src = row.get("source_genesymbol", "")
        tgt = row.get("target_genesymbol", "")
        if src and tgt:
            edges.append((src, tgt))

    return edges


# ======================================================================
# Probe 1: Cell-type classification (Layer 1)
# ======================================================================

def cell_type_probe(
    X: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    model_name: str = "unknown",
    min_cells_per_class: int = 10,
) -> ProbeResult:
    """Donor-stratified linear probe for cell-type identity. Returns per-fold macro F1."""
    labels = np.asarray(labels, dtype=str)
    folds = np.asarray(folds)

    class_counts = {}
    for lab in labels:
        class_counts[lab] = class_counts.get(lab, 0) + 1
    keep_classes = {c for c, n in class_counts.items() if n >= min_cells_per_class}
    keep_mask = np.array([lab in keep_classes for lab in labels])

    X = X[keep_mask]
    labels = labels[keep_mask]
    folds = folds[keep_mask]

    le = LabelEncoder()
    y = le.fit_transform(labels)
    n_classes = len(le.classes_)

    unique_folds = sorted(np.unique(folds))
    fold_scores = []

    for fold_id in unique_folds:
        test_mask = folds == fold_id
        train_mask = ~test_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        y_train, y_test = y[train_mask], y[test_mask]
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        if len(train_classes) < 2 or not test_classes.issubset(train_classes):
            continue

        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_scores.append(f1)

    scores = np.array(fold_scores)
    mean = float(scores.mean()) if len(scores) > 0 else 0.0
    ci_lo, ci_hi = _bootstrap_ci(scores) if len(scores) >= 2 else (float(np.nan), float(np.nan))

    return ProbeResult(
        model_name=model_name,
        probe_name="cell_type",
        scores=fold_scores,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_items=int(X.shape[0]),
        details={
            "metric": "macro_f1",
            "n_classes": n_classes,
            "classes": list(le.classes_),
            "cells_per_fold": [int((folds == f).sum()) for f in unique_folds],
            "dropped_classes": sorted(set(class_counts) - keep_classes),
        },
    )


# ======================================================================
# Probe 2: TF-target regulatory structure (Layer 2)
# ======================================================================

def tf_target_probe(
    gene_embeddings: np.ndarray,
    gene_ids: np.ndarray,
    regulatory_edges: list[tuple[str, str]],
    model_name: str = "unknown",
    min_targets: int = 30,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> ProbeResult:
    """Per-TF binary classification: distinguish targets from random non-targets.

    For each TF with >= min_targets target genes, trains a logistic
    regression to separate target gene embeddings from size-matched
    random non-target embeddings.  Reports per-TF AUC.
    """
    rng = np.random.default_rng(seed)
    gene_ids = np.asarray(gene_ids, dtype=str)
    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}

    tf_targets: dict[str, set[str]] = {}
    for tf, target in regulatory_edges:
        if tf in gene_id_to_idx and target in gene_id_to_idx:
            tf_targets.setdefault(tf, set()).add(target)

    all_genes = set(gene_ids)
    all_tfs = set(tf_targets.keys())
    probed_tfs = []
    tf_aucs = []

    for tf in sorted(tf_targets):
        targets = tf_targets[tf]
        if len(targets) < min_targets:
            continue

        pos_genes = sorted(targets)
        pool = sorted(all_genes - targets - all_tfs)
        n_neg = min(len(pos_genes), len(pool))
        if n_neg < 5:
            continue
        neg_genes = list(rng.choice(pool, size=n_neg, replace=False))

        probe_genes = pos_genes + neg_genes
        X = np.array([gene_embeddings[gene_id_to_idx[g]] for g in probe_genes])
        y = np.array([1] * len(pos_genes) + [0] * len(neg_genes))

        n = len(y)
        idx = rng.permutation(n)
        split = max(1, int(n * (1 - test_fraction)))
        train_idx, test_idx = idx[:split], idx[split:]

        if len(test_idx) < 2:
            continue
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_test)) < 2:
            continue

        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_score = clf.decision_function(X_test)
        auc = float(roc_auc_score(y_test, y_score))
        tf_aucs.append(auc)
        probed_tfs.append(tf)

    scores = np.array(tf_aucs)
    mean = float(scores.mean()) if len(scores) > 0 else 0.0
    ci_lo, ci_hi = _bootstrap_ci(scores) if len(scores) >= 2 else (float(np.nan), float(np.nan))

    return ProbeResult(
        model_name=model_name,
        probe_name="tf_target",
        scores=tf_aucs,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_items=len(tf_aucs),
        details={
            "metric": "auc",
            "n_tfs_probed": len(tf_aucs),
            "n_tfs_total": len(tf_targets),
            "min_targets": min_targets,
            "probed_tfs": probed_tfs,
        },
    )


# ======================================================================
# Probe 3: Hub-TF identity (Layer 3)
# ======================================================================

def hub_tf_probe(
    gene_embeddings: np.ndarray,
    gene_ids: np.ndarray,
    regulatory_edges: list[tuple[str, str]],
    model_name: str = "unknown",
    hub_quantile: float = 0.90,
    k: int = 5,
    seed: int = 0,
) -> ProbeResult:
    """Binary classification: predict hub TFs (top 10% by out-degree) from gene embeddings."""
    rng = np.random.default_rng(seed)
    gene_ids = np.asarray(gene_ids, dtype=str)
    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}

    out_degree: dict[str, int] = {}
    for tf, target in regulatory_edges:
        if tf in gene_id_to_idx:
            out_degree[tf] = out_degree.get(tf, 0) + 1

    if len(out_degree) < 10:
        return ProbeResult(
            model_name=model_name, probe_name="hub_tf",
            scores=[], mean=0.0, ci_lo=float("nan"), ci_hi=float("nan"),
            n_items=0, details={"error": f"Only {len(out_degree)} TFs found"},
        )

    tfs = sorted(out_degree.keys())
    degrees = np.array([out_degree[tf] for tf in tfs])
    threshold = np.quantile(degrees, hub_quantile)
    is_hub = (degrees >= threshold).astype(int)

    if is_hub.sum() < 2 or (len(is_hub) - is_hub.sum()) < 2:
        return ProbeResult(
            model_name=model_name, probe_name="hub_tf",
            scores=[], mean=0.0, ci_lo=float("nan"), ci_hi=float("nan"),
            n_items=0, details={"error": "Too few hub or non-hub TFs"},
        )

    X = np.array([gene_embeddings[gene_id_to_idx[tf]] for tf in tfs])
    y = is_hub

    fold_ids = rng.integers(0, k, size=len(tfs))
    fold_aucs = []

    for fold in range(k):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        if test_mask.sum() < 2 or train_mask.sum() < 2:
            continue
        y_train, y_test = y[train_mask], y[test_mask]
        if len(set(y_test)) < 2:
            continue

        scaler = StandardScaler(with_mean=False)
        X_train = scaler.fit_transform(X[train_mask])
        X_test = scaler.transform(X[test_mask])

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)
        y_score = clf.decision_function(X_test)
        auc = float(roc_auc_score(y_test, y_score))
        fold_aucs.append(auc)

    scores = np.array(fold_aucs)
    mean = float(scores.mean()) if len(scores) > 0 else 0.0
    ci_lo, ci_hi = _bootstrap_ci(scores) if len(scores) >= 2 else (float(np.nan), float(np.nan))

    return ProbeResult(
        model_name=model_name,
        probe_name="hub_tf",
        scores=fold_aucs,
        mean=mean,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_items=len(tfs),
        details={
            "metric": "auc",
            "n_tfs": len(tfs),
            "n_hubs": int(is_hub.sum()),
            "hub_threshold_degree": int(threshold),
            "hub_quantile": hub_quantile,
        },
    )


# ======================================================================
# Probe 4: Representational Similarity Analysis (RSA)
# ======================================================================

def rsa_probe(
    gene_embeddings: np.ndarray,
    gene_ids: np.ndarray,
    regulatory_edges: list[tuple[str, str]],
    model_name: str = "unknown",
) -> ProbeResult:
    """Spearman correlation between pairwise embedding similarity and regulatory adjacency.

    Restricts to TF genes, builds a symmetrized binary adjacency
    matrix from regulatory edges, and correlates with cosine similarity.
    """
    gene_ids = np.asarray(gene_ids, dtype=str)
    gene_id_to_idx = {g: i for i, g in enumerate(gene_ids)}

    tf_set = set()
    for tf, target in regulatory_edges:
        if tf in gene_id_to_idx:
            tf_set.add(tf)

    tfs = sorted(tf_set)
    if len(tfs) < 10:
        return ProbeResult(
            model_name=model_name, probe_name="rsa",
            scores=[], mean=0.0, ci_lo=float("nan"), ci_hi=float("nan"),
            n_items=0, details={"error": f"Only {len(tfs)} TFs found"},
        )

    tf_to_local = {tf: i for i, tf in enumerate(tfs)}
    X_tf = np.array([gene_embeddings[gene_id_to_idx[tf]] for tf in tfs])

    sim = 1.0 - cdist(X_tf, X_tf, metric="cosine")

    n = len(tfs)
    adj = np.zeros((n, n), dtype=float)
    for tf, target in regulatory_edges:
        if tf in tf_to_local and target in tf_to_local:
            i, j = tf_to_local[tf], tf_to_local[target]
            adj[i, j] = 1.0
            adj[j, i] = 1.0

    upper_idx = np.triu_indices(n, k=1)
    sim_vec = sim[upper_idx]
    adj_vec = adj[upper_idx]

    if adj_vec.sum() < 5:
        return ProbeResult(
            model_name=model_name, probe_name="rsa",
            scores=[], mean=0.0, ci_lo=float("nan"), ci_hi=float("nan"),
            n_items=0, details={"error": "Too few TF-TF regulatory edges"},
        )

    rho, p = stats.spearmanr(sim_vec, adj_vec)

    return ProbeResult(
        model_name=model_name,
        probe_name="rsa",
        scores=[float(rho)],
        mean=float(rho),
        ci_lo=float("nan"),
        ci_hi=float("nan"),
        n_items=len(upper_idx[0]),
        details={
            "metric": "spearman_rho",
            "rho": float(rho),
            "p_value": float(p),
            "n_tfs": len(tfs),
            "n_pairs": len(upper_idx[0]),
            "n_adjacent_pairs": int(adj_vec.sum()),
        },
    )


# ======================================================================
# Paired comparison
# ======================================================================

def compare_probes(results: list[ProbeResult], probe_name: str) -> list[PairedComparison]:
    """Paired Wilcoxon signed-rank test + bootstrap CI between all model pairs for a probe."""
    comparisons = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            n = min(len(a.scores), len(b.scores))
            if n < 3:
                continue
            deltas = [a.scores[j] - b.scores[j] for j in range(n)]
            deltas_arr = np.array(deltas)

            try:
                stat_result = stats.wilcoxon(deltas_arr, alternative="two-sided")
                w_stat = float(stat_result.statistic)
                p_val = float(stat_result.pvalue)
            except ValueError:
                w_stat = None
                p_val = None

            ci_lo, ci_hi = _bootstrap_ci(deltas_arr)

            comparisons.append(PairedComparison(
                model_a=a.model_name,
                model_b=b.model_name,
                probe_name=probe_name,
                n_pairs=n,
                deltas=deltas,
                mean_delta=float(deltas_arr.mean()),
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                wilcoxon_stat=w_stat,
                p_value=p_val,
            ))
    return comparisons


# ======================================================================
# Backward compat aliases
# ======================================================================

def linear_probe(X, labels, folds, model_name="unknown", min_cells_per_class=10):
    return cell_type_probe(X, labels, folds, model_name, min_cells_per_class)

def compare_models(results):
    return compare_probes(results, probe_name="cell_type")


# ======================================================================
# Main entry point
# ======================================================================

def run_probes(
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray,
    batch_ids: np.ndarray,
    k: int = 5,
    seed: int = 0,
    min_cells_per_class: int = 10,
    gene_embeddings: dict[str, np.ndarray] | None = None,
    gene_ids: np.ndarray | None = None,
    regulatory_edges: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Run all applicable probes across multiple models and compare.

    Args:
        embeddings: {model_name: (n_cells, d)} cell embedding matrices.
        labels: (n_cells,) cell type labels.
        batch_ids: (n_cells,) donor/batch IDs for stratified CV.
        k: Number of CV folds.
        seed: Random seed for split and bootstrap.
        min_cells_per_class: Drop classes with fewer cells.
        gene_embeddings: {model_name: (n_genes, d)} gene embedding matrices.
        gene_ids: (n_genes,) gene identifiers (symbols or Ensembl IDs).
        regulatory_edges: [(tf, target), ...] regulatory relationships.

    Returns:
        Dict with per-probe results, paired comparisons, and summary.
    """
    folds = donor_stratified_splits(batch_ids, k=k, seed=seed)

    all_results: dict[str, dict[str, ProbeResult]] = {}
    all_comparisons: list[PairedComparison] = []

    # --- Cell-type probe (always runs) ---
    ct_results = {}
    for model_name, X in embeddings.items():
        ct_results[model_name] = cell_type_probe(
            X, labels, folds,
            model_name=model_name,
            min_cells_per_class=min_cells_per_class,
        )
    all_results["cell_type"] = ct_results
    all_comparisons.extend(compare_probes(list(ct_results.values()), "cell_type"))

    # --- Gene-level probes (when gene_embeddings + regulatory_edges provided) ---
    has_gene = (gene_embeddings is not None and gene_ids is not None
                and regulatory_edges is not None and len(regulatory_edges) > 0)

    if has_gene:
        for probe_name, probe_fn in [
            ("tf_target", lambda ge, gid, mn: tf_target_probe(ge, gid, regulatory_edges, model_name=mn, seed=seed)),
            ("hub_tf", lambda ge, gid, mn: hub_tf_probe(ge, gid, regulatory_edges, model_name=mn, k=k, seed=seed)),
            ("rsa", lambda ge, gid, mn: rsa_probe(ge, gid, regulatory_edges, model_name=mn)),
        ]:
            probe_results = {}
            for model_name, X_genes in gene_embeddings.items():
                probe_results[model_name] = probe_fn(X_genes, gene_ids, model_name)
            all_results[probe_name] = probe_results
            all_comparisons.extend(compare_probes(list(probe_results.values()), probe_name))

    # --- Summary table ---
    lines = _format_summary(all_results, all_comparisons)

    # --- Flatten for backward compat ---
    flat_results = {}
    for probe_name, probe_dict in all_results.items():
        for model_name, result in probe_dict.items():
            flat_results[f"{model_name}/{probe_name}"] = result

    return {
        "results": flat_results,
        "results_by_probe": all_results,
        "comparisons": all_comparisons,
        "summary": "\n".join(lines),
    }


def _format_summary(
    all_results: dict[str, dict[str, ProbeResult]],
    comparisons: list[PairedComparison],
) -> list[str]:
    lines = []
    for probe_name, probe_results in all_results.items():
        metric = "macro_f1" if probe_name == "cell_type" else "auc"
        if probe_name == "rsa":
            metric = "spearman_rho"

        lines.append(f"=== {probe_name} ({metric}) ===")
        lines.append(f"{'Model':<24} | {'Score':>8} | {'95% CI':<19} | {'n':>6}")
        lines.append("-" * 65)
        for name in sorted(probe_results):
            r = probe_results[name]
            ci = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if not np.isnan(r.ci_lo) else "[n/a]"
            lines.append(f"{name:<24} | {r.mean:>8.3f} | {ci:<19} | {r.n_items:>6}")
        lines.append("")

    probe_comps = {}
    for c in comparisons:
        probe_comps.setdefault(c.probe_name, []).append(c)

    if probe_comps:
        lines.append("=== Paired comparisons ===")
        for pname, comps in probe_comps.items():
            for c in comps:
                sig = f"p={c.p_value:.4f}" if c.p_value is not None else "p=n/a"
                lines.append(f"  [{pname}] {c.model_a} vs {c.model_b}: "
                             f"delta={c.mean_delta:+.3f} [{c.ci_lo:+.3f}, {c.ci_hi:+.3f}], {sig}")
        lines.append("")

    return lines
