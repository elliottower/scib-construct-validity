"""Exp17: Marker gene recovery as non-classification ground truth.

For each (tissue, model) condition, measures whether Leiden clusters
derived from the embedding recover the same differentially expressed
marker genes as the true cell-type labels. This is a non-classification
ground truth: it tests whether embedding geometry preserves gene-level
biological signal, not just label separability.

Algorithm:
1. Compute reference markers: Wilcoxon rank-sum on raw counts with
   true cell-type labels (source cells) -> top N genes per type
2. Cluster target cells: Leiden on embedding kNN graph
3. Compute recovered markers: Wilcoxon rank-sum on raw counts with
   Leiden labels (target cells) -> top N genes per cluster
4. Match clusters to cell types by majority vote
5. Score = mean of best-per-cluster Jaccard overlap of top-N marker
   gene sets (max over clusters mapping to each type, not union)

Ceiling control: marker_gene_recovery_ceiling() uses true target labels
instead of Leiden clusters, measuring maximum achievable MGR given
cross-assay expression shift alone.

Modal wrapper in modal_exp17_marker_gene_recovery.py calls this.
"""
import numpy as np
from collections import Counter
from scipy import stats

import scanpy as sc
import anndata as ad


SEED = 20260801
TOP_N_GENES = 50


def wilcoxon_markers(X_raw, labels, types, top_n=TOP_N_GENES):
    """Compute top-N marker genes per cell type via Wilcoxon rank-sum.

    Returns dict: {cell_type: set of gene indices}.
    """
    markers = {}
    for t in types:
        mask = labels == t
        if mask.sum() < 3:
            markers[t] = set()
            continue
        rest = ~mask
        if rest.sum() < 3:
            markers[t] = set()
            continue

        n_genes = X_raw.shape[1]
        pvals = np.ones(n_genes)
        fold_changes = np.zeros(n_genes)
        x_in = X_raw[mask]
        x_out = X_raw[rest]

        for g in range(n_genes):
            in_vals = x_in[:, g]
            out_vals = x_out[:, g]
            if in_vals.std() == 0 and out_vals.std() == 0:
                continue
            try:
                stat, p = stats.ranksums(in_vals, out_vals)
                pvals[g] = p
                mean_in = in_vals.mean()
                mean_out = out_vals.mean()
                fold_changes[g] = mean_in - mean_out
            except Exception:
                pass

        upregulated = fold_changes > 0
        score = np.where(upregulated, -np.log10(pvals + 1e-300), 0.0)
        top_idx = np.argsort(score)[-top_n:]
        markers[t] = set(top_idx[score[top_idx] > 0])
    return markers


def _leiden_clusters(X_emb, resolution=1.0, n_neighbors=15):
    """Cluster embedding using Leiden via scanpy."""
    adata = ad.AnnData(X=X_emb)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X")
    sc.tl.leiden(adata, resolution=resolution, random_state=SEED)
    return adata.obs["leiden"].values.astype(str)


def _match_clusters_to_types(cluster_labels, true_labels, shared_types):
    """Match each cluster to its majority cell type."""
    cluster_ids = np.unique(cluster_labels)
    matching = {}
    for c in cluster_ids:
        mask = cluster_labels == c
        type_counts = Counter(true_labels[mask])
        shared_counts = {t: type_counts.get(t, 0) for t in shared_types}
        if sum(shared_counts.values()) > 0:
            matching[c] = max(shared_counts, key=shared_counts.get)
    return matching


def marker_gene_recovery(X_raw_tgt, X_emb_tgt, labels_tgt, shared_types,
                         ref_markers, top_n=TOP_N_GENES, resolution=1.0):
    """Compute marker gene recovery score.

    Uses max-over-clusters Jaccard: when multiple Leiden clusters map to
    the same cell type, the best single-cluster Jaccard is taken (not the
    union, which mechanically penalizes over-clustering).

    Args:
        X_raw_tgt: raw count matrix for target cells (n_tgt, n_genes)
        X_emb_tgt: embedding matrix for target cells (n_tgt, d_emb)
        labels_tgt: true cell-type labels for target cells
        shared_types: list of cell types shared between source and target
        ref_markers: dict from wilcoxon_markers() on source cells
        top_n: number of top marker genes per type/cluster
        resolution: Leiden resolution parameter

    Returns dict with mgr_score, n_matched_types, n_clusters,
    per_type_jaccard.
    """
    cluster_labels = _leiden_clusters(X_emb_tgt, resolution=resolution)
    n_clusters = len(np.unique(cluster_labels))

    cluster_ids = np.unique(cluster_labels)
    cluster_markers = wilcoxon_markers(X_raw_tgt, cluster_labels, cluster_ids, top_n)

    matching = _match_clusters_to_types(cluster_labels, labels_tgt, shared_types)

    type_to_clusters = {}
    for c, t in matching.items():
        if t not in type_to_clusters:
            type_to_clusters[t] = []
        type_to_clusters[t].append(c)

    jaccards = {}
    for t in shared_types:
        ref = ref_markers.get(t, set())
        if not ref:
            continue
        clusters_for_type = type_to_clusters.get(t, [])
        if not clusters_for_type:
            jaccards[t] = 0.0
            continue
        best_jaccard = 0.0
        for c in clusters_for_type:
            cmarks = cluster_markers.get(c, set())
            union = ref | cmarks
            if len(union) == 0:
                continue
            j = len(ref & cmarks) / len(union)
            if j > best_jaccard:
                best_jaccard = j
        jaccards[t] = best_jaccard

    mgr_score = float(np.mean(list(jaccards.values()))) if jaccards else 0.0

    return {
        "mgr_score": mgr_score,
        "n_matched_types": len(jaccards),
        "n_clusters": n_clusters,
        "per_type_jaccard": {str(k): float(v) for k, v in jaccards.items()},
    }


def marker_gene_recovery_ceiling(X_raw_tgt, labels_tgt, shared_types,
                                 ref_markers, top_n=TOP_N_GENES):
    """Ceiling control: MGR using true target labels instead of Leiden.

    Measures the maximum achievable MGR given cross-assay expression
    shift, with the embedding removed from the picture. Computed once
    per tissue (not per embedding).
    """
    tgt_markers = wilcoxon_markers(X_raw_tgt, labels_tgt, shared_types, top_n)

    jaccards = {}
    for t in shared_types:
        ref = ref_markers.get(t, set())
        tgt = tgt_markers.get(t, set())
        if not ref:
            continue
        union = ref | tgt
        if len(union) == 0:
            jaccards[t] = 0.0
        else:
            jaccards[t] = len(ref & tgt) / len(union)

    ceiling_score = float(np.mean(list(jaccards.values()))) if jaccards else 0.0

    return {
        "ceiling_score": ceiling_score,
        "n_types": len(jaccards),
        "per_type_jaccard": {str(k): float(v) for k, v in jaccards.items()},
    }
