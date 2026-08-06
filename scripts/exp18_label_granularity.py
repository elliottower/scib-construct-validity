"""Experiment 18: Label Granularity Robustness [CONFIRMATORY].

Prereg: scripts/exp18_label_granularity_prereg.md (frozen before results).

scIB bio-conservation metrics score an embedding against a cell-type labelling.
Cell types are nested, so the same cells support a family of coarser labellings,
and no published evaluation declares which one it used. This asks whether the
embedding ranking survives coarsening.

Inputs are exp10's: same tissues, embeddings, assay pair, cell budget, seed.
The single manipulated variable is the label vector.

Usage:
    python scripts/exp18_label_granularity.py
    OUT_DIR=/vol/exp18_label_granularity python scripts/exp18_label_granularity.py
"""
import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from tqdm import tqdm

# --- exp10 parameters, held fixed -------------------------------------------
CENSUS_VERSION = "2023-12-15"
ORGANISM = "homo_sapiens"
MAX_CELLS = 2000
SEED = 20260713
N_BOOTSTRAP = 1000

OBS_COLUMNS = [
    "cell_type", "tissue", "assay", "disease",
    "dataset_id", "donor_id", "is_primary_data",
]
TISSUES = {
    "lung":   "UBERON:0002048",
    "liver":  "UBERON:0002107",
    "kidney": "UBERON:0002113",
    "brain":  "UBERON:0000955",
}
SOURCE_ASSAY = "EFO:0009922"   # 10x 3' v3
TARGET_ASSAY = "EFO:0008931"   # Smart-seq2
REAL_EMBEDDINGS = ["geneformer", "scvi", "scgpt"]
BIO_METRICS = [
    "nmi_leiden", "ari_leiden", "silhouette_label",
    "clisi", "isolated_label_asw",
]

# --- exp18 parameters --------------------------------------------------------
GRANULARITY_FRACTIONS = [1.0, 0.75, 0.50, 0.25]
MIN_LEAF_TYPES = 6          # abort condition 2
N_FLOOR_SUBSAMPLES = 3
FLOOR_FRACTION = 0.90
FLOOR_CEILING = 0.40        # abort condition 4

OUT_DIR = Path(os.environ.get("OUT_DIR", "results/exp18_label_granularity"))


def _ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Pure helpers — locally testable without Census or scib-metrics
# ---------------------------------------------------------------------------

def build_granularity_ladder(X_src_raw, labels_src, all_labels, fractions=GRANULARITY_FRACTIONS):
    """Nested label maps from source-domain centroids in log1p CP10K gene space.

    Returns {fraction: {leaf_label -> group_label}}. Types absent from the source
    stay singletons at every level, per the prereg.
    """
    counts = np.asarray(X_src_raw, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    X_log = np.log1p(counts / totals * 1e4)

    src_types = [t for t in pd.unique(labels_src) if t is not None]
    src_types = sorted(str(t) for t in src_types)
    centroids = np.vstack([
        X_log[np.asarray(labels_src).astype(str) == t].mean(axis=0) for t in src_types
    ])

    orphans = sorted(set(str(t) for t in all_labels) - set(src_types))
    L = len(src_types)

    ladder = {}
    if L >= 2:
        Z = linkage(pdist(centroids, metric="correlation"), method="average")
    for frac in fractions:
        k = int(np.ceil(frac * L))
        k = max(1, min(k, L))
        if L < 2:
            assign = {t: t for t in src_types}
        else:
            cl = fcluster(Z, k, criterion="maxclust")
            assign = {t: f"G{frac}_{c}" for t, c in zip(src_types, cl)}
        for o in orphans:
            assign[o] = f"orphan_{o}"
        ladder[frac] = assign
    return ladder, src_types, orphans


def apply_ladder(labels, assign):
    lab = np.asarray(labels).astype(str)
    return np.array([assign.get(x, f"orphan_{x}") for x in lab], dtype=object)


def inversion_rate(scores_a, scores_b, embeddings):
    """Fraction of embedding pairs whose sign(m_A - m_B) flips between two conditions.

    scores_* are dict[embedding -> float]. Pairs with a None/NaN on either side,
    or an exact tie in either condition, are skipped and counted as excluded.
    """
    inverted, total, excluded = 0, 0, 0
    for a, b in combinations(embeddings, 2):
        va, vb = scores_a.get(a), scores_a.get(b)
        wa, wb = scores_b.get(a), scores_b.get(b)
        vals = [va, vb, wa, wb]
        if any(v is None or not np.isfinite(v) for v in vals):
            excluded += 1
            continue
        s1, s2 = np.sign(va - vb), np.sign(wa - wb)
        if s1 == 0 or s2 == 0:
            excluded += 1
            continue
        total += 1
        if s1 != s2:
            inverted += 1
    return inverted, total, excluded


def bootstrap_diff_ci(pairs_gran, pairs_floor, n_boot=N_BOOTSTRAP, alpha=0.05):
    """CI on (granularity inversion rate - floor inversion rate) by resampling pairs."""
    g = np.asarray(pairs_gran, dtype=float)
    f = np.asarray(pairs_floor, dtype=float)
    if len(g) == 0 or len(f) == 0:
        return None, None
    rng = np.random.default_rng(SEED)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        gi = rng.choice(g, size=len(g), replace=True)
        fi = rng.choice(f, size=len(f), replace=True)
        diffs[i] = gi.mean() - fi.mean()
    return float(np.percentile(diffs, 100 * alpha / 2)), float(np.percentile(diffs, 100 * (1 - alpha / 2)))


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_bio_metrics(X_emb, labels, nn_cache=None):
    """The five label-dependent scIB bio metrics. nn_cache reuses kNN across granularities."""
    from scib_metrics import (
        nmi_ari_cluster_labels_leiden,
        silhouette_label,
        isolated_labels,
        clisi_knn,
    )
    from scib_metrics.nearest_neighbors import pynndescent

    if nn_cache is None:
        nn_cache = {}
    if "nn_15" not in nn_cache:
        nn_cache["nn_15"] = pynndescent(X_emb, n_neighbors=15)
        nn_cache["nn_90"] = pynndescent(X_emb, n_neighbors=90)

    labels = np.asarray(labels).astype(str)
    batch = nn_cache["batch"]
    out = {}

    try:
        r = nmi_ari_cluster_labels_leiden(nn_cache["nn_15"], labels, optimize_resolution=True)
        out["nmi_leiden"] = float(r["nmi"])
        out["ari_leiden"] = float(r["ari"])
    except Exception as e:
        print(f"  {_ts()} WARNING nmi/ari: {e}")
        out["nmi_leiden"] = None
        out["ari_leiden"] = None

    for name, fn in (
        ("silhouette_label", lambda: float(silhouette_label(X_emb, labels))),
        ("isolated_label_asw", lambda: float(isolated_labels(X_emb, labels, batch))),
        ("clisi", lambda: float(np.nanmean(clisi_knn(nn_cache["nn_90"], labels)))),
    ):
        try:
            out[name] = fn()
        except Exception as e:
            print(f"  {_ts()} WARNING {name}: {e}")
            out[name] = None
    return out, nn_cache


# ---------------------------------------------------------------------------
# Embedding builders — copied from exp10 so nulls are bit-identical
# ---------------------------------------------------------------------------

def random_projection(X_raw, d_out=512):
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal((X_raw.shape[1], d_out)) / np.sqrt(d_out)
    return (np.log1p(X_raw.astype(np.float64)) @ R).astype(np.float32)


def untrained_encoder(X_raw, d_out=512, d_hidden=256):
    rng = np.random.default_rng(SEED)
    n_genes = X_raw.shape[1]
    W1 = rng.standard_normal((n_genes, d_hidden)) * np.sqrt(2.0 / (n_genes + d_hidden))
    W2 = rng.standard_normal((d_hidden, d_out)) * np.sqrt(2.0 / (d_hidden + d_out))
    X_log = np.log1p(X_raw.astype(np.float64))
    return (np.maximum(0, X_log @ W1) @ W2).astype(np.float32)


def bag_of_genes_pca_combined(X_src_raw, X_tgt_raw, d_out=512):
    from sklearn.decomposition import PCA
    X = np.log1p(np.vstack([X_src_raw, X_tgt_raw]).astype(np.float64))
    n_comp = min(d_out, X.shape[0], X.shape[1])
    Xp = PCA(n_components=n_comp, random_state=0).fit_transform(X).astype(np.float32)
    return Xp[:X_src_raw.shape[0]], Xp[X_src_raw.shape[0]:], n_comp


# ---------------------------------------------------------------------------

def _completed_keys(path):
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["tissue"], r["embedding"], r["condition"]))
    return done


def run_experiment(out_dir=OUT_DIR, checkpoint_cb=None):
    import cellxgene_census

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inc_path = out_dir / "incremental.jsonl"
    done = _completed_keys(inc_path)
    if done:
        print(f"{_ts()} resuming — {len(done)} records already complete")

    manifest = {"tissues": {}, "aborts": []}
    inc = open(inc_path, "a")

    def emit(rec):
        inc.write(json.dumps(rec) + "\n")
        inc.flush()
        os.fsync(inc.fileno())
        if checkpoint_cb:
            checkpoint_cb()

    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        for tissue, tid in TISSUES.items():
            print(f"\n{'='*64}\n{_ts()} tissue: {tissue}")
            src_filter = (f"tissue_ontology_term_id == '{tid}' and is_primary_data == True "
                          f"and assay_ontology_term_id == '{SOURCE_ASSAY}'")
            tgt_filter = (f"tissue_ontology_term_id == '{tid}' and is_primary_data == True "
                          f"and assay_ontology_term_id == '{TARGET_ASSAY}'")

            obs_src = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=src_filter, column_names=["soma_joinid", "donor_id"])
            src_ids = obs_src["soma_joinid"].values
            if len(src_ids) > MAX_CELLS:
                idx = np.random.default_rng(SEED).choice(len(src_ids), MAX_CELLS, replace=False)
                idx.sort()
                src_ids = src_ids[idx]
            obs_tgt = cellxgene_census.get_obs(
                census, ORGANISM, value_filter=tgt_filter, column_names=["soma_joinid", "donor_id"])
            tgt_ids = obs_tgt["soma_joinid"].values
            if len(tgt_ids) == 0:
                manifest["aborts"].append(f"{tissue}: no target cells")
                print(f"  {_ts()} SKIP — no target cells")
                continue
            if len(tgt_ids) > MAX_CELLS:
                idx = np.random.default_rng(SEED + 1).choice(len(tgt_ids), MAX_CELLS, replace=False)
                idx.sort()
                tgt_ids = tgt_ids[idx]

            src = cellxgene_census.get_anndata(
                census, organism=ORGANISM, obs_value_filter=src_filter, obs_coords=src_ids,
                obs_column_names=OBS_COLUMNS, obs_embeddings=REAL_EMBEDDINGS)
            tgt = cellxgene_census.get_anndata(
                census, organism=ORGANISM, obs_value_filter=tgt_filter, obs_coords=tgt_ids,
                obs_column_names=OBS_COLUMNS, obs_embeddings=REAL_EMBEDDINGS)

            X_src_raw = src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X)
            X_tgt_raw = tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X)
            labels_src = src.obs["cell_type"].values
            labels_tgt = tgt.obs["cell_type"].values
            all_labels = np.concatenate([np.asarray(labels_src).astype(str),
                                         np.asarray(labels_tgt).astype(str)])

            ladder, src_types, orphans = build_granularity_ladder(X_src_raw, labels_src, all_labels)
            L = len(src_types)
            sizes = {f: len(set(ladder[f].values())) for f in GRANULARITY_FRACTIONS}
            manifest["tissues"][tissue] = {
                "n_src": int(src.n_obs), "n_tgt": int(tgt.n_obs),
                "n_leaf_types_source": L, "n_orphan_types": len(orphans),
                "ladder_sizes": sizes,
            }
            print(f"  {_ts()} src={src.n_obs} tgt={tgt.n_obs} leaf_types={L} "
                  f"orphans={len(orphans)} ladder={sizes}")

            if L < MIN_LEAF_TYPES:
                manifest["aborts"].append(f"{tissue}: L={L} < {MIN_LEAF_TYPES}, dropped (abort cond. 2)")
                print(f"  {_ts()} DROPPED — abort condition 2")
                continue

            embeddings = {}
            for name in REAL_EMBEDDINGS:
                if name in src.obsm and name in tgt.obsm:
                    embeddings[name] = (np.asarray(src.obsm[name]), np.asarray(tgt.obsm[name]))
                else:
                    print(f"  {_ts()} {name}: unavailable for {tissue}")
            embeddings["random_projection"] = (random_projection(X_src_raw), random_projection(X_tgt_raw))
            embeddings["untrained_encoder"] = (untrained_encoder(X_src_raw), untrained_encoder(X_tgt_raw))
            bs, bt, _ = bag_of_genes_pca_combined(X_src_raw, X_tgt_raw)
            embeddings["bog_pca"] = (bs, bt)

            batch = np.concatenate([np.full(len(labels_src), "source"),
                                    np.full(len(labels_tgt), "target")])

            for emb_name, (Xs, Xt) in tqdm(embeddings.items(), desc=f"  {tissue}", leave=False):
                X_emb = np.vstack([Xs, Xt]).astype(np.float32)
                nn_cache = {"batch": batch}

                for frac in GRANULARITY_FRACTIONS:
                    cond = f"G{frac}"
                    if (tissue, emb_name, cond) in done:
                        continue
                    lab = apply_ladder(all_labels, ladder[frac])
                    print(f"{_ts()} {tissue}/{emb_name}/{cond} "
                          f"({len(set(lab))} classes)")
                    scores, nn_cache = compute_bio_metrics(X_emb, lab, nn_cache)
                    emit({"tissue": tissue, "embedding": emb_name, "condition": cond,
                          "fraction": frac, "n_classes": int(len(set(lab))),
                          "scores": scores, "timestamp": _ts()})

                # stability floor: 90% subsamples at leaf granularity
                lab_leaf = apply_ladder(all_labels, ladder[1.0])
                for s in range(N_FLOOR_SUBSAMPLES):
                    cond = f"floor_{s}"
                    if (tissue, emb_name, cond) in done:
                        continue
                    rng = np.random.default_rng(SEED + 100 + s)
                    keep = rng.choice(X_emb.shape[0],
                                      int(FLOOR_FRACTION * X_emb.shape[0]), replace=False)
                    keep.sort()
                    print(f"{_ts()} {tissue}/{emb_name}/{cond}")
                    sub_cache = {"batch": batch[keep]}
                    scores, _ = compute_bio_metrics(X_emb[keep], lab_leaf[keep], sub_cache)
                    emit({"tissue": tissue, "embedding": emb_name, "condition": cond,
                          "subsample": s, "scores": scores, "timestamp": _ts()})

    inc.close()
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    if checkpoint_cb:
        checkpoint_cb()
    return analyze(out_dir)


def analyze(out_dir=OUT_DIR):
    """Inversion rates, stability floor, bootstrap CI, rank correlations."""
    out_dir = Path(out_dir)
    recs = [json.loads(l) for l in open(out_dir / "incremental.jsonl")]
    df = {}
    for r in recs:
        df[(r["tissue"], r["embedding"], r["condition"])] = r["scores"]

    tissues = sorted({k[0] for k in df})
    embs = sorted({k[1] for k in df})
    real = [e for e in embs if e in REAL_EMBEDDINGS]

    def scores_at(tissue, cond, metric):
        return {e: (df.get((tissue, e, cond), {}) or {}).get(metric) for e in embs}

    results = {"per_metric": {}, "pooled": {}, "rank_corr": {}, "n_real_embeddings": len(real)}

    for use_real, tag in ((True, "real_only"), (False, "all_six")):
        pool = real if use_real else embs
        gran_flags, floor_flags = [], []
        per_metric = {}
        for m in BIO_METRICS:
            g_inv = g_tot = 0
            f_inv = f_tot = 0
            for t in tissues:
                i, n, _ = inversion_rate(scores_at(t, "G1.0", m), scores_at(t, "G0.25", m), pool)
                g_inv += i; g_tot += n
                gran_flags += [1] * i + [0] * (n - i)
                for a, b in combinations(range(N_FLOOR_SUBSAMPLES), 2):
                    i2, n2, _ = inversion_rate(
                        scores_at(t, f"floor_{a}", m), scores_at(t, f"floor_{b}", m), pool)
                    f_inv += i2; f_tot += n2
                    floor_flags += [1] * i2 + [0] * (n2 - i2)
            per_metric[m] = {
                "granularity_inversions": g_inv, "granularity_pairs": g_tot,
                "granularity_rate": (g_inv / g_tot) if g_tot else None,
                "floor_inversions": f_inv, "floor_pairs": f_tot,
                "floor_rate": (f_inv / f_tot) if f_tot else None,
            }
        lo, hi = bootstrap_diff_ci(gran_flags, floor_flags)
        results["per_metric"][tag] = per_metric
        results["pooled"][tag] = {
            "granularity_rate": float(np.mean(gran_flags)) if gran_flags else None,
            "floor_rate": float(np.mean(floor_flags)) if floor_flags else None,
            "diff_ci95": [lo, hi],
            "n_granularity_pairs": len(gran_flags), "n_floor_pairs": len(floor_flags),
        }

    for m in BIO_METRICS:
        rhos = []
        for t in tissues:
            for frac in (0.75, 0.50, 0.25):
                a = scores_at(t, "G1.0", m)
                b = scores_at(t, f"G{frac}", m)
                ok = [e for e in embs
                      if a.get(e) is not None and b.get(e) is not None
                      and np.isfinite(a[e]) and np.isfinite(b[e])]
                if len(ok) >= 3:
                    rho = spearmanr([a[e] for e in ok], [b[e] for e in ok]).statistic
                    if np.isfinite(rho):
                        rhos.append({"tissue": t, "fraction": frac, "rho": float(rho)})
        results["rank_corr"][m] = rhos

    with open(out_dir / "exp18_results.json", "w") as f:
        json.dump(results, f, indent=2)

    p = results["pooled"]["real_only"]
    print("\n" + "=" * 64)
    print(f"POOLED (real embeddings only, n={len(real)})")
    print(f"  granularity inversion rate : {p['granularity_rate']}")
    print(f"  stability floor            : {p['floor_rate']}")
    print(f"  diff 95% CI                : {p['diff_ci95']}")
    if p["floor_rate"] is not None and p["floor_rate"] > FLOOR_CEILING:
        print(f"  ABORT CONDITION 4 — floor {p['floor_rate']:.3f} > {FLOOR_CEILING}; "
              f"report as uninformative")
    print("\nper metric (real only):")
    for m, v in results["per_metric"]["real_only"].items():
        print(f"  {m:<20} gran={v['granularity_rate']}  floor={v['floor_rate']}")
    return results


if __name__ == "__main__":
    run_experiment()
