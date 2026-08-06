"""Experiment 18: Label Granularity Robustness [CONFIRMATORY].

Prereg: scripts/exp18_label_granularity_prereg.md (frozen; amendments A1-A8
recorded before any metric value was computed).

scIB bio-conservation metrics score an embedding against a cell-type labelling.
Cell types nest, so the same cells support a family of coarser labellings, and no
published evaluation declares which one it used. This asks whether the embedding
ranking survives coarsening.

Inputs are exp10's: same tissues, embeddings, assay pair, cell budget, seed.
The single manipulated variable is the label vector.

Three label conditions per (tissue, embedding):
  G{1.0,0.75,0.5,0.25}  the centroid-derived granularity ladder
  floor_{0,1,2}         noise floor - 90% cell subsamples at leaf granularity
  randB_{0,1,2}         Null B - random coarsening size-matched to G0.25

Usage:
    python scripts/exp18_label_granularity.py
    OUT_DIR=/vol/exp18_label_granularity python scripts/exp18_label_granularity.py
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
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

# exp10's published per-tissue (source, target) cell counts. Abort condition 1
# (amendment A5: the leaf-type-count half is struck, exp10 never recorded it).
EXP10_CELL_COUNTS = {
    "lung":   (2000, 2000),
    "liver":  (2000, 2000),
    "kidney": (2000, 370),
    "brain":  (2000, 59),
}

# --- exp18 parameters --------------------------------------------------------
GRANULARITY_FRACTIONS = [1.0, 0.75, 0.50, 0.25]
PRIMARY_COARSE = 0.25
MIN_LEAF_TYPES = 6              # abort condition 2
N_FLOOR_SUBSAMPLES = 3
FLOOR_FRACTION = 0.90
N_RANDOM_DRAWS = 3              # Null B (amendment A3)
FLOOR_CEILING = 0.40            # abort condition 4, applied PER TISSUE (A4)
MIN_TARGET_CELLS_PER_TYPE = 10  # amendment A4
N_BOOTSTRAP = 1000

OUT_DIR = Path(os.environ.get("OUT_DIR", "results/exp18_label_granularity"))
SCRIPT_PATH = Path(__file__).resolve()


def _ts():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def script_sha():
    """Provenance stamp (amendment A2) so a code fix cannot be silently skipped."""
    return hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()[:16]


class AbortCondition(Exception):
    """Raised where the prereg says stop, so the run dies loudly."""


# ---------------------------------------------------------------------------
# Ladder construction
# ---------------------------------------------------------------------------

def build_granularity_ladder(X_src_raw, labels_src, all_labels, fractions=GRANULARITY_FRACTIONS):
    """Nested label maps from source-domain centroids in log1p CP10K gene space.

    frac == 1.0 is the identity labelling, assigned directly rather than through
    fcluster (amendment A1: fcluster(Z, L, 'maxclust') returns fewer than L
    clusters on scipy < ~1.14, which silently made the reference level a
    two-merge coarsening).

    Returns (ladder, src_types, orphans, achieved_k) where achieved_k[frac] is
    the realised source-group count, excluding orphan singletons.
    """
    counts = np.asarray(X_src_raw, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    X_log = np.log1p(counts / totals * 1e4)

    labs = np.asarray(labels_src).astype(str)
    src_types = sorted(set(labs))
    centroids = np.vstack([X_log[labs == t].mean(axis=0) for t in src_types])
    orphans = sorted(set(np.asarray(all_labels).astype(str)) - set(src_types))
    L = len(src_types)

    Z = linkage(pdist(centroids, metric="correlation"), method="average") if L >= 2 else None

    ladder, achieved_k = {}, {}
    for frac in fractions:
        k = max(1, min(int(np.ceil(frac * L)), L))
        if frac == 1.0 or L < 2:
            assign = {t: f"G1.0_{i}" for i, t in enumerate(src_types)}
            got = L
        else:
            cl = fcluster(Z, k, criterion="maxclust")
            got = int(len(np.unique(cl)))
            if got != k:
                raise AbortCondition(
                    f"fcluster returned {got} clusters for requested k={k} "
                    f"(L={L}, frac={frac}, scipy={scipy.__version__}). The ladder "
                    f"is not the registered one; see amendment A1."
                )
            assign = {t: f"G{frac}_{c}" for t, c in zip(src_types, cl)}
        for o in orphans:
            assign[o] = f"orphan_{o}"
        ladder[frac] = assign
        achieved_k[frac] = got
    return ladder, src_types, orphans, achieved_k


def build_random_ladders(src_types, orphans, target_sizes, n_draws=N_RANDOM_DRAWS, seed=SEED):
    """Null B (amendment A3): random coarsenings size-matched to G_0.25.

    target_sizes is the multiset of source-group sizes realised at G_0.25. Each
    draw shuffles which leaf types land in which group, preserving that size
    distribution exactly and destroying centroid structure.
    """
    ladders = []
    for d in range(n_draws):
        rng = np.random.default_rng(seed + 500 + d)
        perm = list(rng.permutation(src_types))
        assign, i = {}, 0
        for g, size in enumerate(target_sizes):
            for t in perm[i:i + size]:
                assign[t] = f"randB_{g}"
            i += size
        for t in perm[i:]:                      # size rounding safety
            assign[t] = f"randB_{len(target_sizes) - 1}"
        for o in orphans:
            assign[o] = f"orphan_{o}"
        ladders.append(assign)
    return ladders


def apply_ladder(labels, assign):
    lab = np.asarray(labels).astype(str)
    return np.array([assign.get(x, f"orphan_{x}") for x in lab], dtype=object)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def inversion_rate(scores_a, scores_b, embeddings):
    """Fraction of embedding pairs whose sign(m_A - m_B) flips between conditions.

    Returns (inverted, total, excluded) where excluded splits by cause so a
    shrinking denominator is attributable (abort condition 3 / amendment A7).
    """
    inverted = total = 0
    exc = {"missing": 0, "nonfinite": 0, "tie": 0}
    for a, b in combinations(embeddings, 2):
        va, vb, wa, wb = scores_a.get(a), scores_a.get(b), scores_b.get(a), scores_b.get(b)
        if any(v is None for v in (va, vb, wa, wb)):
            exc["missing"] += 1
            continue
        if any(not np.isfinite(v) for v in (va, vb, wa, wb)):
            exc["nonfinite"] += 1
            continue
        s1, s2 = np.sign(va - vb), np.sign(wa - wb)
        if s1 == 0 or s2 == 0:
            exc["tie"] += 1
            continue
        total += 1
        inverted += int(s1 != s2)
    return inverted, total, exc


def block_bootstrap_diff(blocks_a, blocks_b, n_boot=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    """CI on (rate_a - rate_b), resampling (tissue, metric) BLOCKS not pairs (A6).

    blocks_* are lists of (inverted, total) per block. Pair-level flags inside a
    block are determined by an ordering of three items and are not independent.
    """
    ba = [b for b in blocks_a if b[1] > 0]
    bb = [b for b in blocks_b if b[1] > 0]
    if not ba or not bb:
        return None, None, 0, 0
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    ia, ib = np.arange(len(ba)), np.arange(len(bb))
    for i in range(n_boot):
        sa = [ba[j] for j in rng.choice(ia, len(ba), replace=True)]
        sb = [bb[j] for j in rng.choice(ib, len(bb), replace=True)]
        ra = sum(x[0] for x in sa) / max(1, sum(x[1] for x in sa))
        rb = sum(x[0] for x in sb) / max(1, sum(x[1] for x in sb))
        diffs[i] = ra - rb
    return (float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))),
            len(ba), len(bb))


def _rate(blocks):
    tot = sum(b[1] for b in blocks)
    return (sum(b[0] for b in blocks) / tot) if tot else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_bio_metrics(X_emb, labels, batch, nn_cache=None):
    """The five label-dependent scIB bio metrics.

    nn_cache carries kNN graphs across label conditions. Verified label-
    independent: nmi_ari consumes only knn_graph_connectivities, clisi only
    distances/indices, silhouette and isolated_labels take X_emb directly.
    """
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
        try:
            nn_cache["nn_15"] = pynndescent(X_emb, n_neighbors=15)
            nn_cache["nn_90"] = pynndescent(X_emb, n_neighbors=90)
        except Exception as e:                       # S8: do not kill the run
            print(f"  {_ts()} WARNING kNN build failed: {e}")
            nn_cache["nn_15"] = nn_cache["nn_90"] = None

    labels = np.asarray(labels).astype(str)
    out = {}

    if nn_cache["nn_15"] is not None:
        try:
            r = nmi_ari_cluster_labels_leiden(nn_cache["nn_15"], labels, optimize_resolution=True)
            out["nmi_leiden"] = float(r["nmi"])
            out["ari_leiden"] = float(r["ari"])
        except Exception as e:
            print(f"  {_ts()} WARNING nmi/ari: {e}")
            out["nmi_leiden"] = out["ari_leiden"] = None
    else:
        out["nmi_leiden"] = out["ari_leiden"] = None

    for name, fn in (
        ("silhouette_label", lambda: float(silhouette_label(X_emb, labels))),
        ("isolated_label_asw", lambda: float(isolated_labels(X_emb, labels, batch))),
        ("clisi", lambda: (float(np.nanmean(clisi_knn(nn_cache["nn_90"], labels)))
                           if nn_cache["nn_90"] is not None else None)),
    ):
        try:
            out[name] = fn()
        except Exception as e:
            print(f"  {_ts()} WARNING {name}: {e}")
            out[name] = None
    return out, nn_cache


# --- null embeddings, bit-identical to exp10 --------------------------------

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

def _load_records(path, require_stamp=None):
    """Read records, tolerating a truncated tail line (S3). Stale stamps dropped (A2)."""
    recs, bad, stale = [], 0, 0
    if not path.exists():
        return recs, bad, stale
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if require_stamp is not None and r.get("script_sha") != require_stamp:
            stale += 1
            continue
        recs.append(r)
    return recs, bad, stale


def run_experiment(out_dir=OUT_DIR, checkpoint_cb=None):
    import cellxgene_census

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = script_sha()
    inc_path = out_dir / f"incremental_{stamp}.jsonl"
    prior, bad, stale = _load_records(inc_path, require_stamp=stamp)
    done = {(r["tissue"], r["embedding"], r["condition"]) for r in prior}
    if done or bad or stale:
        print(f"{_ts()} resume: {len(done)} records reusable, {bad} malformed, {stale} stale-stamp")

    manifest = {
        "script_sha": stamp, "scipy": scipy.__version__, "started": _ts(),
        "tissues": {}, "aborts": [], "excluded_tissues": {},
    }
    man_path = out_dir / "manifest.json"

    def save_manifest():
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        if checkpoint_cb:
            checkpoint_cb()

    inc = open(inc_path, "a")

    def emit(rec):
        rec["script_sha"] = stamp
        inc.write(json.dumps(rec) + "\n")
        inc.flush()
        os.fsync(inc.fileno())
        if checkpoint_cb:
            checkpoint_cb()

    save_manifest()

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
            if len(tgt_ids) > MAX_CELLS:
                idx = np.random.default_rng(SEED + 1).choice(len(tgt_ids), MAX_CELLS, replace=False)
                idx.sort()
                tgt_ids = tgt_ids[idx]

            # Abort condition 1 (amendment A5) - raise, do not warn.
            exp10_src, exp10_tgt = EXP10_CELL_COUNTS[tissue]
            if (len(src_ids), len(tgt_ids)) != (exp10_src, exp10_tgt):
                raise AbortCondition(
                    f"{tissue}: pulled {len(src_ids)}/{len(tgt_ids)} cells, exp10 published "
                    f"{exp10_src}/{exp10_tgt}. Not the same subset; results would not be "
                    f"comparable to the audit. Abort condition 1."
                )

            src = cellxgene_census.get_anndata(
                census, organism=ORGANISM, obs_value_filter=src_filter, obs_coords=src_ids,
                obs_column_names=OBS_COLUMNS, obs_embeddings=REAL_EMBEDDINGS)
            tgt = cellxgene_census.get_anndata(
                census, organism=ORGANISM, obs_value_filter=tgt_filter, obs_coords=tgt_ids,
                obs_column_names=OBS_COLUMNS, obs_embeddings=REAL_EMBEDDINGS)

            labels_src = np.asarray(src.obs["cell_type"].values).astype(str)
            labels_tgt = np.asarray(tgt.obs["cell_type"].values).astype(str)
            all_labels = np.concatenate([labels_src, labels_tgt])
            n_tgt_types = len(set(labels_tgt))
            need = MIN_TARGET_CELLS_PER_TYPE * n_tgt_types

            info = {
                "n_src": int(src.n_obs), "n_tgt": int(tgt.n_obs),
                "n_leaf_types_source": int(len(set(labels_src))),
                "n_leaf_types_target": int(n_tgt_types),
                "min_target_cells_required": int(need),
            }
            manifest["tissues"][tissue] = info
            print(f"  {_ts()} src={src.n_obs} tgt={tgt.n_obs} "
                  f"src_types={info['n_leaf_types_source']} tgt_types={n_tgt_types} need>={need}")

            # Amendment A4 - declared minimum target density.
            if tgt.n_obs < need:
                manifest["excluded_tissues"][tissue] = (
                    f"target {tgt.n_obs} cells < {MIN_TARGET_CELLS_PER_TYPE} x {n_tgt_types} "
                    f"target leaf types = {need} (amendment A4)")
                print(f"  {_ts()} EXCLUDED — {manifest['excluded_tissues'][tissue]}")
                save_manifest()
                continue
            if info["n_leaf_types_source"] < MIN_LEAF_TYPES:
                manifest["excluded_tissues"][tissue] = (
                    f"L={info['n_leaf_types_source']} < {MIN_LEAF_TYPES} (abort condition 2)")
                print(f"  {_ts()} EXCLUDED — abort condition 2")
                save_manifest()
                continue

            X_src_raw = src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X)
            X_tgt_raw = tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X)

            ladder, src_types, orphans, achieved_k = build_granularity_ladder(
                X_src_raw, labels_src, all_labels)
            coarse = ladder[PRIMARY_COARSE]
            sizes = list(pd.Series([coarse[t] for t in src_types]).value_counts().values)
            rand_ladders = build_random_ladders(src_types, orphans, sizes)

            info.update({
                "achieved_k": achieved_k,
                "n_orphan_types": len(orphans),
                "g025_group_sizes": [int(s) for s in sizes],
            })
            print(f"  {_ts()} achieved_k={achieved_k} orphans={len(orphans)} "
                  f"G0.25 sizes={sizes}")
            save_manifest()

            embeddings = {}
            for name in REAL_EMBEDDINGS:
                if name in src.obsm and name in tgt.obsm:
                    embeddings[name] = (np.asarray(src.obsm[name]), np.asarray(tgt.obsm[name]))
                else:
                    print(f"  {_ts()} {name}: unavailable for {tissue}")
            embeddings["random_projection"] = (random_projection(X_src_raw), random_projection(X_tgt_raw))
            embeddings["untrained_encoder"] = (untrained_encoder(X_src_raw), untrained_encoder(X_tgt_raw))
            bs, bt, bog_d = bag_of_genes_pca_combined(X_src_raw, X_tgt_raw)
            embeddings[f"bog_pca_{bog_d}"] = (bs, bt)      # amendment A8

            batch = np.concatenate([np.full(len(labels_src), "source"),
                                    np.full(len(labels_tgt), "target")])
            conds = ([(f"G{f}", ladder[f]) for f in GRANULARITY_FRACTIONS]
                     + [(f"randB_{d}", rl) for d, rl in enumerate(rand_ladders)])

            for emb_name, (Xs, Xt) in tqdm(embeddings.items(), desc=f"  {tissue}", leave=False):
                wanted = {c for c, _ in conds} | {f"floor_{s}" for s in range(N_FLOOR_SUBSAMPLES)}
                if all((tissue, emb_name, c) in done for c in wanted):
                    continue                                # S/B3: skip completed embedding
                X_emb = np.vstack([Xs, Xt]).astype(np.float32)
                nn_cache = None

                for cond, assign in conds:
                    if (tissue, emb_name, cond) in done:
                        continue
                    lab = apply_ladder(all_labels, assign)
                    print(f"{_ts()} {tissue}/{emb_name}/{cond} ({len(set(lab))} classes)")
                    scores, nn_cache = compute_bio_metrics(X_emb, lab, batch, nn_cache)
                    emit({"tissue": tissue, "embedding": emb_name, "condition": cond,
                          "n_classes": int(len(set(lab))), "scores": scores, "timestamp": _ts()})

                lab_leaf = apply_ladder(all_labels, ladder[1.0])
                for s in range(N_FLOOR_SUBSAMPLES):
                    cond = f"floor_{s}"
                    if (tissue, emb_name, cond) in done:
                        continue
                    rng = np.random.default_rng(SEED + 100 + s)
                    keep = np.sort(rng.choice(X_emb.shape[0],
                                              int(FLOOR_FRACTION * X_emb.shape[0]), replace=False))
                    print(f"{_ts()} {tissue}/{emb_name}/{cond}")
                    scores, _ = compute_bio_metrics(X_emb[keep], lab_leaf[keep], batch[keep], None)
                    emit({"tissue": tissue, "embedding": emb_name, "condition": cond,
                          "subsample": s, "scores": scores, "timestamp": _ts()})

    inc.close()
    manifest["finished"] = _ts()
    save_manifest()
    return analyze(out_dir, stamp)


def analyze(out_dir=OUT_DIR, stamp=None):
    out_dir = Path(out_dir)
    if stamp is None:
        cands = sorted(out_dir.glob("incremental_*.jsonl"))
        if not cands:
            raise FileNotFoundError(f"no incremental file in {out_dir}")
        stamp = cands[-1].stem.split("_", 1)[1]
    recs, bad, stale = _load_records(out_dir / f"incremental_{stamp}.jsonl", require_stamp=stamp)
    man = json.loads((out_dir / "manifest.json").read_text()) if (out_dir / "manifest.json").exists() else {}

    df = {(r["tissue"], r["embedding"], r["condition"]): r["scores"] for r in recs}
    tissues = sorted({k[0] for k in df})
    embs = sorted({k[1] for k in df})
    real = [e for e in embs if e in REAL_EMBEDDINGS]

    def s_at(t, c, m):
        return {e: (df.get((t, e, c), {}) or {}).get(m) for e in embs}

    res = {
        "script_sha": stamp, "malformed_lines": bad, "stale_stamp_lines": stale,
        "scipy": man.get("scipy"), "excluded_tissues": man.get("excluded_tissues", {}),
        "tissues_analyzed": tissues, "n_real_embeddings": len(real),
        "per_tissue": {}, "pooled": {}, "rank_corr": {}, "flags": [],
    }

    for tag, pool in (("real_only", real), ("all_six", embs)):
        gran_blocks, floor_blocks, rand_blocks = [], [], []
        exc_tot = {"missing": 0, "nonfinite": 0, "tie": 0}
        per_metric, per_tissue = {}, {}

        for m in BIO_METRICS:
            gb, fb, rb = [], [], []
            for t in tissues:
                i, n, e = inversion_rate(s_at(t, "G1.0", m), s_at(t, f"G{PRIMARY_COARSE}", m), pool)
                gb.append((i, n))
                for k_ in exc_tot:
                    exc_tot[k_] += e[k_]
                fpair = [inversion_rate(s_at(t, f"floor_{a}", m), s_at(t, f"floor_{b}", m), pool)
                         for a, b in combinations(range(N_FLOOR_SUBSAMPLES), 2)]
                fb.append((sum(x[0] for x in fpair), sum(x[1] for x in fpair)))
                rpair = [inversion_rate(s_at(t, "G1.0", m), s_at(t, f"randB_{d}", m), pool)
                         for d in range(N_RANDOM_DRAWS)]
                rb.append((sum(x[0] for x in rpair), sum(x[1] for x in rpair)))
                per_tissue.setdefault(t, {})[m] = {
                    "granularity": {"inv": i, "n": n, "excluded": e},
                    "floor": {"inv": fb[-1][0], "n": fb[-1][1]},
                    "randomB": {"inv": rb[-1][0], "n": rb[-1][1]},
                }
            per_metric[m] = {
                "granularity_rate": _rate(gb), "floor_rate": _rate(fb),
                "randomB_rate": _rate(rb),
                "granularity_pairs": sum(x[1] for x in gb),
            }
            gran_blocks += gb
            floor_blocks += fb
            rand_blocks += rb

        # Support check (S3): P1 is a difference of two rates, so unequal support voids it.
        g_sup = {i for i, b in enumerate(gran_blocks) if b[1] > 0}
        f_sup = {i for i, b in enumerate(floor_blocks) if b[1] > 0}
        support_ok = g_sup == f_sup
        if not support_ok:
            res["flags"].append(
                f"{tag}: granularity and floor arms cover different (tissue,metric) blocks "
                f"({len(g_sup)} vs {len(f_sup)}); pooled difference is not reportable")

        lo, hi, nga, nfb = block_bootstrap_diff(gran_blocks, floor_blocks)
        rlo, rhi, _, nrb = block_bootstrap_diff(gran_blocks, rand_blocks)
        res["pooled"][tag] = {
            "granularity_rate": _rate(gran_blocks),
            "floor_rate": _rate(floor_blocks),
            "randomB_rate": _rate(rand_blocks),
            "P1_diff_ci95_vs_floor": [lo, hi],
            "P5_diff_ci95_vs_randomB": [rlo, rhi],
            "n_blocks": {"granularity": nga, "floor": nfb, "randomB": nrb},
            "n_pairs": {"granularity": sum(b[1] for b in gran_blocks),
                        "floor": sum(b[1] for b in floor_blocks),
                        "randomB": sum(b[1] for b in rand_blocks)},
            "excluded_pairs": exc_tot,
            "support_matched": support_ok,
        }
        res["per_metric_" + tag] = per_metric
        if tag == "real_only":
            res["per_tissue"] = per_tissue

    # Abort condition 4, per tissue (amendment A4)
    for t in tissues:
        blocks = [(v["floor"]["inv"], v["floor"]["n"]) for v in res["per_tissue"].get(t, {}).values()]
        fr = _rate(blocks)
        res.setdefault("per_tissue_floor", {})[t] = fr
        if fr is not None and fr > FLOOR_CEILING:
            res["flags"].append(
                f"{t}: floor {fr:.3f} > {FLOOR_CEILING} — abort condition 4, "
                f"tissue reported as uninformative")

    # Secondary: rank correlation, six-embedding rows only (amendment A7)
    for m in BIO_METRICS:
        rows, skipped = [], 0
        for t in tissues:
            for frac in GRANULARITY_FRACTIONS[1:]:
                a, b = s_at(t, "G1.0", m), s_at(t, f"G{frac}", m)
                ok = [e for e in embs if a.get(e) is not None and b.get(e) is not None
                      and np.isfinite(a[e]) and np.isfinite(b[e])]
                if len(ok) < len(embs):
                    skipped += 1
                    continue
                rho = spearmanr([a[e] for e in ok], [b[e] for e in ok]).statistic
                if np.isfinite(rho):
                    rows.append({"tissue": t, "fraction": frac, "rho": float(rho)})
                else:
                    skipped += 1
        passed = sum(1 for r in rows if r["rho"] >= 0.70)
        res["rank_corr"][m] = {
            "rows": rows, "n_rows": len(rows), "n_skipped_incomplete": skipped,
            "n_pass_070": passed,
            "pass_rate_070": (passed / len(rows)) if rows else None,
        }

    # Intermediate-level inversion rates (amendment A7)
    inter = {}
    for frac in GRANULARITY_FRACTIONS[1:]:
        blocks = []
        for m in BIO_METRICS:
            for t in tissues:
                i, n, _ = inversion_rate(s_at(t, "G1.0", m), s_at(t, f"G{frac}", m), real)
                blocks.append((i, n))
        inter[f"G1.0_vs_G{frac}"] = _rate(blocks)
    res["intermediate_inversion_real_only"] = inter

    with open(out_dir / "exp18_results.json", "w") as f:
        json.dump(res, f, indent=2)

    p = res["pooled"]["real_only"]
    print("\n" + "=" * 64)
    print(f"tissues analyzed : {tissues}")
    print(f"excluded         : {res['excluded_tissues']}")
    print(f"POOLED (real embeddings only, n={len(real)}, blocks={p['n_blocks']})")
    print(f"  granularity : {p['granularity_rate']}")
    print(f"  noise floor : {p['floor_rate']}   P1 diff CI {p['P1_diff_ci95_vs_floor']}")
    print(f"  null B      : {p['randomB_rate']}   P5 diff CI {p['P5_diff_ci95_vs_randomB']}")
    print(f"  excluded    : {p['excluded_pairs']}   support_matched={p['support_matched']}")
    for f_ in res["flags"]:
        print(f"  FLAG: {f_}")
    return res


if __name__ == "__main__":
    run_experiment()
