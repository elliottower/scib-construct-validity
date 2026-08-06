"""Synthetic data generators for validating Preflight modules.

Six levels of synthetic data:
  1. Embedding-level: Gaussian arrays with planted subspace geometry.
     Directly tests Modules 1, 3, 4 (geodesic, DI, collapse).
  2. Gene-expression-level: Synthetic count matrices.
     Tests the full pipeline including extractors.
  3. Missingness-level: Embeddings with planted MCAR/MAR/MNAR patterns.
     Tests Module 5 domain-of-validity robustness.
  4. Network-level: nx.Graph with planted hub/bridge topology.
     Tests Module 6 ORC curvature.
  5. Ecological-level: Tabular records with site-level aggregation bias.
     Tests Module 7 Mundlak decomposition and E-value sensitivity.
  6. Cross-design-level: Scalar estimates with planted discordance.
     Tests Module 2 sheaf Q-test.

All generators produce known ground truth so assertions can verify
the modules recover the planted values.
"""
import math

import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from preflight.core.preregister import DatasetSpec


@dataclass
class SyntheticGroundTruth:
    """Known ground truth for a synthetic dataset pair."""
    planted_geodesic: float
    should_transfer: bool
    rotation_angle: float
    confound_strength: float
    class_separation: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SyntheticPair:
    """A source-target pair with known ground truth."""
    source: np.ndarray
    target: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray
    source_batch: np.ndarray
    target_batch: np.ndarray
    ground_truth: SyntheticGroundTruth
    dataset_spec: DatasetSpec


def _random_rotation(d: int, angle: float, rng: np.random.Generator, n_planes: int = 5) -> np.ndarray:
    """Generate a rotation matrix in R^d that rotates by `angle` radians
    in multiple random 2D planes for stronger subspace disruption."""
    R = np.eye(d)
    if abs(angle) < 1e-12:
        return R
    for _ in range(n_planes):
        v1 = rng.standard_normal(d)
        v1 /= np.linalg.norm(v1)
        v2 = rng.standard_normal(d)
        v2 -= v2 @ v1 * v1
        v2 /= np.linalg.norm(v2)
        c, s = np.cos(angle), np.sin(angle)
        Ri = np.eye(d) + (c - 1) * np.outer(v1, v1) + (c - 1) * np.outer(v2, v2)
        Ri += s * (np.outer(v1, v2) - np.outer(v2, v1))
        R = Ri @ R
    return R


# ======================================================================
# Level 1: Embedding-level synthetic data
# ======================================================================

def generate_embedding_pair(
    n_source: int = 200,
    n_target: int = 200,
    d: int = 100,
    k: int = 10,
    rotation_angle: float = 0.1,
    class_separation: float = 2.0,
    confound_strength: float = 0.0,
    noise_scale: float = 0.5,
    seed: int = 42,
) -> SyntheticPair:
    """Generate a source-target embedding pair with planted subspace geometry.

    The source data lives in a k-dimensional subspace of R^d with a class
    signal along the first principal direction. The target data lives in a
    rotated version of this subspace. The rotation angle controls the
    geodesic distance between the two subspaces.

    Args:
        n_source: number of source samples
        n_target: number of target samples
        d: embedding dimension
        k: subspace dimension (for PCA to recover)
        rotation_angle: angle of subspace rotation (radians).
            0 = identical subspaces, pi/2 = orthogonal.
        class_separation: distance between class centroids in the
            discriminant direction (higher = easier classification)
        confound_strength: magnitude of batch-correlated shift
            (0 = no confound, >0 = batch effect)
        noise_scale: scale of isotropic noise added to all samples
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    basis = np.linalg.qr(rng.standard_normal((d, k)))[0]

    source_labels = np.array([0] * (n_source // 2) + [1] * (n_source - n_source // 2))
    target_labels = np.array([0] * (n_target // 2) + [1] * (n_target - n_target // 2))

    coeffs_source = rng.standard_normal((n_source, k))
    signal_dir = np.zeros(k)
    signal_dir[0] = 1.0
    for i in range(n_source):
        coeffs_source[i] += class_separation * signal_dir * (source_labels[i] - 0.5)

    source = coeffs_source @ basis.T + noise_scale * rng.standard_normal((n_source, d))

    R = _random_rotation(d, rotation_angle, rng)
    basis_target = R @ basis
    basis_target = np.linalg.qr(basis_target)[0]

    coeffs_target = rng.standard_normal((n_target, k))
    for i in range(n_target):
        coeffs_target[i] += class_separation * signal_dir * (target_labels[i] - 0.5)

    target = coeffs_target @ basis_target.T + noise_scale * rng.standard_normal((n_target, d))

    if confound_strength > 0:
        confound_dir = rng.standard_normal(d)
        confound_dir /= np.linalg.norm(confound_dir)
        target += confound_strength * confound_dir

    source_batch = np.zeros(n_source)
    target_batch = np.ones(n_target)

    planted_geodesic = rotation_angle * np.sqrt(min(k, 2))

    ground_truth = SyntheticGroundTruth(
        planted_geodesic=planted_geodesic,
        should_transfer=rotation_angle < 0.3,
        rotation_angle=rotation_angle,
        confound_strength=confound_strength,
        class_separation=class_separation,
    )

    spec = DatasetSpec(
        name=f"synthetic_embedding_seed{seed}",
        source_description=f"Gaussian in R^{d}, k={k} subspace, class_sep={class_separation}",
        target_description=f"Rotated by {rotation_angle:.3f} rad, confound={confound_strength}",
        n_source=n_source,
        n_target=n_target,
        d=d,
        extra={"seed": seed, "k": k, "rotation_angle": rotation_angle},
    )

    return SyntheticPair(
        source=source,
        target=target,
        source_labels=source_labels,
        target_labels=target_labels,
        source_batch=source_batch,
        target_batch=target_batch,
        ground_truth=ground_truth,
        dataset_spec=spec,
    )


def generate_transfer_positive(seed: int = 100, **kwargs) -> SyntheticPair:
    """A pair that should transfer: small rotation, no confound."""
    defaults = dict(
        rotation_angle=0.05,
        class_separation=3.0,
        confound_strength=0.0,
        noise_scale=0.3,
    )
    defaults.update(kwargs)
    return generate_embedding_pair(seed=seed, **defaults)


def generate_transfer_negative(seed: int = 200, **kwargs) -> SyntheticPair:
    """A pair that should NOT transfer: large rotation + confound."""
    defaults = dict(
        rotation_angle=1.2,
        class_separation=2.0,
        confound_strength=5.0,
        noise_scale=0.5,
    )
    defaults.update(kwargs)
    return generate_embedding_pair(seed=seed, **defaults)


# ======================================================================
# Level 2: Gene-expression-level synthetic data
# ======================================================================

def generate_expression_pair(
    n_source: int = 300,
    n_target: int = 300,
    n_genes: int = 2000,
    n_programs: int = 20,
    rotation_angle: float = 0.1,
    class_separation: float = 1.5,
    confound_strength: float = 0.0,
    dropout_rate: float = 0.3,
    seed: int = 42,
) -> SyntheticPair:
    """Generate synthetic gene-expression count matrices.

    Simulates two "tissues" (source/target) with:
    - Shared gene programs (latent structure)
    - Sparse, non-negative counts (like real scRNA-seq)
    - Planted class signal (cell type)
    - Optional batch effect (confound)

    Returns SyntheticPair with integer count matrices.
    """
    rng = np.random.default_rng(seed)

    programs = np.abs(rng.standard_normal((n_programs, n_genes)))
    programs /= programs.sum(axis=1, keepdims=True)

    source_labels = np.array([0] * (n_source // 2) + [1] * (n_source - n_source // 2))
    target_labels = np.array([0] * (n_target // 2) + [1] * (n_target - n_target // 2))

    source_activations = np.abs(rng.standard_normal((n_source, n_programs)))
    signal_program = np.zeros(n_programs)
    signal_program[0] = 1.0
    for i in range(n_source):
        source_activations[i] += class_separation * signal_program * (source_labels[i] - 0.5)
    source_activations = np.maximum(source_activations, 0)

    R = _random_rotation(n_programs, rotation_angle, rng)
    programs_target = np.abs(R @ programs)
    programs_target /= programs_target.sum(axis=1, keepdims=True)

    target_activations = np.abs(rng.standard_normal((n_target, n_programs)))
    for i in range(n_target):
        target_activations[i] += class_separation * signal_program * (target_labels[i] - 0.5)
    target_activations = np.maximum(target_activations, 0)

    source_rates = source_activations @ programs * 1000
    target_rates = target_activations @ programs_target * 1000

    if confound_strength > 0:
        batch_genes = rng.choice(n_genes, size=n_genes // 10, replace=False)
        target_rates[:, batch_genes] *= (1 + confound_strength)

    source_counts = rng.poisson(np.maximum(source_rates, 0.01))
    target_counts = rng.poisson(np.maximum(target_rates, 0.01))

    dropout_mask_s = rng.random((n_source, n_genes)) > dropout_rate
    dropout_mask_t = rng.random((n_target, n_genes)) > dropout_rate
    source_counts = source_counts * dropout_mask_s
    target_counts = target_counts * dropout_mask_t

    source_batch = np.zeros(n_source)
    target_batch = np.ones(n_target)

    planted_geodesic = rotation_angle * np.sqrt(min(n_programs, 2))

    ground_truth = SyntheticGroundTruth(
        planted_geodesic=planted_geodesic,
        should_transfer=rotation_angle < 0.3,
        rotation_angle=rotation_angle,
        confound_strength=confound_strength,
        class_separation=class_separation,
        extra={"n_programs": n_programs, "dropout_rate": dropout_rate},
    )

    spec = DatasetSpec(
        name=f"synthetic_expression_seed{seed}",
        source_description=f"Poisson counts, {n_genes} genes, {n_programs} programs",
        target_description=f"Rotated programs by {rotation_angle:.3f} rad, dropout={dropout_rate}",
        n_source=n_source,
        n_target=n_target,
        d=n_genes,
        extra={"seed": seed, "n_programs": n_programs, "rotation_angle": rotation_angle},
    )

    return SyntheticPair(
        source=source_counts.astype(np.float64),
        target=target_counts.astype(np.float64),
        source_labels=source_labels,
        target_labels=target_labels,
        source_batch=source_batch,
        target_batch=target_batch,
        ground_truth=ground_truth,
        dataset_spec=spec,
    )


def generate_expression_transfer_positive(seed: int = 300, **kwargs) -> SyntheticPair:
    """Expression pair that should transfer: small rotation, no confound."""
    defaults = dict(
        rotation_angle=0.05,
        class_separation=2.0,
        confound_strength=0.0,
        dropout_rate=0.2,
    )
    defaults.update(kwargs)
    return generate_expression_pair(seed=seed, **defaults)


def generate_expression_transfer_negative(seed: int = 400, **kwargs) -> SyntheticPair:
    """Expression pair that should NOT transfer: large rotation + batch effect."""
    defaults = dict(
        rotation_angle=1.0,
        class_separation=1.5,
        confound_strength=3.0,
        dropout_rate=0.4,
    )
    defaults.update(kwargs)
    return generate_expression_pair(seed=seed, **defaults)


# ======================================================================
# Level 3: Missingness-level synthetic data (Module 5)
# ======================================================================

@dataclass
class SyntheticMissingnessPair:
    """Embedding pair with planted missingness patterns."""
    source: np.ndarray
    target: np.ndarray
    source_complete: np.ndarray
    target_complete: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray
    ground_truth: SyntheticGroundTruth
    dataset_spec: DatasetSpec


def generate_missingness_pair(
    n_source: int = 200,
    n_target: int = 200,
    d: int = 50,
    k: int = 5,
    missingness_type: str = "MCAR",
    missingness_rate: float = 0.2,
    class_separation: float = 2.0,
    noise_scale: float = 0.3,
    seed: int = 42,
) -> SyntheticMissingnessPair:
    """Generate an embedding pair with planted missingness.

    Args:
        missingness_type: "MCAR" (completely at random), "MAR" (depends
            on observed features), or "MNAR" (depends on missing values).
        missingness_rate: fraction of entries set to NaN.
    """
    rng = np.random.default_rng(seed)

    pair = generate_embedding_pair(
        n_source=n_source, n_target=n_target, d=d, k=k,
        rotation_angle=0.1, class_separation=class_separation,
        noise_scale=noise_scale, seed=seed,
    )
    source_complete = pair.source.copy()
    target_complete = pair.target.copy()
    source = pair.source.copy()
    target = pair.target.copy()

    if missingness_type == "MCAR":
        mask_s = rng.random((n_source, d)) < missingness_rate
        mask_t = rng.random((n_target, d)) < missingness_rate
    elif missingness_type == "MAR":
        prob_s = np.zeros((n_source, d))
        for j in range(d):
            ref_col = (j + 1) % d
            quantile = np.percentile(source[:, ref_col], 75)
            prob_s[:, j] = np.where(
                source[:, ref_col] > quantile,
                missingness_rate * 3,
                missingness_rate * 0.5,
            )
        prob_t = np.zeros((n_target, d))
        for j in range(d):
            ref_col = (j + 1) % d
            quantile = np.percentile(target[:, ref_col], 75)
            prob_t[:, j] = np.where(
                target[:, ref_col] > quantile,
                missingness_rate * 3,
                missingness_rate * 0.5,
            )
        mask_s = rng.random((n_source, d)) < prob_s
        mask_t = rng.random((n_target, d)) < prob_t
    elif missingness_type == "MNAR":
        prob_s = np.zeros((n_source, d))
        for j in range(d):
            quantile = np.percentile(source[:, j], 80)
            prob_s[:, j] = np.where(
                source[:, j] > quantile,
                missingness_rate * 4,
                missingness_rate * 0.25,
            )
        prob_t = np.zeros((n_target, d))
        for j in range(d):
            quantile = np.percentile(target[:, j], 80)
            prob_t[:, j] = np.where(
                target[:, j] > quantile,
                missingness_rate * 4,
                missingness_rate * 0.25,
            )
        mask_s = rng.random((n_source, d)) < prob_s
        mask_t = rng.random((n_target, d)) < prob_t
    else:
        raise ValueError(f"Unknown missingness_type: {missingness_type}")

    source[mask_s] = np.nan
    target[mask_t] = np.nan

    actual_rate_s = mask_s.mean()
    actual_rate_t = mask_t.mean()

    ground_truth = SyntheticGroundTruth(
        planted_geodesic=0.1 * np.sqrt(min(k, 2)),
        should_transfer=True,
        rotation_angle=0.1,
        confound_strength=0.0,
        class_separation=class_separation,
        extra={
            "missingness_type": missingness_type,
            "missingness_rate": missingness_rate,
            "actual_rate_source": float(actual_rate_s),
            "actual_rate_target": float(actual_rate_t),
        },
    )

    spec = DatasetSpec(
        name=f"synthetic_missingness_{missingness_type}_seed{seed}",
        source_description=f"Gaussian R^{d}, {missingness_type} missing @ {missingness_rate:.0%}",
        target_description=f"Same with independent missingness pattern",
        n_source=n_source,
        n_target=n_target,
        d=d,
        extra={"seed": seed, "missingness_type": missingness_type},
    )

    return SyntheticMissingnessPair(
        source=source,
        target=target,
        source_complete=source_complete,
        target_complete=target_complete,
        source_labels=pair.source_labels,
        target_labels=pair.target_labels,
        ground_truth=ground_truth,
        dataset_spec=spec,
    )


def generate_missingness_mcar(seed: int = 500, **kwargs) -> SyntheticMissingnessPair:
    """MCAR missingness: random entries missing, no bias."""
    defaults = dict(missingness_type="MCAR", missingness_rate=0.2)
    defaults.update(kwargs)
    return generate_missingness_pair(seed=seed, **defaults)


def generate_missingness_mnar(seed: int = 550, **kwargs) -> SyntheticMissingnessPair:
    """MNAR missingness: high values preferentially missing, biased."""
    defaults = dict(missingness_type="MNAR", missingness_rate=0.2)
    defaults.update(kwargs)
    return generate_missingness_pair(seed=seed, **defaults)


# ======================================================================
# Level 4: Network-level synthetic data (Module 6)
# ======================================================================

@dataclass
class SyntheticNetworkData:
    """Network with planted hub/bridge topology for ORC testing."""
    graph: nx.Graph
    hub_nodes: list[int]
    bridge_edges: list[tuple[int, int]]
    ground_truth: SyntheticGroundTruth
    dataset_spec: DatasetSpec


def generate_network_pair(
    n_clusters: int = 4,
    cluster_size: int = 10,
    p_within: float = 0.6,
    p_between: float = 0.05,
    hub_extra_edges: int = 5,
    weight_range: tuple[float, float] = (0.5, 2.0),
    seed: int = 42,
) -> SyntheticNetworkData:
    """Generate a network with planted hub/bridge structure.

    Creates a stochastic block model with dense within-cluster connectivity
    and sparse between-cluster bridges. Hub nodes get extra within-cluster
    edges. Bridge edges connect clusters and should have negative ORC
    (bottleneck), while hub neighborhoods should have positive ORC
    (redundant paths).

    Args:
        n_clusters: number of clusters
        cluster_size: nodes per cluster
        p_within: edge probability within clusters
        p_between: edge probability between clusters
        hub_extra_edges: extra edges added to each cluster's hub node
        weight_range: (min, max) for random edge weights
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    n_total = n_clusters * cluster_size

    G = nx.Graph()
    G.add_nodes_from(range(n_total))

    hub_nodes = []
    bridge_edges = []

    for c in range(n_clusters):
        start = c * cluster_size
        end = start + cluster_size
        nodes = list(range(start, end))
        hub_node = start
        hub_nodes.append(hub_node)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if rng.random() < p_within:
                    w = rng.uniform(*weight_range)
                    G.add_edge(nodes[i], nodes[j], weight=w)

        for target_node in rng.choice(
            [n for n in nodes if n != hub_node],
            size=min(hub_extra_edges, len(nodes) - 1),
            replace=False,
        ):
            if not G.has_edge(hub_node, target_node):
                w = rng.uniform(*weight_range)
                G.add_edge(hub_node, target_node, weight=w)

    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            start1, start2 = c1 * cluster_size, c2 * cluster_size
            nodes1 = list(range(start1, start1 + cluster_size))
            nodes2 = list(range(start2, start2 + cluster_size))
            for n1 in nodes1:
                for n2 in nodes2:
                    if rng.random() < p_between:
                        w = rng.uniform(*weight_range)
                        G.add_edge(n1, n2, weight=w)
                        bridge_edges.append((n1, n2))

    if not bridge_edges:
        for c1 in range(n_clusters):
            c2 = (c1 + 1) % n_clusters
            n1 = c1 * cluster_size + rng.integers(cluster_size)
            n2 = c2 * cluster_size + rng.integers(cluster_size)
            w = rng.uniform(*weight_range)
            G.add_edge(n1, n2, weight=w)
            bridge_edges.append((n1, n2))

    mean_hub_degree = np.mean([G.degree(h) for h in hub_nodes])
    mean_degree = np.mean([G.degree(n) for n in G.nodes()])

    ground_truth = SyntheticGroundTruth(
        planted_geodesic=0.0,
        should_transfer=True,
        rotation_angle=0.0,
        confound_strength=0.0,
        class_separation=0.0,
        extra={
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "p_within": p_within,
            "p_between": p_between,
            "n_hub_nodes": len(hub_nodes),
            "n_bridge_edges": len(bridge_edges),
            "mean_hub_degree": float(mean_hub_degree),
            "mean_degree": float(mean_degree),
        },
    )

    spec = DatasetSpec(
        name=f"synthetic_network_seed{seed}",
        source_description=f"SBM: {n_clusters} clusters x {cluster_size} nodes",
        target_description=f"p_within={p_within}, p_between={p_between}",
        n_source=n_total,
        n_target=0,
        d=0,
        extra={"seed": seed, "n_clusters": n_clusters},
    )

    return SyntheticNetworkData(
        graph=G,
        hub_nodes=hub_nodes,
        bridge_edges=bridge_edges,
        ground_truth=ground_truth,
        dataset_spec=spec,
    )


def generate_network_clustered(seed: int = 600, **kwargs) -> SyntheticNetworkData:
    """Well-separated clusters: clear hub/bridge distinction."""
    defaults = dict(
        n_clusters=4, cluster_size=12, p_within=0.7,
        p_between=0.03, hub_extra_edges=6,
    )
    defaults.update(kwargs)
    return generate_network_pair(seed=seed, **defaults)


def generate_network_dense(seed: int = 650, **kwargs) -> SyntheticNetworkData:
    """Dense between-cluster connections: blurred hub/bridge distinction."""
    defaults = dict(
        n_clusters=3, cluster_size=10, p_within=0.5,
        p_between=0.3, hub_extra_edges=3,
    )
    defaults.update(kwargs)
    return generate_network_pair(seed=seed, **defaults)


# ======================================================================
# Level 5: Ecological-level synthetic data (Module 7)
# ======================================================================

@dataclass
class SyntheticEcologicalData:
    """Tabular data with planted ecological (aggregation) bias."""
    records: list[dict]
    site_means: dict[str, float]
    ground_truth: SyntheticGroundTruth
    dataset_spec: DatasetSpec


def generate_ecological_data(
    n_sites: int = 20,
    n_per_site: int = 200,
    within_effect: float = 0.5,
    between_effect: float = 2.0,
    baseline_rate: float = 0.1,
    seed: int = 42,
) -> SyntheticEcologicalData:
    """Generate tabular data with planted ecological bias.

    Creates individual-level records grouped by site where the
    within-site and between-site effects of a binary covariate differ.
    The difference (between - within) is the ecological bias that
    Mundlak decomposition should recover.

    The data model:
      logit(P(y=1)) = baseline + within_effect * (x_i - x_bar_site)
                                + between_effect * x_bar_site

    When within_effect != between_effect, aggregation bias exists.

    Args:
        n_sites: number of sites/groups
        n_per_site: individuals per site
        within_effect: log-OR for within-site x effect
        between_effect: log-OR for between-site mean(x) effect
        baseline_rate: base probability of outcome when x=0
        seed: random seed
    """
    rng = np.random.default_rng(seed)

    site_elderly_rates = rng.beta(2, 5, size=n_sites)

    baseline_logit = math.log(baseline_rate / (1 - baseline_rate))

    records = []
    for s in range(n_sites):
        site_name = f"site_{s:02d}"
        rate = site_elderly_rates[s]

        x_vals = rng.binomial(1, rate, size=n_per_site)
        x_bar = x_vals.mean()

        for i in range(n_per_site):
            x_within = x_vals[i] - x_bar
            logit_p = baseline_logit + within_effect * x_within + between_effect * x_bar
            p = 1 / (1 + math.exp(-logit_p))
            y = int(rng.random() < p)
            records.append({
                "site": site_name,
                "elderly": int(x_vals[i]),
                "died": y,
                "age": int(rng.normal(50 + 25 * x_vals[i], 10)),
            })

    site_means = {}
    for s in range(n_sites):
        site_name = f"site_{s:02d}"
        site_records = [r for r in records if r["site"] == site_name]
        site_means[site_name] = np.mean([r["elderly"] for r in site_records])

    ecological_bias = between_effect - within_effect

    ground_truth = SyntheticGroundTruth(
        planted_geodesic=0.0,
        should_transfer=abs(ecological_bias) < 0.5,
        rotation_angle=0.0,
        confound_strength=abs(ecological_bias),
        class_separation=0.0,
        extra={
            "within_effect": within_effect,
            "between_effect": between_effect,
            "ecological_bias": ecological_bias,
            "n_sites": n_sites,
            "n_per_site": n_per_site,
            "baseline_rate": baseline_rate,
            "within_or": math.exp(within_effect),
            "between_or": math.exp(between_effect),
        },
    )

    spec = DatasetSpec(
        name=f"synthetic_ecological_seed{seed}",
        source_description=f"{n_sites} sites x {n_per_site} individuals",
        target_description=f"within={within_effect:.2f}, between={between_effect:.2f}",
        n_source=n_sites * n_per_site,
        n_target=0,
        d=0,
        extra={"seed": seed, "n_sites": n_sites, "ecological_bias": ecological_bias},
    )

    return SyntheticEcologicalData(
        records=records,
        site_means=site_means,
        ground_truth=ground_truth,
        dataset_spec=spec,
    )


def generate_ecological_biased(seed: int = 700, **kwargs) -> SyntheticEcologicalData:
    """Large ecological bias: between-site effect >> within-site effect."""
    defaults = dict(
        within_effect=0.3, between_effect=2.5,
        n_sites=25, n_per_site=300,
    )
    defaults.update(kwargs)
    return generate_ecological_data(seed=seed, **defaults)


def generate_ecological_unbiased(seed: int = 750, **kwargs) -> SyntheticEcologicalData:
    """No ecological bias: within and between effects are equal."""
    defaults = dict(
        within_effect=1.0, between_effect=1.0,
        n_sites=25, n_per_site=300,
    )
    defaults.update(kwargs)
    return generate_ecological_data(seed=seed, **defaults)


# ======================================================================
# Level 6: Cross-design-level synthetic data (Module 2)
# ======================================================================

@dataclass
class SyntheticEstimates:
    """Scalar estimates with planted discordance for sheaf Q-test."""
    estimates: dict[str, dict[str, float]]
    ground_truth: SyntheticGroundTruth
    dataset_spec: DatasetSpec


def generate_multidesign_estimates(
    n_designs: int = 5,
    true_effect: float = 0.5,
    se_range: tuple[float, float] = (0.05, 0.2),
    discordance_fraction: float = 0.0,
    bias_magnitude: float = 1.0,
    seed: int = 42,
) -> SyntheticEstimates:
    """Generate scalar estimates from multiple study designs.

    Creates beta/se pairs as if from different study designs (RCT, cohort,
    case-control, MR, etc.) estimating the same causal effect. When
    discordance_fraction > 0, some designs have a systematic bias,
    causing the sheaf Q-test to reject consistency.

    Args:
        n_designs: number of study designs
        true_effect: true causal effect (log-OR scale)
        se_range: (min, max) standard errors across designs
        discordance_fraction: fraction of designs that are biased (0-1)
        bias_magnitude: how far biased designs deviate from truth
        seed: random seed
    """
    rng = np.random.default_rng(seed)

    design_names = [
        "RCT", "prospective_cohort", "case_control",
        "MR_primary", "MR_replication", "cross_sectional",
        "nested_case_control", "ecological",
    ][:n_designs]

    n_discordant = max(0, int(round(discordance_fraction * n_designs)))
    discordant_idx = set(rng.choice(n_designs, size=n_discordant, replace=False))
    discordant_designs = [design_names[i] for i in discordant_idx]

    estimates = {}
    for i, name in enumerate(design_names):
        se = rng.uniform(*se_range)
        if i in discordant_idx:
            sign = 1 if rng.random() > 0.5 else -1
            beta = true_effect + sign * bias_magnitude + rng.normal(0, se)
        else:
            beta = true_effect + rng.normal(0, se)
        estimates[name] = {"beta": float(beta), "se": float(se)}

    should_be_consistent = discordance_fraction == 0.0

    ground_truth = SyntheticGroundTruth(
        planted_geodesic=0.0,
        should_transfer=should_be_consistent,
        rotation_angle=0.0,
        confound_strength=discordance_fraction,
        class_separation=0.0,
        extra={
            "true_effect": true_effect,
            "n_designs": n_designs,
            "discordance_fraction": discordance_fraction,
            "bias_magnitude": bias_magnitude,
            "discordant_designs": discordant_designs,
            "should_be_consistent": should_be_consistent,
        },
    )

    spec = DatasetSpec(
        name=f"synthetic_multidesign_seed{seed}",
        source_description=f"{n_designs} designs, true_effect={true_effect:.2f}",
        target_description=f"discordance={discordance_fraction:.0%}, bias={bias_magnitude:.1f}",
        n_source=n_designs,
        n_target=0,
        d=0,
        extra={"seed": seed, "n_designs": n_designs, "true_effect": true_effect},
    )

    return SyntheticEstimates(
        estimates=estimates,
        ground_truth=ground_truth,
        dataset_spec=spec,
    )


def generate_concordant_estimates(seed: int = 803, **kwargs) -> SyntheticEstimates:
    """All designs agree: sheaf Q-test should NOT reject."""
    defaults = dict(
        n_designs=5, true_effect=0.5,
        se_range=(0.2, 0.5),
        discordance_fraction=0.0, bias_magnitude=0.0,
    )
    defaults.update(kwargs)
    return generate_multidesign_estimates(seed=seed, **defaults)


def generate_discordant_estimates(seed: int = 850, **kwargs) -> SyntheticEstimates:
    """40% of designs are biased: sheaf Q-test SHOULD reject."""
    defaults = dict(
        n_designs=5, true_effect=0.5,
        discordance_fraction=0.4, bias_magnitude=2.0,
    )
    defaults.update(kwargs)
    return generate_multidesign_estimates(seed=seed, **defaults)
