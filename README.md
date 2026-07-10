# Preflight Bio

Transportability diagnostics for foundation models in biology.

Given source and target embeddings (or expression matrices), Preflight runs geometric diagnostic modules that predict whether a model will generalize — without retraining.

## Install

```bash
pip install preflight-bio
```

For CELLxGENE Census data (requires Python <3.13):

```bash
pip install "preflight-bio[census]"
```

## Quick start

### Python API

```python
import numpy as np
from preflight import run

source = np.random.randn(500, 512)  # (n_cells, embedding_dim)
target = np.random.randn(500, 512)

report = run(source, target)
print(f"Tier {report.overall_tier}/7 — {report.overall_score:.3f}")

for name, result in report.module_results.items():
    print(f"  {name}: Tier {result.tier} ({result.score:.3f})")
```

### CLI

```bash
# From .npy files
preflight run --source embeddings_a.npy --target embeddings_b.npy

# From .h5ad files with hosted embeddings
preflight run --source a.h5ad --target b.h5ad --obsm-key X_geneformer

# With preregistration (SHA-freezes the scorer before the run)
preflight run --source a.npy --target b.npy --preregister --output report.json

# Select specific modules
preflight run --source a.npy --target b.npy --modules m1_grassmannian m4_embedding_collapse

# Verify a preregistration
preflight verify prereg.json
```

### CELLxGENE Census

Pull hosted foundation model embeddings directly from Census:

```python
from preflight.extractors.cellxgene_loader import query_census
from preflight import run

X_source, labels_s, meta_s = query_census(
    census_version="2023-12-15",
    obs_value_filter="tissue_ontology_term_id == 'UBERON:0002048' "
                     "and assay_ontology_term_id == 'EFO:0009922'",
    embedding="geneformer",
    obs_label_key="cell_type",
    max_cells=2000,
)

X_target, labels_t, meta_t = query_census(
    census_version="2023-12-15",
    obs_value_filter="tissue_ontology_term_id == 'UBERON:0002048' "
                     "and assay_ontology_term_id == 'EFO:0011025'",
    embedding="geneformer",
    obs_label_key="cell_type",
    max_cells=2000,
)

report = run(X_source, X_target)
print(report.to_markdown())
```

## Diagnostic modules

| Module | What it measures | Inputs |
|--------|-----------------|--------|
| **M1 Grassmannian** | Subspace alignment via geodesic distance on Gr(k, d) | embeddings |
| **M2 Domain shift** | Domain classifier AUC detecting distribution shift | embeddings |
| **M3 Direction stability** | Stability of learned directions across conditions | embeddings |
| **M4 Domain validity** | Data completeness and missingness | embeddings |
| **M5 Cross-design** | Effect estimate discordance across study designs | metadata |
| **M6 Curvature** | Ollivier-Ricci curvature on embedding graphs | metadata |
| **M7 Ecological bias** | Site-level vs individual-level effect distortion | metadata |

Modules M1--M4 run by default on any pair of embedding matrices. Modules M5--M7 run when their metadata inputs are provided:

- **M5**: `metadata["estimates"]` — dict of scalar estimates per study design
- **M6**: `metadata["graph"]` — a `networkx.Graph` (e.g., k-NN graph on embeddings)
- **M7**: `metadata["records"]` — list of dicts with `site_key` and `outcome_key` fields

## Probing suite

Alongside geometric diagnostics, Preflight includes a multi-model probing suite for supervised evaluation of embedding quality:

```python
from preflight.probes import run_probes

results = run_probes(
    embeddings={"geneformer": X_gf, "scvi": X_scvi},
    labels=cell_types,       # cell type annotations
    batch_ids=donor_ids,     # donor IDs for stratified CV
    k=5,                     # number of CV folds
)
print(results["summary"])
```

| Probe | What it measures | Metric | Level |
|-------|-----------------|--------|-------|
| **cell_type** | Cell type classification | macro F1 | cell embeddings |
| **tf_target** | Per-TF target vs non-target | mean AUC | gene embeddings |
| **hub_tf** | Hub TF vs non-hub TF | AUC | gene embeddings |
| **rsa** | Embedding similarity vs regulatory adjacency | Spearman rho | gene embeddings |

Gene-level probes activate when `gene_embeddings`, `gene_ids`, and `regulatory_edges` are provided. Regulatory edges can be pulled from the OmniPath REST API:

```python
from preflight.probes import load_regulatory_edges
edges = load_regulatory_edges()  # DoRothEA + CollecTRI
```

Multi-model comparison uses paired Wilcoxon signed-rank tests with bootstrap 95% CIs on fold-level scores.

## Tiering

Reports produce a 1–7 tier score:

| Tier | Label | Score range |
|------|-------|-------------|
| 7 | Excellent | ≥ 0.85 |
| 6 | Good | 0.70 – 0.85 |
| 5 | Acceptable | 0.55 – 0.70 |
| 4 | Marginal | 0.40 – 0.55 |
| 3 | Poor | 0.25 – 0.40 |
| 2 | Very Poor | 0.10 – 0.25 |
| 1 | Failure | < 0.10 |

## Preregistration

Preflight supports SHA-256 preregistration: the scorer's source code, hyperparameters, and dataset specification are hashed before any data is accessed. The same SHA appears in the report, allowing third parties to verify the analysis was not modified after seeing the results.

```python
from preflight import run
from preflight.core.preregister import DatasetSpec

spec = DatasetSpec(
    name="lung_assay_shift",
    source_description="Lung 10x 3' v3 (Geneformer)",
    target_description="Lung 10x 5' v2 (Geneformer)",
    n_source=2000,
    n_target=2000,
    extra={"census_version": "2023-12-15", "embedding": "geneformer"},
)

report = run(
    source, target,
    preregister=True,
    preregister_path="prereg.json",
    dataset_spec=spec,
)

# Verify
from preflight.core.preregister import load_preregistration, verify_preregistration
from preflight.core.runner import ALL_MODULE_SOURCES, resolve_hyperparameters

record = load_preregistration("prereg.json")
hp, weights = resolve_hyperparameters({"k": 5})
match, msg = verify_preregistration(record, ALL_MODULE_SOURCES, {**hp, "weights": weights}, spec)
assert match
```

## Example result

Lung tissue, Geneformer embeddings, assay shift (10x 3' v3 → 10x 5' v2), 2000 cells per condition:

```
Overall Tier: 3/7 (Poor) — Score: 0.457

  m1_grassmannian:        Tier 3 (0.309) — geodesic distance 2.43
  m3_direction_instability: Tier 7 (1.000) — directions stable
  m4_embedding_collapse:  Tier 1 (0.062) — domain AUC 0.969
  m5_domain_validity:     Tier 7 (1.000) — data complete

SHA verified: aaac31a11c21e017...
```

The assay shift produces a strong domain-shift signal (M4 domain AUC = 0.97) while direction stability remains excellent (M3) — the framework detects the shift you'd expect and doesn't false-alarm on the stable axis.

## Development

```bash
git clone https://github.com/your-org/preflight-bio.git
cd preflight-bio
pip install -e ".[dev]"
pytest tests/
```

## License

CC-BY-4.0
