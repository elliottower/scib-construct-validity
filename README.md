# scib-validity

Construct validity diagnostics for single-cell embedding evaluation metrics. CKA null saturation bounds and source classifier confidence (SCC).

[![PyPI](https://img.shields.io/pypi/v/scib-validity)](https://pypi.org/project/scib-validity/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Open Quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elliottower/scib-construct-validity/blob/main/notebooks/quickstart.ipynb)

## Install

```bash
pip install scib-validity
```

## Quick start

```python
from scib_validity import scc, cka_null, cka_certifiable

# Is CKA informative at your dimensionality?
print(cka_null(k=15, d=512))        # 0.973 -- null floor
print(cka_certifiable(0.95, 15, 512))  # False -- below random

# SCC: does the source classifier transfer?
score = scc(X_source, X_target, labels_source, labels_target)
print(f"SCC = {score:.3f}")
```

## The problem

CKA on cell-type centroids exhibits **null saturation**: its analytic expectation under a Gaussian null,

$$\mathbb{E}[\text{CKA}] \approx 1 - 1.06 \cdot \frac{k-1}{d+k}$$

exceeds every trained-model CKA score at foundation-model dimensionality (d >= 512). At d = 50, 86% of conditions clear the null; at d >= 512, none do.

Five relational consistency score variants avoid null saturation yet predict cross-technology transfer only marginally (rho <= 0.25).

## The alternative

Source classifier confidence (SCC) trains a cell-type classifier on source embeddings and reports the mean maximum predicted probability on target cells. It directly measures whether learned decision boundaries transfer.

| Classifier | Key | Spearman rho with transfer F1 |
|-----------|-----|------|
| Logistic regression | `"logreg"` | 0.67 |
| k-nearest neighbors | `"knn"` | 0.56 |
| Random forest | `"rf"` | 0.57 |
| Support vector machine | `"svm"` | 0.55 |

All four pass Benjamini-Hochberg correction across 23 human tissues and 92 embedding conditions in CELLxGENE Census.

## API

### `scc(X_source, X_target, labels_source, labels_target=None, shared_types=None, classifier="logreg", seed=0)`

Returns the mean maximum predicted probability of a source-trained classifier applied to target cells.

### `scc_multi(X_source, X_target, labels_source, ...)`

Returns a dict mapping classifier name to SCC score across all four families.

### `cka_null(k, d)`

Returns the analytic expected CKA between independent Gaussian centroid matrices.

### `cka_certifiable(observed_cka, k, d, margin=0.0)`

Returns True if the observed CKA exceeds the null floor.

## Cross-classifier stress test

```python
from scib_validity import scc_multi

scores = scc_multi(X_source, X_target, labels_source, labels_target)
# {'logreg': 0.87, 'knn': 0.82, 'rf': 0.84, 'svm': 0.85}
```

Running SCC with multiple classifier families rules out shared-machinery confounding. If all classifiers agree, the signal is an embedding-space property.

## Empirical CKA null

Verify the closed-form bound against Monte Carlo simulation:

```python
from scib_validity.metrics.cka_null import cka_null_empirical

empirical = cka_null_empirical(k=15, d=512, n_trials=5000)
print(f"Empirical mean: {empirical['mean']:.4f}")
print(f"Analytic bound: {cka_null(15, 512):.4f}")
```

## Citation

```bibtex
@software{tower2026scib_validity,
  author = {Tower, Elliot},
  title = {Construct Validity Failure in Single-Cell Embedding Evaluation},
  year = {2026},
  url = {https://github.com/elliottower/scib-construct-validity},
  license = {MIT}
}
```

## License

MIT
