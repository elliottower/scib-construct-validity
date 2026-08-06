"""Source Classifier Confidence (SCC) for embedding transfer evaluation.

SCC trains a cell-type classifier on source-domain embeddings and reports
the mean maximum predicted probability on target-domain cells.  Higher
SCC indicates that the source decision boundaries transfer cleanly to
the target embedding space.

Reference:
    Tower (2026). Construct Validity Failure in Single-Cell Embedding
    Evaluation: Null Saturation Bounds and Source Classifier Confidence.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


_CLASSIFIERS = {
    "logreg": lambda seed: LogisticRegression(max_iter=1000, random_state=seed),
    "knn": lambda seed: KNeighborsClassifier(n_neighbors=15),
    "rf": lambda seed: RandomForestClassifier(n_estimators=100, random_state=seed),
    "svm": lambda seed: SVC(kernel="linear", probability=True, max_iter=2000, random_state=seed),
}


def scc(
    X_source: np.ndarray,
    X_target: np.ndarray,
    labels_source: np.ndarray,
    labels_target: np.ndarray | None = None,
    shared_types: list | np.ndarray | None = None,
    classifier: str = "logreg",
    seed: int = 0,
) -> float:
    """Compute source classifier confidence.

    Args:
        X_source: Source embeddings, shape (n_source, d).
        X_target: Target embeddings, shape (n_target, d).
        labels_source: Cell-type labels for source cells.
        labels_target: Cell-type labels for target cells.  Only used to
            restrict target cells to shared types.  If None, all target
            cells are scored.
        shared_types: Cell types present in both domains.  If None,
            inferred from ``labels_source`` (and ``labels_target`` if
            provided).
        classifier: One of "logreg", "knn", "rf", "svm".
        seed: Random seed for the classifier.

    Returns:
        Mean maximum predicted probability on (shared-type) target cells.
    """
    if classifier not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier {classifier!r}. Choose from {list(_CLASSIFIERS)}")

    if shared_types is None:
        if labels_target is not None:
            shared_types = np.intersect1d(np.unique(labels_source), np.unique(labels_target))
        else:
            shared_types = np.unique(labels_source)
    shared_types = np.asarray(shared_types)

    mask_src = np.isin(labels_source, shared_types)
    le = LabelEncoder()
    le.fit(shared_types)

    clf = _CLASSIFIERS[classifier](seed)
    clf.fit(X_source[mask_src], le.transform(labels_source[mask_src]))

    if labels_target is not None:
        mask_tgt = np.isin(labels_target, shared_types)
        X_score = X_target[mask_tgt]
    else:
        X_score = X_target

    probs = clf.predict_proba(X_score)
    return float(np.mean(np.max(probs, axis=1)))


def scc_multi(
    X_source: np.ndarray,
    X_target: np.ndarray,
    labels_source: np.ndarray,
    labels_target: np.ndarray | None = None,
    shared_types: list | np.ndarray | None = None,
    classifiers: list[str] | None = None,
    seed: int = 0,
) -> dict[str, float]:
    """Compute SCC with multiple classifier families.

    Args:
        X_source: Source embeddings, shape (n_source, d).
        X_target: Target embeddings, shape (n_target, d).
        labels_source: Cell-type labels for source cells.
        labels_target: Cell-type labels for target cells (optional).
        shared_types: Cell types present in both domains (optional).
        classifiers: List of classifier names.  Defaults to all four:
            ["logreg", "knn", "rf", "svm"].
        seed: Random seed for classifiers that use one.

    Returns:
        Dict mapping classifier name to SCC score.
    """
    if classifiers is None:
        classifiers = list(_CLASSIFIERS)
    return {
        name: scc(
            X_source, X_target, labels_source, labels_target,
            shared_types=shared_types, classifier=name, seed=seed,
        )
        for name in classifiers
    }
