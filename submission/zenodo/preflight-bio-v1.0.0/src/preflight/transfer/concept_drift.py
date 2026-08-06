"""#2: Concept drift detection — does P(Y|X) change across domains?

Covariate shift: P(X) changes, P(Y|X) stays. Correctable by reweighting.
Concept drift: P(Y|X) changes. No feature correction will help.

Detection approach: train on source, predict on target (if labels available),
check if residuals are random (covariate shift) or structured (concept drift).
"""

import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, roc_auc_score


@dataclass
class ConceptDriftResult:
    has_concept_drift: bool
    drift_severity: float
    residual_predictability: float
    details: dict

    def summary(self) -> str:
        status = "CONCEPT DRIFT DETECTED" if self.has_concept_drift else "No concept drift"
        return (
            f"{status}\n"
            f"  Drift severity: {self.drift_severity:.3f}\n"
            f"  Residual predictability: {self.residual_predictability:.3f}\n"
            f"  (>0.1 = residuals are structured = P(Y|X) differs)"
        )


def detect_concept_drift(
    source_X: np.ndarray,
    source_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    task: str = "regression",
) -> ConceptDriftResult:
    """Test whether P(Y|X) differs between source and target.

    Method: train on source, predict on target, then check if the
    residuals (target_y - predicted_y) are predictable from target_X.
    If residuals are random, the model transfers (pure covariate shift).
    If residuals are structured, P(Y|X) has changed (concept drift).

    Args:
        source_X, source_y: labeled source data.
        target_X, target_y: labeled target data.
        task: "regression" or "classification".
    """
    if task == "regression":
        model = Ridge(alpha=1.0)
        model.fit(source_X, source_y)
        target_pred = model.predict(target_X)
        residuals = target_y - target_pred

        # Can we predict the residuals from features?
        if len(target_y) >= 20:
            cv = KFold(n_splits=min(5, len(target_y)), shuffle=True, random_state=42)
            residual_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42)
            oof_residuals = cross_val_predict(residual_model, target_X, residuals, cv=cv)
            predictability = max(0.0, r2_score(residuals, oof_residuals))
        else:
            residual_model = Ridge(alpha=10.0)
            residual_model.fit(target_X, residuals)
            pred_res = residual_model.predict(target_X)
            predictability = max(0.0, float(np.corrcoef(residuals, pred_res)[0, 1] ** 2))

        raw_transfer_r2 = r2_score(target_y, target_pred)
        source_cv = cross_val_predict(Ridge(alpha=1.0), source_X, source_y,
                                       cv=KFold(n_splits=5, shuffle=True, random_state=42))
        source_r2 = r2_score(source_y, source_cv)
        performance_drop = max(0, source_r2 - raw_transfer_r2)

    else:
        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(source_X, source_y)
        target_proba = model.predict_proba(target_X)[:, 1]
        residuals = target_y.astype(float) - target_proba

        if len(target_y) >= 20:
            cv = KFold(n_splits=min(5, len(target_y)), shuffle=True, random_state=42)
            residual_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=42)
            oof_residuals = cross_val_predict(residual_model, target_X, residuals, cv=cv)
            predictability = max(0.0, r2_score(residuals, oof_residuals))
        else:
            predictability = 0.0

        performance_drop = 0.0

    # Concept drift if residuals are predictable
    has_drift = predictability > 0.1
    severity = min(1.0, predictability * 2)

    return ConceptDriftResult(
        has_concept_drift=has_drift,
        drift_severity=severity,
        residual_predictability=predictability,
        details={
            "task": task,
            "n_source": len(source_y),
            "n_target": len(target_y),
            "performance_drop": performance_drop,
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
        },
    )
