from __future__ import annotations

"""
Utilities for training and logging fraud detection models with MLflow.

The design goal is to keep these functions:
- **Config-aware** via `common.config.get_config` (for default MLflow tracking URI)
- **Test-friendly** by allowing in-memory DataFrames and a file-based tracking URI
- **Extensible** so later phases can swap in real Feast-powered datasets
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import optuna
import shap
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

from common.config import get_config


FeatureNames = List[str]


@dataclass
class TrainingResult:
    model: xgb.XGBClassifier
    feature_names: FeatureNames
    best_params: Dict[str, object]
    roc_auc: float
    accuracy: float
    run_id: str


def build_fraud_training_dataframe(
    user_metrics: pd.DataFrame,
    *,
    fraud_threshold: float = 0.05,
    label_column: str = "is_fraud_label",
) -> pd.DataFrame:
    """
    Construct a simple binary classification dataset from user-level metrics.

    This treats `fraud_risk_score` as a proxy for the ground-truth label by
    thresholding it into a boolean column.
    """
    if "fraud_risk_score" not in user_metrics.columns:
        raise ValueError("user_metrics must contain a 'fraud_risk_score' column")

    df = user_metrics.copy()
    df[label_column] = (df["fraud_risk_score"] >= fraud_threshold).astype(int)

    # Keep a compact feature set that is stable across phases.
    keep_cols: List[str] = []
    for col in ("lifetime_purchases", "fraud_risk_score"):
        if col in df.columns:
            keep_cols.append(col)
    keep_cols.append(label_column)

    return df[keep_cols]


def _resolve_feature_columns(
    df: pd.DataFrame, target_column: str, feature_columns: Optional[Sequence[str]]
) -> FeatureNames:
    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Requested feature columns missing from DataFrame: {missing}")
        return list(feature_columns)

    return [c for c in df.columns if c != target_column]


def _set_tracking_uri(tracking_uri: Optional[str]) -> str:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        return tracking_uri

    cfg = get_config()
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    return cfg.mlflow.tracking_uri


def _log_shap_artifacts(
    model: xgb.XGBClassifier,
    X_sample: np.ndarray,
    feature_names: FeatureNames,
) -> None:
    """
    Compute SHAP values for a small sample and log them as MLflow artifacts.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        shap_dir = Path(tmpdir) / "shap"
        shap_dir.mkdir(parents=True, exist_ok=True)

        np.save(shap_dir / "shap_values.npy", shap_values)
        np.save(shap_dir / "sample_X.npy", X_sample)
        (shap_dir / "feature_names.txt").write_text("\n".join(feature_names), encoding="utf-8")

        mlflow.log_artifacts(str(shap_dir), artifact_path="shap")


def train_fraud_model(
    df: pd.DataFrame,
    target_column: str,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "fraud_detection",
    n_trials: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingResult:
    """
    Train an XGBoost fraud classifier with Optuna hyperparameter tuning and
    log everything to MLflow, including a SHAP explainer.

    The caller is responsible for constructing `df` (e.g. via Feast or
    `build_fraud_training_dataframe`). This function does not talk to Feast
    directly and is therefore easy to test.
    """
    feature_names = _resolve_feature_columns(df, target_column, feature_columns)

    X = df[feature_names].to_numpy()
    y = df[target_column].to_numpy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    _set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 20, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = xgb.XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            tree_method="hist",
            random_state=random_state,
        )

        with mlflow.start_run(run_name="fraud_trial", nested=True):
            mlflow.log_params(params)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, y_pred_proba)
            acc = accuracy_score(y_valid, (y_pred_proba >= 0.5).astype(int))
            mlflow.log_metrics({"roc_auc": float(auc), "accuracy": float(acc)})
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_trials)

    best_params = study.best_params
    best_model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist",
        random_state=random_state,
    )

    best_model.fit(X_train, y_train)

    # Evaluate on validation split.
    y_pred_proba = best_model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred_proba)
    acc = accuracy_score(y_valid, (y_pred_proba >= 0.5).astype(int))

    # Use a small sample for SHAP to keep tests fast and artifacts compact.
    sample_size = min(128, X_train.shape[0])
    X_sample = X_train[:sample_size]

    with mlflow.start_run(run_name="fraud_best") as run:
        mlflow.log_params(best_params)
        mlflow.log_metrics({"roc_auc": float(auc), "accuracy": float(acc)})

        # SHAP artifacts + model
        _log_shap_artifacts(best_model, X_sample, feature_names)
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        run_id = run.info.run_id

    return TrainingResult(
        model=best_model,
        feature_names=feature_names,
        best_params=best_params,
        roc_auc=float(auc),
        accuracy=float(acc),
        run_id=run_id,
    )


__all__ = [
    "TrainingResult",
    "build_fraud_training_dataframe",
    "train_fraud_model",
]

