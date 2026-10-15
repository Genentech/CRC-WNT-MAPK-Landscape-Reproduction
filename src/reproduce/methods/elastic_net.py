"""Elastic-net regression with cross-validated, out-of-fold performance.

`elastic_net_regression` fits one target; `elastic_net_regression_multiple_targets` fits
several targets on the same predictors (optionally in parallel) and returns a per-target
table of cross-validated R^2 (regression) or AUC (classification) with coefficients.
"""

from __future__ import annotations

from typing import Dict, Any, Union, Tuple, Optional
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.base import clone
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _oof_regression_r2(
    model,
    X,
    y,
    *,
    random_state: int = 42,
    cv_splitter=None,
    n_splits: int = 5,
) -> float:
    """Compute true out-of-fold R^2 via ``cross_val_predict``.

    The estimator is cloned for each fold so the already-fitted ``model``
    instance passed by the caller is not mutated. ``cv_splitter`` lets the
    caller pass a pre-built ``KFold`` instance (useful for multi-target
    runs that need consistent folds across targets); when omitted a fresh
    ``KFold(n_splits, shuffle=True, random_state=random_state)`` is built.
    """
    if cv_splitter is None:
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    estimator = clone(model)
    y_oof = cross_val_predict(estimator, X, y, cv=cv_splitter)
    return float(r2_score(y, y_oof))


def _oof_classification_auc(
    model,
    X,
    y,
    *,
    random_state: int = 42,
    cv_splitter=None,
    n_splits: int = 5,
) -> float:
    """Compute true out-of-fold ROC-AUC via ``cross_val_predict``.

    Uses ``StratifiedKFold`` by default (binary AUC needs both classes in
    every fold). ``cv_splitter`` may be supplied to override.
    """
    if cv_splitter is None:
        cv_splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    estimator = clone(model)
    y_proba = cross_val_predict(
        estimator, X, y, cv=cv_splitter, method="predict_proba"
    )[:, 1]
    return float(roc_auc_score(y, y_proba))


def _warn_if_leakage(in_sample: float, oof: float, *, label: str = "") -> None:
    """Log a warning if OOF score is suspiciously higher than in-sample score.

    A well-behaved model should have ``in_sample >= oof``. If ``oof`` exceeds
    ``in_sample`` by more than a small tolerance it suggests the OOF predictions
    were contaminated by test-fold statistics (data leakage).
    """
    if np.isfinite(in_sample) and np.isfinite(oof) and oof > in_sample + 0.01:
        logger.warning(
            "OOF score (%.4f) exceeds in-sample score (%.4f) by >0.01%s — "
            "possible data leakage in the preprocessing pipeline.",
            oof,
            in_sample,
            f" for {label}" if label else "",
        )


def elastic_net_regression(
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    task: str = "regression",
    cv: int = 5,
    random_state: int = 42,
    l1_ratio: float = 0.5,
    alphas: Optional[np.ndarray] = None,
    scale_features: bool = True,
    return_model: bool = True,
    compute_oof: bool = True,
) -> Dict[str, Any]:
    """
    Perform Elastic Net regression (combination of LASSO and Ridge).

    Elastic Net combines L1 (LASSO) and L2 (Ridge) penalties, providing
    a balance between feature selection and handling correlated features.
    The l1_ratio parameter controls the mixing: 0 = pure Ridge, 1 = pure LASSO.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features)
    y : pd.Series or np.ndarray
        Target variable (binary for classification, continuous for regression)
    task : str, default 'regression'
        Type of task: 'regression' or 'classification'
    cv : int, default 5
        Number of cross-validation folds for hyperparameter selection
    random_state : int, default 42
        Random state for reproducibility
    l1_ratio : float, default 0.5
        Mixing parameter between L1 and L2 penalties (0 = Ridge, 1 = LASSO)
    alphas : np.ndarray, optional
        Array of alpha values to test. If None, uses default sklearn range.
    scale_features : bool, default True
        Whether to standardize features before fitting
    return_model : bool, default True
        Whether to return the fitted model object

    Returns
    -------
    dict
        Results dictionary containing:
        - model: Fitted model object (if return_model=True)
        - coefficients: pd.Series of coefficients (sparse due to L1 component)
        - alpha_optimal: Selected regularization parameter
        - cv_score: Out-of-fold R² (regression) or ROC-AUC (classification)
          via ``cross_val_predict``. Sensitivity metric.
        - in_sample_score: In-sample R² or ROC-AUC. Descriptive metric.
        - l1_ratio: L1/L2 mixing parameter used
        - n_features_selected: Number of non-zero coefficients
        - feature_names: List of features with non-zero coefficients
        - performance_metrics: Additional performance metrics
    """
    # Input validation
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")

    if task not in ["regression", "classification"]:
        raise ValueError("task must be 'regression' or 'classification'")

    if not 0 <= l1_ratio <= 1:
        raise ValueError("l1_ratio must be between 0 and 1")

    # Align X and y
    if isinstance(y, pd.Series):
        common_index = X.index.intersection(y.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index]
    else:
        X_aligned = X.copy()
        y_aligned = np.asarray(y)

    # Remove any remaining missing values
    if X_aligned.isna().any().any() or pd.Series(y_aligned).isna().any():
        combined_data = pd.concat(
            [X_aligned, pd.Series(y_aligned, index=X_aligned.index)], axis=1
        )
        combined_clean = combined_data.dropna()
        X_clean = combined_clean.iloc[:, :-1]
        y_clean = combined_clean.iloc[:, -1].values
    else:
        X_clean = X_aligned
        y_clean = y_aligned

    if len(X_clean) == 0:
        raise ValueError("No valid samples after removing missing values")

    # Set up base model
    if task == "regression":
        if alphas is None:
            base_model = ElasticNetCV(
                cv=cv, random_state=random_state, l1_ratio=l1_ratio, max_iter=10000
            )
        else:
            base_model = ElasticNetCV(
                cv=cv,
                random_state=random_state,
                l1_ratio=l1_ratio,
                max_iter=10000,
                alphas=alphas,
            )
    else:  # classification
        # For classification, use LogisticRegressionCV with elasticnet penalty
        if alphas is None:
            base_model = LogisticRegressionCV(
                cv=cv,
                random_state=random_state,
                penalty="elasticnet",
                solver="saga",  # saga solver supports elasticnet
                l1_ratios=[l1_ratio],  # Fixed l1_ratio
                max_iter=10000,
            )
        else:
            # Convert alphas to C values
            Cs = 1.0 / alphas
            base_model = LogisticRegressionCV(
                cv=cv,
                random_state=random_state,
                penalty="elasticnet",
                solver="saga",
                l1_ratios=[l1_ratio],
                max_iter=10000,
                Cs=Cs,
            )

    # Build pipeline so that scaler is fitted per fold (no leakage).
    # When scale_features=False the pipeline contains only the model.
    if scale_features:
        pipeline = make_pipeline(StandardScaler(), base_model)
    else:
        pipeline = make_pipeline(base_model)

    # Fit pipeline on full data (used for in-sample metrics and coefficient extraction)
    try:
        pipeline.fit(X_clean, y_clean)
    except Exception as e:
        raise RuntimeError(f"Model fitting failed: {str(e)}")

    # Retrieve the fitted model step for coefficient / alpha extraction
    model = pipeline.steps[-1][1]

    # Retain scaler for backward-compatible 'scaler' key in results
    scaler = pipeline.named_steps.get("standardscaler", None)

    # Extract results
    if task == "regression":
        coefficients = model.coef_
        alpha_optimal = model.alpha_
        in_sample_r2 = float(r2_score(y_clean, pipeline.predict(X_clean)))
        if compute_oof:
            cv_score = _oof_regression_r2(
                pipeline, X_clean, y_clean, random_state=random_state, n_splits=cv
            )
            _warn_if_leakage(in_sample_r2, cv_score, label="elastic_net_regression")
        else:
            cv_score = np.nan

        # Calculate additional metrics
        y_pred = pipeline.predict(X_clean)
        mse = mean_squared_error(y_clean, y_pred)
        r2 = r2_score(y_clean, y_pred)

        performance_metrics = {"r2_score": r2, "mse": mse, "rmse": np.sqrt(mse)}
        in_sample_score = in_sample_r2

    else:  # classification
        coefficients = model.coef_[0]  # Extract from 2D array
        alpha_optimal = 1.0 / model.C_[0]  # Convert back to alpha

        # True out-of-fold AUC (replaces in-sample roc_auc_score on fitted X)
        auc_computed = False
        in_sample_auc = np.nan
        try:
            in_sample_auc = float(
                roc_auc_score(y_clean, pipeline.predict_proba(X_clean)[:, 1])
            )
            auc_computed = True
            if compute_oof:
                cv_score = _oof_classification_auc(
                    pipeline, X_clean, y_clean, random_state=random_state, n_splits=cv
                )
            else:
                cv_score = np.nan
            _warn_if_leakage(in_sample_auc, cv_score, label="elastic_net_regression")
        except Exception:
            cv_score = pipeline.score(X_clean, y_clean)  # Fallback to accuracy

        # Calculate additional metrics
        y_pred = pipeline.predict(X_clean)
        accuracy = (y_pred == y_clean).mean()

        performance_metrics = {
            "accuracy": accuracy,
            "auc": in_sample_auc if auc_computed else np.nan,
        }
        in_sample_score = in_sample_auc

    # Create coefficient series
    coef_series = pd.Series(coefficients, index=X_clean.columns)

    # Filter to non-zero coefficients for feature selection info
    non_zero_coefs = coef_series[coef_series != 0]

    # Prepare results
    results = {
        "coefficients": coef_series,  # All coefficients (Ridge doesn't zero out)
        "coefficients_nonzero": non_zero_coefs,  # For compatibility with LASSO
        "alpha_optimal": alpha_optimal,
        "cv_score": cv_score,
        "in_sample_score": in_sample_score,
        "l1_ratio": l1_ratio,
        "n_features_selected": len(non_zero_coefs),
        "n_features_total": len(coef_series),
        "feature_names": list(coef_series.index),
        "feature_names_selected": list(non_zero_coefs.index),
        "performance_metrics": performance_metrics,
        "task": task,
        "n_samples": len(X_clean),
    }

    if return_model:
        results["model"] = model

    if scaler is not None:
        results["scaler"] = scaler

    return results


def elastic_net_regression_multiple_targets(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    task: str = "regression",
    cv: int = 5,
    random_state: int = 42,
    l1_ratio: float = 0.5,
    alphas: Optional[np.ndarray] = None,
    scale_features: bool = True,
    show_progress: bool = True,
    compute_oof: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Perform Elastic Net regression for multiple target variables.

    This function applies Elastic Net regression to each target variable independently
    and aggregates the results. Elastic Net combines LASSO and Ridge penalties,
    providing both feature selection and coefficient shrinkage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (samples x features)
    Y : pd.DataFrame
        Target matrix (samples x targets)
    task : str, default 'regression'
        Type of task: 'regression' or 'classification'
    cv : int, default 5
        Number of cross-validation folds
    random_state : int, default 42
        Random state for reproducibility
    l1_ratio : float, default 0.5
        Mixing parameter between L1 and L2 penalties (0 = Ridge, 1 = LASSO)
    alphas : np.ndarray, optional
        Array of alpha values to test
    scale_features : bool, default True
        Whether to standardize features
    show_progress : bool, default True
        Whether to show progress bar

    Returns
    -------
    tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]
        - results_dict: Dictionary with results for each target
        - metrics_df: DataFrame with performance metrics for each target
        - coefficients_df: DataFrame with coefficients (features x targets)
    """
    # Input validation
    if not isinstance(X, pd.DataFrame) or not isinstance(Y, pd.DataFrame):
        raise TypeError("X and Y must be pandas DataFrames")

    # Align X and Y by index
    common_index = X.index.intersection(Y.index)
    X_aligned = X.loc[common_index]
    Y_aligned = Y.loc[common_index]

    if len(common_index) == 0:
        raise ValueError("No common samples between X and Y")

    # Initialize results containers
    results_dict = {}
    metrics_list = []
    coefficients_dict = {}

    # Set up progress bar
    targets = Y_aligned.columns
    if show_progress:
        targets = tqdm(targets, desc="Elastic Net regression", leave=False)

    # Fit model for each target
    for target in targets:
        y_target = Y_aligned[target]

        try:
            # Run single-target Elastic Net
            result = elastic_net_regression(
                X_aligned,
                y_target,
                task=task,
                cv=cv,
                random_state=random_state,
                l1_ratio=l1_ratio,
                compute_oof=compute_oof,
                alphas=alphas,
                scale_features=scale_features,
                return_model=False,  # Don't store models to save memory
            )

            # Store results
            results_dict[target] = result

            # Extract metrics
            metrics_row = {
                "target": target,
                "cv_score": result["cv_score"],
                "in_sample_score": result.get("in_sample_score", np.nan),
                "alpha_optimal": result["alpha_optimal"],
                "l1_ratio": result["l1_ratio"],
                "n_features_selected": result["n_features_selected"],
                "n_features_total": result["n_features_total"],
                "n_samples": result["n_samples"],
            }
            metrics_row.update(result["performance_metrics"])
            metrics_list.append(metrics_row)

            # Store coefficients (use non-zero coefficients for sparsity)
            coefficients_dict[target] = result["coefficients_nonzero"]

        except Exception as e:
            # Handle individual target failures
            warning_msg = f"Elastic Net regression failed for target {target}: {str(e)}"
            warnings.warn(warning_msg, RuntimeWarning)

            # Store NaN results
            results_dict[target] = {
                "coefficients": pd.Series(dtype=float),
                "coefficients_nonzero": pd.Series(dtype=float),
                "alpha_optimal": np.nan,
                "cv_score": np.nan,
                "l1_ratio": l1_ratio,
                "n_features_selected": 0,
                "n_features_total": len(X_aligned.columns),
                "error": str(e),
            }

            metrics_row = {
                "target": target,
                "cv_score": np.nan,
                "alpha_optimal": np.nan,
                "l1_ratio": l1_ratio,
                "n_features_selected": 0,
                "n_features_total": len(X_aligned.columns),
                "n_samples": len(X_aligned),
                "error": str(e),
            }
            metrics_list.append(metrics_row)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Create coefficients DataFrame (sparse like LASSO due to L1 component)
    if coefficients_dict:
        # Get all unique features that were selected
        all_features = set()
        for coefs in coefficients_dict.values():
            all_features.update(coefs.index)

        # Create coefficient matrix
        coef_matrix = pd.DataFrame(
            index=sorted(all_features), columns=Y_aligned.columns, dtype=float
        )

        for target, coefs in coefficients_dict.items():
            coef_matrix.loc[coefs.index, target] = coefs.values

        # Fill missing values with 0 (features not selected)
        coef_matrix = coef_matrix.fillna(0)
    else:
        coef_matrix = pd.DataFrame()

    return results_dict, metrics_df, coef_matrix


__all__ = [
    "elastic_net_regression",
    "elastic_net_regression_multiple_targets",
]
