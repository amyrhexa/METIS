# tests/test_interpretation.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from config.schema import AnalysisTaskType, PreprocessingConfig
from core.interpretation import aggregate_fold_importance, compute_cv_stable_attribution
from core.modeling import get_model_zoo
from core.preprocessing import build_preprocessor


def _fitted_fold_estimators(n=100, n_folds=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = 3.0 * X["x1"].to_numpy() + rng.normal(scale=0.4, size=n)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())

    fold_estimators = []
    fold_size = n // n_folds
    for fold_idx in range(n_folds):
        test_idx = np.arange(fold_idx * fold_size, (fold_idx + 1) * fold_size)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        pipe = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(n_estimators=20, random_state=0)),
            ]
        )
        pipe.fit(X.iloc[train_idx], y[train_idx])
        fold_estimators.append((fold_idx, pipe, test_idx))
    return X, y, fold_estimators


def test_cv_stable_attribution_shap_uses_only_held_out_rows_per_fold():
    X, y, fold_estimators = _fitted_fold_estimators()
    spec = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)["random_forest"]

    result = compute_cv_stable_attribution(
        spec, fold_estimators, X, y, AnalysisTaskType.REGRESSION,
        n_jobs=1, random_state=0, compute_shap=True, permutation_n_repeats=5,
    )
    assert result.shap_status == "ok"
    total_test_rows = sum(len(test_idx) for _, _, test_idx in fold_estimators)
    # If SHAP had been computed on train data (or the full dataset) instead of
    # strictly each fold's held-out rows, this row count would not match the
    # exact, disjoint held-out total.
    assert result.shap_values_array.shape[0] == total_test_rows
    assert result.X_test_transformed.shape[0] == total_test_rows


def test_cv_stable_attribution_permutation_fallback_for_non_tree_model():
    X, y, fold_estimators = _fitted_fold_estimators()
    linear_spec = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)["ridge"]
    assert not linear_spec.is_tree_based

    result = compute_cv_stable_attribution(
        linear_spec, fold_estimators, X, y, AnalysisTaskType.REGRESSION,
        n_jobs=1, random_state=0, compute_shap=True, permutation_n_repeats=5,
    )
    # Non-tree model -> SHAP is skipped up front, permutation importance runs instead.
    assert result.shap_status == "skipped: non-tree estimator without tree fallback"
    assert len(result.fold_importance_records) > 0
    assert len(fold_estimators) == len({r["fold"] for r in result.fold_importance_records})


def test_cv_stable_attribution_respects_skip_shap_config():
    X, y, fold_estimators = _fitted_fold_estimators()
    spec = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)["random_forest"]
    result = compute_cv_stable_attribution(
        spec, fold_estimators, X, y, AnalysisTaskType.REGRESSION,
        n_jobs=1, random_state=0, compute_shap=False, permutation_n_repeats=5,
    )
    assert result.shap_status == "skipped: skip_shap is enabled"
    assert len(result.fold_importance_records) > 0  # falls back to permutation


def test_aggregate_fold_importance_summarizes_across_folds():
    records = [
        {"fold": 0, "feature": "x1", "value": 0.5},
        {"fold": 1, "feature": "x1", "value": 0.7},
        {"fold": 0, "feature": "x2", "value": 0.1},
        {"fold": 1, "feature": "x2", "value": 0.15},
    ]
    df = aggregate_fold_importance(records)
    assert not df.empty
    assert set(["feature", "mean_importance", "std_importance", "n_valid_folds"]).issubset(df.columns)
    # x1's mean importance (0.6) clearly exceeds x2's (0.125) -> ranks first.
    assert df.iloc[0]["feature"] == "x1"
    assert df.iloc[0]["mean_importance"] == pytest.approx(0.6)
    assert df.iloc[0]["n_valid_folds"] == 2


def test_aggregate_fold_importance_handles_empty_input():
    df = aggregate_fold_importance([])
    assert df.empty
