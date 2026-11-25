# tests/test_modeling.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from config.schema import AnalysisTaskType, PreprocessingConfig
from core.modeling import (
    ModelSpec,
    compute_corrected_resampled_ttest,
    evaluate_model_nested_cv,
    extract_linear_model_coefficients,
    get_model_zoo,
)
from core.preprocessing import build_preprocessor


def _regression_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = 2.0 * X["x1"].to_numpy() - 1.5 * X["x2"].to_numpy() + rng.normal(
        scale=0.5, size=n
    )
    return X, y


def test_get_model_zoo_returns_valid_specs_for_both_task_types():
    for task in (AnalysisTaskType.REGRESSION, AnalysisTaskType.CLASSIFICATION):
        zoo = get_model_zoo(task, random_state=0, include_full=True)
        assert len(zoo) > 0
        for key, spec in zoo.items():
            assert isinstance(spec, ModelSpec)
            assert spec.task_type == task
            assert spec.name == key
            assert spec.estimator is not None


def test_nested_cv_oof_predictions_cover_every_sample_exactly_once():
    X, y = _regression_data()
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    spec = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)["ridge"]

    res = evaluate_model_nested_cv(
        spec, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=5, inner_folds=3, n_iter=3, n_jobs=1, random_state=0,
        compute_learning_curve=False,
    )
    # Every outer fold must claim a disjoint set of test rows, and together
    # they must cover the whole dataset exactly once (no leakage-through-
    # duplication, no samples silently skipped).
    assert res.execution_status == "SUCCESS"
    assert np.all(res.fold_assignment >= 0)  # no sample left unassigned
    assert set(res.fold_assignment.tolist()) == set(range(5))  # all 5 folds used
    assert not np.isnan(res.oof_predictions).any()
    # A ridge model on this strongly linear signal should clearly beat chance.
    from core.evaluation import calculate_comprehensive_metrics
    m = calculate_comprehensive_metrics(y, res.oof_predictions, AnalysisTaskType.REGRESSION)
    assert m["r2"] > 0.5


def test_nested_cv_preprocessing_statistics_are_fit_only_on_the_training_fold():
    """Directly proves no leakage: for every outer fold, the fitted numeric
    imputer's statistics must equal what you'd get fitting on that fold's
    training rows alone -- not the full dataset (which would include the
    held-out test rows and bias the imputed values)."""
    n = 100
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    # Missing values scattered non-uniformly so per-fold medians genuinely differ
    # from the full-dataset median.
    X.loc[rng.choice(n, size=15, replace=False), "x1"] = np.nan
    y = X["x1"].fillna(0).to_numpy() + X["x2"].to_numpy() + rng.normal(scale=0.3, size=n)

    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    spec = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)["ridge"]

    res = evaluate_model_nested_cv(
        spec, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=5, inner_folds=3, n_iter=3, n_jobs=1, random_state=0,
        compute_learning_curve=False, retain_fold_estimators=True,
    )

    # Reconstruct the same outer KFold split used internally (shuffle=True,
    # random_state=0, 5 splits, non-grouped regression path) to know each
    # fold's true train/test partition.
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    splits = list(outer_cv.split(X, y))

    assert len(res.fold_estimators) == 5
    full_median_x1 = X["x1"].median()

    saw_a_fold_that_differs_from_full_dataset_median = False
    for fold_idx, pipe, test_idx in res.fold_estimators:
        train_idx, expected_test_idx = splits[fold_idx]
        assert set(test_idx.tolist()) == set(expected_test_idx.tolist())

        train_only_median = X.iloc[train_idx]["x1"].median()
        fitted_imputer = pipe.named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"]
        # x1 is the first numeric column (alphabetical/positional per numeric_cols).
        fitted_median = fitted_imputer.statistics_[0]

        assert fitted_median == pytest.approx(train_only_median)
        if abs(fitted_median - full_median_x1) > 1e-9:
            saw_a_fold_that_differs_from_full_dataset_median = True

    # If every fold's fitted median happened to exactly equal the full-dataset
    # median, this test wouldn't actually be discriminating; make sure the
    # setup produced genuine per-fold variation.
    assert saw_a_fold_that_differs_from_full_dataset_median


def test_nested_cv_reports_cancellation_when_should_stop_fires_immediately():
    X, y = _regression_data(n=60)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    spec = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)["ridge"]
    res = evaluate_model_nested_cv(
        spec, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=3, inner_folds=2, n_iter=2, n_jobs=1, random_state=0,
        compute_learning_curve=False, should_stop=lambda: True,
    )
    assert res.execution_status == "CANCELLED"
    assert res.cancelled is True


def test_compute_corrected_resampled_ttest_identical_scores_give_no_difference():
    scores = [0.5, 0.52, 0.48, 0.51, 0.49]
    t_stat, p_val = compute_corrected_resampled_ttest(scores, scores, n_train=80, n_test=20)
    assert t_stat == pytest.approx(0.0)
    assert p_val == pytest.approx(1.0)


def test_compute_corrected_resampled_ttest_large_consistent_difference_is_significant():
    scores_a = [0.9, 0.91, 0.89, 0.92, 0.88]
    scores_b = [0.5, 0.48, 0.52, 0.49, 0.51]
    t_stat, p_val = compute_corrected_resampled_ttest(scores_a, scores_b, n_train=80, n_test=20)
    assert abs(t_stat) > 2
    assert p_val < 0.10


def test_extract_linear_model_coefficients_shape_and_content():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    X, y = _regression_data(n=80)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    fold_estimators = []
    for fold_idx in range(3):
        pipe = Pipeline([("preprocess", preprocessor), ("model", Ridge(alpha=1.0))])
        pipe.fit(X, y)
        fold_estimators.append((fold_idx, pipe, np.arange(10)))

    coefs = extract_linear_model_coefficients(fold_estimators)
    assert not coefs.empty
    assert set(
        ["feature", "mean_beta", "std_beta", "fold_range_lower", "fold_range_upper", "count"]
    ).issubset(coefs.columns)
    # Fix 8: must NOT be labeled as a conventional confidence interval --
    # "ci_lower"/"ci_upper" would misleadingly imply a calibrated interval
    # when this is a [2.5, 97.5] percentile range across as few as 3-10
    # fold-level point estimates.
    assert "ci_lower" not in coefs.columns
    assert "ci_upper" not in coefs.columns
    assert (coefs["interval_type"] == "descriptive_fold_range").all()
    assert (coefs["count"] == 3).all()
    # x1 has by far the largest true coefficient (2.0) -> should rank first.
    assert coefs.iloc[0]["feature"] == "x1"
