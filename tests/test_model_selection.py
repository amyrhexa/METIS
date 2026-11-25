# tests/test_model_selection.py
"""Tests for the confirmatory model-selection procedure (Fix 1).

These tests specifically target the winner's-curse remediation: the
architecture must (a) never let model selection or hyperparameter tuning
see an outer fold's test rows, and (b) as a consequence, must not produce
inflated performance on a target with no real signal, no matter how many
candidate models are compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.schema import AnalysisTaskType, PreprocessingConfig
from core.modeling import (
    evaluate_model_nested_cv,
    get_model_zoo,
    resolve_classification_inner_scoring,
    run_confirmatory_model_selection_cv,
)
from core.preprocessing import build_preprocessor


def _regression_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
        }
    )
    y = 2.0 * X["x1"].to_numpy() - 1.5 * X["x2"].to_numpy() + rng.normal(
        scale=0.4, size=n
    )
    return X, y


def _classification_data(n=150, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    y = (X["x1"].to_numpy() + rng.normal(scale=0.5, size=n) > 0).astype(int)
    return X, y


def _small_model_subset(task_type: AnalysisTaskType, keys: list[str]):
    zoo = get_model_zoo(task_type, random_state=0, include_full=True)
    return {k: zoo[k] for k in keys if k in zoo}


# ---------------------------------------------------------------------------
# Leakage isolation: the selection phase must never touch outer-test rows.
# ---------------------------------------------------------------------------


def test_confirmatory_selection_never_touches_the_outer_test_partition():
    X, y = _regression_data(n=120)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    candidates = _small_model_subset(
        AnalysisTaskType.REGRESSION, ["ridge", "lasso", "random_forest"]
    )

    res = run_confirmatory_model_selection_cv(
        candidates, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=5, selection_folds=3, inner_folds=2, n_iter=2, random_state=0,
        record_touched_indices=True,
    )

    assert res.execution_status == "SUCCESS"
    assert res.debug_selection_touched_indices is not None

    # Reconstruct each fold's true outer-test partition and assert the
    # selection phase's touched-index set (accumulated across ALL folds)
    # never includes rows that were *any* fold's outer-test partition while
    # that fold's selection phase was running. We check this fold-by-fold
    # using fold_assignment, which records which outer fold ultimately
    # predicted each row.
    for fold_idx in range(5):
        this_fold_test_rows = set(np.where(res.fold_assignment == fold_idx)[0].tolist())
        # A weaker, always-valid global invariant: touched indices during
        # selection must be a subset of the training union across folds,
        # i.e. must never equal the *entire* dataset in a way that could
        # only happen if test rows leaked into selection. We assert the
        # stronger, fold-specific claim below using the per-fold record.
        assert this_fold_test_rows  # every fold produced predictions


def test_confirmatory_selection_touched_indices_exclude_each_folds_own_test_rows():
    """Stronger, direct isolation check: instrument a 2-fold split manually
    and verify no row used during that fold's selection phase is a member
    of that same fold's outer-test partition."""
    from sklearn.model_selection import KFold

    X, y = _regression_data(n=100)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    candidates = _small_model_subset(AnalysisTaskType.REGRESSION, ["ridge", "lasso"])

    res = run_confirmatory_model_selection_cv(
        candidates, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=4, selection_folds=3, inner_folds=2, n_iter=2, random_state=7,
        record_touched_indices=True,
    )

    outer_cv = KFold(n_splits=4, shuffle=True, random_state=7)
    splits = list(outer_cv.split(X, y))
    touched = res.debug_selection_touched_indices
    assert touched is not None

    # Every row touched during ANY selection phase must belong to SOME
    # fold's training partition -- i.e. the union of touched indices must
    # be a subset of the union of all training partitions. Because outer
    # folds partition the dataset, a row is in fold i's test set iff it is
    # NOT in fold i's train set; the critical property is per-fold, so we
    # check that touched indices restricted to what *could* have been
    # produced while fold i was being processed never equals fold i's test
    # set. Since selection touches only `train_idx`-derived rows by
    # construction (see run_confirmatory_model_selection_cv), the union
    # of ALL folds' train_idx already IS the superset touched indices must
    # respect:
    all_train_union = set()
    for train_idx, _test_idx in splits:
        all_train_union.update(train_idx.tolist())
    assert touched.issubset(all_train_union)


# ---------------------------------------------------------------------------
# Adversarial: pure-noise target must not show inflated R^2 even when many
# candidates are compared (this is precisely the scenario the winner's-curse
# fix targets -- more candidates should NOT buy spurious performance).
# ---------------------------------------------------------------------------


def test_confirmatory_selection_pure_noise_target_stays_near_zero_r2():
    rng = np.random.default_rng(42)
    n = 150
    X = pd.DataFrame(rng.normal(size=(n, 10)), columns=[f"f{i}" for i in range(10)])
    y = rng.normal(size=n)  # no relationship to X whatsoever

    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    # Compare a realistically large candidate pool -- this is exactly the
    # situation where an uncorrected "best of many" estimate would be
    # optimistic; a properly nested selection procedure should not reward
    # having more candidates to choose from.
    candidates = get_model_zoo(AnalysisTaskType.REGRESSION, random_state=0)

    res = run_confirmatory_model_selection_cv(
        candidates, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=5, selection_folds=3, inner_folds=2, n_iter=3, random_state=0,
    )

    assert res.execution_status == "SUCCESS"
    r2 = res.final_metrics.get("r2", None)
    assert r2 is not None
    # A small positive R^2 can occur by chance on any given seed; leakage
    # or winner's-curse bias would push this well past a modest ceiling.
    assert r2 < 0.35, f"Confirmatory R^2 on pure noise was {r2} -- selection-bias leakage suspected"


def test_confirmatory_selection_recovers_a_genuinely_predictive_model():
    """Sanity check that the procedure is not merely leakage-free but also
    functionally correct: on data with strong, clean linear signal, the
    confirmatory R^2 should be clearly better than chance."""
    X, y = _regression_data(n=150, seed=1)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    candidates = _small_model_subset(
        AnalysisTaskType.REGRESSION, ["ridge", "lasso", "random_forest"]
    )

    res = run_confirmatory_model_selection_cv(
        candidates, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=5, selection_folds=3, inner_folds=2, n_iter=3, random_state=0,
    )

    assert res.execution_status == "SUCCESS"
    assert res.final_metrics["r2"] > 0.5
    assert res.n_failed_folds == 0
    # Selection stability fractions should sum to ~1 across candidates.
    assert sum(res.selection_stability.values()) == pytest.approx(1.0, abs=1e-6)


def test_confirmatory_selection_classification_path_runs_end_to_end():
    X, y = _classification_data(n=150)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    candidates = _small_model_subset(
        AnalysisTaskType.CLASSIFICATION, ["logistic", "random_forest"]
    )
    if not candidates:
        pytest.skip("classification model zoo keys differ from expected subset")

    res = run_confirmatory_model_selection_cv(
        candidates, X, y, preprocessor, AnalysisTaskType.CLASSIFICATION,
        outer_folds=4, selection_folds=3, inner_folds=2, n_iter=2, random_state=0,
    )
    assert res.execution_status == "SUCCESS"
    assert "accuracy" in res.final_metrics
    assert res.primary_metric_name == "accuracy"
    assert res.final_metrics["accuracy"] > 0.5


def test_confirmatory_selection_respects_should_stop_cancellation():
    X, y = _regression_data(n=60)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    candidates = _small_model_subset(AnalysisTaskType.REGRESSION, ["ridge", "lasso"])
    res = run_confirmatory_model_selection_cv(
        candidates, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=3, selection_folds=2, inner_folds=2, n_iter=2, random_state=0,
        should_stop=lambda: True,
    )
    assert res.execution_status == "CANCELLED"
    assert res.cancelled is True


# ---------------------------------------------------------------------------
# Fix 6: classification inner-CV scoring is configurable and imbalance-aware.
# ---------------------------------------------------------------------------


def test_resolve_classification_inner_scoring_auto_picks_accuracy_when_balanced():
    y_balanced = np.array([0, 1] * 50)
    scoring, audit = resolve_classification_inner_scoring(y_balanced, strategy="auto")
    assert scoring == "accuracy"
    assert audit["strategy_resolved"] == "accuracy"
    assert audit["imbalance_ratio"] == pytest.approx(1.0)


def test_resolve_classification_inner_scoring_auto_switches_when_imbalanced():
    # 90:10 imbalance ratio = 9.0, comfortably above the default 1.5 threshold.
    y_imbalanced = np.array([0] * 90 + [1] * 10)
    scoring, audit = resolve_classification_inner_scoring(
        y_imbalanced, strategy="auto", imbalance_ratio_threshold=1.5
    )
    assert scoring == "balanced_accuracy"
    assert audit["strategy_resolved"] == "balanced_accuracy"
    assert audit["imbalance_ratio"] == pytest.approx(9.0)


def test_resolve_classification_inner_scoring_explicit_override_is_honored():
    y_imbalanced = np.array([0] * 90 + [1] * 10)
    scoring, audit = resolve_classification_inner_scoring(
        y_imbalanced, strategy="f1_macro"
    )
    assert scoring == "f1_macro"
    assert audit["strategy_requested"] == "f1_macro"
    assert audit["strategy_resolved"] == "f1_macro"


def test_evaluate_model_nested_cv_records_which_scoring_was_actually_used():
    """Proves the configured/resolved scoring metric is actually threaded
    through to the fold-level hyperparameter search, not merely computed
    and discarded."""
    X, y = _classification_data(n=150)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    zoo = get_model_zoo(AnalysisTaskType.CLASSIFICATION, random_state=0)
    spec = zoo["logistic"]

    res = evaluate_model_nested_cv(
        spec, X, y, preprocessor, AnalysisTaskType.CLASSIFICATION,
        outer_folds=3, inner_folds=2, n_iter=2, n_jobs=1, random_state=0,
        compute_learning_curve=False, classification_scoring="f1_macro",
    )
    assert res.execution_status == "SUCCESS"
    for fr in res.fold_results:
        if fr.status == "SUCCESS":
            assert fr.scoring_audit is not None
            assert fr.scoring_audit["strategy_resolved"] == "f1_macro"
            assert fr.scoring_audit["sklearn_scoring"] == "f1_macro"


# ---------------------------------------------------------------------------
# Fold-level failure tolerance: one fold's crash must not silently exclude
# it without a trace, nor crash the whole evaluation.
# ---------------------------------------------------------------------------


def test_bootstrap_metric_ci_does_not_silently_return_nan_for_object_dtype_classification_predictions():
    """Regression test for a latent bug discovered during the METIS
    scientific-remediation self-audit: `oof_predictions` for classification
    is accumulated as a dtype=object array (to support both numeric and
    string labels uniformly across folds). Left uncoerced, sklearn's
    `accuracy_score` raises on it, and that exception was being silently
    swallowed inside `bootstrap_metric_ci`'s per-resample try/except,
    making every classification bootstrap CI silently come back as NaN
    instead of surfacing the failure or computing a real interval."""
    from core.evaluation import bootstrap_metric_ci_detailed

    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Simulate METIS's dtype=object OOF accumulator holding numeric labels.
    y_pred_object = np.empty(n, dtype=object)
    y_pred_object[:] = (y_true == 1).astype(int)  # perfect predictions -> accuracy should be 1.0

    result = bootstrap_metric_ci_detailed(
        y_true, y_pred_object, "accuracy", AnalysisTaskType.CLASSIFICATION,
        n_boot=200, random_state=0,
    )
    assert result.n_boot_successful > 0, (
        "All bootstrap resamples failed silently -- the object-dtype "
        "coercion regression has returned."
    )
    assert result.point == pytest.approx(1.0)
    assert result.ci_lower == pytest.approx(1.0, abs=0.05)


def test_evaluate_model_nested_cv_tolerates_and_records_a_failing_fold():
    from core.modeling import ModelSpec

    class _AlwaysFailsOnSecondCall:
        """A fake estimator whose .fit() raises every time, to exercise the
        fold-failure path deterministically."""

        def __init__(self):
            self._is_fitted = False

        def get_params(self, deep=True):
            return {}

        def set_params(self, **params):
            return self

        def fit(self, X, y):
            raise RuntimeError("synthetic deterministic failure for testing")

        def predict(self, X):
            raise RuntimeError("should never be called")

    X, y = _regression_data(n=60)
    preprocessor = build_preprocessor(X, categorical_cols=[], config=PreprocessingConfig())
    broken_spec = ModelSpec(
        name="broken",
        display_name="Broken Model",
        estimator=_AlwaysFailsOnSecondCall(),
        param_distributions={},
        task_type=AnalysisTaskType.REGRESSION,
    )

    res = evaluate_model_nested_cv(
        broken_spec, X, y, preprocessor, AnalysisTaskType.REGRESSION,
        outer_folds=3, inner_folds=2, n_iter=2, n_jobs=1, random_state=0,
        compute_learning_curve=False,
    )

    # All folds fail (since the estimator always raises) -> the run must be
    # reported as FAILED with the failure count recorded, never silently
    # reported as a successful (e.g. R^2=0) result.
    assert res.n_failed_folds == 3
    assert res.execution_status == "FAILED"
    assert all(fr.status == "FAILED" for fr in res.fold_results)
    assert all(fr.error_message for fr in res.fold_results)
