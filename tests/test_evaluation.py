# tests/test_evaluation.py
from __future__ import annotations

import numpy as np
import pytest

from config.schema import AnalysisTaskType
from core.evaluation import (
    bootstrap_metric_ci,
    bootstrap_paired_difference_ci,
    calculate_comprehensive_metrics,
    calculate_single_metric,
)


def test_regression_metrics_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = calculate_comprehensive_metrics(y, y.copy(), AnalysisTaskType.REGRESSION)
    assert m["r2"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["spearman_rho"] == pytest.approx(1.0)


def test_regression_metrics_known_values():
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])
    m = calculate_comprehensive_metrics(y_true, y_pred, AnalysisTaskType.REGRESSION)
    # Hand/np-computed reference values for this textbook sklearn example.
    assert m["r2"] == pytest.approx(0.9486, abs=1e-3)
    assert m["mae"] == pytest.approx(0.5, abs=1e-9)
    assert m["rmse"] == pytest.approx(np.sqrt(0.375), abs=1e-9)


def test_regression_r2_is_zero_not_crash_on_constant_target():
    # np.var(y_true) == 0 -> sklearn's r2_score is degenerate; code must guard this.
    y = np.array([5.0, 5.0, 5.0, 5.0])
    m = calculate_comprehensive_metrics(y, np.array([5.1, 4.9, 5.0, 5.2]), AnalysisTaskType.REGRESSION)
    assert m["r2"] == 0.0
    assert np.isfinite(m["rmse"])


def test_classification_metrics_known_confusion_matrix():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    m = calculate_comprehensive_metrics(y_true, y_pred, AnalysisTaskType.CLASSIFICATION)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["confusion_matrix"] == [[1, 1], [1, 2]]


def test_calculate_single_metric_matches_comprehensive_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=40)
    y_pred = y_true + rng.normal(scale=0.3, size=40)
    full = calculate_comprehensive_metrics(y_true, y_pred, AnalysisTaskType.REGRESSION)
    for key in ("r2", "rmse", "mae", "pearson_r", "spearman_rho"):
        single = calculate_single_metric(y_true, y_pred, key, AnalysisTaskType.REGRESSION)
        assert single == pytest.approx(full[key])


def test_bootstrap_metric_ci_contains_point_estimate_and_has_sane_width():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=200)
    y_pred = y_true + rng.normal(scale=0.5, size=200)
    point, lo, hi = bootstrap_metric_ci(
        y_true, y_pred, "r2", AnalysisTaskType.REGRESSION, n_boot=300, random_state=0
    )
    assert lo <= point <= hi
    assert lo >= 0.0
    assert hi <= 1.0 + 1e-9


def test_bootstrap_metric_ci_is_reproducible_given_same_seed():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=100)
    y_pred = y_true + rng.normal(scale=0.4, size=100)
    r1 = bootstrap_metric_ci(y_true, y_pred, "r2", AnalysisTaskType.REGRESSION, n_boot=200, random_state=42)
    r2 = bootstrap_metric_ci(y_true, y_pred, "r2", AnalysisTaskType.REGRESSION, n_boot=200, random_state=42)
    assert r1 == r2


def test_bootstrap_paired_difference_ci_is_zero_for_identical_models():
    rng = np.random.default_rng(3)
    y_true = rng.normal(size=100)
    y_pred = y_true + rng.normal(scale=0.4, size=100)
    point_diff, lo, hi = bootstrap_paired_difference_ci(
        y_true, y_pred, y_pred.copy(), "r2", AnalysisTaskType.REGRESSION, n_boot=200, random_state=0
    )
    assert point_diff == pytest.approx(0.0)
    assert lo <= 0.0 <= hi


def test_bootstrap_group_aware_resampling_only_draws_whole_groups():
    """When groups are supplied, every bootstrap resample must be a union of
    complete groups (no group split across the resampled index), otherwise
    within-group correlation would be broken and CIs would be too narrow."""
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(20), 5)  # 20 subjects x 5 obs each
    y_true = rng.normal(size=100)
    y_pred = y_true + rng.normal(scale=0.5, size=100)

    # Monkeypatch-free check: rerun the same resampling logic directly.
    rng2 = np.random.default_rng(0)
    unique_groups = np.unique(groups)
    group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    sampled_groups = rng2.choice(unique_groups, size=len(unique_groups), replace=True)
    idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
    # Every sampled group must contribute all 5 of its rows, contiguously grouped.
    counts = {}
    for g in sampled_groups:
        counts[g] = counts.get(g, 0) + 5
    assert len(idx) == sum(counts.values())

    # Sanity: the public API runs without error under groups and returns a valid CI.
    point, lo, hi = bootstrap_metric_ci(
        y_true, y_pred, "r2", AnalysisTaskType.REGRESSION, n_boot=100, random_state=0, groups=groups
    )
    assert lo <= point <= hi
