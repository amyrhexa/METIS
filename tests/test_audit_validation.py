# tests/test_audit_validation.py
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.schema import (
    AnalysisTaskType,
    PreprocessingConfig,
    SEMTargetConfig,
)
from core.data_loader import inspect_dataset
from core.evaluation import (
    bootstrap_metric_ci,
    bootstrap_paired_difference_ci,
    calculate_comprehensive_metrics,
)
from core.modeling import get_model_zoo
from core.preprocessing import build_preprocessor
from core.sem import (
    build_sem_model_syntax,
    build_sem_variable_map,
    compute_srmr,
    run_sem_target,
)


def test_binary_columns_remain_numeric_when_explicitly_configured():
    df = pd.DataFrame(
        {
            "age": [25.0, 30.0, 45.0, 60.0],
            "sex_male": [1.0, 0.0, 1.0, 0.0],
            "target": [10.0, 12.0, 15.0, 20.0],
        }
    )
    profile = inspect_dataset(df)
    assert "sex_male" in profile.binary_numeric_columns
    assert "sex_male" in profile.numerical_columns
    assert "sex_male" not in profile.categorical_columns

    cats: list[str] = []
    preproc = build_preprocessor(df[["age", "sex_male"]], cats, PreprocessingConfig())
    trans = preproc.fit_transform(df[["age", "sex_male"]])
    out_features = list(preproc.get_feature_names_out())
    assert len(out_features) == 2
    assert "age" in out_features
    assert "sex_male" in out_features
    assert "sex_male_0" not in out_features and "sex_male_1" not in out_features
    assert trans.shape == (4, 2)


def test_catboost_exists_in_zoo():
    zoo_reg = get_model_zoo(AnalysisTaskType.REGRESSION)
    zoo_clf = get_model_zoo(AnalysisTaskType.CLASSIFICATION)
    assert "catboost" in zoo_reg
    assert "catboost" in zoo_clf
    assert zoo_reg["catboost"].is_tree_based
    assert zoo_clf["catboost"].is_tree_based


def test_srmr_calculation():
    s_cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    imp_cov = np.array([[1.0, 0.4], [0.4, 1.0]])
    val = compute_srmr(s_cov, imp_cov)
    assert not np.isnan(val)
    assert val > 0.0
    assert val == pytest.approx(0.1, rel=1e-2)

    # Ill-formed or mismatched inputs return NaN safely
    assert np.isnan(compute_srmr(np.array([[1.0]]), np.array([[1.0]])))
    assert np.isnan(compute_srmr(s_cov, np.array([[1.0]])))


def test_classification_metrics_and_bootstrap():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1, 1, 1])
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.1, 0.9],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )

    metrics = calculate_comprehensive_metrics(
        y_true, y_pred, AnalysisTaskType.CLASSIFICATION, y_prob
    )
    assert "accuracy" in metrics
    assert "balanced_acc" in metrics
    assert "f1_macro" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 0.8
    assert not np.isnan(metrics["roc_auc"])

    pt, lo, hi = bootstrap_metric_ci(
        y_true,
        y_pred,
        metric_name="accuracy",
        task_type=AnalysisTaskType.CLASSIFICATION,
        n_boot=100,
        random_state=42,
    )
    assert pt == 0.8
    assert 0.0 <= lo <= hi <= 1.0

    diff, diff_lo, diff_hi = bootstrap_paired_difference_ci(
        y_true,
        y_pred,
        y_true,  # Perfect predictions
        metric_name="accuracy",
        task_type=AnalysisTaskType.CLASSIFICATION,
        n_boot=100,
        random_state=42,
    )
    assert diff == pytest.approx(-0.2)
    assert diff_lo <= diff_hi


def test_sem_no_hardcoded_placeholders(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 60
    x = rng.normal(size=n)
    m = 0.5 * x + rng.normal(scale=0.5, size=n)
    y = 0.4 * m + 0.3 * x + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"x": x, "m": m, "y": y})

    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x"],
        mediators=["m"],
        bootstrap=True,
        bootstrap_n=50,
        random_state=42,
    )
    res = run_sem_target("y", cfg, df, Path(tmp_path))
    assert res["status"] == "ok"
    assert "parameters" in res
    assert "structural_paths" in res

    fit_idx = res["fit_indices"]
    assert fit_idx.srmr != 0.04
    assert fit_idx.p_val != 0.5
    assert not np.isnan(fit_idx.aic) or fit_idx.verdict != "Good Fit (semopy)"


def test_parallel_mediators_get_residual_covariance():
    """build_sem_model_syntax must free the residual covariance between every
    pair of parallel mediators -- leaving it fixed at zero (semopy/lavaan's
    default) is a false assumption for correlated indicators (e.g. brain
    region volumes) and was the primary driver of a previously catastrophic
    CFI/TLI/RMSEA misfit."""
    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x"],
        mediators=["m1", "m2", "m3"],
    )
    vmap = build_sem_variable_map(cfg, available_cols=["x", "y", "m1", "m2", "m3"])
    syntax = build_sem_model_syntax(cfg, vmap)
    for pair in ("m1 ~~ m2", "m1 ~~ m3", "m2 ~~ m3"):
        assert pair in syntax, f"missing residual covariance line: {pair}"
    # The target itself must never be treated as its own mediator, even if
    # a user config mistakenly lists it among the mediators.
    assert "y ~~ y" not in syntax
    assert "y ~ y" not in syntax


def test_sem_r2_is_nan_not_false_zero_when_data_is_missing(tmp_path: Path):
    """With missing_data_method='fiml', NaNs are preserved through to the R^2
    computation. A previous bug computed R^2 via np.var() on arrays
    containing NaN (-> nan), then did max(0.0, nan), which Python evaluates
    to 0.0 -- silently reporting '0% variance explained' instead of
    surfacing the failure. R^2 must now either be a real, NaN-safe value
    computed over complete cases, or explicitly reported as unavailable --
    never a false 0."""
    rng = np.random.default_rng(7)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 * x1 + 0.4 * x2 + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    # Light missingness in one predictor, mirroring real-world covariate gaps.
    missing_idx = rng.choice(n, size=max(3, n // 50), replace=False)
    df.loc[missing_idx, "x1"] = np.nan

    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2"],
        missing_data_method="fiml",
        bootstrap=False,
        random_state=42,
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    assert res["status"] == "ok"

    r2_df = res["variance_explained"]
    y_row = r2_df[r2_df["Endogenous_Variable"] == "y"].iloc[0]
    # The true model explains a substantial share of variance (betas 0.5/0.4
    # on standardized data); a value pinned at exactly 0.0 indicates the
    # NaN-masking bug has regressed.
    assert not (y_row["R_Squared"] == 0.0 and y_row["Variance_Explained_Pct"] == "0.0%")
    if np.isfinite(y_row["R_Squared"]):
        assert y_row["R_Squared"] > 0.2
    else:
        assert y_row["Variance_Explained_Pct"] == "N/A"


def test_listwise_deletion_actually_drops_incomplete_rows(tmp_path: Path):
    """missing_data_method='listwise' must drop incomplete rows, not silently
    median-impute them (the previous, mislabeled behavior)."""
    rng = np.random.default_rng(3)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.4 * x1 + 0.3 * x2 + rng.normal(scale=0.7, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    n_missing = 10
    df.loc[: n_missing - 1, "x1"] = np.nan

    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2"],
        missing_data_method="listwise",
        bootstrap=False,
        random_state=42,
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    assert res["status"] == "ok"
    assert res["n"] == n - n_missing
    assert not res["sem_data"].isna().any().any()
