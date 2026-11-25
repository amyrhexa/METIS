# tests/test_sem.py
"""Dedicated SEM test suite (previously absent -- filled in as part of the
METIS scientific remediation). Covers Fix 5 (missing-data method
validation, no silent fallback), Fix 2 (multiplicity correction for
structural paths and mediation effects, with raw p-values retained), Fix
11 (z- vs t-statistic labeling), and SEM convergence/identification
hardening.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config.exceptions import SEMConfigurationError
from config.schema import SEMTargetConfig
from core.sem import (
    assess_sem_convergence,
    build_sem_variable_map,
    compute_comprehensive_ols_sem,
    run_sem_target,
    validate_missing_data_method,
)


def _mediation_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    m = 0.5 * x1 + rng.normal(scale=0.5, size=n)
    y = 0.4 * m + 0.3 * x2 + rng.normal(scale=0.6, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "m": m, "y": y})


def _multi_path_dataset(n=250, seed=1):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 0.35 * x1 + rng.normal(scale=0.9, size=n)  # x2, x3 have no real effect
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


# ---------------------------------------------------------------------------
# Fix 5: missing_data_method validation -- fail loudly, never silently
# substitute single-value imputation for an unrecognized method.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["fiml", "FIML", " Fiml ", "listwise", "LISTWISE"])
def test_validate_missing_data_method_accepts_known_values_case_insensitively(method):
    normalized = validate_missing_data_method(method)
    assert normalized in ("fiml", "listwise")


@pytest.mark.parametrize("method", ["mice", "mean", "none", "", None, "median", "unknown"])
def test_validate_missing_data_method_rejects_everything_else(method):
    with pytest.raises(SEMConfigurationError):
        validate_missing_data_method(method)


def test_run_sem_target_raises_for_invalid_missing_data_method(tmp_path: Path):
    """The critical end-to-end guarantee: a config with an unrecognized
    missing_data_method must cause run_sem_target to fail loudly rather
    than silently median-imputing and returning status='ok' as if nothing
    unusual happened."""
    df = _mediation_dataset()
    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2"],
        mediators=["m"],
        missing_data_method="mean",  # not in ALLOWED_MISSING_DATA_METHODS
        bootstrap=False,
    )
    with pytest.raises(SEMConfigurationError):
        run_sem_target("y", cfg, df, tmp_path)


def test_run_sem_target_records_actual_missing_data_method_used(tmp_path: Path):
    df = _mediation_dataset()
    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2"],
        mediators=["m"],
        missing_data_method="fiml",
        bootstrap=False,
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    assert res["status"] == "ok"
    assert res["missing_data_method_requested"] == "fiml"
    assert res["missing_data_method_used"] == "fiml"


# ---------------------------------------------------------------------------
# Fix 2: multiplicity correction for structural paths -- raw p retained,
# adjusted p used for significance, correction method/family size reported.
# ---------------------------------------------------------------------------


def test_structural_paths_table_has_raw_and_adjusted_p_columns(tmp_path: Path):
    df = _multi_path_dataset()
    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2", "x3"],
        bootstrap=False,
        multiplicity_correction="holm",
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    assert res["status"] == "ok"
    struct_df = res["structural_paths"]
    assert not struct_df.empty
    for col in (
        "p_value_raw",
        "p_value_adjusted",
        "Correction_Method",
        "Family_Size",
        "Test_Statistic",
        "Statistic_Type",
        "DF",
        "Significance",
    ):
        assert col in struct_df.columns

    assert (struct_df["Correction_Method"] == "holm").all()
    assert (struct_df["Family_Size"] == len(struct_df)).all()
    # Holm-adjusted p-values are never smaller than the raw p-value for the
    # same test.
    valid = struct_df.dropna(subset=["p_value_raw", "p_value_adjusted"])
    assert (valid["p_value_adjusted"] >= valid["p_value_raw"] - 1e-9).all()


def test_multiplicity_correction_none_leaves_p_values_unchanged(tmp_path: Path):
    df = _multi_path_dataset()
    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2", "x3"],
        bootstrap=False,
        multiplicity_correction="none",
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    struct_df = res["structural_paths"]
    valid = struct_df.dropna(subset=["p_value_raw", "p_value_adjusted"])
    assert np.allclose(valid["p_value_raw"], valid["p_value_adjusted"], atol=1e-9)


def test_significance_stars_are_computed_from_adjusted_not_raw_p(tmp_path: Path):
    """Construct a case where correction plausibly changes a borderline
    call, and assert Significance always matches what p_value_adjusted
    alone would imply (never p_value_raw)."""
    df = _multi_path_dataset()
    cfg = SEMTargetConfig(
        target="y", feature_columns=["x1", "x2", "x3"], bootstrap=False,
        multiplicity_correction="bonferroni",
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    struct_df = res["structural_paths"].dropna(subset=["p_value_adjusted"])
    for _, row in struct_df.iterrows():
        p_adj = row["p_value_adjusted"]
        sig = row["Significance"]
        if p_adj < 0.001:
            assert sig == "***"
        elif p_adj < 0.01:
            assert sig == "**"
        elif p_adj < 0.05:
            assert sig == "*"
        else:
            assert sig == "ns"


def test_mediation_table_has_raw_and_adjusted_indirect_p(tmp_path: Path):
    df = _mediation_dataset()
    cfg = SEMTargetConfig(
        target="y",
        feature_columns=["x1", "x2"],
        mediators=["m"],
        bootstrap=True,
        bootstrap_n=100,
        random_state=0,
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    assert res["status"] == "ok"
    med_df = res["mediation_effects"]
    assert not med_df.empty
    for col in (
        "Indirect_p_value_raw",
        "Indirect_p_value_adjusted",
        "Correction_Method",
        "Family_Size",
    ):
        assert col in med_df.columns
    # Deprecated alias still present and equal to the raw value.
    assert "Indirect_p_value" in med_df.columns
    valid = med_df.dropna(subset=["Indirect_p_value", "Indirect_p_value_raw"])
    assert np.allclose(valid["Indirect_p_value"], valid["Indirect_p_value_raw"], atol=1e-9)


# ---------------------------------------------------------------------------
# Fix 11: a t-statistic (OLS fallback) must never be labeled as a z-statistic.
# ---------------------------------------------------------------------------


def test_ols_fallback_labels_its_statistic_as_t_not_z():
    df = _multi_path_dataset()
    cfg = SEMTargetConfig(target="y", feature_columns=["x1", "x2", "x3"], bootstrap=False)
    vmap = build_sem_variable_map(cfg, available_cols=list(df.columns))
    struct_df, _med_df, _r2_df, _, _params_df, fit_idx = compute_comprehensive_ols_sem(
        df, cfg, vmap
    )
    assert not struct_df.empty
    assert (struct_df["Statistic_Type"] == "t (OLS)").all()
    assert struct_df["DF"].notna().all()
    assert (struct_df["DF"] > 0).all()
    # Deprecated alias column must still exist and match Test_Statistic.
    assert np.allclose(struct_df["z_value"], struct_df["Test_Statistic"], atol=1e-9)
    # A t-labeled statistic must never claim to be a Wald z-statistic.
    assert not (struct_df["Statistic_Type"] == "z (ML/Wald)").any()
    assert math.isnan(fit_idx.cfi)  # OLS fallback never fabricates a covariance-fit CFI


# ---------------------------------------------------------------------------
# SEM convergence/identification hardening
# ---------------------------------------------------------------------------


def test_assess_sem_convergence_flags_negative_dof_as_under_identified():
    status, notes = assess_sem_convergence(pd.DataFrame({"Beta_Std": [0.1], "SE": [0.05]}), dof=-1)
    assert status == "under_identified"
    assert "identified" in notes.lower()


def test_assess_sem_convergence_flags_implausible_coefficients():
    struct_df = pd.DataFrame(
        {"Beta_Std": [0.2, 12.5, 0.1], "SE": [0.05, 0.1, 0.05]}
    )
    status, notes = assess_sem_convergence(struct_df, dof=2)
    assert status == "unstable"
    assert "Heywood" in notes or "exceed" in notes


def test_assess_sem_convergence_reports_converged_for_well_behaved_fit():
    struct_df = pd.DataFrame(
        {"Beta_Std": [0.2, 0.35, -0.1], "SE": [0.05, 0.06, 0.04]}
    )
    status, _notes = assess_sem_convergence(struct_df, dof=3)
    assert status == "converged"


def test_run_sem_target_reports_convergence_status_in_result(tmp_path: Path):
    df = _multi_path_dataset()
    cfg = SEMTargetConfig(target="y", feature_columns=["x1", "x2", "x3"], bootstrap=False)
    res = run_sem_target("y", cfg, df, tmp_path)
    assert res["status"] == "ok"
    assert "convergence_status" in res
    assert res["convergence_status"] in (
        "converged",
        "unstable",
        "under_identified",
        "not_applicable_ols_fallback",
        "fit_error",
        "no_paths",
    )


# ---------------------------------------------------------------------------
# Narrative transparency: the generated narrative must mention the
# missing-data method and multiplicity correction rather than leaving them
# implicit.
# ---------------------------------------------------------------------------


def test_narrative_mentions_missing_data_method_and_correction(tmp_path: Path):
    df = _multi_path_dataset()
    cfg = SEMTargetConfig(
        target="y", feature_columns=["x1", "x2", "x3"], bootstrap=False,
        missing_data_method="fiml", multiplicity_correction="holm",
    )
    res = run_sem_target("y", cfg, df, tmp_path)
    narrative = res["narrative"].lower()
    assert "fiml" in narrative or "full-information maximum likelihood" in narrative
    assert "multiplicity-corrected" in narrative or "holm" in narrative
