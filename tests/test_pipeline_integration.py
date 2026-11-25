# tests/test_pipeline_integration.py
"""End-to-end QA/QC: run the real execute_pipeline() entry point top-to-bottom
on small synthetic datasets and verify the whole thing actually works --
manifest completes, expected output artifacts exist, and reported performance
is consistent with what the underlying data-generating process should allow
(a strong practical check against leakage: a pure-noise target must not score
suspiciously well)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from config.exceptions import PipelineValidationError
from config.schema import AnalysisTaskType, DatasetConfig, get_preset_config
from core.pipeline_runner import execute_pipeline


def _make_regression_df(n=150, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "group": rng.choice(["a", "b"], size=n),
        }
    )
    df["y"] = 3.0 * df["x1"] - 2.0 * df["x2"] + rng.normal(scale=0.5, size=n)
    return df


def _make_classification_df(n=150, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    logits = 2.0 * df["x1"] - 1.0 * df["x2"]
    prob = 1 / (1 + np.exp(-logits))
    df["y"] = (rng.uniform(size=n) < prob).astype(int)
    return df


def test_full_pipeline_regression_completes_and_produces_expected_artifacts(tmp_path):
    df = _make_regression_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"],
        feature_columns=["x1", "x2", "x3", "group"],
        categorical_columns=["group"],
    )
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["execution"]["completed_targets"] == ["y"]
    assert manifest["execution"]["error"] is None

    assert (tmp_path / "derived_dataset.csv").exists()
    assert (tmp_path / "metrics" / "all_models_benchmark.csv").exists()
    assert (tmp_path / "metrics" / "model_performance_summary.csv").exists()
    assert any((tmp_path / "reports").glob("*.html"))

    summary = pd.read_csv(tmp_path / "metrics" / "model_performance_summary.csv")
    assert len(summary) == 1
    # Strong true linear signal (betas 3.0 / -2.0) should be recovered well
    # under correct nested CV -- this is a floor, not a tight bound.
    r2_col = "r2" if "r2" in summary.columns else next(c for c in summary.columns if "r2" in c.lower())
    assert summary.iloc[0][r2_col] > 0.6


def test_full_pipeline_classification_completes(tmp_path):
    df = _make_classification_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(target_columns=["y"], feature_columns=["x1", "x2"])
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["execution"]["completed_targets"] == ["y"]


def test_full_pipeline_on_pure_noise_target_does_not_report_inflated_performance(tmp_path):
    """Practical, whole-pipeline leakage smoke-test: if the nested-CV loop
    (preprocessing + hyperparameter search + model fit) leaked test-fold
    information anywhere, a model would be able to report deceptively good
    performance predicting *pure noise*. With correct leave-fold-out CV,
    reported R^2 on an unpredictable target must stay low."""
    rng = np.random.default_rng(7)
    n = 150
    df = pd.DataFrame(
        {f"x{i}": rng.normal(size=n) for i in range(10)}
    )
    df["y"] = rng.normal(size=n)  # target is independent noise, unrelated to any x

    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(target_columns=["y"], feature_columns=[f"x{i}" for i in range(10)])
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    summary = pd.read_csv(tmp_path / "metrics" / "model_performance_summary.csv")
    r2_col = "r2" if "r2" in summary.columns else next(c for c in summary.columns if "r2" in c.lower())
    # A small positive R^2 can occur by chance with n=150, ~10 noise features,
    # and inner-CV hyperparameter selection, but leakage would blow this well
    # past a modest ceiling (e.g. > 0.5 on pure noise is not a "close call").
    assert summary.iloc[0][r2_col] < 0.35


def test_full_pipeline_run_sem_alongside_ml_does_not_crash(tmp_path):
    df = _make_regression_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(target_columns=["y"], feature_columns=["x1", "x2", "x3"])
    cfg.output_dir = str(tmp_path)
    cfg.run_sem = True  # exercise the auto-generated SEMTargetConfig fallback path

    execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert (tmp_path / "sem").exists()


def test_full_pipeline_fails_gracefully_with_no_targets(tmp_path):
    df = _make_regression_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(target_columns=[], feature_columns=["x1"])
    cfg.output_dir = str(tmp_path)

    with pytest.raises(ValueError):
        execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["execution"]["error"] is not None


# ---------------------------------------------------------------------------
# Fix 1: confirmatory model-selection procedure is actually produced by a
# full pipeline run, with its own audit trail and CI.
# ---------------------------------------------------------------------------


def test_full_pipeline_produces_confirmatory_model_selection_outputs(tmp_path):
    df = _make_regression_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"],
        feature_columns=["x1", "x2", "x3", "group"],
        categorical_columns=["group"],
    )
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    conf_csv = tmp_path / "metrics" / "confirmatory_model_selection_summary.csv"
    assert conf_csv.exists()
    conf_df = pd.read_csv(conf_csv)
    assert len(conf_df) == 1
    assert conf_df.iloc[0]["execution_status"] in ("SUCCESS", "PARTIAL_FAILURE")
    assert conf_df.iloc[0]["primary_metric_name"] == "r2"
    # Strong true signal should be recoverable by the confirmatory
    # procedure too, not just the exploratory leaderboard.
    assert conf_df.iloc[0]["primary_metric_mean"] > 0.4

    audit_path = tmp_path / "metrics" / "y_confirmatory_selection_audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text())
    assert "selection_stability" in audit
    assert "fold_records" in audit
    assert len(audit["fold_records"]) == cfg.cv.outer_folds

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert "methodology" in manifest
    assert "confirmatory_model_selection" in manifest["methodology"]
    assert "y" in manifest["methodology"]["confirmatory_model_selection"]
    assert "model_selection_architecture" in manifest["methodology"]


def test_full_pipeline_confirmatory_r2_on_pure_noise_also_stays_low(tmp_path):
    """The winner's-curse fix should hold at the full-pipeline level too:
    the confirmatory R^2 (not just the exploratory leaderboard's) must not
    be inflated by comparing many candidate models on a target with no
    real signal."""
    rng = np.random.default_rng(11)
    n = 150
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(10)})
    df["y"] = rng.normal(size=n)

    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"], feature_columns=[f"x{i}" for i in range(10)]
    )
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    conf_df = pd.read_csv(
        tmp_path / "metrics" / "confirmatory_model_selection_summary.csv"
    )
    assert conf_df.iloc[0]["primary_metric_mean"] < 0.35


# ---------------------------------------------------------------------------
# Fix 3: ambiguous task-type targets are gated at the pipeline level.
# ---------------------------------------------------------------------------


def _make_ambiguous_target_df(n=150, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    # 4 unique integer values -- squarely in the ambiguous [3, 10] band.
    df["y"] = rng.integers(0, 4, size=n)
    return df


def test_full_pipeline_raises_for_unconfirmed_ambiguous_target(tmp_path):
    df = _make_ambiguous_target_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"],
        feature_columns=["x1", "x2"],
        task_type=AnalysisTaskType.AUTO,
    )
    cfg.output_dir = str(tmp_path)

    with pytest.raises(PipelineValidationError):
        execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "ambiguous" in manifest["execution"]["error"].lower()
    assert "y" in manifest["methodology"]["task_type_resolutions"]
    assert manifest["methodology"]["task_type_resolutions"]["y"]["is_ambiguous"] is True


def test_full_pipeline_proceeds_when_ambiguous_target_is_confirmed(tmp_path):
    df = _make_ambiguous_target_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"],
        feature_columns=["x1", "x2"],
        task_type=AnalysisTaskType.AUTO,
        confirmed_ambiguous_targets=["y"],
    )
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["methodology"]["task_type_resolutions"]["y"]["user_confirmed"] is True


def test_full_pipeline_proceeds_when_ambiguous_target_has_explicit_override(tmp_path):
    df = _make_ambiguous_target_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"],
        feature_columns=["x1", "x2"],
        task_type=AnalysisTaskType.AUTO,
        task_type_overrides={"y": "classification"},
    )
    cfg.output_dir = str(tmp_path)

    execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "completed"
    resolution = manifest["methodology"]["task_type_resolutions"]["y"]
    assert resolution["is_ambiguous"] is False
    assert resolution["resolved_task_type"] == "classification"


# ---------------------------------------------------------------------------
# Invalid configuration is rejected before any computation starts.
# ---------------------------------------------------------------------------


def test_generate_classification_diagnostic_plots_handles_object_dtype_predictions(tmp_path):
    """Regression test for a second instance of the object-dtype
    classification-array bug (found via the explicit-override integration
    test above): confusion_matrix (and, transitively, roc_curve /
    precision_recall_curve, which reuse the same coerced y_true in this
    function) previously raised on METIS's dtype=object OOF prediction
    accumulator."""
    from core.reporting import generate_classification_diagnostic_plots

    rng = np.random.default_rng(0)
    n = 60
    y_true = rng.integers(0, 3, size=n)
    y_pred_object = np.empty(n, dtype=object)
    y_pred_object[:] = rng.integers(0, 3, size=n)
    y_prob = rng.dirichlet(np.ones(3), size=n)

    out_path = generate_classification_diagnostic_plots(
        y_true, y_pred_object, y_prob, "target", "model", tmp_path
    )
    assert out_path.exists()


def test_full_pipeline_rejects_invalid_multiplicity_correction_upfront(tmp_path):
    df = _make_regression_df()
    cfg = get_preset_config("smoke_test")
    cfg.dataset = DatasetConfig(
        target_columns=["y"], feature_columns=["x1", "x2", "x3"]
    )
    cfg.cv.multiplicity_correction = "not_a_real_method"
    cfg.output_dir = str(tmp_path)

    with pytest.raises(PipelineValidationError):
        execute_pipeline(cfg, dataset_df=df)

    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
