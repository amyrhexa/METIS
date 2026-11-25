# core/pipeline_runner.py
"""Headless execution engine for Nested CV tournaments and Structural Equation Modeling."""

from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import time
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from config.exceptions import PipelineValidationError, SEMConfigurationError
from config.schema import (
    MANIFEST_SCHEMA_VERSION,
    AnalysisTaskType,
    PipelineConfig,
    SEMTargetConfig,
)
from core.data_loader import (
    clean_dataframe,
    inspect_dataset,
    load_dataset,
    resolve_task_type,
    resolve_task_type_detailed,
)
from core.evaluation import (
    BOOTSTRAP_CI_METHOD_DESCRIPTION,
    apply_multiplicity_correction,
    bootstrap_metric_ci,
    bootstrap_paired_difference_ci,
    mask_valid_oof_predictions,
)
from core.interpretation import (
    aggregate_fold_importance,
    compute_cv_stable_attribution,
    plot_attribution_stability,
    plot_feature_importance_bar,
    plot_shap_beeswarm,
    save_plot,
)
from core.modeling import (
    ModelSpec,
    build_stacking_spec,
    compute_corrected_resampled_ttest,
    evaluate_model_nested_cv,
    get_model_zoo,
    run_confirmatory_model_selection_cv,
)
from core.preprocessing import build_preprocessor
from core.progress import PipelineStage, ProgressCallback, ProgressTracker
from core.reporting import (
    generate_calibration_plot,
    generate_classification_diagnostic_plots,
    generate_html_report,
    generate_learning_curve_plot,
    generate_regression_diagnostic_plots,
    generate_sem_covariance_plot,
    generate_sem_effect_forest_plot,
    generate_sem_effects_bar_plot,
    generate_sem_path_diagram_plot,
)
from core.sem import run_sem_target


def _get_metis_version() -> str:
    """Resolve METIS's own installed version for the manifest, without a
    module-level `from core import __version__` (which would be a
    circular import: `core/__init__.py` imports this module).
    """
    try:
        return importlib.metadata.version("METIS")
    except Exception:
        return "unknown"


def get_environment_metadata() -> dict[str, Any]:
    packages = [
        "scikit-learn",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "joblib",
        "xgboost",
        "lightgbm",
        "catboost",
        "shap",
        "semopy",
        "PySide6",
    ]
    pkg_versions: dict[str, str] = {}
    for pkg in packages:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "not-installed"

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": pkg_versions,
    }


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def setup_logger(
    out_dir: Path, extra_handler: logging.Handler | None = None
) -> logging.Logger:
    logger = logging.getLogger("ml_sem_pipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    out_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(out_dir / "pipeline.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)-8s | %(message)s")
    )
    logger.addHandler(file_handler)

    if extra_handler:
        extra_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(extra_handler)

    return logger


def calculate_pipeline_total_steps(
    targets: list[str],
    config: PipelineConfig,
    active_estimator_count: int,
) -> int:
    steps = 1
    if config.run_sem or len(config.sem_configs) > 0:
        steps += len(targets) * 2

    for _ in targets:
        steps += 1
        steps += active_estimator_count * config.cv.outer_folds
        steps += 4

    steps += 1
    return max(1, steps)


def _make_fold_callback(
    tracker: ProgressTracker,
    target_name: str,
    model_spec: ModelSpec,
) -> Callable[[int, int, str], None]:
    def fold_callback(fold_idx: int, total_f: int, msg: str) -> None:
        tracker.advance(
            1,
            stage=PipelineStage.NESTED_CV,
            message=msg,
            target_name=target_name,
            model_name=model_spec.display_name,
            fold=fold_idx + 1,
            total_folds=total_f,
        )

    return fold_callback


def execute_pipeline(
    config: PipelineConfig,
    dataset_df: pd.DataFrame | None = None,
    progress_callback: ProgressCallback | None = None,
    log_handler: logging.Handler | None = None,
    should_stop: Any = None,
) -> None:
    """Execute complete nested CV tournament and SEM path analyses with research-grade reproducibility."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings(
        "ignore", message="Singular matrix in solving dual problem.*"
    )

    t0 = time.time()
    run_id = f"metis-run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out_dir = Path(config.output_dir).resolve()
    for sub in [
        "figures/dataset",
        "figures/performance",
        "figures/explainability",
        "feature_importance",
        "metrics",
        "reports",
        "sem",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "end_time": None,
        "elapsed_seconds": 0.0,
        "metis_version": _get_metis_version(),
        "random_seed": config.cv.random_state,
        "environment": get_environment_metadata(),
        "config": config.to_dict(),
        "dataset_summary": {
            "input_path": config.dataset.input_path,
            "target_columns": list(config.dataset.target_columns),
            "feature_columns": list(config.dataset.feature_columns),
            "active_columns": list(config.dataset.active_columns),
            "id_column": config.dataset.id_column,
            "task_type": config.dataset.task_type.value,
        },
        "execution": {
            "current_stage": "INITIALIZING",
            "completed_targets": [],
            "error": None,
        },
        # Scientific-remediation audit trail: what METIS actually decided
        # (as opposed to what was merely requested), for every methodology
        # choice capable of affecting conclusions. Populated incrementally
        # as the run progresses. See the remediation report for the full
        # rationale behind each field.
        "methodology": {
            "model_selection_architecture": (
                "Per-target confirmatory performance is estimated via "
                "run_confirmatory_model_selection_cv: candidate-model "
                "comparison and hyperparameter tuning happen only within "
                "each outer fold's training partition; that fold's "
                "outer-test partition is touched exactly once, by the "
                "fold's selected-and-tuned winner. The per-candidate "
                "leaderboard (evaluate_model_nested_cv, one run per "
                "candidate) remains available separately as an "
                "EXPLORATORY/DESCRIPTIVE comparison -- selecting the max "
                "of many such per-candidate nested-CV scores and "
                "reporting it as if it were a single pre-specified "
                "model's score would be an optimistic 'winner's curse' "
                "estimate; it is not used as the confirmatory number."
            ),
            "cv_execution": (
                "Single-threaded per fold (deterministic) unless "
                "cv.parallel_outer_folds=true, in which case outer folds "
                "run concurrently via joblib using cv.n_jobs workers "
                "(each fold is an independent computation, so this "
                "changes wall-clock time only, not results). "
                "Hyperparameter search (RandomizedSearchCV) always runs "
                "single-threaded internally regardless, to avoid CPU "
                "oversubscription when outer folds are parallelized."
            ),
            "multiplicity_correction": {
                "ml_model_comparisons": config.cv.multiplicity_correction,
                "sem_default": config.cv.multiplicity_correction,
            },
            "task_type_resolutions": {},
            "sem_decisions": {},
            "classification_scoring_decisions": {},
            "warnings": [],
        },
    }
    write_manifest(out_dir, manifest)

    logger = setup_logger(out_dir, log_handler)
    logger.info("Starting METIS Scientific Pipeline (Run ID: %s)", run_id)

    def _record_warning(msg: str) -> None:
        """Log a warning AND record it in the manifest's methodology
        warnings list, so scientifically relevant caveats are never only
        visible in a log file a reader might not open (manifest/report
        transparency requirement)."""
        logger.warning(msg)
        manifest["methodology"]["warnings"].append(msg)

    try:
        config_errors = config.validate()
        if config_errors:
            msg = "Invalid pipeline configuration: " + "; ".join(config_errors)
            manifest["status"] = "failed"
            manifest["execution"]["error"] = msg
            manifest["execution"]["current_stage"] = "VALIDATION_FAILED"
            manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_manifest(out_dir, manifest)
            logger.error(msg)
            raise PipelineValidationError(msg)

        if dataset_df is not None:
            dataset_df = clean_dataframe(dataset_df, logger)
            profile = inspect_dataset(dataset_df)
            dataset_df.to_csv(out_dir / "derived_dataset.csv", index=False)
            if config.transformation_log:
                (out_dir / "transformations.json").write_text(
                    json.dumps(config.transformation_log, indent=2), encoding="utf-8"
                )
        else:
            dataset_df, profile = load_dataset(config.dataset.input_path, logger)

        manifest["dataset_summary"]["total_rows"] = profile.total_rows
        manifest["dataset_summary"]["total_columns"] = profile.total_columns
        manifest["dataset_summary"]["memory_usage_mb"] = profile.memory_usage_mb
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_manifest(out_dir, manifest)

        targets = config.dataset.target_columns
        if not targets:
            msg = "No targets (Y) specified for modeling pipeline."
            raise ValueError(msg)

        features = config.dataset.feature_columns or [
            c for c in dataset_df.columns if c not in targets
        ]
        if config.dataset.id_column and config.dataset.id_column in features:
            features.remove(config.dataset.id_column)

        if (
            config.dataset.categorical_columns is not None
            and len(config.dataset.categorical_columns) > 0
        ):
            cats = [c for c in config.dataset.categorical_columns if c in features]
        elif (
            config.dataset.categorical_columns is not None
            and len(config.dataset.categorical_columns) == 0
        ):
            cats = []
        else:
            cats = [c for c in features if c in profile.categorical_columns]

        # --- Fix 3: pre-flight ambiguous task-type validation -----------
        # Every target's task-type resolution is checked *before* any
        # expensive CV/SEM computation starts, so an ambiguous numeric
        # target (3-10 unique values under AUTO) is caught immediately and
        # explicitly, rather than discovered after minutes of computation
        # -- or worse, not discovered at all.
        task_type_resolutions: dict[str, dict[str, Any]] = {}
        unconfirmed_ambiguous: list[str] = []
        for t in targets:
            requested_str = config.dataset.task_type_overrides.get(
                t, config.dataset.task_type.value
            )
            requested_enum = AnalysisTaskType(requested_str)
            detailed = resolve_task_type_detailed(
                dataset_df[t], requested_enum, column_name=t
            )
            resolution_dict = detailed.to_dict()
            resolution_dict["user_confirmed"] = (
                t in config.dataset.confirmed_ambiguous_targets
            )
            task_type_resolutions[t] = resolution_dict
            if detailed.is_ambiguous and t not in config.dataset.confirmed_ambiguous_targets:
                unconfirmed_ambiguous.append(t)

        manifest["methodology"]["task_type_resolutions"] = task_type_resolutions
        write_manifest(out_dir, manifest)

        if unconfirmed_ambiguous:
            details = "; ".join(
                f"'{t}' ({task_type_resolutions[t]['unique_count']} unique values)"
                for t in unconfirmed_ambiguous
            )
            msg = (
                f"The following target(s) have scientifically ambiguous "
                f"cardinality under task_type='auto' (3-10 unique numeric "
                f"values -- could be ordinal, nominal, or continuous) and "
                f"have not been explicitly resolved: {details}. For each, "
                f"either set an explicit type in "
                f"dataset.task_type_overrides (e.g. "
                f"{{'{unconfirmed_ambiguous[0]}': 'classification'}}), or "
                f"add it to dataset.confirmed_ambiguous_targets to proceed "
                f"with the default resolution. METIS will not silently "
                f"guess which one is correct."
            )
            manifest["status"] = "failed"
            manifest["execution"]["error"] = msg
            manifest["execution"]["current_stage"] = "VALIDATION_FAILED"
            manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_manifest(out_dir, manifest)
            logger.error(msg)
            raise PipelineValidationError(msg)

        sample_task = resolve_task_type(
            dataset_df[targets[0]], config.dataset.task_type, allow_ambiguous=True
        )
        zoo = get_model_zoo(sample_task, config.cv.random_state, include_full=True)
        if config.models.use_all_models:
            est_count = len(zoo) + (1 if config.models.enable_stacking else 0)
        else:
            chosen = config.models.selected_models or list(zoo.keys())
            est_count = len(chosen) + (
                1 if config.models.enable_stacking and len(chosen) >= 2 else 0
            )

        total_steps = calculate_pipeline_total_steps(targets, config, est_count)
        tracker = ProgressTracker(
            total_steps=total_steps,
            callback=progress_callback,
            enable_cli_bar=False,
        )

        tracker.advance(
            1,
            stage=PipelineStage.LOADING_DATA,
            message=f"Dataset verified: {len(dataset_df)} samples, {len(features)} features.",
        )

        # 1. Structural Equation Modeling
        sem_result_dict: dict[str, dict[str, Any]] = {}
        if config.run_sem or len(config.sem_configs) > 0:
            manifest["execution"]["current_stage"] = "SEM_ANALYSIS"
            write_manifest(out_dir, manifest)

            for t in targets:
                if should_stop and should_stop():
                    logger.warning("Pipeline stop requested during SEM execution.")
                    break

                tracker.advance(
                    1,
                    stage=PipelineStage.SEM_ANALYSIS,
                    message=f"Fitting Structural Equation Model for {t}",
                    target_name=t,
                )
                sem_cfg = config.sem_configs.get(t) or SEMTargetConfig(
                    target=t,
                    feature_columns=features[: min(6, len(features))],
                    standardize=True,
                    missing_data_method="fiml",
                    bootstrap=True,
                    bootstrap_n=2000,
                    random_state=config.cv.random_state,
                )
                try:
                    s_res = run_sem_target(t, sem_cfg, dataset_df, out_dir, logger)
                except SEMConfigurationError:
                    # Configuration errors are not per-target numerical
                    # failures -- they indicate the run's setup needs
                    # attention (Fix 5: never silently substitute a weaker
                    # missing-data method). config.validate() should have
                    # already caught this before any computation started;
                    # this re-raise is defense-in-depth in case a
                    # SEMTargetConfig was constructed programmatically
                    # after that pre-flight check.
                    raise
                except Exception as exc:
                    logger.error("SEM fit failed for target '%s': %s", t, exc)
                    s_res = {
                        "target": t,
                        "status": "failed",
                        "error": str(exc)[:500],
                    }
                    _record_warning(
                        f"SEM analysis for target '{t}' failed and was skipped: "
                        f"{str(exc)[:200]}"
                    )
                sem_result_dict[t] = s_res
                manifest["methodology"]["sem_decisions"][t] = {
                    "status": s_res.get("status"),
                    "missing_data_method_requested": s_res.get(
                        "missing_data_method_requested", sem_cfg.missing_data_method
                    ),
                    "missing_data_method_used": s_res.get("missing_data_method_used"),
                    "multiplicity_correction": s_res.get(
                        "multiplicity_correction", sem_cfg.multiplicity_correction
                    ),
                    "fit_method": s_res.get("fit_method"),
                    "convergence_status": s_res.get("convergence_status"),
                }
                if s_res.get("status") == "ok":
                    params_df = s_res.get("parameters")
                    effects_df = s_res.get("effects")
                    sem_data_df = s_res.get("sem_data")
                    if params_df is not None and not params_df.empty:
                        generate_sem_effect_forest_plot(params_df, t, out_dir)
                        generate_sem_path_diagram_plot(params_df, t, out_dir)
                    if effects_df is not None and not effects_df.empty:
                        generate_sem_effects_bar_plot(effects_df, t, out_dir)
                    if sem_data_df is not None and not sem_data_df.empty:
                        generate_sem_covariance_plot(sem_data_df, t, out_dir)

                tracker.advance(
                    1,
                    stage=PipelineStage.SEM_ANALYSIS,
                    message=f"Generated SEM figures and reports for {t}",
                    target_name=t,
                )

        # 2. Machine Learning Tournament
        all_models_benchmark_rows: list[dict[str, Any]] = []
        summary_winner_rows: list[dict[str, Any]] = []
        # Fix 1: the confirmatory (winner's-curse-corrected) performance
        # estimate per target -- see
        # manifest["methodology"]["model_selection_architecture"]. This is
        # the primary number that should be quoted as "how well does this
        # analysis generalize", not the exploratory leaderboard above.
        confirmatory_summary_rows_all: list[dict[str, Any]] = []

        groups = (
            dataset_df[config.dataset.id_column].to_numpy()
            if config.cv.use_group_cv
            and config.dataset.id_column
            and config.dataset.id_column in dataset_df.columns
            else None
        )

        manifest["execution"]["current_stage"] = "NESTED_CV"
        write_manifest(out_dir, manifest)

        for target in targets:
            if should_stop and should_stop():
                logger.warning("Pipeline stop requested. Terminating tournament loop.")
                break

            tracker.advance(
                1,
                stage=PipelineStage.PREPROCESSING,
                message=f"Partitioning matrix for {target}",
                target_name=target,
            )

            required_columns = (
                [target]
                + features
                + ([config.dataset.id_column] if config.dataset.id_column else [])
            )
            target_df = dataset_df[required_columns].dropna(subset=[target])
            if len(target_df) < config.cv.min_n:
                logger.warning(
                    "Target %s omitted: N=%d < minimum %d.",
                    target,
                    len(target_df),
                    config.cv.min_n,
                )
                continue

            X = target_df[features].reset_index(drop=True)
            y_raw = target_df[target].to_numpy()
            target_groups = (
                target_df[config.dataset.id_column].to_numpy()
                if groups is not None
                else None
            )

            # Task type was already resolved (and, if ambiguous, explicitly
            # confirmed) in the pre-flight validation pass above -- reuse
            # that decision rather than re-resolving (which would raise
            # again for an ambiguous-but-now-confirmed target, since a
            # bare re-call has no way to know it was confirmed).
            task = AnalysisTaskType(task_type_resolutions[target]["resolved_task_type"])
            y = (
                y_raw
                if task == AnalysisTaskType.CLASSIFICATION
                else y_raw.astype(float)
            )
            prim_metric_name = (
                "accuracy" if task == AnalysisTaskType.CLASSIFICATION else "r2"
            )

            preprocessor = build_preprocessor(X, cats, config.preprocessing)
            target_zoo = get_model_zoo(task, config.cv.random_state, include_full=True)

            if config.models.use_all_models:
                active_specs = dict(target_zoo)
            else:
                model_keys = config.models.selected_models or list(target_zoo.keys())
                active_specs = {}
                for k in model_keys:
                    if k in target_zoo:
                        active_specs[k] = target_zoo[k]
                    else:
                        all_models_benchmark_rows.append(
                            {
                                "target": target,
                                "task_type": task.value,
                                "rank": 999,
                                "model_key": k,
                                "model_name": k.replace("_", " ").title(),
                                "execution_status": "NOT_AVAILABLE",
                                "error_message": f"Estimator '{k}' is not installed or recognized in model zoo.",
                                "is_best": False,
                                "n_subjects": len(y),
                                "primary_metric_mean": np.nan,
                                "primary_metric_std": np.nan,
                                "r2_mean": np.nan,
                                "r2_std": np.nan,
                                "rmse_mean": np.nan,
                                "mae_mean": np.nan,
                                "pearson_r_mean": np.nan,
                                "spearman_rho_mean": np.nan,
                                "accuracy_mean": np.nan,
                                "balanced_acc_mean": np.nan,
                                "f1_macro_mean": np.nan,
                                "roc_auc_mean": np.nan,
                                "mcc_mean": np.nan,
                                "oof_primary_metric": np.nan,
                                "oof_ci_lower": np.nan,
                                "oof_ci_upper": np.nan,
                                "delta_vs_best": np.nan,
                                "delta_ci_lower": np.nan,
                                "delta_ci_upper": np.nan,
                                "p_val_vs_best": np.nan,
                                "complexity_tier": "N/A",
                                "interpretability_tier": "N/A",
                                "recommendation": "Failed",
                                "execution_time_sec": 0.0,
                                "tuning_status": "Failed",
                            }
                        )

            for m_key, m_overrides in config.models.per_model_overrides.items():
                if m_key in active_specs and m_overrides:
                    for p_name, p_val in m_overrides.items():
                        if hasattr(active_specs[m_key].estimator, p_name):
                            setattr(active_specs[m_key].estimator, p_name, p_val)

            if (
                config.models.enable_stacking
                and len(active_specs) >= 2
                and "stacking" not in active_specs
            ):
                active_specs["stacking"] = build_stacking_spec(
                    list(active_specs.values())[: config.models.stacking_top_k],
                    task,
                    config.cv.random_state,
                )

            model_evals = {}
            for m_name, m_spec in active_specs.items():
                if should_stop and should_stop():
                    break

                fold_cb = _make_fold_callback(tracker, target, m_spec)

                try:
                    res = evaluate_model_nested_cv(
                        m_spec,
                        X,
                        y,
                        preprocessor,
                        task,
                        outer_folds=config.cv.outer_folds,
                        inner_folds=config.cv.inner_folds,
                        n_iter=config.cv.n_iter,
                        n_jobs=config.cv.n_jobs,
                        random_state=config.cv.random_state,
                        retain_fold_estimators=True,
                        groups=target_groups,
                        progress_callback=fold_cb,
                        logger=logger,
                        should_stop=should_stop,
                        classification_scoring=config.cv.classification_scoring,
                        imbalance_ratio_threshold=config.cv.imbalance_ratio_threshold,
                        parallel_outer_folds=config.cv.parallel_outer_folds,
                    )
                    if res.cancelled:
                        break
                    if res.execution_status == "FAILED" or not res.fold_estimators:
                        # Every fold failed -- this is not a partial/tolerable
                        # failure, it means the model never produced a usable
                        # fit. Report it as failed rather than silently
                        # including it (with an all-NaN score) in the
                        # leaderboard.
                        logger.error(
                            "[%s] Estimator '%s' failed on all %d fold(s).",
                            target,
                            m_name,
                            res.n_failed_folds,
                        )
                        all_models_benchmark_rows.append(
                            {
                                "target": target,
                                "task_type": task.value,
                                "rank": 999,
                                "model_key": m_name,
                                "model_name": m_spec.display_name,
                                "execution_status": "FAILED",
                                "error_message": (
                                    f"All {res.n_failed_folds} fold(s) failed. "
                                    + "; ".join(
                                        (f.error_message or "")[:120]
                                        for f in res.fold_results
                                        if f.status == "FAILED"
                                    )[:400]
                                ),
                                "is_best": False,
                                "n_subjects": len(y),
                                "n_failed_folds": res.n_failed_folds,
                                "primary_metric_mean": np.nan,
                                "primary_metric_std": np.nan,
                                "r2_mean": np.nan,
                                "r2_std": np.nan,
                                "rmse_mean": np.nan,
                                "mae_mean": np.nan,
                                "pearson_r_mean": np.nan,
                                "spearman_rho_mean": np.nan,
                                "accuracy_mean": np.nan,
                                "balanced_acc_mean": np.nan,
                                "f1_macro_mean": np.nan,
                                "roc_auc_mean": np.nan,
                                "mcc_mean": np.nan,
                                "oof_primary_metric": np.nan,
                                "oof_ci_lower": np.nan,
                                "oof_ci_upper": np.nan,
                                "delta_vs_best": np.nan,
                                "delta_ci_lower": np.nan,
                                "delta_ci_upper": np.nan,
                                "p_val_vs_best": np.nan,
                                "p_val_vs_best_adjusted": np.nan,
                                "complexity_tier": m_spec.complexity_tier,
                                "interpretability_tier": m_spec.interpretability_tier,
                                "recommendation": "Failed",
                                "execution_time_sec": res.total_execution_time,
                                "tuning_status": "Failed",
                            }
                        )
                        continue
                    if res.n_failed_folds > 0:
                        _record_warning(
                            f"[{target}] Estimator '{m_name}': {res.n_failed_folds} "
                            f"of {len(res.fold_results)} outer fold(s) failed and "
                            f"were excluded from this model's aggregated metrics "
                            f"(remaining folds' results are still reported)."
                        )
                    model_evals[m_name] = (m_spec, res)
                except Exception as exc:
                    logger.error("[%s] Estimator '%s' error: %s", target, m_name, exc)
                    all_models_benchmark_rows.append(
                        {
                            "target": target,
                            "task_type": task.value,
                            "rank": 999,
                            "model_key": m_name,
                            "model_name": m_spec.display_name,
                            "execution_status": "FAILED",
                            "error_message": f"{type(exc).__name__}: {exc}",
                            "is_best": False,
                            "n_subjects": len(y),
                            "primary_metric_mean": np.nan,
                            "primary_metric_std": np.nan,
                            "r2_mean": np.nan,
                            "r2_std": np.nan,
                            "rmse_mean": np.nan,
                            "mae_mean": np.nan,
                            "pearson_r_mean": np.nan,
                            "spearman_rho_mean": np.nan,
                            "accuracy_mean": np.nan,
                            "balanced_acc_mean": np.nan,
                            "f1_macro_mean": np.nan,
                            "roc_auc_mean": np.nan,
                            "mcc_mean": np.nan,
                            "oof_primary_metric": np.nan,
                            "oof_ci_lower": np.nan,
                            "oof_ci_upper": np.nan,
                            "delta_vs_best": np.nan,
                            "delta_ci_lower": np.nan,
                            "delta_ci_upper": np.nan,
                            "p_val_vs_best": np.nan,
                            "complexity_tier": m_spec.complexity_tier,
                            "interpretability_tier": m_spec.interpretability_tier,
                            "recommendation": "Failed",
                            "execution_time_sec": 0.0,
                            "tuning_status": "Failed",
                        }
                    )

            if not model_evals:
                logger.error("No estimators converged for target %s.", target)
                continue

            def _success_scores(res: Any) -> list[float]:
                return [
                    f.primary_metric
                    for f in res.fold_results
                    if f.status == "SUCCESS" and np.isfinite(f.primary_metric)
                ]

            sorted_models = sorted(
                model_evals.keys(),
                key=lambda k: np.mean(_success_scores(model_evals[k][1])),
                reverse=True,
            )
            best_name = sorted_models[0]
            best_spec, best_res = model_evals[best_name]
            best_success_folds = {
                f.fold: f.primary_metric
                for f in best_res.fold_results
                if f.status == "SUCCESS" and np.isfinite(f.primary_metric)
            }
            n_tr = best_res.fold_results[0].n_train
            n_te = best_res.fold_results[0].n_test

            tracker.advance(
                1,
                stage=PipelineStage.NESTED_CV,
                message=f"Pairwise hypothesis testing for {target}",
                target_name=target,
            )

            for rank_idx, m_name in enumerate(sorted_models, start=1):
                m_spec_i, m_res_i = model_evals[m_name]
                scores_i = _success_scores(m_res_i)
                mean_cv = float(np.mean(scores_i)) if scores_i else float("nan")
                std_cv = float(np.std(scores_i)) if scores_i else float("nan")
                tuning_failed = any(
                    "_tuning_failed" in f.best_params for f in m_res_i.fold_results
                )

                all_fold_m = [
                    f.metrics for f in m_res_i.fold_results if f.status == "SUCCESS"
                ]

                if task == AnalysisTaskType.REGRESSION:
                    reg_metrics = {
                        k: float(np.mean([m.get(k, np.nan) for m in all_fold_m]))
                        if all_fold_m
                        else float("nan")
                        for k in ["rmse", "mae", "pearson_r", "spearman_rho"]
                    }
                    clf_metrics = {
                        k: np.nan
                        for k in [
                            "accuracy",
                            "balanced_acc",
                            "f1_macro",
                            "roc_auc",
                            "mcc",
                        ]
                    }
                else:
                    clf_metrics = {
                        k: float(np.mean([m.get(k, np.nan) for m in all_fold_m]))
                        if all_fold_m
                        else float("nan")
                        for k in [
                            "accuracy",
                            "balanced_acc",
                            "f1_macro",
                            "roc_auc",
                            "mcc",
                        ]
                    }
                    reg_metrics = {
                        k: np.nan for k in ["rmse", "mae", "pearson_r", "spearman_rho"]
                    }

                # A failed fold leaves its rows unset (NaN/None sentinel) in
                # oof_predictions -- mask those out before computing
                # bootstrap CIs so a partial failure cannot silently
                # produce a spurious or NaN interval (see
                # mask_valid_oof_predictions docstring).
                valid_i = mask_valid_oof_predictions(y, m_res_i.oof_predictions, task)
                pt_i, ci_lo_i, ci_hi_i = bootstrap_metric_ci(
                    y[valid_i],
                    m_res_i.oof_predictions[valid_i],
                    prim_metric_name,
                    task,
                    n_boot=config.cv.bootstrap_n,
                    random_state=config.cv.random_state,
                    groups=target_groups[valid_i] if target_groups is not None else None,
                )

                p_val_vs_best = np.nan
                delta_val, delta_ci_lo, delta_ci_hi = 0.0, 0.0, 0.0
                if rank_idx > 1:
                    # Pair per-fold scores by fold index -- if model A and
                    # the best model failed on *different* folds, only
                    # folds where BOTH succeeded form a valid paired
                    # comparison for the corrected resampled t-test.
                    i_success_folds = {
                        f.fold: f.primary_metric
                        for f in m_res_i.fold_results
                        if f.status == "SUCCESS" and np.isfinite(f.primary_metric)
                    }
                    common_folds = sorted(
                        set(best_success_folds) & set(i_success_folds)
                    )
                    if len(common_folds) >= 2:
                        paired_best = [best_success_folds[f] for f in common_folds]
                        paired_i = [i_success_folds[f] for f in common_folds]
                        _, p_val_vs_best = compute_corrected_resampled_ttest(
                            paired_best, paired_i, n_tr, n_te
                        )
                    valid_both = mask_valid_oof_predictions(
                        y, best_res.oof_predictions, task
                    ) & mask_valid_oof_predictions(y, m_res_i.oof_predictions, task)
                    delta_val, delta_ci_lo, delta_ci_hi = (
                        bootstrap_paired_difference_ci(
                            y[valid_both],
                            best_res.oof_predictions[valid_both],
                            m_res_i.oof_predictions[valid_both],
                            metric_name=prim_metric_name,
                            task_type=task,
                            n_boot=config.cv.bootstrap_n,
                            random_state=config.cv.random_state,
                            groups=target_groups[valid_both]
                            if target_groups is not None
                            else None,
                        )
                    )

                recommendation_text = (
                    "Primary"
                    if rank_idx == 1
                    or (m_spec_i.complexity_tier == "Low" and p_val_vs_best > 0.05)
                    else ("Secondary" if rank_idx <= 3 else "Sensitivity")
                )

                row_data = {
                    "target": target,
                    "task_type": task.value,
                    "rank": rank_idx,
                    "model_key": m_name,
                    "model_name": m_spec_i.display_name,
                    "execution_status": "SUCCESS",
                    "error_message": None,
                    "is_best": (rank_idx == 1),
                    "n_subjects": len(y),
                    "primary_metric_mean": mean_cv,
                    "primary_metric_std": std_cv,
                    "r2_mean": mean_cv
                    if task == AnalysisTaskType.REGRESSION
                    else np.nan,
                    "r2_std": std_cv if task == AnalysisTaskType.REGRESSION else np.nan,
                    "rmse_mean": reg_metrics["rmse"],
                    "mae_mean": reg_metrics["mae"],
                    "pearson_r_mean": reg_metrics["pearson_r"],
                    "spearman_rho_mean": reg_metrics["spearman_rho"],
                    "accuracy_mean": clf_metrics["accuracy"],
                    "balanced_acc_mean": clf_metrics["balanced_acc"],
                    "f1_macro_mean": clf_metrics["f1_macro"],
                    "roc_auc_mean": clf_metrics["roc_auc"],
                    "mcc_mean": clf_metrics["mcc"],
                    "oof_primary_metric": pt_i,
                    "oof_ci_lower": ci_lo_i,
                    "oof_ci_upper": ci_hi_i,
                    "delta_vs_best": delta_val,
                    "delta_ci_lower": delta_ci_lo,
                    "delta_ci_upper": delta_ci_hi,
                    "p_val_vs_best": p_val_vs_best,
                    "p_val_vs_best_adjusted": np.nan,  # filled in below, post-loop
                    "n_failed_folds": m_res_i.n_failed_folds,
                    "complexity_tier": m_spec_i.complexity_tier,
                    "interpretability_tier": m_spec_i.interpretability_tier,
                    "recommendation": recommendation_text,
                    "execution_time_sec": m_res_i.total_execution_time,
                    "tuning_status": "Failed" if tuning_failed else "OK",
                }
                all_models_benchmark_rows.append(row_data)
                if rank_idx == 1:
                    summary_winner_rows.append(row_data)

            # Fix 2 (ML model comparisons): apply multiplicity correction
            # across the family of "vs. best model" pairwise p-values for
            # THIS target's leaderboard (an uncorrected p < .05 threshold
            # per comparison inflates the family-wise false-positive rate
            # when, as here, many candidates are each compared to the
            # best). This is explicitly an EXPLORATORY/DESCRIPTIVE
            # comparison, not the confirmatory performance estimate -- see
            # the confirmatory model-selection section below.
            this_target_rows = [
                r
                for r in all_models_benchmark_rows
                if r["target"] == target and r["execution_status"] == "SUCCESS"
            ]
            raw_p_family = np.array(
                [r["p_val_vs_best"] for r in this_target_rows], dtype=float
            )
            adjusted_p_family = apply_multiplicity_correction(
                raw_p_family, config.cv.multiplicity_correction
            )
            for row, p_adj in zip(this_target_rows, adjusted_p_family, strict=False):
                row["p_val_vs_best_adjusted"] = (
                    float(p_adj) if np.isfinite(p_adj) else np.nan
                )

            manifest["methodology"]["multiplicity_correction"][
                f"ml_comparisons_{target}"
            ] = {
                "method": config.cv.multiplicity_correction,
                "family_size": int(np.sum(np.isfinite(raw_p_family))),
            }

            # Leaderboard Plot
            tracker.advance(
                1,
                stage=PipelineStage.NESTED_CV,
                message=f"Rendering leaderboard for {target}",
                target_name=target,
            )
            fig, ax = plt.subplots(
                figsize=(8.5, max(3.8, 0.48 * len(model_evals))), facecolor="#ffffff"
            )
            ax.set_facecolor("#ffffff")
            names = [model_evals[k][0].display_name for k in sorted_models[::-1]]
            means = [
                np.mean(_success_scores(model_evals[k][1]))
                for k in sorted_models[::-1]
            ]
            stds = [
                np.std(_success_scores(model_evals[k][1]))
                for k in sorted_models[::-1]
            ]
            colors = [
                "#5E81AC" if k != best_name else "#A3BE8C" for k in sorted_models[::-1]
            ]
            ax.barh(
                names,
                means,
                xerr=stds,
                color=colors,
                ecolor="#4C566A",
                capsize=4.0,
                edgecolor="#2E3440",
                linewidth=0.8,
            )
            ax.set_xlabel(
                f"Mean Nested-CV Score ({prim_metric_name.upper()} ± SD)",
                fontsize=9.5,
                color="#2E3440",
            )
            ax.set_title(
                f"Algorithm Tournament Leaderboard: {target}",
                fontsize=11.5,
                fontweight="bold",
                color="#2E3440",
                pad=12,
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", linestyle="--", alpha=0.7, color="#E5E9F0")
            fig.tight_layout()
            save_plot(
                fig, out_dir / "figures" / "performance" / f"{target}__model_comparison"
            )

            # Diagnostics & Calibration
            tracker.advance(
                1,
                stage=PipelineStage.NESTED_CV,
                message=f"Generating calibration & diagnostics for {target}",
                target_name=target,
            )
            if task == AnalysisTaskType.REGRESSION:
                generate_calibration_plot(
                    y, best_res.oof_predictions, target, best_spec.display_name, out_dir
                )
                # Fix 9: the diagnostic-plot covariate overlay column is
                # user-configurable (OutputConfig.diagnostic_covariate_column)
                # rather than hardcoded to "age" -- a general-purpose
                # analysis platform must not silently privilege one
                # domain-specific column name, and datasets without an
                # "age" column previously got no covariate overlay at all
                # with no way to request a different one.
                cov_col = config.output_config.diagnostic_covariate_column
                if cov_col and cov_col in target_df.columns:
                    cov_series = target_df[cov_col]
                    cov_name = cov_col
                else:
                    if cov_col:
                        _record_warning(
                            f"[{target}] output.diagnostic_covariate_column="
                            f"'{cov_col}' was not found in this target's data; "
                            f"the diagnostic plot's covariate overlay is omitted "
                            f"for this target."
                        )
                    cov_series = None
                    cov_name = "Covariate"
                generate_regression_diagnostic_plots(
                    y,
                    best_res.oof_predictions,
                    target,
                    best_spec.display_name,
                    out_dir,
                    covariate_series=cov_series,
                    covariate_name=cov_name,
                )
            else:
                generate_classification_diagnostic_plots(
                    y,
                    best_res.oof_predictions,
                    best_res.oof_probabilities,
                    target,
                    best_spec.display_name,
                    out_dir,
                )

            if best_res.learning_curve:
                generate_learning_curve_plot(
                    best_res.learning_curve, target, best_spec.display_name, out_dir
                )

            if (
                best_res.linear_coefficients is not None
                and not best_res.linear_coefficients.empty
            ):
                best_res.linear_coefficients.to_csv(
                    out_dir
                    / "feature_importance"
                    / f"{target}_linear_standardized_coefficients.csv",
                    index=False,
                )

            # Attribution & SHAP
            tracker.advance(
                1,
                stage=PipelineStage.EXPLAINABILITY,
                message=f"Computing attribution for {target}",
                target_name=target,
            )
            attribution_spec = best_spec
            attribution_estimators = best_res.fold_estimators
            if not best_spec.is_tree_based:
                tree_eval = next(
                    (
                        (m_sp, m_res)
                        for m_k, (m_sp, m_res) in model_evals.items()
                        if m_sp.is_tree_based
                    ),
                    None,
                )
                if tree_eval:
                    attribution_spec, attribution_res = tree_eval
                    attribution_estimators = attribution_res.fold_estimators

            cv_attrib = compute_cv_stable_attribution(
                attribution_spec,
                attribution_estimators,
                X,
                y,
                task,
                n_jobs=config.cv.n_jobs,
                random_state=config.cv.random_state,
                compute_shap=not config.cv.skip_shap,
                permutation_n_repeats=config.cv.permutation_repeats,
                logger=logger,
            )
            agg_importance = aggregate_fold_importance(
                cv_attrib.fold_shap_records or cv_attrib.fold_importance_records
            )
            agg_importance.to_csv(
                out_dir / "feature_importance" / f"{target}_importance_cv.csv",
                index=False,
            )

            if not agg_importance.empty:
                fig_bar = plot_feature_importance_bar(
                    agg_importance, target, best_spec.display_name, top_k=15
                )
                save_plot(
                    fig_bar,
                    out_dir
                    / "figures"
                    / "explainability"
                    / f"{target}__top_features_cv",
                )
                fig_stab = plot_attribution_stability(agg_importance, target, top_k=12)
                save_plot(
                    fig_stab,
                    out_dir
                    / "figures"
                    / "explainability"
                    / f"{target}__attribution_stability",
                )

            if (
                cv_attrib.shap_values_array is not None
                and cv_attrib.X_test_transformed is not None
            ):
                fig_bee = plot_shap_beeswarm(
                    cv_attrib.shap_values_array,
                    cv_attrib.X_test_transformed,
                    cv_attrib.feature_names,
                    target,
                    attribution_spec.display_name,
                    top_k=15,
                    logger=logger,
                )
                if fig_bee is not None:
                    save_plot(
                        fig_bee,
                        out_dir
                        / "figures"
                        / "explainability"
                        / f"{target}__shap_beeswarm",
                    )

            # --- Fix 1: confirmatory model-selection procedure ---------
            # Evaluates the *entire* "compare candidates, pick a winner"
            # procedure via its own nested nested-CV (see
            # run_confirmatory_model_selection_cv docstring). This is the
            # primary, confirmatory performance estimate for this target;
            # the exploratory leaderboard above remains available as a
            # separate, explicitly-labeled descriptive comparison.
            tracker.advance(
                1,
                stage=PipelineStage.NESTED_CV,
                message=f"Running confirmatory model-selection procedure for {target}",
                target_name=target,
            )
            confirmatory_res = None
            try:
                confirmatory_res = run_confirmatory_model_selection_cv(
                    active_specs,
                    X,
                    y,
                    preprocessor,
                    task,
                    outer_folds=config.cv.outer_folds,
                    selection_folds=config.cv.selection_folds,
                    inner_folds=config.cv.inner_folds,
                    n_iter=config.cv.n_iter,
                    random_state=config.cv.random_state,
                    classification_scoring=config.cv.classification_scoring,
                    imbalance_ratio_threshold=config.cv.imbalance_ratio_threshold,
                    groups=target_groups,
                    logger=logger,
                    should_stop=should_stop,
                )
            except Exception as exc:
                logger.error(
                    "[%s] Confirmatory model-selection procedure failed: %s",
                    target,
                    exc,
                )
                _record_warning(
                    f"[{target}] Confirmatory model-selection procedure failed "
                    f"({type(exc).__name__}: {str(exc)[:200]}) -- only the "
                    f"exploratory leaderboard is available for this target; "
                    f"treat its 'best model' figure as descriptive only, not "
                    f"as a bias-corrected confirmatory estimate."
                )

            confirmatory_manifest_entry: dict[str, Any] = {"status": "unavailable"}
            if confirmatory_res is not None and not confirmatory_res.cancelled:
                conf_valid = mask_valid_oof_predictions(
                    y, confirmatory_res.oof_predictions, task
                )
                if conf_valid.sum() > 0:
                    conf_pt, conf_ci_lo, conf_ci_hi = bootstrap_metric_ci(
                        y[conf_valid],
                        confirmatory_res.oof_predictions[conf_valid],
                        confirmatory_res.primary_metric_name,
                        task,
                        n_boot=config.cv.bootstrap_n,
                        random_state=config.cv.random_state,
                        groups=target_groups[conf_valid]
                        if target_groups is not None
                        else None,
                    )
                else:
                    conf_pt, conf_ci_lo, conf_ci_hi = np.nan, np.nan, np.nan

                confirmatory_row = {
                    "target": target,
                    "task_type": task.value,
                    "execution_status": confirmatory_res.execution_status,
                    "primary_metric_name": confirmatory_res.primary_metric_name,
                    "primary_metric_mean": confirmatory_res.final_metrics.get(
                        confirmatory_res.primary_metric_name, np.nan
                    ),
                    "oof_bootstrap_point": conf_pt,
                    "oof_ci_lower": conf_ci_lo,
                    "oof_ci_upper": conf_ci_hi,
                    "oof_ci_method": BOOTSTRAP_CI_METHOD_DESCRIPTION,
                    "n_failed_folds": confirmatory_res.n_failed_folds,
                    "n_outer_folds": len(confirmatory_res.fold_records),
                    "n_candidates_compared": len(active_specs),
                    "execution_time_sec": confirmatory_res.total_execution_time,
                    "most_frequently_selected_model": max(
                        confirmatory_res.selection_stability,
                        key=lambda k: confirmatory_res.selection_stability[k],
                    )
                    if confirmatory_res.selection_stability
                    else None,
                }
                confirmatory_summary_rows_all.append(confirmatory_row)

                fold_audit_records = [
                    {
                        "fold": fr.fold,
                        "n_train": fr.n_train,
                        "n_test": fr.n_test,
                        "selected_model": fr.selected_model,
                        "selected_model_params": fr.selected_model_params,
                        "selected_model_test_metric": fr.selected_model_test_metric,
                        "candidate_selection_scores": fr.candidate_selection_scores,
                        "status": fr.status,
                        "error_message": fr.error_message,
                        "train_time_sec": fr.train_time_sec,
                        "selection_time_sec": fr.selection_time_sec,
                    }
                    for fr in confirmatory_res.fold_records
                ]
                (
                    out_dir
                    / "metrics"
                    / f"{target}_confirmatory_selection_audit.json"
                ).write_text(
                    json.dumps(
                        {
                            "target": target,
                            "primary_metric_name": confirmatory_res.primary_metric_name,
                            "final_metrics": confirmatory_res.final_metrics,
                            "selection_stability": confirmatory_res.selection_stability,
                            "fold_records": fold_audit_records,
                        },
                        indent=2,
                        default=str,
                    ),
                    encoding="utf-8",
                )
                confirmatory_manifest_entry = {
                    "status": confirmatory_res.execution_status,
                    "primary_metric_name": confirmatory_res.primary_metric_name,
                    "primary_metric_value": confirmatory_row["primary_metric_mean"],
                    "n_failed_folds": confirmatory_res.n_failed_folds,
                    "selection_stability": confirmatory_res.selection_stability,
                }
            manifest["methodology"].setdefault("confirmatory_model_selection", {})[
                target
            ] = confirmatory_manifest_entry

            manifest["execution"]["completed_targets"].append(target)
            manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            write_manifest(out_dir, manifest)

        all_models_df = pd.DataFrame(all_models_benchmark_rows)
        winners_df = pd.DataFrame(summary_winner_rows)
        confirmatory_df = pd.DataFrame(confirmatory_summary_rows_all)

        all_models_df.to_csv(
            out_dir / "metrics" / "all_models_benchmark.csv", index=False
        )
        winners_df.to_csv(
            out_dir / "metrics" / "model_performance_summary.csv", index=False
        )
        confirmatory_df.to_csv(
            out_dir / "metrics" / "confirmatory_model_selection_summary.csv",
            index=False,
        )

        elapsed_time = time.time() - t0
        tracker.advance(
            1,
            stage=PipelineStage.REPORTING,
            message="Rendering self-contained HTML research report",
        )

        generate_html_report(
            all_models_df=all_models_df,
            out_dir=out_dir,
            dataset_profile=profile,
            dataset_df=dataset_df,
            config=config,
            sem_result_dict=sem_result_dict,
            execution_time=elapsed_time,
            confirmatory_df=confirmatory_df,
            methodology=manifest["methodology"],
        )

        manifest["status"] = (
            "cancelled" if (should_stop and should_stop()) else "completed"
        )
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["elapsed_seconds"] = round(elapsed_time, 2)
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["execution"]["current_stage"] = (
            "CANCELLED" if (should_stop and should_stop()) else "COMPLETED"
        )
        write_manifest(out_dir, manifest)

        tracker.finish(
            f"Workflow completed across {len(targets)} targets in {elapsed_time:.2f}s."
        )

    except Exception as exc:
        elapsed_fail = round(time.time() - t0, 2)
        manifest["status"] = "failed"
        manifest["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        manifest["elapsed_seconds"] = elapsed_fail
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest["execution"]["current_stage"] = "FAILED"
        manifest["execution"]["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(out_dir, manifest)
        logger.exception("Pipeline execution failed: %s", exc)
        raise
