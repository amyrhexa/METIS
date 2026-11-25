"""Core scientific analytics, model training, cross-validation, SEM, and diagnostics."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from core.data_loader import (
    TaskTypeResolution,
    clean_dataframe,
    inspect_dataset,
    load_dataset,
    resolve_task_type,
    resolve_task_type_detailed,
)
from core.evaluation import (
    apply_multiplicity_correction,
    bootstrap_metric_ci,
    bootstrap_paired_difference_ci,
    calculate_comprehensive_metrics,
    calculate_single_metric,
    mask_valid_oof_predictions,
)
from core.interpretation import (
    aggregate_fold_importance,
    compute_cv_stable_attribution,
    plot_attribution_stability,
    plot_feature_importance_bar,
    plot_shap_beeswarm,
    plot_shap_dependence,
)
from core.modeling import (
    ModelSpec,
    build_stacking_spec,
    compute_corrected_resampled_ttest,
    evaluate_model_nested_cv,
    get_model_zoo,
)
from core.pipeline_runner import execute_pipeline
from core.preprocessing import (
    ConstantColumnFilter,
    MissingnessFilter,
    build_preprocessor,
)
from core.progress import PipelineStage, ProgressTracker
from core.reporting import generate_html_report, generate_latex_tables
from core.sem import (
    assess_sem_convergence,
    run_sem_target,
    validate_missing_data_method,
)

try:
    __version__ = version("METIS")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = [
    "ConstantColumnFilter",
    "MissingnessFilter",
    "ModelSpec",
    "PipelineStage",
    "ProgressTracker",
    "TaskTypeResolution",
    "__version__",
    "aggregate_fold_importance",
    "apply_multiplicity_correction",
    "assess_sem_convergence",
    "bootstrap_metric_ci",
    "bootstrap_paired_difference_ci",
    "build_preprocessor",
    "build_stacking_spec",
    "calculate_comprehensive_metrics",
    "calculate_single_metric",
    "clean_dataframe",
    "compute_corrected_resampled_ttest",
    "compute_cv_stable_attribution",
    "evaluate_model_nested_cv",
    "execute_pipeline",
    "generate_html_report",
    "generate_latex_tables",
    "get_model_zoo",
    "inspect_dataset",
    "load_dataset",
    "mask_valid_oof_predictions",
    "plot_attribution_stability",
    "plot_feature_importance_bar",
    "plot_shap_beeswarm",
    "plot_shap_dependence",
    "resolve_task_type",
    "resolve_task_type_detailed",
    "run_sem_target",
    "validate_missing_data_method",
]
