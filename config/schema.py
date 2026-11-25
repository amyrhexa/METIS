"""Configuration schemas, presets, and validation models."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

# Bumped from 1.1.0 -> 2.0.0 as part of the scientific-remediation rewrite:
# the manifest now records requested-vs-actual methodology decisions
# (task-type resolution, missing-data method, model-selection architecture,
# multiplicity correction, etc.) that did not exist in 1.1.0 manifests.
# Old manifests still load (see PipelineConfig.load_manifest / _from_dict),
# they simply predate these additional audit fields.
MANIFEST_SCHEMA_VERSION = "2.0.0"

T = TypeVar("T")

#: The only missing-data methods METIS will silently accept. Anything else
#: raises SEMConfigurationError rather than falling back to single-value
#: imputation (see core/sem.py::validate_missing_data_method).
ALLOWED_MISSING_DATA_METHODS: frozenset[str] = frozenset({"fiml", "listwise"})

#: Multiplicity-correction strategies available for SEM structural-path
#: p-values and for pairwise model-comparison p-values.
ALLOWED_MULTIPLICITY_METHODS: frozenset[str] = frozenset(
    {"holm", "bonferroni", "fdr_bh", "none"}
)

#: Inner-CV classification scoring strategies. "auto" inspects the training
#: fold's class balance and switches from accuracy to balanced_accuracy when
#: the majority:minority ratio exceeds `CVConfig.imbalance_ratio_threshold`.
ALLOWED_CLASSIFICATION_SCORING: frozenset[str] = frozenset(
    {"auto", "accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"}
)


class AnalysisTaskType(StrEnum):
    AUTO = "auto"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"


@dataclass
class DatasetConfig:
    input_path: str = ""
    target_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] | None = None
    numerical_columns: list[str] = field(default_factory=list)
    active_columns: list[str] = field(default_factory=list)
    id_column: str | None = None
    task_type: AnalysisTaskType = AnalysisTaskType.REGRESSION
    # Per-target override of `task_type`. Keys are target column names,
    # values are "regression" | "classification". Takes precedence over the
    # global `task_type` for that specific column. Introduced so that a
    # dataset with multiple targets is not forced to analyze all of them
    # under the same task type.
    task_type_overrides: dict[str, str] = field(default_factory=dict)
    # Targets for which the user has explicitly acknowledged that AUTO
    # cardinality-based task-type resolution is ambiguous (numeric target
    # with ~3-10 unique values) and has chosen to proceed with the default
    # resolution anyway. See core.exceptions.AmbiguousTaskTypeError and
    # core.data_loader.resolve_task_type_detailed.
    confirmed_ambiguous_targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["task_type"] = self.task_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetConfig:
        return _from_dict(cls, data)


@dataclass
class PreprocessingConfig:
    missingness_threshold: float = 0.25
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    scale_numeric: bool = True
    scaler_type: str = "standard"
    handle_unknown_categories: str = "ignore"
    drop_constant_columns: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingConfig:
        return _from_dict(cls, data)


@dataclass
class CVConfig:
    outer_folds: int = 5
    inner_folds: int = 3
    n_iter: int = 25
    # `n_jobs` semantics (Fix 10 / n_jobs honesty): this value is used for
    # (a) permutation-importance parallelism (unchanged from before), and
    # (b) outer-fold parallelism *only if* `parallel_outer_folds=True`. It
    # is intentionally NOT threaded into per-fold hyperparameter search
    # (RandomizedSearchCV always runs with n_jobs=1 internally) to avoid
    # CPU oversubscription when outer folds themselves are parallelized.
    n_jobs: int = -1
    # When True, the outer CV folds of `evaluate_model_nested_cv` and of
    # the confirmatory model-selection procedure are executed concurrently
    # via joblib, using `n_jobs` workers. Each outer fold is a fully
    # independent computation (its own clone()'d preprocessor/estimator,
    # its own fold-local random_state derivation), so parallel execution
    # does not change results — only wall-clock time. Default is False so
    # that out-of-the-box behavior remains single-threaded/deterministic
    # and does not risk oversubscribing shared machines.
    parallel_outer_folds: bool = False
    bootstrap_n: int = 1000
    min_n: int = 25
    random_state: int = 42
    fast_mode: bool = False
    skip_shap: bool = False
    use_group_cv: bool = False
    permutation_repeats: int = 20
    # Number of within-outer-training-fold CV splits used by the
    # confirmatory model-selection procedure (Fix 1) to rank candidate
    # models using default hyperparameters before the winning candidate is
    # tuned and refit on the full outer-training partition. Kept separate
    # from `inner_folds` (which is used for the winning candidate's own
    # hyperparameter search) to bound the cost of comparing many
    # candidates.
    selection_folds: int = 3
    # Classification inner-CV scoring strategy (Fix 6). One of
    # ALLOWED_CLASSIFICATION_SCORING. "auto" inspects the outer-training
    # fold's class balance and uses `balanced_accuracy` instead of
    # `accuracy` when the majority:minority ratio exceeds
    # `imbalance_ratio_threshold`.
    classification_scoring: str = "auto"
    imbalance_ratio_threshold: float = 1.5
    # Multiplicity-correction method applied to the family of pairwise
    # "vs. best model" comparison p-values (Fix 2). One of
    # ALLOWED_MULTIPLICITY_METHODS.
    multiplicity_correction: str = "holm"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CVConfig:
        return _from_dict(cls, data)

    def validate(self) -> list[str]:
        """Returns a list of human-readable validation error strings (empty
        if the configuration is valid). Does not raise, so callers (GUI/CLI)
        can decide how to surface multiple problems at once."""
        errors: list[str] = []
        if self.classification_scoring not in ALLOWED_CLASSIFICATION_SCORING:
            errors.append(
                f"cv.classification_scoring='{self.classification_scoring}' is "
                f"not one of {sorted(ALLOWED_CLASSIFICATION_SCORING)}."
            )
        if self.multiplicity_correction not in ALLOWED_MULTIPLICITY_METHODS:
            errors.append(
                f"cv.multiplicity_correction='{self.multiplicity_correction}' is "
                f"not one of {sorted(ALLOWED_MULTIPLICITY_METHODS)}."
            )
        if self.selection_folds < 2:
            errors.append("cv.selection_folds must be >= 2.")
        if self.outer_folds < 2:
            errors.append("cv.outer_folds must be >= 2.")
        if self.inner_folds < 2:
            errors.append("cv.inner_folds must be >= 2.")
        if self.imbalance_ratio_threshold <= 1.0:
            errors.append("cv.imbalance_ratio_threshold must be > 1.0.")
        return errors


@dataclass
class ModelZooConfig:
    selected_models: list[str] = field(default_factory=list)
    use_all_models: bool = False
    enable_stacking: bool = True
    stacking_top_k: int = 3
    per_model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelZooConfig:
        return _from_dict(cls, data)


@dataclass
class SEMTargetConfig:
    target: str = ""
    enabled: bool = True
    feature_columns: list[str] = field(default_factory=list)
    covariates: list[str] = field(default_factory=list)
    mediators: list[str] = field(default_factory=list)
    latent_variables: dict[str, list[str]] = field(default_factory=dict)
    custom_model_syntax: str | None = None
    # Must be one of ALLOWED_MISSING_DATA_METHODS ("fiml" | "listwise").
    # An unrecognized value is a hard configuration error (see
    # core.sem.validate_missing_data_method) — METIS never silently
    # substitutes single-value (e.g. median) imputation for a requested
    # method it doesn't recognize.
    missing_data_method: str = "fiml"
    standardize: bool = True
    estimator: str = "ML"
    robust_se: bool = False
    bootstrap: bool = True
    bootstrap_n: int = 2000
    random_state: int = 42
    min_subjects: int = 30
    # Multiplicity-correction method applied independently to (a) the
    # family of structural-path p-values and (b) the family of mediation
    # indirect-effect p-values, within this SEM run (Fix 2). One of
    # ALLOWED_MULTIPLICITY_METHODS. Raw p-values are always retained
    # alongside the adjusted ones; significance stars are based on the
    # adjusted p-value.
    multiplicity_correction: str = "holm"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(
        cls, target_or_data: str | dict[str, Any], data: dict[str, Any] | None = None
    ) -> SEMTargetConfig:
        if isinstance(target_or_data, str):
            target = target_or_data
            payload = data or {}
        else:
            target = ""
            payload = target_or_data
        obj = _from_dict(cls, payload)
        if target:
            obj.target = target
        return obj

    def validate(self) -> list[str]:
        """Returns a list of human-readable validation error strings (empty
        if valid). Raising happens at the point of use (core.sem) so that
        the exact `SEMConfigurationError` type is raised there; this method
        exists so the GUI/CLI can pre-validate without importing core.sem."""
        errors: list[str] = []
        method = (self.missing_data_method or "").strip().lower()
        if method not in ALLOWED_MISSING_DATA_METHODS:
            errors.append(
                f"sem missing_data_method='{self.missing_data_method}' is not "
                f"one of {sorted(ALLOWED_MISSING_DATA_METHODS)}."
            )
        if self.multiplicity_correction not in ALLOWED_MULTIPLICITY_METHODS:
            errors.append(
                f"sem multiplicity_correction='{self.multiplicity_correction}' is "
                f"not one of {sorted(ALLOWED_MULTIPLICITY_METHODS)}."
            )
        return errors


@dataclass
class ReportingConfig:
    generate_html: bool = True
    generate_latex: bool = True
    generate_csv_tables: bool = True
    generate_json_manifest: bool = True
    title: str = "Academic Machine Learning & Structural Equation Modeling Report"
    dpi: int = 300
    color_palette: str = "nord"
    embed_interactive_plots: bool = True

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReportingConfig:
        return _from_dict(cls, data)


@dataclass
class OutputConfig:
    output_dir: str = "results"
    report_title: str = (
        "Academic Machine Learning & Structural Equation Modeling Report"
    )
    mode: str = "exploratory"
    table_palette: str = "nord"
    table_style: str = "publication"
    decimal_precision: int = 3
    generate_html: bool = True
    generate_latex: bool = True
    generate_csv: bool = True
    generate_json_manifest: bool = True
    figure_dpi: int = 300
    figure_formats: list[str] = field(default_factory=lambda: ["png", "pdf"])
    # Optional column name to overlay as a covariate in the regression
    # residual-diagnostic plot (previously hardcoded to "age" — Fix 9).
    # None disables the covariate-overlay panel (falls back to the
    # scale-location plot). Must name a column present in the target's
    # feature/dataset frame to have any effect; unresolved names are
    # ignored with a logged warning rather than silently substituting a
    # different column.
    diagnostic_covariate_column: str | None = None
    include_ml_sections: dict[str, bool] = field(
        default_factory=lambda: {
            "leaderboard": True,
            "model_comparison": True,
            "calibration": True,
            "regression_diagnostics": True,
            "classification_diagnostics": True,
            "feature_importance": True,
            "attribution_stability": True,
            "shap_beeswarm": True,
            "shap_dependence": True,
            "learning_curve": True,
            "linear_coefficients": True,
            "execution_status": True,
        }
    )
    include_sem_sections: dict[str, bool] = field(
        default_factory=lambda: {
            "structural_paths": True,
            "mediation_effects": True,
            "variance_explained": True,
            "fit_indices": True,
            "reliability_validity": True,
            "sem_narrative": True,
            "path_diagram": True,
            "forest_plot": True,
            "mediation_plot": True,
            "covariance_heatmap": True,
        }
    )
    include_metrics: dict[str, bool] = field(
        default_factory=lambda: {
            "r2": True,
            "rmse": True,
            "mae": True,
            "medae": True,
            "max_error": True,
            "explained_variance": True,
            "mape": True,
            "pearson_r": True,
            "spearman_rho": True,
            "calibration_slope": True,
            "accuracy": True,
            "balanced_accuracy": True,
            "f1_macro": True,
            "f1_weighted": True,
            "precision_macro": True,
            "recall_macro": True,
            "mcc": True,
            "cohen_kappa": True,
            "roc_auc": True,
            "log_loss": True,
            "bootstrap_ci": True,
            "paired_difference": True,
            "corrected_ttest": True,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputConfig:
        return _from_dict(cls, data)


def _from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    """Recursively instantiates dataclasses from dictionary payloads."""
    if not isinstance(data, dict):
        return cls()

    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}

    for name, _ in fields.items():
        if name not in data:
            if name == "active_columns" and "selected_columns" in data:
                val = data.get("selected_columns")
            else:
                continue
        else:
            val = data[name]

        if name == "task_type":
            if isinstance(val, str) and val in AnalysisTaskType._value2member_map_:
                kwargs[name] = AnalysisTaskType(val)
            elif isinstance(val, AnalysisTaskType):
                kwargs[name] = val
            else:
                kwargs[name] = AnalysisTaskType.REGRESSION
        elif name == "sem_configs" and isinstance(val, dict):
            kwargs[name] = {
                k: _from_dict(SEMTargetConfig, v if isinstance(v, dict) else {})
                for k, v in val.items()
            }
        elif name == "dataset" and isinstance(val, dict):
            kwargs[name] = _from_dict(DatasetConfig, val)
        elif name == "preprocessing" and isinstance(val, dict):
            kwargs[name] = _from_dict(PreprocessingConfig, val)
        elif name == "cv" and isinstance(val, dict):
            kwargs[name] = _from_dict(CVConfig, val)
        elif name == "models" and isinstance(val, dict):
            kwargs[name] = _from_dict(ModelZooConfig, val)
        elif name == "reporting" and isinstance(val, dict):
            kwargs[name] = _from_dict(ReportingConfig, val)
        elif name == "output_config" and isinstance(val, dict):
            kwargs[name] = _from_dict(OutputConfig, val)
        else:
            kwargs[name] = val

    return cls(**kwargs)


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    models: ModelZooConfig = field(default_factory=ModelZooConfig)
    sem_configs: dict[str, SEMTargetConfig] = field(default_factory=dict)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    run_sem: bool = False
    output_dir: str = "results"
    mode: str = "exploratory"
    transformation_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if isinstance(self.dataset.task_type, AnalysisTaskType):
            d["dataset"]["task_type"] = self.dataset.task_type.value
        return d

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        return _from_dict(cls, data)

    @classmethod
    def load_json(cls, path: Path | str) -> PipelineConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def validate(self) -> list[str]:
        """Aggregate configuration validation across sub-configs. Returns a
        list of human-readable error strings; empty means valid. This does
        not raise — `execute_pipeline` is responsible for calling this and
        raising `MetisConfigError` with the joined messages, so both GUI and
        headless/CLI callers can decide how to present validation failures."""
        errors: list[str] = list(self.cv.validate())
        for target, sem_cfg in self.sem_configs.items():
            errors.extend(f"sem_configs['{target}']: {e}" for e in sem_cfg.validate())
        for target, requested in self.dataset.task_type_overrides.items():
            if requested not in (AnalysisTaskType.REGRESSION.value, AnalysisTaskType.CLASSIFICATION.value):
                errors.append(
                    f"dataset.task_type_overrides['{target}']='{requested}' must be "
                    f"'regression' or 'classification' (not 'auto' — an override "
                    f"exists specifically to resolve an ambiguous auto-detection)."
                )
        return errors

    @classmethod
    def load_manifest(cls, path: Path | str) -> tuple[PipelineConfig, dict[str, Any]]:
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, dict):
            msg = f"Invalid manifest format in {path}: expected a JSON object."
            raise ValueError(msg)

        if "config" in raw_data and isinstance(raw_data["config"], dict):
            config_dict = raw_data["config"]
            metadata = raw_data
        else:
            config_dict = raw_data
            metadata = {"config": raw_data, "manifest_version": MANIFEST_SCHEMA_VERSION}

        return cls.from_dict(config_dict), metadata


def get_preset_config(preset_name: str) -> PipelineConfig:
    if preset_name == "smoke_test":
        return PipelineConfig(
            cv=CVConfig(
                outer_folds=2,
                inner_folds=2,
                selection_folds=2,
                n_iter=3,
                bootstrap_n=100,
                fast_mode=True,
                skip_shap=True,
                permutation_repeats=5,
            ),
            models=ModelZooConfig(
                selected_models=["ridge", "random_forest", "logistic"],
                enable_stacking=False,
            ),
            mode="smoke_test",
        )
    if preset_name == "high_precision":
        return PipelineConfig(
            cv=CVConfig(
                outer_folds=10,
                inner_folds=5,
                n_iter=50,
                bootstrap_n=2000,
                fast_mode=False,
                skip_shap=False,
                permutation_repeats=50,
            ),
            models=ModelZooConfig(use_all_models=True, enable_stacking=True),
            mode="confirmatory",
        )
    return PipelineConfig()
