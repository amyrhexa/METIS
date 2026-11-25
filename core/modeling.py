# core/modeling.py
from __future__ import annotations

import contextlib
import logging
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR

from config.schema import AnalysisTaskType
from core.evaluation import calculate_comprehensive_metrics, calculate_single_metric

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    cb = None
    HAS_CATBOOST = False


@dataclass
class ModelSpec:
    name: str
    display_name: str
    estimator: Any
    param_distributions: dict[str, Any] = field(default_factory=dict)
    is_tree_based: bool = False
    is_linear: bool = False
    complexity_tier: str = "Low"
    interpretability_tier: str = "High"
    task_type: AnalysisTaskType = AnalysisTaskType.REGRESSION
    description: str = ""
    strengths: str = ""
    weaknesses: str = ""


def get_model_zoo(
    task_type: AnalysisTaskType,
    random_state: int = 42,
    include_full: bool = True,
) -> dict[str, ModelSpec]:
    zoo: dict[str, ModelSpec] = {}

    if task_type == AnalysisTaskType.REGRESSION:
        zoo["ridge"] = ModelSpec(
            "ridge",
            "Ridge Regression",
            Ridge(random_state=random_state, solver="lsqr", alpha=10.0),
            {"model__alpha": list(np.logspace(-1, 4, 20))},
            is_linear=True,
            complexity_tier="Low",
            interpretability_tier="High",
            task_type=task_type,
            description="L2-regularized linear model with LSQR solver.",
        )
        zoo["lasso"] = ModelSpec(
            "lasso",
            "Lasso Regression",
            Lasso(random_state=random_state, max_iter=15000, tol=1e-3),
            {"model__alpha": list(np.logspace(-3, 1, 20))},
            is_linear=True,
            complexity_tier="Low",
            interpretability_tier="High",
            task_type=task_type,
            description="L1-regularized sparse regression model.",
        )
        zoo["elasticnet"] = ModelSpec(
            "elasticnet",
            "ElasticNet",
            ElasticNet(random_state=random_state, max_iter=15000, tol=1e-3),
            {
                "model__alpha": list(np.logspace(-3, 1, 15)),
                "model__l1_ratio": list(np.linspace(0.1, 0.9, 8)),
            },
            is_linear=True,
            complexity_tier="Low",
            interpretability_tier="High",
            task_type=task_type,
        )
        zoo["bayesian_ridge"] = ModelSpec(
            "bayesian_ridge",
            "Bayesian Ridge",
            BayesianRidge(compute_score=True, tol=1e-3),
            {
                "model__alpha_1": list(np.logspace(-6, -2, 8)),
                "model__lambda_1": list(np.logspace(-6, -2, 8)),
            },
            is_linear=True,
            complexity_tier="Low",
            interpretability_tier="High",
            task_type=task_type,
        )
        zoo["random_forest"] = ModelSpec(
            "random_forest",
            "Random Forest Regressor",
            RandomForestRegressor(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [100, 250],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [2, 4],
            },
            is_tree_based=True,
            complexity_tier="Medium",
            interpretability_tier="Medium",
            task_type=task_type,
        )
        zoo["extra_trees"] = ModelSpec(
            "extra_trees",
            "Extra Trees Regressor",
            ExtraTreesRegressor(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [100, 250],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [2, 4],
            },
            is_tree_based=True,
            complexity_tier="Medium",
            interpretability_tier="Medium",
            task_type=task_type,
        )
        zoo["hist_gradient_boosting"] = ModelSpec(
            "hist_gradient_boosting",
            "HistGBDT Regressor",
            HistGradientBoostingRegressor(random_state=random_state),
            {
                "model__max_iter": [100, 200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.01, 0.05, 0.1],
            },
            is_tree_based=True,
            complexity_tier="High",
            interpretability_tier="Medium",
            task_type=task_type,
        )
        if HAS_XGBOOST and xgb is not None:
            zoo["xgboost"] = ModelSpec(
                "xgboost",
                "XGBoost Regressor",
                xgb.XGBRegressor(
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=0,
                    objective="reg:squarederror",
                ),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                },
                is_tree_based=True,
                complexity_tier="High",
                interpretability_tier="Medium",
                task_type=task_type,
            )
        if HAS_LIGHTGBM and lgb is not None:
            zoo["lightgbm"] = ModelSpec(
                "lightgbm",
                "LightGBM Regressor",
                lgb.LGBMRegressor(random_state=random_state, n_jobs=1, verbosity=-1),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [-1, 4, 8],
                    "model__learning_rate": [0.01, 0.05, 0.1],
                },
                is_tree_based=True,
                complexity_tier="High",
                interpretability_tier="Medium",
                task_type=task_type,
            )
        if HAS_CATBOOST and cb is not None:
            zoo["catboost"] = ModelSpec(
                "catboost",
                "CatBoost Regressor",
                cb.CatBoostRegressor(
                    random_seed=random_state,
                    thread_count=1,
                    verbose=0,
                ),
                {
                    "model__iterations": [100, 250],
                    "model__depth": [4, 6],
                    "model__learning_rate": [0.02, 0.08],
                },
                is_tree_based=True,
                complexity_tier="High",
                interpretability_tier="Medium",
                task_type=task_type,
                description="Gradient boosted decision trees using oblivious trees and ordered boosting.",
            )
        if include_full:
            zoo["svr"] = ModelSpec(
                "svr",
                "Support Vector Regressor (RBF)",
                SVR(kernel="rbf", C=1.0),
                {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.01, 0.1]},
                complexity_tier="Medium",
                interpretability_tier="Low",
                task_type=task_type,
            )
            zoo["mlp"] = ModelSpec(
                "mlp",
                "Multi-Layer Perceptron Regressor",
                MLPRegressor(
                    random_state=random_state,
                    max_iter=1000,
                    early_stopping=True,
                    n_iter_no_change=10,
                    learning_rate_init=0.001,
                ),
                {
                    "model__hidden_layer_sizes": [(32,), (64, 32)],
                    "model__alpha": [0.01, 0.1, 1.0],
                },
                complexity_tier="High",
                interpretability_tier="Low",
                task_type=task_type,
            )
    else:  # CLASSIFICATION
        zoo["logistic"] = ModelSpec(
            "logistic",
            "Logistic Regression",
            LogisticRegression(random_state=random_state, max_iter=2000),
            {"model__C": list(np.logspace(-3, 3, 20))},
            is_linear=True,
            complexity_tier="Low",
            interpretability_tier="High",
            task_type=task_type,
        )
        zoo["random_forest"] = ModelSpec(
            "random_forest",
            "Random Forest Classifier",
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [100, 250],
                "model__max_depth": [None, 6, 12],
            },
            is_tree_based=True,
            complexity_tier="Medium",
            interpretability_tier="Medium",
            task_type=task_type,
        )
        zoo["extra_trees"] = ModelSpec(
            "extra_trees",
            "Extra Trees Classifier",
            ExtraTreesClassifier(random_state=random_state, n_jobs=1),
            {
                "model__n_estimators": [100, 250],
                "model__max_depth": [None, 8, 16],
            },
            is_tree_based=True,
            complexity_tier="Medium",
            interpretability_tier="Medium",
            task_type=task_type,
        )
        zoo["hist_gradient_boosting"] = ModelSpec(
            "hist_gradient_boosting",
            "HistGBDT Classifier",
            HistGradientBoostingClassifier(random_state=random_state),
            {
                "model__max_iter": [100, 200],
                "model__learning_rate": [0.01, 0.1],
            },
            is_tree_based=True,
            complexity_tier="High",
            interpretability_tier="Medium",
            task_type=task_type,
        )
        if HAS_XGBOOST and xgb is not None:
            zoo["xgboost"] = ModelSpec(
                "xgboost",
                "XGBoost Classifier",
                xgb.XGBClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    verbosity=0,
                    eval_metric="logloss",
                ),
                {"model__n_estimators": [100, 200], "model__max_depth": [3, 6]},
                is_tree_based=True,
                complexity_tier="High",
                interpretability_tier="Medium",
                task_type=task_type,
            )
        if HAS_LIGHTGBM and lgb is not None:
            zoo["lightgbm"] = ModelSpec(
                "lightgbm",
                "LightGBM Classifier",
                lgb.LGBMClassifier(random_state=random_state, n_jobs=1, verbosity=-1),
                {"model__n_estimators": [100, 200], "model__max_depth": [-1, 6]},
                is_tree_based=True,
                complexity_tier="High",
                interpretability_tier="Medium",
                task_type=task_type,
            )
        if HAS_CATBOOST and cb is not None:
            zoo["catboost"] = ModelSpec(
                "catboost",
                "CatBoost Classifier",
                cb.CatBoostClassifier(
                    random_seed=random_state,
                    thread_count=1,
                    verbose=0,
                ),
                {"model__iterations": [100, 250], "model__depth": [4, 6]},
                is_tree_based=True,
                complexity_tier="High",
                interpretability_tier="Medium",
                task_type=task_type,
            )
        if include_full:
            zoo["svc"] = ModelSpec(
                "svc",
                "Support Vector Classifier (RBF)",
                SVC(kernel="rbf", probability=True, random_state=random_state),
                {"model__C": list(np.logspace(-1, 2, 6))},
                complexity_tier="Medium",
                interpretability_tier="Low",
                task_type=task_type,
            )
            zoo["mlp"] = ModelSpec(
                "mlp",
                "Multi-Layer Perceptron Classifier",
                MLPClassifier(
                    random_state=random_state,
                    max_iter=1500,
                    early_stopping=True,
                    n_iter_no_change=15,
                    learning_rate_init=0.003,
                    tol=1e-3,
                ),
                {
                    "model__hidden_layer_sizes": [(64, 32), (32, 16)],
                    "model__alpha": [0.001, 0.01, 0.1],
                },
                complexity_tier="High",
                interpretability_tier="Low",
                task_type=task_type,
            )

    return zoo


def build_stacking_spec(
    base_specs: list[ModelSpec],
    task_type: AnalysisTaskType,
    random_state: int,
) -> ModelSpec:
    estimators = [
        (s.name, clone(s.estimator)) for s in base_specs if s.name != "stacking"
    ]
    if not estimators:
        estimators = [("ridge", Ridge(random_state=random_state))]

    if task_type == AnalysisTaskType.REGRESSION:
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(alpha=1.0, random_state=random_state),
            n_jobs=1,
            passthrough=False,
        )
    else:
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=random_state),
            n_jobs=1,
            passthrough=False,
        )

    return ModelSpec(
        name="stacking",
        display_name="Stacking Super-Learner Ensemble",
        estimator=model,
        param_distributions={},
        is_tree_based=False,
        is_linear=False,
        complexity_tier="High",
        interpretability_tier="Low",
        task_type=task_type,
    )


@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_test: int
    primary_metric: float
    metrics: dict[str, Any]
    best_params: dict[str, Any]
    train_time_sec: float
    status: str = "SUCCESS"  # "SUCCESS" | "FAILED"
    error_message: str | None = None
    # Audit trail for Fix 6 (classification inner-CV scoring): which
    # sklearn scoring string was actually used to drive this fold's
    # hyperparameter search, and why (requested strategy, resolved
    # strategy, class distribution, imbalance ratio). None for regression
    # folds, where the scoring objective is fixed
    # (neg_root_mean_squared_error) and not subject to this decision.
    scoring_audit: dict[str, Any] | None = None


@dataclass
class LearningCurveData:
    train_sizes: list[int] = field(default_factory=list)
    train_scores_mean: list[float] = field(default_factory=list)
    train_scores_std: list[float] = field(default_factory=list)
    val_scores_mean: list[float] = field(default_factory=list)
    val_scores_std: list[float] = field(default_factory=list)


@dataclass
class NestedCVResult:
    fold_results: list[FoldResult]
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray | None
    fold_assignment: np.ndarray
    fold_estimators: list[tuple[int, Pipeline, np.ndarray]] = field(
        default_factory=list
    )
    learning_curve: LearningCurveData | None = None
    linear_coefficients: pd.DataFrame | None = None
    total_execution_time: float = 0.0
    execution_status: str = "SUCCESS"
    error_message: str | None = None
    cancelled: bool = False
    # How many of the outer folds raised an exception during fit/predict
    # and were excluded from the out-of-fold prediction array (Fix: fold
    # failures must be recorded/quantified, never silently dropped). 0 in
    # the overwhelming majority of runs.
    n_failed_folds: int = 0


def compute_corrected_resampled_ttest(
    scores_a: list[float], scores_b: list[float], n_train: int, n_test: int
) -> tuple[float, float]:
    diffs = np.array(scores_a, dtype=np.float64) - np.array(scores_b, dtype=np.float64)
    k = len(diffs)
    if k <= 1:
        return 0.0, 1.0
    mean_diff = float(np.mean(diffs))
    var_diff = float(np.var(diffs, ddof=1))
    if var_diff == 0.0 or np.isnan(var_diff):
        return mean_diff, 1.0
    corrected_var = ((1.0 / k) + (n_test / max(1, n_train))) * var_diff
    t_stat = mean_diff / np.sqrt(max(1e-12, corrected_var))
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=k - 1))
    return float(t_stat), float(p_val)


#: Explains what `extract_linear_model_coefficients`'s "fold_range_*"
#: columns are (and, just as importantly, are not) -- see Fix 8. Surfaced
#: verbatim in the HTML/LaTeX report next to any table built from this
#: function's output.
COEFFICIENT_RANGE_METHOD_DESCRIPTION = (
    "Descriptive [2.5, 97.5] percentile range of the standardized "
    "coefficient across the small number of outer CV folds (typically "
    "5-10 values) -- NOT a formal bootstrap or analytic confidence "
    "interval. With few folds this range is a rough sense of stability, "
    "not a calibrated interval; treat it as descriptive only."
)


def extract_linear_model_coefficients(
    fold_estimators: list[tuple[int, Pipeline, np.ndarray]],
) -> pd.DataFrame:
    """Aggregate standardized linear-model coefficients across outer CV
    folds.

    IMPORTANT (Fix 8): the `fold_range_lower`/`fold_range_upper` columns
    are the [2.5, 97.5] percentile of the coefficient's value across
    outer-fold-fitted models -- typically only `outer_folds` values (5 by
    default). This is a descriptive range across a handful of point
    estimates, not a statistically calibrated confidence interval, and is
    deliberately *not* named "ci_lower"/"ci_upper" to avoid implying a
    precision the small sample size cannot support. See
    `COEFFICIENT_RANGE_METHOD_DESCRIPTION`.
    """
    coef_records = []
    for fold_idx, pipe, _ in fold_estimators:
        if "preprocess" not in pipe.named_steps or "model" not in pipe.named_steps:
            continue
        preproc = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        if not hasattr(model, "coef_"):
            continue

        raw_coef = np.asarray(model.coef_)
        try:
            feat_names = [str(fn) for fn in preproc.get_feature_names_out()]
        except Exception:
            feat_names = [f"Feature_{i}" for i in range(raw_coef.shape[-1])]

        classes = getattr(model, "classes_", None)

        if raw_coef.ndim == 1:
            for fn, c_val in zip(feat_names, raw_coef, strict=False):
                coef_records.append(
                    {"fold": fold_idx, "feature": fn, "beta": float(c_val)}
                )
        elif raw_coef.ndim == 2:
            if raw_coef.shape[0] == 1:
                for fn, c_val in zip(feat_names, raw_coef[0], strict=False):
                    coef_records.append(
                        {"fold": fold_idx, "feature": fn, "beta": float(c_val)}
                    )
            else:
                for c_idx in range(raw_coef.shape[0]):
                    c_label = (
                        str(classes[c_idx]) if classes is not None else f"Class_{c_idx}"
                    )
                    for fn, c_val in zip(feat_names, raw_coef[c_idx], strict=False):
                        coef_records.append(
                            {
                                "fold": fold_idx,
                                "feature": f"{fn} [{c_label}]",
                                "beta": float(c_val),
                            }
                        )

    if not coef_records:
        return pd.DataFrame()

    df_c = pd.DataFrame(coef_records)
    summary = (
        df_c.groupby("feature")["beta"]
        .agg(
            mean_beta="mean",
            std_beta="std",
            fold_range_lower=lambda s: (
                float(np.percentile(s, 2.5)) if len(s) > 1 else float(np.min(s))
            ),
            fold_range_upper=lambda s: (
                float(np.percentile(s, 97.5)) if len(s) > 1 else float(np.max(s))
            ),
            count="count",
        )
        .reset_index()
    )
    summary["std_beta"] = summary["std_beta"].fillna(0.0)
    summary["interval_type"] = "descriptive_fold_range"
    return summary.sort_values("mean_beta", key=abs, ascending=False).reset_index(
        drop=True
    )


def compute_nested_learning_curve(
    best_pipe: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: AnalysisTaskType,
    fractions: list[float] | None = None,
    cv_splits: int = 4,
    random_state: int = 42,
) -> LearningCurveData:
    fracs = fractions or [0.2, 0.4, 0.6, 0.8, 1.0]
    n_total = len(y)
    n_splits = max(2, min(cv_splits, n_total // 2)) if n_total >= 4 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_sizes_out = []
    train_means, train_stds = [], []
    val_means, val_stds = [], []

    for frac in fracs:
        train_fold_scores = []
        val_fold_scores = []
        max_train_samples = int(n_total * (1.0 - 1.0 / n_splits))
        n_sub = max(4, min(max_train_samples, int(max_train_samples * frac)))

        for train_idx, val_idx in kf.split(X, y):
            sub_train_idx = train_idx[: min(len(train_idx), n_sub)]
            if len(sub_train_idx) < 2:
                continue
            X_tr, y_tr = X.iloc[sub_train_idx], y[sub_train_idx]
            X_va, y_va = X.iloc[val_idx], y[val_idx]

            pipe_inst = clone(best_pipe)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe_inst.fit(X_tr, y_tr)

            yp_tr = cast(Any, pipe_inst).predict(X_tr)
            yp_va = cast(Any, pipe_inst).predict(X_va)

            m_tr = calculate_comprehensive_metrics(y_tr, yp_tr, task_type)
            m_va = calculate_comprehensive_metrics(y_va, yp_va, task_type)

            metric_key = (
                "r2" if task_type == AnalysisTaskType.REGRESSION else "accuracy"
            )
            train_fold_scores.append(m_tr.get(metric_key, 0.0))
            val_fold_scores.append(m_va.get(metric_key, 0.0))

        if train_fold_scores:
            train_sizes_out.append(n_sub)
            train_means.append(float(np.mean(train_fold_scores)))
            train_stds.append(float(np.std(train_fold_scores)))
            val_means.append(float(np.mean(val_fold_scores)))
            val_stds.append(float(np.std(val_fold_scores)))

    return LearningCurveData(
        train_sizes=train_sizes_out,
        train_scores_mean=train_means,
        train_scores_std=train_stds,
        val_scores_mean=val_means,
        val_scores_std=val_stds,
    )


def _build_cv_splitter(
    task_type: AnalysisTaskType,
    n_splits: int,
    random_state: int,
    groups: np.ndarray | None,
) -> tuple[Any, bool]:
    """Construct the appropriate CV splitter (stratified/group-aware as
    needed) for a given task type and split count.

    Shared by every outer/selection/inner split constructed in this module
    so that exactly the same splitting logic (stratification, grouping,
    shuffling, seeding) is used everywhere, rather than being re-derived
    slightly differently in several places (which previously risked subtle
    inconsistencies between, e.g., the outer and inner CV construction).
    """
    has_groups = groups is not None and len(np.unique(groups)) >= n_splits
    if has_groups:
        splitter = (
            StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            if task_type == AnalysisTaskType.CLASSIFICATION
            else GroupKFold(n_splits=n_splits)
        )
    else:
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            if task_type == AnalysisTaskType.CLASSIFICATION
            else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        )
    return splitter, has_groups


def _safe_n_splits(
    requested: int,
    task_type: AnalysisTaskType,
    y: np.ndarray,
    minimum: int = 2,
) -> int:
    """Clamp a requested fold count so it never exceeds what the data can
    actually support (e.g. StratifiedKFold requires n_splits <= the
    smallest class count). Used for the confirmatory model-selection
    procedure's inner "selection CV", which runs on a training partition
    that is smaller than the full dataset and, for classification tasks,
    may have small class counts."""
    if task_type == AnalysisTaskType.CLASSIFICATION:
        _, counts = np.unique(y, return_counts=True)
        ceiling = max(minimum, int(counts.min())) if len(counts) else minimum
    else:
        ceiling = max(minimum, len(y) // 2) if len(y) else minimum
    return max(minimum, min(requested, ceiling))


#: Maps METIS's `classification_scoring` config values to the sklearn
#: `scoring=` string RandomizedSearchCV expects.
_CLASSIFICATION_SCORING_TO_SKLEARN: dict[str, str] = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "f1_macro": "f1_macro",
    "mcc": "matthews_corrcoef",
    "roc_auc": "roc_auc",
}


def resolve_classification_inner_scoring(
    y_train: np.ndarray,
    strategy: str = "auto",
    imbalance_ratio_threshold: float = 1.5,
) -> tuple[str, dict[str, Any]]:
    """Resolve the sklearn `scoring=` string used to drive classification
    inner-CV hyperparameter search (Fix 6).

    Previously this was unconditionally hardcoded to "accuracy" regardless
    of class balance. `strategy="auto"` (the default) now inspects the
    *training fold's own* class distribution and switches to
    "balanced_accuracy" once the majority:minority class-count ratio
    exceeds `imbalance_ratio_threshold`; any other value in
    `config.schema.ALLOWED_CLASSIFICATION_SCORING` is used as-is
    (explicit user override). The returned audit dict is recorded per-fold
    in `FoldResult.scoring_audit` and surfaced in the run manifest/report
    so the actual scoring decision is never invisible.
    """
    values, counts = np.unique(np.asarray(y_train), return_counts=True)
    class_counts = {str(v): int(c) for v, c in zip(values, counts, strict=False)}
    n_classes = len(values)
    imbalance_ratio = float(counts.max() / max(1, counts.min())) if len(counts) else 1.0

    requested = (strategy or "auto").strip().lower()
    resolved = (
        (
            "balanced_accuracy"
            if imbalance_ratio > imbalance_ratio_threshold
            else "accuracy"
        )
        if requested == "auto"
        else requested
    )

    sklearn_scoring = _CLASSIFICATION_SCORING_TO_SKLEARN.get(resolved, "accuracy")
    if resolved == "roc_auc" and n_classes > 2:
        sklearn_scoring = "roc_auc_ovr_weighted"

    audit = {
        "strategy_requested": requested,
        "strategy_resolved": resolved,
        "sklearn_scoring": sklearn_scoring,
        "class_counts": class_counts,
        "imbalance_ratio": round(imbalance_ratio, 3),
        "imbalance_ratio_threshold": imbalance_ratio_threshold,
    }
    return sklearn_scoring, audit


@dataclass
class TunedFitResult:
    estimator: Any
    best_params: dict[str, Any]
    inner_scoring_used: str
    scoring_audit: dict[str, Any]
    tuning_failed: bool = False


def _tune_and_fit_single_model(
    model_spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    preprocessor: ColumnTransformer,
    task_type: AnalysisTaskType,
    inner_folds: int,
    n_iter: int,
    random_state: int,
    classification_scoring: str,
    imbalance_ratio_threshold: float,
    groups_train: np.ndarray | None,
    logger: logging.Logger | None,
) -> TunedFitResult:
    """Hyperparameter-tune (if `model_spec` defines a search space) and fit
    a single candidate model strictly on `X_train`/`y_train`.

    Factored out of `evaluate_model_nested_cv` so this exact tuning logic
    is reused, unmodified, by the confirmatory model-selection procedure's
    "tune the selected winner" phase (`run_confirmatory_model_selection_cv`,
    Fix 1) -- guaranteeing the two share one implementation rather than
    risking silent drift between two independently-maintained copies.
    """
    candidate_pipe = Pipeline(
        [("preprocess", clone(preprocessor)), ("model", clone(model_spec.estimator))]
    )

    if task_type == AnalysisTaskType.CLASSIFICATION:
        inner_scoring, scoring_audit = resolve_classification_inner_scoring(
            y_train, classification_scoring, imbalance_ratio_threshold
        )
    else:
        inner_scoring = "neg_root_mean_squared_error"
        scoring_audit = {
            "strategy_requested": inner_scoring,
            "strategy_resolved": inner_scoring,
            "sklearn_scoring": inner_scoring,
        }

    if not model_spec.param_distributions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = candidate_pipe.fit(X_train, y_train)
        return TunedFitResult(fitted, {}, inner_scoring, scoring_audit)

    inner_cv, _ = _build_cv_splitter(task_type, inner_folds, random_state, groups_train)

    n_candidates = 1
    for pvals in model_spec.param_distributions.values():
        n_candidates *= len(pvals) if hasattr(pvals, "__len__") else 10

    try:
        search = RandomizedSearchCV(
            candidate_pipe,
            param_distributions=model_spec.param_distributions,
            n_iter=min(n_iter, max(1, n_candidates)),
            cv=inner_cv,
            scoring=inner_scoring,
            n_jobs=1,
            random_state=random_state,
            refit=True,
            error_score="raise",
        )
        fit_params = {"groups": groups_train} if groups_train is not None else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            search.fit(X_train, y_train, **fit_params)
        return TunedFitResult(
            search.best_estimator_, search.best_params_, inner_scoring, scoring_audit
        )
    except Exception as exc:
        if logger:
            logger.warning(
                "Hyperparameter search failed for '%s': %s. Falling back to default params.",
                model_spec.display_name,
                exc,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = candidate_pipe.fit(X_train, y_train)
        return TunedFitResult(
            fitted,
            {"_tuning_failed": str(exc)[:200]},
            inner_scoring,
            scoring_audit,
            tuning_failed=True,
        )


def evaluate_model_nested_cv(
    model_spec: ModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    task_type: AnalysisTaskType,
    outer_folds: int = 5,
    inner_folds: int = 3,
    n_iter: int = 25,
    n_jobs: int = -1,
    random_state: int = 42,
    retain_fold_estimators: bool = True,
    groups: np.ndarray | None = None,
    compute_learning_curve: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
    logger: logging.Logger | None = None,
    should_stop: Callable[[], bool] | None = None,
    classification_scoring: str = "auto",
    imbalance_ratio_threshold: float = 1.5,
    parallel_outer_folds: bool = False,
) -> NestedCVResult:
    """Fix 10 (n_jobs honesty): by default every outer fold runs
    sequentially, single-threaded (deterministic, and avoids CPU
    oversubscription). Passing `parallel_outer_folds=True` runs the outer
    folds concurrently using `n_jobs` threads via joblib -- safe because
    each fold is a fully independent computation (its own `clone()`'d
    preprocessor/estimator, its own disjoint slice of the shared
    pre-allocated output arrays), so parallelizing changes wall-clock time
    only, never results. Hyperparameter search inside each fold always
    runs with n_jobs=1 regardless, specifically to avoid oversubscribing
    CPUs when outer folds are themselves running in parallel.
    """
    t_start = time.time()
    outer_cv, has_groups = _build_cv_splitter(task_type, outer_folds, random_state, groups)

    n_samples = len(y)
    oof_predictions = np.full(
        n_samples,
        np.nan if task_type == AnalysisTaskType.REGRESSION else None,
        dtype=object if task_type == AnalysisTaskType.CLASSIFICATION else float,
    )
    oof_probabilities: np.ndarray | None = None
    if task_type == AnalysisTaskType.CLASSIFICATION:
        n_classes = len(np.unique(y))
        oof_probabilities = np.full((n_samples, max(2, n_classes)), np.nan)

    fold_assignment = np.full(n_samples, -1, dtype=int)

    splits = list(
        outer_cv.split(X, y, groups=groups) if has_groups else outer_cv.split(X, y)
    )
    total_splits = len(splits)

    def _process_one_outer_fold(
        fold_idx: int, train_idx: np.ndarray, test_idx: np.ndarray
    ) -> tuple[
        int, np.ndarray, FoldResult, Pipeline | None, np.ndarray | None, np.ndarray | None
    ]:
        """Runs one outer fold end-to-end and returns everything the
        caller needs to merge into the shared result arrays. Written as a
        pure function of its arguments (no mutation of outer-scope state)
        so it can be safely dispatched either sequentially or in a worker
        thread."""
        f_t0 = time.time()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx] if has_groups and groups is not None else None

        try:
            tuned = _tune_and_fit_single_model(
                model_spec,
                X_train,
                y_train,
                preprocessor,
                task_type,
                inner_folds,
                n_iter,
                random_state,
                classification_scoring,
                imbalance_ratio_threshold,
                groups_train,
                logger,
            )
            best_estimator = tuned.estimator
            y_pred = np.asarray(cast(Any, best_estimator).predict(X_test))

            y_prob_fold = None
            if task_type == AnalysisTaskType.CLASSIFICATION:
                predict_proba_fn = getattr(best_estimator, "predict_proba", None)
                if callable(predict_proba_fn):
                    with contextlib.suppress(Exception):
                        y_prob_fold = np.asarray(predict_proba_fn(X_test))

            fold_metrics = calculate_comprehensive_metrics(
                y_test, y_pred, task_type, y_prob_fold
            )
            primary_metric_val = (
                fold_metrics.get("r2", 0.0)
                if task_type == AnalysisTaskType.REGRESSION
                else fold_metrics.get("accuracy", 0.0)
            )
            fold_dur = round(time.time() - f_t0, 3)
            fold_result = FoldResult(
                fold=fold_idx,
                n_train=len(train_idx),
                n_test=len(test_idx),
                primary_metric=primary_metric_val,
                metrics=fold_metrics,
                best_params=tuned.best_params,
                train_time_sec=fold_dur,
                status="SUCCESS",
                scoring_audit=tuned.scoring_audit,
            )
            if progress_callback:
                progress_callback(
                    fold_idx,
                    total_splits,
                    f"Evaluated fold {fold_idx + 1}/{total_splits} (Score: {primary_metric_val:.3f}, {fold_dur}s)",
                )
            return (
                fold_idx,
                test_idx,
                fold_result,
                cast(Pipeline, best_estimator),
                y_pred,
                y_prob_fold,
            )
        except Exception as exc:
            # Fold-level hardening: one fold's unexpected failure (e.g. a
            # solver crash on a pathological training partition) must not
            # silently discard, nor silently bias, the model's overall
            # result. The failure is recorded explicitly with status
            # "FAILED"; the corresponding rows are left unset (NaN/None) in
            # oof_predictions and excluded from the final OOF metric
            # computation, rather than contributing a spurious value.
            if logger:
                logger.error(
                    "Fold %d failed for model '%s': %s",
                    fold_idx,
                    model_spec.display_name,
                    exc,
                )
            fold_dur = round(time.time() - f_t0, 3)
            fold_result = FoldResult(
                fold=fold_idx,
                n_train=len(train_idx),
                n_test=len(test_idx),
                primary_metric=float("nan"),
                metrics={},
                best_params={},
                train_time_sec=fold_dur,
                status="FAILED",
                error_message=str(exc)[:500],
            )
            if progress_callback:
                progress_callback(
                    fold_idx,
                    total_splits,
                    f"Fold {fold_idx + 1}/{total_splits} FAILED: {str(exc)[:80]}",
                )
            return (fold_idx, test_idx, fold_result, None, None, None)

    if should_stop and should_stop():
        return NestedCVResult(
            fold_results=[],
            oof_predictions=oof_predictions,
            oof_probabilities=oof_probabilities,
            fold_assignment=fold_assignment,
            execution_status="CANCELLED",
            cancelled=True,
        )

    if parallel_outer_folds and n_jobs != 1 and total_splits > 1:
        fold_outputs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_process_one_outer_fold)(fold_idx, train_idx, test_idx)
            for fold_idx, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        fold_outputs = []
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if should_stop and should_stop():
                break
            fold_outputs.append(_process_one_outer_fold(fold_idx, train_idx, test_idx))

    fold_results: list[FoldResult] = []
    fold_estimators: list[tuple[int, Pipeline, np.ndarray]] = []
    for fold_idx, test_idx, fold_result, best_estimator, y_pred, y_prob_fold in sorted(
        fold_outputs, key=lambda item: item[0]
    ):
        fold_results.append(fold_result)
        fold_assignment[test_idx] = fold_idx
        if fold_result.status == "SUCCESS" and best_estimator is not None:
            oof_predictions[test_idx] = y_pred
            if (
                task_type == AnalysisTaskType.CLASSIFICATION
                and y_prob_fold is not None
                and oof_probabilities is not None
                and y_prob_fold.shape[1] <= oof_probabilities.shape[1]
            ):
                oof_probabilities[test_idx, : y_prob_fold.shape[1]] = y_prob_fold
            if retain_fold_estimators:
                fold_estimators.append((fold_idx, best_estimator, test_idx))

    n_failed_folds = sum(1 for fr in fold_results if fr.status == "FAILED")

    learning_curve_res = None
    linear_coefs = None
    if fold_estimators:
        if model_spec.is_linear:
            linear_coefs = extract_linear_model_coefficients(fold_estimators)
        if compute_learning_curve:
            sample_best_pipe = fold_estimators[0][1]
            learning_curve_res = compute_nested_learning_curve(
                sample_best_pipe, X, y, task_type, random_state=random_state
            )

    execution_status = "SUCCESS" if fold_estimators else "FAILED"

    return NestedCVResult(
        fold_results=fold_results,
        oof_predictions=oof_predictions,
        oof_probabilities=oof_probabilities,
        fold_assignment=fold_assignment,
        fold_estimators=fold_estimators,
        learning_curve=learning_curve_res,
        linear_coefficients=linear_coefs,
        total_execution_time=round(time.time() - t_start, 2),
        execution_status=execution_status,
        n_failed_folds=n_failed_folds,
    )


# ---------------------------------------------------------------------------
# Fix 1 -- confirmatory model-selection procedure (winner's-curse remediation)
#
# `evaluate_model_nested_cv` above gives an unbiased nested-CV estimate of
# ONE pre-specified model. It does not, by itself, give an unbiased
# estimate of "compare N candidate models and report the best one" -- that
# is a distinct, larger procedure, and selecting the max of N nested-CV
# scores and reporting that max as if it were the score of a single
# pre-specified model is an optimistic ("winner's curse") estimate whose
# bias grows with N (Cawley & Talbot, 2010).
#
# `run_confirmatory_model_selection_cv` below evaluates the *entire
# selection procedure* via its own outer CV, so the reported number
# answers "how well does the procedure of comparing these models and
# picking a winner generalize?" rather than "how well did the model that
# happened to win perform, according to the same numbers used to pick it?"
# ---------------------------------------------------------------------------


@dataclass
class SelectionFoldRecord:
    """Full audit trail for one outer fold of the confirmatory
    model-selection procedure: which candidates were compared, what each
    scored during the *selection* phase (within the training partition
    only), which one won, how it was tuned, and how the tuned winner then
    performed on the untouched outer-test partition."""

    fold: int
    n_train: int
    n_test: int
    candidate_selection_scores: dict[str, float]
    selected_model: str
    selected_model_params: dict[str, Any]
    selected_model_test_metric: float
    train_time_sec: float
    selection_time_sec: float
    status: str = "SUCCESS"
    error_message: str | None = None


@dataclass
class ModelSelectionResult:
    oof_predictions: np.ndarray
    oof_probabilities: np.ndarray | None
    fold_assignment: np.ndarray
    fold_records: list[SelectionFoldRecord]
    # Fraction of outer folds in which each candidate won the selection
    # phase. A stable procedure has one candidate near 1.0; a procedure
    # where the "winner" changes fold-to-fold is itself an important,
    # honest diagnostic that a single point estimate would hide.
    selection_stability: dict[str, float]
    final_metrics: dict[str, Any]
    primary_metric_name: str
    n_failed_folds: int = 0
    total_execution_time: float = 0.0
    execution_status: str = "SUCCESS"
    cancelled: bool = False
    # Populated only when `record_touched_indices=True` -- the set of
    # original-dataset row indices ever touched during any fold's
    # *selection* phase (i.e. while comparing/tuning candidates within that
    # fold's training partition). Used exclusively by the leakage-isolation
    # regression test to assert this set never intersects that fold's
    # outer-test partition; left None in normal runs to avoid the memory
    # overhead of tracking it.
    debug_selection_touched_indices: set[int] | None = None


def run_confirmatory_model_selection_cv(
    candidate_specs: dict[str, ModelSpec],
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    task_type: AnalysisTaskType,
    outer_folds: int = 5,
    selection_folds: int = 3,
    inner_folds: int = 3,
    n_iter: int = 25,
    random_state: int = 42,
    classification_scoring: str = "auto",
    imbalance_ratio_threshold: float = 1.5,
    groups: np.ndarray | None = None,
    logger: logging.Logger | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    record_touched_indices: bool = False,
) -> ModelSelectionResult:
    """Evaluate the entire "compare candidate models, pick a winner"
    procedure via its own outer cross-validation (Fix 1).

    Per outer fold:
      1. Split into an outer-training partition and an outer-test
         partition. The outer-test partition is touched exactly once,
         right at the end, and never used for selection or tuning.
      2. **Selection phase** (within the outer-training partition only):
         every candidate in `candidate_specs` is scored via a
         `selection_folds`-fold CV using that candidate's *default*
         hyperparameters (no search -- this bounds the cost of comparing
         many candidates). The candidate with the best mean score wins.
      3. **Tuning phase**: the single winning candidate is hyperparameter-
         tuned (`_tune_and_fit_single_model`, the same routine
         `evaluate_model_nested_cv` uses for a single model) and refit on
         the *entire* outer-training partition. Still no exposure to the
         outer-test partition.
      4. **Prediction phase**: the tuned winner predicts the untouched
         outer-test partition. Only these predictions are accumulated into
         the returned out-of-fold array.

    Because both model selection and hyperparameter tuning in every outer
    fold are confined to that fold's training partition, and the
    outer-test partition is used only for a single, final prediction from
    a model that was never evaluated against it, the metrics computed from
    the aggregated out-of-fold predictions are an unbiased estimate of the
    full selection-and-tuning procedure -- not merely of whichever
    candidate happens to win overall.

    This is the primary, confirmatory performance number METIS reports
    per target (see core.pipeline_runner and core.reporting); the
    per-candidate leaderboard from `evaluate_model_nested_cv` remains
    available as a separate, explicitly-labeled *exploratory/descriptive*
    comparison, not as the confirmatory estimate.
    """
    t_start = time.time()
    outer_cv, has_groups = _build_cv_splitter(
        task_type, outer_folds, random_state, groups
    )
    primary_metric_name = "r2" if task_type == AnalysisTaskType.REGRESSION else "accuracy"

    n_samples = len(y)
    oof_predictions = np.full(
        n_samples,
        np.nan if task_type == AnalysisTaskType.REGRESSION else None,
        dtype=object if task_type == AnalysisTaskType.CLASSIFICATION else float,
    )
    oof_probabilities: np.ndarray | None = None
    if task_type == AnalysisTaskType.CLASSIFICATION:
        n_classes = len(np.unique(y))
        oof_probabilities = np.full((n_samples, max(2, n_classes)), np.nan)

    fold_assignment = np.full(n_samples, -1, dtype=int)
    fold_records: list[SelectionFoldRecord] = []
    selection_counts: dict[str, int] = dict.fromkeys(candidate_specs, 0)
    touched_selection_indices: set[int] = set()

    splits = list(
        outer_cv.split(X, y, groups=groups) if has_groups else outer_cv.split(X, y)
    )
    total_splits = len(splits)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        if should_stop and should_stop():
            return ModelSelectionResult(
                oof_predictions=oof_predictions,
                oof_probabilities=oof_probabilities,
                fold_assignment=fold_assignment,
                fold_records=fold_records,
                selection_stability={},
                final_metrics={},
                primary_metric_name=primary_metric_name,
                execution_status="CANCELLED",
                cancelled=True,
            )

        f_t0 = time.time()
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train = groups[train_idx] if has_groups and groups is not None else None

        try:
            # --- Phase A: selection, strictly within the training partition ---
            sel_t0 = time.time()
            n_sel_splits = _safe_n_splits(selection_folds, task_type, y_train)
            sel_cv, sel_has_groups = _build_cv_splitter(
                task_type, n_sel_splits, random_state, groups_train
            )
            sel_splits = list(
                sel_cv.split(X_train, y_train, groups=groups_train)
                if sel_has_groups
                else sel_cv.split(X_train, y_train)
            )

            candidate_scores: dict[str, float] = {}
            for name, spec in candidate_specs.items():
                fold_scores: list[float] = []
                for sel_tr, sel_val in sel_splits:
                    if record_touched_indices:
                        touched_selection_indices.update(
                            train_idx[sel_tr].tolist()
                        )
                        touched_selection_indices.update(
                            train_idx[sel_val].tolist()
                        )
                    pipe = Pipeline(
                        [
                            ("preprocess", clone(preprocessor)),
                            ("model", clone(spec.estimator)),
                        ]
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pipe.fit(X_train.iloc[sel_tr], y_train[sel_tr])
                    pred = pipe.predict(X_train.iloc[sel_val])
                    score = calculate_single_metric(
                        y_train[sel_val], pred, primary_metric_name, task_type
                    )
                    if np.isfinite(score):
                        fold_scores.append(float(score))
                candidate_scores[name] = (
                    float(np.mean(fold_scores)) if fold_scores else float("-inf")
                )

            selected_name = max(candidate_scores, key=lambda k: candidate_scores[k])
            selection_counts[selected_name] = selection_counts.get(selected_name, 0) + 1
            sel_dur = round(time.time() - sel_t0, 3)

            # --- Phase B: tune + refit the winner on the FULL training partition ---
            tuned = _tune_and_fit_single_model(
                candidate_specs[selected_name],
                X_train,
                y_train,
                preprocessor,
                task_type,
                inner_folds,
                n_iter,
                random_state,
                classification_scoring,
                imbalance_ratio_threshold,
                groups_train,
                logger,
            )

            # --- Phase C: predict the untouched outer-test partition ---
            y_pred = np.asarray(tuned.estimator.predict(X_test))
            oof_predictions[test_idx] = y_pred
            fold_assignment[test_idx] = fold_idx

            if task_type == AnalysisTaskType.CLASSIFICATION:
                predict_proba_fn = getattr(tuned.estimator, "predict_proba", None)
                if callable(predict_proba_fn):
                    with contextlib.suppress(Exception):
                        y_prob_fold = np.asarray(predict_proba_fn(X_test))
                        if (
                            oof_probabilities is not None
                            and y_prob_fold.shape[1] <= oof_probabilities.shape[1]
                        ):
                            oof_probabilities[test_idx, : y_prob_fold.shape[1]] = (
                                y_prob_fold
                            )

            test_metric = calculate_single_metric(
                y_test, y_pred, primary_metric_name, task_type
            )

            fold_records.append(
                SelectionFoldRecord(
                    fold=fold_idx,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    candidate_selection_scores=candidate_scores,
                    selected_model=selected_name,
                    selected_model_params=tuned.best_params,
                    selected_model_test_metric=float(test_metric),
                    train_time_sec=round(time.time() - f_t0, 3),
                    selection_time_sec=sel_dur,
                    status="SUCCESS",
                )
            )
        except Exception as exc:
            if logger:
                logger.error(
                    "Confirmatory model-selection fold %d failed: %s", fold_idx, exc
                )
            fold_assignment[test_idx] = fold_idx
            fold_records.append(
                SelectionFoldRecord(
                    fold=fold_idx,
                    n_train=len(train_idx),
                    n_test=len(test_idx),
                    candidate_selection_scores={},
                    selected_model="",
                    selected_model_params={},
                    selected_model_test_metric=float("nan"),
                    train_time_sec=round(time.time() - f_t0, 3),
                    selection_time_sec=0.0,
                    status="FAILED",
                    error_message=str(exc)[:500],
                )
            )

        if progress_callback:
            progress_callback(
                fold_idx,
                total_splits,
                f"Confirmatory selection fold {fold_idx + 1}/{total_splits}",
            )

    n_failed = sum(1 for r in fold_records if r.status == "FAILED")

    if task_type == AnalysisTaskType.REGRESSION:
        valid_mask = ~pd.isna(oof_predictions.astype(float))
    else:
        valid_mask = np.array(
            [v is not None and not (isinstance(v, float) and np.isnan(v))
             for v in oof_predictions]
        ) if n_samples else np.array([], dtype=bool)

    if valid_mask.sum() == 0:
        final_metrics: dict[str, Any] = {}
        execution_status = "FAILED"
    else:
        final_metrics = calculate_comprehensive_metrics(
            y[valid_mask],
            oof_predictions[valid_mask],
            task_type,
            oof_probabilities[valid_mask] if oof_probabilities is not None else None,
        )
        execution_status = "SUCCESS" if n_failed == 0 else "PARTIAL_FAILURE"

    selection_stability = {
        name: round(count / total_splits, 3) if total_splits else 0.0
        for name, count in selection_counts.items()
    }

    return ModelSelectionResult(
        oof_predictions=oof_predictions,
        oof_probabilities=oof_probabilities,
        fold_assignment=fold_assignment,
        fold_records=fold_records,
        selection_stability=selection_stability,
        final_metrics=final_metrics,
        primary_metric_name=primary_metric_name,
        n_failed_folds=n_failed,
        total_execution_time=round(time.time() - t_start, 2),
        execution_status=execution_status,
        debug_selection_touched_indices=(
            touched_selection_indices if record_touched_indices else None
        ),
    )
