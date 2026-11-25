from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config.schema import AnalysisTaskType


def _coerce_classification_labels(arr: Any) -> np.ndarray:
    """Re-infer a proper numpy dtype for classification label/prediction
    arrays that may have been constructed as `dtype=object`.

    METIS accumulates out-of-fold classification predictions into a
    dtype=object array so it can hold either numeric or string class
    labels uniformly across folds. Left as dtype=object when handed to
    sklearn, `type_of_target` cannot reliably infer "binary"/"multiclass"
    and raises "Classification metrics can't handle a mix of binary and
    unknown targets" -- and, critically, that exception was previously
    being silently swallowed by `bootstrap_metric_ci`'s per-resample
    try/except, turning *every* classification bootstrap CI into silent
    NaNs rather than a visible failure (discovered during the METIS
    scientific-remediation self-audit). Rebuilding the array from its own
    contents lets numpy re-infer the correct concrete dtype (int64,
    float64, or a string/object dtype for genuinely mixed/text labels)
    without altering any label values.
    """
    a = np.asarray(arr)
    if a.dtype == object:
        try:
            return np.array(a.tolist())
        except Exception:
            return a
    return a


def calculate_single_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    task_type: AnalysisTaskType = AnalysisTaskType.REGRESSION,
    y_prob: np.ndarray | None = None,
) -> float:
    """Evaluate only the specific target metric directly for optimal bootstrap throughput."""
    if task_type == AnalysisTaskType.REGRESSION:
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        if len(y_t) == 0:
            return float(np.nan)

        if metric_name == "r2":
            var_t = float(np.var(y_t))
            return float(cast(Any, r2_score(y_t, y_p))) if var_t > 0 else 0.0
        if metric_name == "rmse":
            return float(cast(Any, np.sqrt(mean_squared_error(y_t, y_p))))
        if metric_name == "mae":
            return float(cast(Any, mean_absolute_error(y_t, y_p)))
        if metric_name == "medae":
            return float(cast(Any, median_absolute_error(y_t, y_p)))
        if metric_name == "max_error":
            return float(cast(Any, max_error(y_t, y_p)))
        if metric_name == "explained_variance":
            return float(cast(Any, explained_variance_score(y_t, y_p)))
        if metric_name == "mape":
            with np.errstate(all="ignore"):
                return float(cast(Any, mean_absolute_percentage_error(y_t, y_p)))

        var_t = float(np.var(y_t))
        var_p = float(np.var(y_p))
        has_variance = len(y_t) > 2 and var_p > 1e-12 and var_t > 1e-12

        if metric_name == "pearson_r":
            return float(stats.pearsonr(y_t, y_p)[0]) if has_variance else 0.0
        if metric_name == "pearson_p":
            return float(stats.pearsonr(y_t, y_p)[1]) if has_variance else 1.0
        if metric_name == "spearman_rho":
            return float(stats.spearmanr(y_t, y_p)[0]) if has_variance else 0.0
        if metric_name == "spearman_p":
            return float(stats.spearmanr(y_t, y_p)[1]) if has_variance else 1.0
        if metric_name in (
            "calibration_slope",
            "calibration_intercept",
            "calibration_se",
        ):
            if has_variance:
                res = stats.linregress(y_p, y_t)
                if metric_name == "calibration_slope":
                    return float(res.slope)
                if metric_name == "calibration_intercept":
                    return float(res.intercept)
                return float(res.stderr)
            if metric_name == "calibration_slope":
                return 1.0
            return 0.0
    else:
        if len(y_true) == 0:
            return float(np.nan)

        y_true = _coerce_classification_labels(y_true)
        y_pred = _coerce_classification_labels(y_pred)

        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_name in ("balanced_acc", "balanced_accuracy"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(balanced_accuracy_score(y_true, y_pred))
        if metric_name == "f1_macro":
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "f1_weighted":
            return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        if metric_name == "precision_macro":
            return float(
                precision_score(y_true, y_pred, average="macro", zero_division=0)
            )
        if metric_name == "recall_macro":
            return float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        if metric_name == "mcc":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(matthews_corrcoef(y_true, y_pred))
        if metric_name == "cohen_kappa":
            return float(cohen_kappa_score(y_true, y_pred))
        if metric_name in ("roc_auc", "log_loss") and y_prob is not None:
            try:
                if metric_name == "roc_auc":
                    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                        return float(roc_auc_score(y_true, y_prob[:, 1]))
                    if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                        return float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
                elif metric_name == "log_loss":
                    return float(log_loss(y_true, y_prob))
            except Exception:
                return float(np.nan)

    res_dict = calculate_comprehensive_metrics(y_true, y_pred, task_type, y_prob)
    return float(cast(Any, res_dict.get(metric_name, np.nan)))


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: AnalysisTaskType,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute rigorous scientific performance, discrimination, and calibration metrics."""
    metrics: dict[str, Any] = {}
    if task_type == AnalysisTaskType.REGRESSION:
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_pred, dtype=float)

        var_t = float(np.var(y_t)) if len(y_t) > 0 else 0.0
        var_p = float(np.var(y_p)) if len(y_p) > 0 else 0.0

        metrics["r2"] = float(cast(Any, r2_score(y_t, y_p))) if var_t > 0 else 0.0
        metrics["rmse"] = float(cast(Any, np.sqrt(mean_squared_error(y_t, y_p))))
        metrics["mae"] = float(cast(Any, mean_absolute_error(y_t, y_p)))
        metrics["medae"] = float(cast(Any, median_absolute_error(y_t, y_p)))
        metrics["max_error"] = float(cast(Any, max_error(y_t, y_p)))
        metrics["explained_variance"] = float(
            cast(Any, explained_variance_score(y_t, y_p))
        )
        with np.errstate(all="ignore"):
            metrics["mape"] = float(cast(Any, mean_absolute_percentage_error(y_t, y_p)))

        if len(y_t) > 2 and var_p > 1e-12 and var_t > 1e-12:
            pr, pr_pval = stats.pearsonr(y_t, y_p)
            sr, sr_pval = stats.spearmanr(y_t, y_p)
            metrics["pearson_r"] = float(pr)
            metrics["pearson_p"] = float(pr_pval)
            metrics["spearman_rho"] = float(sr)
            metrics["spearman_p"] = float(sr_pval)

            slope, intercept, _r_val, _, se_slope = stats.linregress(y_p, y_t)
            metrics["calibration_slope"] = float(slope)
            metrics["calibration_intercept"] = float(intercept)
            metrics["calibration_se"] = float(se_slope)
        else:
            metrics["pearson_r"] = 0.0
            metrics["pearson_p"] = 1.0
            metrics["spearman_rho"] = 0.0
            metrics["spearman_p"] = 1.0
            metrics["calibration_slope"] = 1.0
            metrics["calibration_intercept"] = 0.0
            metrics["calibration_se"] = 0.0
    else:
        y_true = _coerce_classification_labels(y_true)
        y_pred = _coerce_classification_labels(y_pred)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics["balanced_acc"] = float(balanced_accuracy_score(y_true, y_pred))
        metrics["f1_macro"] = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["f1_weighted"] = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
        metrics["precision_macro"] = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        metrics["recall_macro"] = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        if y_prob is not None:
            try:
                if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                    metrics["log_loss"] = float(log_loss(y_true, y_prob))
                elif y_prob.ndim == 2 and y_prob.shape[1] > 2:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_prob, multi_class="ovr")
                    )
                    metrics["log_loss"] = float(log_loss(y_true, y_prob))
            except Exception:
                metrics["roc_auc"] = np.nan
                metrics["log_loss"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# Multiplicity (multiple-comparison) correction — Fix 2
#
# Every function below is NaN-safe: p-values that are NaN (e.g. from a
# failed SEM path fit, or a model that failed to train) are excluded from
# the correction's family size and passed through as NaN in the output,
# rather than corrupting the correction applied to the valid p-values or
# silently being treated as p=0/p=1.
# ---------------------------------------------------------------------------


def _nan_safe_correction(pvalues: Any, fn) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    if not np.any(valid_mask):
        return out
    out[valid_mask] = fn(p[valid_mask])
    return out


def bonferroni_correction(pvalues: Any) -> np.ndarray:
    """Bonferroni family-wise-error correction: p_adj = min(p * m, 1)."""

    def _apply(p: np.ndarray) -> np.ndarray:
        m = len(p)
        return np.clip(p * m, 0.0, 1.0)

    return _nan_safe_correction(pvalues, _apply)


def holm_correction(pvalues: Any) -> np.ndarray:
    """Holm-Bonferroni step-down correction (Holm, 1979).

    Uniformly more powerful than plain Bonferroni while still controlling
    the family-wise error rate under arbitrary dependence among the tests,
    which makes it a safe default when the dependence structure among SEM
    structural paths or model-comparison tests is not known. This is the
    correction METIS defaults to for both SEM path significance and
    pairwise model-comparison significance (Fix 2).
    """

    def _apply(p: np.ndarray) -> np.ndarray:
        m = len(p)
        order = np.argsort(p, kind="mergesort")
        adjusted_sorted = np.empty(m, dtype=float)
        running_max = 0.0
        for rank, idx in enumerate(order):
            candidate = (m - rank) * p[idx]
            running_max = max(running_max, candidate)
            adjusted_sorted[rank] = min(running_max, 1.0)
        out = np.empty(m, dtype=float)
        out[order] = adjusted_sorted
        return out

    return _nan_safe_correction(pvalues, _apply)


def benjamini_hochberg_correction(pvalues: Any) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate correction (1995).

    Less conservative than Holm/Bonferroni (controls expected proportion of
    false discoveries rather than the probability of any false discovery),
    offered as an alternative for exploratory contexts where FWER control
    is judged unnecessarily strict.
    """

    def _apply(p: np.ndarray) -> np.ndarray:
        m = len(p)
        order = np.argsort(p, kind="mergesort")
        ranked = p[order]
        raw = ranked * m / (np.arange(m) + 1)
        # Enforce monotonicity: adjusted p-values must be non-decreasing
        # when read in order of *decreasing* rank (standard BH step-up).
        adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]
        adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
        out = np.empty(m, dtype=float)
        out[order] = adjusted_sorted
        return out

    return _nan_safe_correction(pvalues, _apply)


def apply_multiplicity_correction(pvalues: Any, method: str = "holm") -> np.ndarray:
    """Dispatch to the requested multiplicity-correction method.

    `method` must be one of `config.schema.ALLOWED_MULTIPLICITY_METHODS`
    ("holm" | "bonferroni" | "fdr_bh" | "none"). Passing "none" returns the
    raw p-values unchanged (still useful as a uniform code path for callers
    that want to always report both raw and "adjusted" columns, even when
    the adjustment is a no-op by explicit user choice).
    """
    normalized = (method or "none").strip().lower()
    p = np.asarray(pvalues, dtype=float)
    if p.size == 0:
        return p.copy()
    if normalized == "none":
        return p.copy()
    if normalized == "bonferroni":
        return bonferroni_correction(p)
    if normalized == "fdr_bh":
        return benjamini_hochberg_correction(p)
    if normalized == "holm":
        return holm_correction(p)
    msg = (
        f"Unknown multiplicity correction method: {method!r}. Must be one "
        f"of 'holm', 'bonferroni', 'fdr_bh', 'none'."
    )
    raise ValueError(msg)


def significance_stars(p_adjusted: float) -> str:
    """Significance stars computed from an *adjusted* p-value. Centralized
    here so every call site (SEM structural paths, mediation effects, model
    comparisons) uses identical, adjusted-p-based thresholds rather than
    ad hoc raw-p thresholds scattered across the codebase."""
    if p_adjusted is None or not np.isfinite(p_adjusted):
        return ""
    if p_adjusted < 0.001:
        return "***"
    if p_adjusted < 0.01:
        return "**"
    if p_adjusted < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
#
# IMPORTANT INTERPRETATION NOTE (Fix 7): the bootstrap CIs below resample
# rows from an already-fixed set of out-of-fold predictions (y_true,
# y_pred) with replacement, and recompute the metric on each resample. This
# captures sampling variability in *which observations* ended up
# contributing to the metric, conditional on the model(s) that produced
# those predictions already being fixed. It does NOT capture the
# additional uncertainty that would come from refitting the model(s) on
# resampled training data (i.e., cross-validation-and-refit uncertainty).
# It is therefore best described as a "prediction-resampling bootstrap
# interval conditional on the fitted out-of-fold predictions", not as a
# comprehensive predictive-uncertainty interval. This description is
# surfaced verbatim in the HTML/LaTeX report next to every CI produced by
# these functions (see core.reporting) rather than left implicit.
# ---------------------------------------------------------------------------

BOOTSTRAP_CI_METHOD_DESCRIPTION = (
    "Prediction-resampling bootstrap interval, conditional on the fitted "
    "out-of-fold predictions (rows resampled with replacement; models are "
    "not refit per resample). Does not include uncertainty from refitting "
    "models on resampled training data."
)


@dataclass
class BootstrapCIResult:
    point: float
    ci_lower: float
    ci_upper: float
    n_boot_requested: int
    n_boot_successful: int
    method_description: str = BOOTSTRAP_CI_METHOD_DESCRIPTION

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.point, self.ci_lower, self.ci_upper)


def mask_valid_oof_predictions(
    y_true: np.ndarray,
    oof_predictions: np.ndarray,
    task_type: AnalysisTaskType,
) -> np.ndarray:
    """Boolean mask selecting rows where `oof_predictions` holds a real
    prediction rather than the unset sentinel (NaN for regression, None for
    classification) left behind by a fold that failed during
    `evaluate_model_nested_cv`/`run_confirmatory_model_selection_cv`.

    A fold-level failure must never silently bias downstream metrics --
    masking these rows out (rather than, say, letting a stray NaN
    propagate into `accuracy_score`/`r2_score` and either crash or, worse,
    silently produce a NaN/garbage result further downstream) makes the
    exclusion explicit at the point of use, consistent with the failed
    fold already being recorded and reported at the `FoldResult` level.
    """
    if task_type == AnalysisTaskType.REGRESSION:
        arr = np.asarray(oof_predictions, dtype=float)
        return np.isfinite(arr)
    return np.array(
        [
            v is not None and not (isinstance(v, float) and np.isnan(v))
            for v in oof_predictions
        ],
        dtype=bool,
    )


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    task_type: AnalysisTaskType = AnalysisTaskType.REGRESSION,
    n_boot: int = 1000,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    boot_stats = np.empty(n_boot, dtype=np.float64)

    has_groups = groups is not None and len(groups) == n_samples
    unique_groups: np.ndarray = np.array([])
    group_to_indices: dict[Any, np.ndarray] = {}
    n_groups = 0

    if has_groups and groups is not None:
        unique_groups = np.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)

    for i in range(n_boot):
        if has_groups and n_groups > 0:
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        else:
            idx = rng.integers(0, n_samples, n_samples)

        yt_b, yp_b = y_true[idx], y_pred[idx]
        try:
            val = calculate_single_metric(yt_b, yp_b, metric_name, task_type)
        except Exception:
            val = np.nan
        boot_stats[i] = float(cast(Any, val))

    valid = boot_stats[np.isfinite(boot_stats)]
    try:
        point = calculate_single_metric(y_true, y_pred, metric_name, task_type)
        if np.isnan(point) and len(valid) > 0:
            point = float(np.mean(valid))
    except Exception:
        point = float(np.mean(valid)) if len(valid) > 0 else np.nan

    if len(valid) == 0:
        return float(cast(Any, point)), np.nan, np.nan

    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    return float(cast(Any, point)), float(ci_lo), float(ci_hi)


def bootstrap_metric_ci_detailed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: str,
    task_type: AnalysisTaskType = AnalysisTaskType.REGRESSION,
    n_boot: int = 1000,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> BootstrapCIResult:
    """Same computation as `bootstrap_metric_ci`, but returns a
    `BootstrapCIResult` carrying the explicit method description and the
    number of bootstrap resamples that actually produced a finite metric
    value (a resample can fail, e.g. a degenerate resampled class
    distribution for a classification metric), so reporting code can be
    transparent about both rather than presenting a bare CI as if it were
    unconditionally reliable."""
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    boot_stats = np.empty(n_boot, dtype=np.float64)

    has_groups = groups is not None and len(groups) == n_samples
    unique_groups: np.ndarray = np.array([])
    group_to_indices: dict[Any, np.ndarray] = {}
    n_groups = 0
    if has_groups and groups is not None:
        unique_groups = np.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)

    for i in range(n_boot):
        if has_groups and n_groups > 0:
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        else:
            idx = rng.integers(0, n_samples, n_samples)
        yt_b, yp_b = y_true[idx], y_pred[idx]
        try:
            val = calculate_single_metric(yt_b, yp_b, metric_name, task_type)
        except Exception:
            val = np.nan
        boot_stats[i] = float(cast(Any, val))

    valid = boot_stats[np.isfinite(boot_stats)]
    try:
        point = calculate_single_metric(y_true, y_pred, metric_name, task_type)
        if np.isnan(point) and len(valid) > 0:
            point = float(np.mean(valid))
    except Exception:
        point = float(np.mean(valid)) if len(valid) > 0 else np.nan

    if len(valid) == 0:
        return BootstrapCIResult(
            point=float(cast(Any, point)),
            ci_lower=np.nan,
            ci_upper=np.nan,
            n_boot_requested=n_boot,
            n_boot_successful=0,
        )

    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    return BootstrapCIResult(
        point=float(cast(Any, point)),
        ci_lower=float(ci_lo),
        ci_upper=float(ci_hi),
        n_boot_requested=n_boot,
        n_boot_successful=len(valid),
    )


def bootstrap_paired_difference_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric_name: str = "r2",
    task_type: AnalysisTaskType = AnalysisTaskType.REGRESSION,
    n_boot: int = 1000,
    random_state: int = 42,
    groups: np.ndarray | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    diff_stats = np.empty(n_boot, dtype=np.float64)

    has_groups = groups is not None and len(groups) == n_samples
    unique_groups: np.ndarray = np.array([])
    group_to_indices: dict[Any, np.ndarray] = {}
    n_groups = 0

    if has_groups and groups is not None:
        unique_groups = np.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)

    for i in range(n_boot):
        if has_groups and n_groups > 0:
            sampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
            idx = np.concatenate([group_to_indices[g] for g in sampled_groups])
        else:
            idx = rng.integers(0, n_samples, n_samples)

        yt_b = y_true[idx]
        yp_a_b = y_pred_a[idx]
        yp_b_b = y_pred_b[idx]
        try:
            val_a = calculate_single_metric(yt_b, yp_a_b, metric_name, task_type)
            val_b = calculate_single_metric(yt_b, yp_b_b, metric_name, task_type)
            diff_stats[i] = float(cast(Any, val_a)) - float(cast(Any, val_b))
        except Exception:
            diff_stats[i] = np.nan

    valid = diff_stats[np.isfinite(diff_stats)]
    try:
        val_a_pt = calculate_single_metric(y_true, y_pred_a, metric_name, task_type)
        val_b_pt = calculate_single_metric(y_true, y_pred_b, metric_name, task_type)
        point_diff = float(cast(Any, val_a_pt)) - float(cast(Any, val_b_pt))
    except Exception:
        point_diff = float(np.mean(valid)) if len(valid) > 0 else 0.0

    if len(valid) == 0:
        return point_diff, np.nan, np.nan

    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    return point_diff, float(ci_lo), float(ci_hi)
