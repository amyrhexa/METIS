# core/interpretation.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from config.schema import AnalysisTaskType

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": "#2E3440",
        "axes.labelcolor": "#2E3440",
        "axes.linewidth": 1.0,
        "text.color": "#2E3440",
        "xtick.color": "#2E3440",
        "ytick.color": "#2E3440",
        "grid.color": "#E5E9F0",
        "grid.linestyle": "--",
        "grid.alpha": 0.8,
    }
)


@dataclass
class CVStableAttributionResult:
    fold_importance_records: list[dict[str, Any]] = field(default_factory=list)
    fold_shap_records: list[dict[str, Any]] = field(default_factory=list)
    shap_values_array: np.ndarray | None = None
    X_test_transformed: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    shap_status: str = "not_requested"
    permutation_status: str = "ok"


def compute_cv_stable_attribution(
    model_spec: Any,
    fold_estimators: list[tuple[int, Pipeline, np.ndarray]],
    X: pd.DataFrame,
    y: np.ndarray,
    task_type: AnalysisTaskType,
    n_jobs: int = -1,
    random_state: int = 42,
    compute_shap: bool = True,
    permutation_n_repeats: int = 20,
    logger: logging.Logger | None = None,
) -> CVStableAttributionResult:
    fold_importance_records: list[dict[str, Any]] = []
    fold_shap_records: list[dict[str, Any]] = []
    shap_values_list: list[np.ndarray] = []
    X_test_transformed_list: list[np.ndarray] = []

    shap_status = "not_requested"
    perm_ran = False

    if compute_shap:
        if not HAS_SHAP or shap is None:
            shap_status = "skipped: shap not installed"
            if logger:
                logger.warning(
                    "SHAP attribution skipped: 'shap' package is not installed."
                )
        elif not getattr(model_spec, "is_tree_based", False):
            shap_status = "skipped: non-tree estimator without tree fallback"
            if logger:
                logger.warning(
                    "SHAP attribution skipped for estimator '%s' (non-tree based).",
                    getattr(model_spec, "display_name", str(model_spec)),
                )
    else:
        shap_status = "skipped: skip_shap is enabled"
        if logger:
            logger.warning("SHAP attribution skipped: skip_shap configuration is set.")

    # 1. Establish unified master feature index across all fold preprocessors
    master_features_dict: dict[str, None] = {}
    for _, fitted_estimator, _ in fold_estimators:
        if "preprocess" in fitted_estimator.named_steps:
            preproc = fitted_estimator.named_steps["preprocess"]
            try:
                for fn in preproc.get_feature_names_out():
                    master_features_dict[str(fn)] = None
            except Exception:
                pass

    master_features = list(master_features_dict.keys())
    master_feature_map = {fn: idx for idx, fn in enumerate(master_features)}
    last_fold_feature_names: list[str] = []

    for fold_idx, fitted_estimator, test_idx in fold_estimators:
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]
        preproc = fitted_estimator.named_steps.get("preprocess")
        fitted_model = fitted_estimator.named_steps.get("model")

        if preproc is None or fitted_model is None:
            continue

        try:
            feature_names = [str(fn) for fn in preproc.get_feature_names_out()]
        except Exception:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]
        last_fold_feature_names = feature_names

        X_test_transformed = preproc.transform(X_test)
        if hasattr(X_test_transformed, "toarray"):
            X_test_transformed = X_test_transformed.toarray()
        else:
            X_test_transformed = np.asarray(X_test_transformed, dtype=float)

        shap_ok = False
        if (
            compute_shap
            and getattr(model_spec, "is_tree_based", False)
            and HAS_SHAP
            and shap is not None
        ):
            try:
                explainer = shap.TreeExplainer(fitted_model)
                sv = explainer.shap_values(X_test_transformed)

                if isinstance(sv, list):
                    shap_values_array = sv[1] if len(sv) == 2 else sv[0]
                    mean_abs = np.mean([np.abs(s).mean(axis=0) for s in sv], axis=0)
                elif hasattr(sv, "values"):
                    shap_values_array = cast(Any, sv).values
                    if shap_values_array.ndim == 3:
                        shap_values_array = (
                            shap_values_array[:, :, 1]
                            if shap_values_array.shape[2] == 2
                            else shap_values_array[:, :, 0]
                        )
                    mean_abs = np.abs(shap_values_array).mean(axis=0)
                else:
                    shap_values_array = np.asarray(sv, dtype=float)
                    if shap_values_array.ndim == 3:
                        shap_values_array = (
                            shap_values_array[:, :, 1]
                            if shap_values_array.shape[2] == 2
                            else shap_values_array[:, :, 0]
                        )
                    mean_abs = np.abs(shap_values_array).mean(axis=0)

                n_test = X_test_transformed.shape[0]
                if master_features:
                    reindexed_shap = np.zeros(
                        (n_test, len(master_features)), dtype=float
                    )
                    reindexed_x = np.zeros((n_test, len(master_features)), dtype=float)
                    for col_i, fn in enumerate(feature_names):
                        if (
                            fn in master_feature_map
                            and col_i < shap_values_array.shape[1]
                        ):
                            m_idx = master_feature_map[fn]
                            reindexed_shap[:, m_idx] = shap_values_array[:, col_i]
                            reindexed_x[:, m_idx] = X_test_transformed[:, col_i]
                    shap_values_list.append(reindexed_shap)
                    X_test_transformed_list.append(reindexed_x)
                else:
                    shap_values_list.append(shap_values_array)
                    X_test_transformed_list.append(X_test_transformed)

                for fn, val in zip(feature_names, mean_abs, strict=False):
                    fold_shap_records.append(
                        {"fold": fold_idx, "feature": fn, "value": float(val)}
                    )
                shap_status = "ok"
                shap_ok = True
            except Exception as exc:
                shap_status = f"failed: {exc}"
                if logger:
                    logger.warning(
                        "[Fold %d] TreeExplainer failed for model '%s': %s. Falling back to permutation.",
                        fold_idx,
                        getattr(model_spec, "display_name", str(model_spec)),
                        exc,
                    )

        if not shap_ok:
            perm_ran = True
            scoring = (
                "accuracy" if task_type == AnalysisTaskType.CLASSIFICATION else "r2"
            )
            try:
                perm_res: Any = permutation_importance(
                    fitted_model,
                    X_test_transformed,
                    y_test,
                    n_repeats=permutation_n_repeats,
                    random_state=random_state,
                    n_jobs=n_jobs,
                    scoring=scoring,
                )
                for fn, imp in zip(
                    feature_names, perm_res.importances_mean, strict=False
                ):
                    fold_importance_records.append(
                        {"fold": fold_idx, "feature": fn, "value": float(imp)}
                    )
            except Exception as exc:
                if logger:
                    logger.error(
                        "[Fold %d] Permutation importance failed: %s", fold_idx, exc
                    )

    pooled_shap = np.vstack(shap_values_list) if shap_values_list else None
    pooled_x = np.vstack(X_test_transformed_list) if X_test_transformed_list else None
    final_feature_names = (
        master_features
        if master_features
        else (
            last_fold_feature_names
            if last_fold_feature_names
            else [str(c) for c in X.columns]
        )
    )

    return CVStableAttributionResult(
        fold_importance_records=fold_importance_records,
        fold_shap_records=fold_shap_records,
        shap_values_array=pooled_shap,
        X_test_transformed=pooled_x,
        feature_names=final_feature_names,
        shap_status=shap_status,
        permutation_status="ok" if perm_ran else "bypassed_shap_success",
    )


def aggregate_fold_importance(records: list[dict[str, Any]]) -> pd.DataFrame:
    cols = [
        "feature",
        "mean_importance",
        "std_importance",
        "median_importance",
        "coefficient_of_variation",
        "positive_fold_fraction",
        "n_valid_folds",
    ]
    if not records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(records)
    summary = (
        df.groupby("feature")["value"]
        .agg(
            mean_importance="mean",
            std_importance="std",
            median_importance="median",
            n_valid_folds="count",
        )
        .reset_index()
    )
    summary["std_importance"] = summary["std_importance"].fillna(0.0)
    abs_mean = summary["mean_importance"].abs()
    summary["coefficient_of_variation"] = np.where(
        abs_mean > 1e-12,
        summary["std_importance"] / abs_mean,
        np.nan,
    )
    pos_frac = (
        df.groupby("feature")["value"]
        .apply(lambda s: float((s > 0).mean()))
        .rename("positive_fold_fraction")
        .reset_index()
    )
    summary = summary.merge(pos_frac, on="feature")
    return (
        summary[cols]
        .sort_values("mean_importance", ascending=False)
        .reset_index(drop=True)
    )


def plot_feature_importance_bar(
    agg_importance: pd.DataFrame,
    target_name: str,
    model_name: str,
    top_k: int = 15,
) -> Figure:
    top_df = agg_importance.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(
        figsize=(9, max(4.2, 0.4 * max(1, len(top_df)))), facecolor="#ffffff"
    )
    ax.set_facecolor("#ffffff")

    if top_df.empty:
        ax.text(
            0.5,
            0.5,
            "No feature attribution records available",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        return fig

    colors = sns.color_palette("mako", len(top_df))
    bars = ax.barh(
        top_df["feature"],
        top_df["mean_importance"],
        xerr=top_df["std_importance"],
        color=colors,
        edgecolor="#2E3440",
        linewidth=0.8,
        capsize=4,
        ecolor="#4C566A",
    )

    ax.set_title(
        f"Cross-Validation Stable Feature Attribution\nModel: {model_name} | Target: {target_name}",
        fontsize=11.5,
        fontweight="bold",
        color="#2E3440",
        pad=14,
    )
    ax.set_xlabel(
        "Mean Absolute Attribution (± SD across CV folds)",
        fontsize=9.5,
        color="#2E3440",
    )
    ax.set_ylabel("Predictor Variable", fontsize=9.5, color="#2E3440")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.7, color="#E5E9F0")

    for bar, mean_val in zip(bars, top_df["mean_importance"], strict=False):
        ax.text(
            bar.get_width() * 1.02 if mean_val >= 0 else bar.get_width() * 0.98,
            bar.get_y() + bar.get_height() / 2.0,
            f"{mean_val:.3f}",
            va="center",
            ha="left" if mean_val >= 0 else "right",
            fontsize=8.5,
            color="#2E3440",
            fontweight="bold",
        )

    fig.tight_layout()
    return fig


def plot_shap_beeswarm(
    shap_values_array: np.ndarray,
    X_data: np.ndarray,
    feature_names: list[str],
    target_name: str,
    model_name: str,
    top_k: int = 15,
    logger: logging.Logger | None = None,
) -> Figure | None:
    if not HAS_SHAP or shap is None or shap_values_array is None or X_data is None:
        if logger:
            logger.warning(
                "plot_shap_beeswarm skipped: SHAP not available or input arrays are None."
            )
        return None
    try:
        sv_arr = np.atleast_2d(shap_values_array)
        xd_arr = np.atleast_2d(X_data)

        if sv_arr.ndim == 2 and xd_arr.ndim == 2:
            if sv_arr.shape[1] != xd_arr.shape[1]:
                if logger:
                    logger.error(
                        "Shape mismatch in plot_shap_beeswarm: shap_values_array columns (%d) != X_data columns (%d)",
                        sv_arr.shape[1],
                        xd_arr.shape[1],
                    )
                return None
            if len(feature_names) != xd_arr.shape[1]:
                if logger:
                    logger.error(
                        "Feature names length (%d) != X_data columns (%d)",
                        len(feature_names),
                        xd_arr.shape[1],
                    )
                return None

        plt.figure(
            figsize=(9.5, max(4.8, 0.44 * min(top_k, max(1, len(feature_names))))),
            facecolor="#ffffff",
        )
        explanation = shap.Explanation(
            values=sv_arr,
            data=xd_arr,
            feature_names=feature_names,
        )
        shap.plots.beeswarm(
            explanation,
            max_display=top_k,
            show=False,
            plot_size=None,
            color_bar=True,
        )
        fig = plt.gcf()
        fig.patch.set_facecolor("#ffffff")
        if len(fig.axes) > 0:
            main_ax = fig.axes[0]
            main_ax.set_xlabel("SHAP value (impact on model output)")
        if len(fig.axes) > 1:
            for ax_i in fig.axes[1:]:
                lo, hi = ax_i.get_ylim()
                ax_i.set_yticks([lo, hi])
                ax_i.set_yticklabels(["Low", "High"])
                ax_i.set_ylabel("Feature value")
        for ax_i in fig.axes:
            ax_i.set_facecolor("#ffffff")
        fig.tight_layout()
        return fig
    except Exception as exc:
        if logger:
            logger.error(
                "Failed to generate SHAP beeswarm plot for model '%s' (target '%s'): %s",
                model_name,
                target_name,
                exc,
            )
        return None


def plot_shap_dependence(
    shap_values_array: np.ndarray,
    X_data: np.ndarray,
    feature_names: list[str],
    target_name: str,
    top_feature_idx: int = 0,
) -> Figure | None:
    if not HAS_SHAP or shap is None or shap_values_array is None or X_data is None:
        return None

    try:
        sv_arr = np.atleast_2d(shap_values_array)
        xd_arr = np.atleast_2d(X_data)

        if (
            top_feature_idx >= len(feature_names)
            or top_feature_idx >= xd_arr.shape[1]
            or top_feature_idx >= sv_arr.shape[1]
        ):
            return None

        feat_name = feature_names[top_feature_idx]
        fig, ax = plt.subplots(figsize=(7.5, 5.2), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")

        x_vals = xd_arr[:, top_feature_idx]
        sv_vals = sv_arr[:, top_feature_idx]

        scatter = ax.scatter(
            x_vals,
            sv_vals,
            c=x_vals,
            cmap="viridis",
            alpha=0.8,
            edgecolors="none",
            s=42,
        )

        sns.regplot(
            x=x_vals,
            y=sv_vals,
            scatter=False,
            ax=ax,
            color="#BF616A",
            line_kws={"linewidth": 2, "linestyle": "-"},
        )

        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(f"{feat_name} Value", fontsize=9, color="#2E3440")
        cbar.ax.yaxis.set_tick_params(color="#2E3440")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#2E3440")

        ax.set_title(
            f"SHAP Dependence Profile: {feat_name}\nTarget: {target_name}",
            fontsize=11,
            fontweight="bold",
            color="#2E3440",
            pad=12,
        )
        ax.set_xlabel(
            f"Observed {feat_name} (Standardized)", fontsize=9.5, color="#2E3440"
        )
        ax.set_ylabel(
            f"SHAP Marginal Attribution ({target_name})", fontsize=9.5, color="#2E3440"
        )
        ax.axhline(0, color="#D8DEE9", linestyle="--", linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", alpha=0.6, color="#E5E9F0")

        fig.tight_layout()
        return fig
    except Exception:
        return None


def plot_attribution_stability(
    agg_importance: pd.DataFrame,
    target_name: str,
    top_k: int = 12,
) -> Figure:
    top_df = agg_importance.head(top_k).copy()
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")

    if top_df.empty or "coefficient_of_variation" not in top_df.columns:
        ax.text(
            0.5,
            0.5,
            "No attribution stability records available",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        return fig

    valid_df = top_df.dropna(subset=["coefficient_of_variation"])
    if valid_df.empty:
        ax.text(
            0.5,
            0.5,
            "Insufficient variance across folds to compute stability",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        return fig

    sns.barplot(
        data=valid_df,
        x="feature",
        y="coefficient_of_variation",
        hue="feature",
        palette="crest",
        legend=False,
        ax=ax,
        edgecolor="#2E3440",
        linewidth=0.8,
    )
    ax.set_title(
        f"Cross-Validation Attribution Stability (CV = SD / |Mean|)\nTarget: {target_name} (Lower = More Generalizable)",
        fontsize=11,
        fontweight="bold",
        color="#2E3440",
        pad=14,
    )
    ax.set_xlabel("Top Predictors", fontsize=9.5, color="#2E3440")
    ax.set_ylabel("Coefficient of Variation", fontsize=9.5, color="#2E3440")
    plt.setp(
        ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5, color="#2E3440"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.7, color="#E5E9F0")

    fig.tight_layout()
    return fig


def save_plot(fig: Figure, path_no_ext: Path, dpi: int = 300) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path_no_ext) + ".png", dpi=dpi, bbox_inches="tight", facecolor="#ffffff"
    )
    fig.savefig(str(path_no_ext) + ".pdf", bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
