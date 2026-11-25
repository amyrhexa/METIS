# core/reporting.py
from __future__ import annotations

import base64
import html
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.patches as patches

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from config.schema import PipelineConfig
from core.data_loader import DatasetProfile
from core.evaluation import (
    BOOTSTRAP_CI_METHOD_DESCRIPTION,
    _coerce_classification_labels,
)
from core.interpretation import HAS_SHAP, save_plot
from core.modeling import LearningCurveData
from core.sem import _safe_float

BOOTSTRAP_CI_NOTE_HTML = (
    f"<i>Uncertainty note: {html.escape(BOOTSTRAP_CI_METHOD_DESCRIPTION)}</i>"
)

logger = logging.getLogger("metis.reporting")

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
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

_REPORT_CSS = """
<style>
  :root {
    --nord0: #2E3440;
    --nord1: #3B4252;
    --nord2: #434C5E;
    --nord3: #4C566A;
    --nord4: #D8DEE9;
    --nord5: #E5E9F0;
    --nord6: #ECEFF4;
    --nord7: #8FBCBB;
    --nord8: #88C0D0;
    --nord9: #81A1C1;
    --nord10: #5E81AC;
    --nord11: #BF616A;
    --nord12: #D08770;
    --nord13: #EBCB8B;
    --nord14: #A3BE8C;
    --nord15: #B48EAD;
  }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Inter, Helvetica, sans-serif;
    background: var(--nord0);
    color: var(--nord6);
    margin: 0;
    padding: 0;
    line-height: 1.6;
    font-size: 14px;
    -webkit-font-smoothing: antialiased;
  }
  .container { max-width: 1480px; margin: 0 auto; padding: 48px 32px 96px; }
  .header {
    border-bottom: 2px solid var(--nord2);
    padding-bottom: 28px;
    margin-bottom: 36px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 16px;
  }
  h1 { font-size: 28px; margin: 0 0 8px 0; color: var(--nord6); font-weight: 700; letter-spacing: -0.03em; }
  .sub-title { color: var(--nord4); font-size: 14px; }
  h2 { font-size: 20px; border-left: 4px solid var(--nord8); padding-left: 14px; margin: 48px 0 22px; color: var(--nord6); font-weight: 600; }
  h3 { font-size: 16px; margin: 26px 0 14px; color: var(--nord8); font-weight: 600; }
  h4 { font-size: 13px; margin: 18px 0 10px; color: var(--nord4); text-transform: uppercase; letter-spacing: 0.06em; }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-bottom: 28px; }
  .card { background: var(--nord1); border: 1px solid var(--nord2); border-radius: 10px; padding: 20px; }
  .card:hover { border-color: var(--nord9); }
  .card-label { font-size: 11px; text-transform: uppercase; font-weight: 700; color: var(--nord4); letter-spacing: 0.05em; }
  .card-val { font-size: 24px; font-weight: 700; color: var(--nord8); margin-top: 6px; }
  .card-sub { font-size: 12px; color: var(--nord4); margin-top: 4px; }

  .panel { background: var(--nord1); border: 1px solid var(--nord2); border-radius: 10px; padding: 24px; margin-bottom: 28px; }
  .badge { display: inline-flex; align-items: center; padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600; }
  .badge-success { background: rgba(163, 190, 140, 0.2); color: var(--nord14); border: 1px solid var(--nord14); }
  .badge-warning { background: rgba(235, 203, 139, 0.2); color: var(--nord13); border: 1px solid var(--nord13); }
  .badge-info { background: rgba(136, 192, 208, 0.2); color: var(--nord8); border: 1px solid var(--nord8); }
  .badge-purple { background: rgba(180, 142, 173, 0.2); color: var(--nord15); border: 1px solid var(--nord15); }

  .table-wrap { overflow-x: auto; border: 1px solid var(--nord2); border-radius: 10px; margin-bottom: 24px; background: var(--nord1); }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; text-align: left; }
  th, td { padding: 12px 16px; white-space: nowrap; }
  th { background: var(--nord0); color: var(--nord4); font-size: 11.5px; text-transform: uppercase; font-weight: 700; border-bottom: 1px solid var(--nord2); }
  td { border-top: 1px solid var(--nord2); color: var(--nord6); }
  tr:hover td { background: rgba(136, 192, 208, 0.05); }
  .winner-row td { background: rgba(163, 190, 140, 0.12); font-weight: 600; }

  .img-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 22px; margin: 20px 0; }
  .img-card { border: 1px solid var(--nord2); border-radius: 10px; padding: 16px; background: #ffffff; color: var(--nord0); }
  .img-card h4 { color: var(--nord0); margin-top: 0; }
  .img-card img { display: block; width: 100%; height: auto; border-radius: 4px; background: #ffffff; }

  details.target-card { background: var(--nord1); border: 1px solid var(--nord2); border-radius: 10px; margin-bottom: 24px; overflow: hidden; }
  details.target-card > summary { padding: 20px 24px; cursor: pointer; font-weight: 600; display: flex; justify-content: space-between; align-items: center; user-select: none; }
  details.target-card > summary:hover { background: rgba(255, 255, 255, 0.03); }
  .details-body { padding: 26px; border-top: 1px solid var(--nord2); background: #2b313d; }

  .callout { background: rgba(136, 192, 208, 0.08); border-left: 4px solid var(--nord8); padding: 16px 20px; border-radius: 0 8px 8px 0; margin-bottom: 22px; }
  .callout p { margin: 0; color: var(--nord5); font-size: 13.5px; }
  .latex-box { background: var(--nord0); border: 1px solid var(--nord2); border-left: 4px solid var(--nord15); padding: 18px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--nord5); overflow-x: auto; white-space: pre; border-radius: 0 8px 8px 0; margin-bottom: 20px; }
</style>
"""


def _apply_base_plot_style(
    ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = ""
) -> None:
    ax.set_facecolor("#ffffff")
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#2E3440", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color="#2E3440")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color="#2E3440")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def encode_img_base64(path: Path, title: str | None = None) -> str:
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    try:
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        header = f'<h4 style="margin-top:0;">{html.escape(title or path.stem.replace("__", " - "))}</h4>'
        return f'<div class="img-card">{header}<img src="data:image/png;base64,{b64}" alt="{html.escape(path.name)}"/></div>'
    except Exception as exc:
        logger.warning("Failed to encode base64 image %s: %s", path.name, exc)
        return ""


def generate_calibration_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
    out_dir: Path,
) -> Path:
    fig_dir = out_dir / "figures" / "explainability"
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    slope, intercept, r_val, _, _ = stats.linregress(y_p, y_t)
    r2 = float(r_val) ** 2
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = np.mean(np.abs(y_t - y_p))

    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor="#ffffff")

    ax.scatter(
        y_p,
        y_t,
        alpha=0.55,
        color="#5E81AC",
        edgecolors="none",
        s=38,
        label="Out-of-Fold Observations",
    )

    lims = [
        min(float(y_t.min()), float(y_p.min())),
        max(float(y_t.max()), float(y_p.max())),
    ]
    ax.plot(
        lims,
        lims,
        "--",
        color="#4C566A",
        linewidth=1.5,
        label="Perfect Calibration (Identity 1:1)",
    )

    x_line = np.linspace(lims[0], lims[1], 100)
    ax.plot(
        x_line,
        float(intercept) + float(slope) * x_line,
        color="#BF616A",
        linewidth=2.0,
        label=f"Fit (Slope={float(slope):.2f}, Intercept={float(intercept):.2f})",
    )

    _apply_base_plot_style(
        ax,
        title=f"Prediction Calibration Analysis\nTarget: {target_name} | Estimator: {model_name}",
        xlabel="Predicted Value (Out-of-Fold Ŷ)",
        ylabel="Observed Ground Truth (Y)",
    )
    ax.title.set_fontsize(11.5)

    stat_box = (
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\nSlope = {float(slope):.3f}"
    )
    ax.text(
        0.05,
        0.95,
        stat_box,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9.0,
        color="#2E3440",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="#ECEFF4", edgecolor="#D8DEE9", lw=0.8
        ),
    )

    ax.legend(loc="lower right", fontsize=8.5, facecolor="#ffffff", edgecolor="#D8DEE9")
    ax.grid(True, linestyle="--", alpha=0.6, color="#E5E9F0")

    fig.tight_layout()
    out_path = fig_dir / f"{target_name}__calibration_plot"
    save_plot(fig, out_path)
    return out_path.with_suffix(".png")


def generate_learning_curve_plot(
    lc_data: LearningCurveData,
    target_name: str,
    model_name: str,
    out_dir: Path,
) -> Path | None:
    if not lc_data.train_sizes:
        return None

    try:
        fig_dir = out_dir / "figures" / "performance"
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8.0, 5.2), facecolor="#ffffff")

        sizes = np.array(lc_data.train_sizes)
        tr_m = np.array(lc_data.train_scores_mean)
        tr_s = np.array(lc_data.train_scores_std)
        va_m = np.array(lc_data.val_scores_mean)
        va_s = np.array(lc_data.val_scores_std)

        ax.plot(
            sizes, tr_m, "o-", color="#5E81AC", linewidth=2.0, label="Training Score"
        )
        ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="#5E81AC")

        ax.plot(
            sizes,
            va_m,
            "s-",
            color="#A3BE8C",
            linewidth=2.0,
            label="Cross-Validation Score",
        )
        ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15, color="#A3BE8C")

        _apply_base_plot_style(
            ax,
            title=f"Empirical Learning Curve & Sample Efficiency\nTarget: {target_name} ({model_name})",
            xlabel="Training Sample Size (N)",
            ylabel="Cross-Validation Generalization Score",
        )
        ax.title.set_fontsize(11.5)

        ax.legend(
            loc="lower right", fontsize=9.0, facecolor="#ffffff", edgecolor="#D8DEE9"
        )
        ax.grid(True, linestyle="--", alpha=0.7, color="#E5E9F0")

        fig.tight_layout()
        out_path = fig_dir / f"{target_name}__learning_curve"
        save_plot(fig, out_path)
        return out_path.with_suffix(".png")
    except Exception as exc:
        logger.warning(
            "Failed to generate learning curve plot for %s: %s", target_name, exc
        )
        return None


def generate_dataset_eda_plots(
    dataset_df: pd.DataFrame,
    profile: DatasetProfile,
    out_dir: Path,
    selected_columns: list[str] | None = None,
) -> dict[str, Path]:
    fig_dir = out_dir / "figures" / "dataset"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if selected_columns:
        valid_cols = [c for c in selected_columns if c in dataset_df.columns]
        eda_df = dataset_df[valid_cols] if valid_cols else dataset_df
    else:
        eda_df = dataset_df

    if eda_df.isna().sum().sum() > 0:
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#ffffff")
        missing_df = eda_df.isna().astype(int)
        if missing_df.shape[1] > 30:
            top_miss = missing_df.sum().sort_values(ascending=False).head(30).index
            missing_df = missing_df[top_miss]
        sns.heatmap(
            missing_df,
            cbar=False,
            cmap=["#ECEFF4", "#BF616A"],
            yticklabels=False,
            ax=ax,
        )
        _apply_base_plot_style(
            ax,
            title=f"Missing Value Matrix ({len(missing_df.columns)} Selected Variables)",
            xlabel="Columns",
            ylabel="Observations",
        )
        ax.title.set_fontsize(11)
        p = fig_dir / "missingness_matrix"
        save_plot(fig, p)
        paths["missingness"] = p.with_suffix(".png")

    num_cols = [c for c in eda_df.columns if pd.api.types.is_numeric_dtype(eda_df[c])]
    if len(num_cols) >= 2:
        plot_num_cols = num_cols[:25]
        fig_dim = min(12, max(4.5, 2.5 + 0.45 * len(plot_num_cols)))
        fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        corr = eda_df[plot_num_cols].corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=len(plot_num_cols) <= 15,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="#E5E9F0",
            cbar_kws={"shrink": 0.8, "label": "Correlation (r)"},
            ax=ax,
        )
        ax.set_title(
            f"Selected Continuous Predictors Inter-Correlation Matrix ({len(plot_num_cols)} Variables)",
            fontsize=11.5,
            fontweight="bold",
            color="#2E3440",
            pad=12,
        )
        p = fig_dir / "correlation_matrix"
        save_plot(fig, p)
        paths["correlation"] = p.with_suffix(".png")

    return paths


def generate_regression_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
    out_dir: Path,
    covariate_series: pd.Series | None = None,
    covariate_name: str = "Age",
) -> Path:
    fig_dir = out_dir / "figures" / "explainability"
    fig_dir.mkdir(parents=True, exist_ok=True)

    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    residuals = y_t - y_p

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), facecolor="#ffffff")

    ax1 = axes[0, 0]
    ax1.scatter(y_p, residuals, alpha=0.55, s=36, color="#5E81AC", edgecolors="none")
    ax1.axhline(0, color="#BF616A", linestyle="--", linewidth=1.5, label="Zero Bias")
    sns.regplot(
        x=y_p,
        y=residuals,
        scatter=False,
        lowess=True,
        ax=ax1,
        color="#D08770",
        line_kws={"linewidth": 1.6, "label": "Lowess"},
    )
    _apply_base_plot_style(
        ax1,
        title="Residuals vs. Fitted (Homoscedasticity)",
        xlabel="Predicted Value (Ŷ)",
        ylabel="Residual (Y - Ŷ)",
    )
    ax1.legend(
        loc="upper right", fontsize=8.0, facecolor="#ffffff", edgecolor="#D8DEE9"
    )

    ax2 = axes[0, 1]
    sns.histplot(
        residuals, kde=True, ax=ax2, color="#A3BE8C", edgecolor="#2E3440", alpha=0.6
    )
    ax2.axvline(0, color="#BF616A", linestyle="--", linewidth=1.5)
    _apply_base_plot_style(
        ax2,
        title="Residual Distribution & Normality",
        xlabel="Residual Value",
        ylabel="Density / Count",
    )

    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.get_lines()[0].set_color("#5E81AC")
    ax3.get_lines()[0].set_markersize(5.0)
    ax3.get_lines()[1].set_color("#BF616A")
    ax3.get_lines()[1].set_linewidth(1.6)
    _apply_base_plot_style(
        ax3,
        title="Normal Q-Q Plot of Residuals",
        xlabel="Theoretical Quantiles",
        ylabel="Ordered Values",
    )

    ax4 = axes[1, 1]
    if covariate_series is not None and len(covariate_series) == len(residuals):
        cov_vals = covariate_series.to_numpy(dtype=float)
        ax4.scatter(
            cov_vals, residuals, alpha=0.55, s=36, color="#B48EAD", edgecolors="none"
        )
        ax4.axhline(0, color="#BF616A", linestyle="--", linewidth=1.2)
        sns.regplot(
            x=cov_vals,
            y=residuals,
            scatter=False,
            ax=ax4,
            color="#5E81AC",
            line_kws={"linewidth": 1.5},
        )
        r_cov, p_cov = stats.pearsonr(cov_vals, residuals)
        _apply_base_plot_style(
            ax4,
            title=f"Error Association: {covariate_name} (r={r_cov:.2f}, p={p_cov:.3f})",
            xlabel=covariate_name,
            ylabel="Residual (Y - Ŷ)",
        )
    else:
        abs_res = np.abs(residuals)
        ax4.scatter(y_p, abs_res, alpha=0.55, s=36, color="#B48EAD", edgecolors="none")
        sns.regplot(
            x=y_p,
            y=abs_res,
            scatter=False,
            lowess=True,
            ax=ax4,
            color="#5E81AC",
            line_kws={"linewidth": 1.5},
        )
        _apply_base_plot_style(
            ax4,
            title="Absolute Residuals vs. Fitted (Scale-Location)",
            xlabel="Predicted Value (Ŷ)",
            ylabel="|Residual|",
        )

    fig.suptitle(
        f"Comprehensive Regression Diagnostic Suite: {target_name} ({model_name})",
        fontsize=12,
        fontweight="bold",
        color="#2E3440",
        y=0.995,
    )
    fig.tight_layout()

    out_path = fig_dir / f"{target_name}__regression_diagnostics"
    save_plot(fig, out_path)
    return out_path.with_suffix(".png")


def generate_classification_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    target_name: str,
    model_name: str,
    out_dir: Path,
) -> Path:
    fig_dir = out_dir / "figures" / "explainability"
    fig_dir.mkdir(parents=True, exist_ok=True)

    has_proba = y_prob is not None and not np.isnan(y_prob).all()
    n_panels = 3 if has_proba else 1
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.8 * n_panels, 4.4), facecolor="#ffffff"
    )
    if n_panels == 1:
        axes = [axes]

    # Same object-dtype coercion applied in core.evaluation's metric
    # functions (Fix / discovered during remediation self-audit): METIS's
    # OOF prediction accumulator is dtype=object to support both numeric
    # and string class labels uniformly across folds, which sklearn's
    # confusion_matrix cannot type correctly without this re-inference.
    y_true = _coerce_classification_labels(y_true)
    y_pred = _coerce_classification_labels(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )
    annot = np.array(
        [
            [f"{count}\n({pct:.1%})" for count, pct in zip(r, r_n, strict=False)]
            for r, r_n in zip(cm, cm_norm, strict=False)
        ]
    )

    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        linewidths=1.2,
        linecolor="#ffffff",
    )
    _apply_base_plot_style(
        axes[0],
        title=f"Confusion Matrix\n{target_name}",
        xlabel="Predicted Class",
        ylabel="True Class",
    )

    if has_proba and y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        prob_pos = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_true, prob_pos)
        axes[1].plot(fpr, tpr, color="#5E81AC", linewidth=2.0, label="OOF ROC Curve")
        axes[1].plot(
            [0, 1], [0, 1], "--", color="#4C566A", linewidth=1.2, label="Chance"
        )
        _apply_base_plot_style(
            axes[1],
            title="Receiver Operating Characteristic (ROC)",
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
        )
        axes[1].legend(
            loc="lower right", fontsize=8.5, facecolor="#ffffff", edgecolor="#D8DEE9"
        )

        prec, rec, _ = precision_recall_curve(y_true, prob_pos)
        axes[2].plot(rec, prec, color="#A3BE8C", linewidth=2.0, label="PR Curve")
        _apply_base_plot_style(
            axes[2],
            title="Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
        )
        axes[2].legend(
            loc="lower left", fontsize=8.5, facecolor="#ffffff", edgecolor="#D8DEE9"
        )

    fig.suptitle(
        f"Classification Diagnostics — {model_name}",
        fontsize=11.5,
        fontweight="bold",
        color="#2E3440",
        y=1.02,
    )
    fig.tight_layout()

    out_path = fig_dir / f"{target_name}__classification_diagnostics"
    save_plot(fig, out_path)
    return out_path.with_suffix(".png")


def generate_sem_effect_forest_plot(
    params_df: pd.DataFrame, target_name: str, out_dir: Path
) -> Path | None:
    try:
        if params_df.empty:
            return None
        paths = (
            params_df[params_df["op"] == "~"].copy()
            if "op" in params_df.columns
            else params_df.copy()
        )
        if paths.empty:
            return None
        fig_dir = out_dir / "sem" / target_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        l_col = (
            "lval_clean"
            if "lval_clean" in paths.columns
            else ("lval" if "lval" in paths.columns else "Outcome")
        )
        r_col = (
            "rval_clean"
            if "rval_clean" in paths.columns
            else ("rval" if "rval" in paths.columns else "Predictor")
        )
        paths["label"] = paths[l_col].astype(str) + " ← " + paths[r_col].astype(str)
        est_col = next(
            (
                c
                for c in ["Est. Std", "Est.Std", "Estimate", "Beta_Std", "beta"]
                if c in paths.columns
            ),
            None,
        )
        se_col = next(
            (c for c in ["Std. Err", "Std.Err", "SE", "se"] if c in paths.columns), None
        )
        if est_col is None:
            return None
        paths[est_col] = pd.to_numeric(paths[est_col], errors="coerce").fillna(0.0)
        se = (
            pd.to_numeric(paths[se_col], errors="coerce").fillna(0.08)
            if se_col
            else pd.Series([0.08] * len(paths))
        )
        fig, ax = plt.subplots(
            figsize=(8.5, max(3.8, 0.45 * len(paths))), facecolor="#ffffff"
        )
        y_pos = np.arange(len(paths))
        ax.errorbar(
            paths[est_col],
            y_pos,
            xerr=1.96 * se,
            fmt="o",
            color="#5E81AC",
            ecolor="#4C566A",
            elinewidth=1.6,
            capsize=4.5,
            markersize=6.5,
        )
        ax.axvline(0, color="#BF616A", linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(paths["label"], fontsize=9, color="#2E3440")

        _apply_base_plot_style(
            ax,
            title=f"Structural Equation Model: Path Parameter Estimates\nTarget: {target_name}",
            xlabel="Standardized Path Coefficient (β ± 95% CI)",
        )
        ax.title.set_fontsize(11)
        ax.grid(axis="x", linestyle="--", alpha=0.7, color="#E5E9F0")
        fig.tight_layout()
        out_path = fig_dir / "sem_forest_plot"
        save_plot(fig, out_path)
        return out_path.with_suffix(".png")
    except Exception as exc:
        logger.warning(
            "Failed to generate SEM effect forest plot for %s: %s", target_name, exc
        )
        return None


def generate_sem_effects_bar_plot(
    effects_df: pd.DataFrame, target_name: str, out_dir: Path
) -> Path | None:
    try:
        if effects_df.empty or "Predictor" not in effects_df.columns:
            return None
        fig_dir = out_dir / "sem" / target_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(
            figsize=(9, max(3.8, 0.45 * len(effects_df))), facecolor="#ffffff"
        )
        df_m = effects_df.copy()
        y_pos = np.arange(len(df_m))
        h = 0.25
        ax.barh(
            y_pos - h,
            df_m["Direct_Effect_c_prime"],
            height=h,
            color="#5E81AC",
            label="Direct Effect (c')",
            edgecolor="#2E3440",
            linewidth=0.6,
        )
        ax.barh(
            y_pos,
            df_m["Indirect_Effect_ab"],
            height=h,
            color="#D08770",
            label="Indirect Effect (a x b)",
            edgecolor="#2E3440",
            linewidth=0.6,
        )
        ax.barh(
            y_pos + h,
            df_m["Total_Effect_c"],
            height=h,
            color="#A3BE8C",
            label="Total Effect (c)",
            edgecolor="#2E3440",
            linewidth=0.6,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_m["Predictor"], fontsize=9.5, color="#2E3440")
        ax.axvline(0, color="#4C566A", linestyle="--", linewidth=1.0)

        _apply_base_plot_style(
            ax,
            title=f"SEM Mediation Decomposition (Direct, Indirect, Total)\nTarget: {target_name}",
            xlabel="Standardized Effect Magnitude (β)",
        )
        ax.title.set_fontsize(11)
        ax.legend(
            loc="lower right", facecolor="#ffffff", edgecolor="#D8DEE9", fontsize=8.5
        )
        ax.grid(axis="x", linestyle="--", alpha=0.7, color="#E5E9F0")
        fig.tight_layout()
        out_path = fig_dir / "sem_effects_plot"
        save_plot(fig, out_path)
        return out_path.with_suffix(".png")
    except Exception as exc:
        logger.warning(
            "Failed to generate SEM effects bar plot for %s: %s", target_name, exc
        )
        return None


def generate_sem_path_diagram_plot(
    params_df: pd.DataFrame, target_name: str, out_dir: Path
) -> Path | None:
    try:
        if params_df.empty:
            return None
        paths = (
            params_df[params_df["op"] == "~"].copy()
            if "op" in params_df.columns
            else params_df.copy()
        )
        if paths.empty:
            return None
        fig_dir = out_dir / "sem" / target_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        l_col = (
            "lval_clean"
            if "lval_clean" in paths.columns
            else ("lval" if "lval" in paths.columns else "Outcome")
        )
        r_col = (
            "rval_clean"
            if "rval_clean" in paths.columns
            else ("rval" if "rval" in paths.columns else "Predictor")
        )
        est_col = next(
            (
                c
                for c in ["Est. Std", "Est.Std", "Estimate", "Beta_Std"]
                if c in paths.columns
            ),
            "Estimate",
        )
        all_targets = list(paths[l_col].unique())
        all_preds = list(paths[r_col].unique())
        all_nodes = list(dict.fromkeys([*all_preds, *all_targets]))
        mediators = [n for n in all_nodes if n in all_targets and n != target_name]
        pure_preds = [n for n in all_preds if n not in all_targets]
        fig, ax = plt.subplots(figsize=(10.5, 6.5), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        pos: dict[str, tuple[float, float]] = {}
        if pure_preds:
            y_step = 8.0 / max(1, len(pure_preds))
            for idx, p in enumerate(pure_preds):
                pos[p] = (1.8, 8.5 - (idx + 0.5) * y_step)
        if mediators:
            y_step = 8.0 / max(1, len(mediators))
            for idx, m in enumerate(mediators):
                pos[m] = (5.2, 8.5 - (idx + 0.5) * y_step)
        pos[target_name] = (8.5, 5.0)
        for node, (x, y) in pos.items():
            box_color = (
                "#ECEFF4"
                if node == target_name
                else ("#E5E9F0" if node in mediators else "#F3F4F6")
            )
            border_color = (
                "#5E81AC"
                if node == target_name
                else ("#D08770" if node in mediators else "#4C566A")
            )
            rect = patches.FancyBboxPatch(
                (x - 1.1, y - 0.45),
                2.2,
                0.9,
                boxstyle="round,pad=0.1",
                facecolor=box_color,
                edgecolor=border_color,
                linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.text(
                x,
                y,
                node,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#2E3440",
            )
        for _, row in paths.iterrows():
            u = str(row[r_col])
            v = str(row[l_col])
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                beta_value = _safe_float(row.get(est_col, 0.0), 0.0)
                pval = _safe_float(row.get("p_value", row.get("p-value", 0.05)), 0.05)
                stars = (
                    "***"
                    if pval < 0.001
                    else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                )
                arrow_color = "#BF616A" if beta_value < 0 else "#5E81AC"
                ax.annotate(
                    "",
                    xy=(x2 - 1.15, y2),
                    xytext=(x1 + 1.15, y1),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=arrow_color,
                        lw=max(1.0, min(3.5, abs(beta_value) * 3.0)),
                        mutation_scale=12,
                    ),
                )
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0 + 0.18
                ax.text(
                    mid_x,
                    mid_y,
                    f"β={beta_value:.2f}{stars}",
                    ha="center",
                    va="center",
                    fontsize=8.0,
                    color="#2E3440",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="#ffffff",
                        edgecolor="#E5E9F0",
                        lw=0.6,
                    ),
                )
        ax.set_title(
            f"Structural Equation Model Path Network Diagram\nTarget: {target_name}",
            fontsize=12,
            fontweight="bold",
            color="#2E3440",
            pad=12,
        )
        fig.tight_layout()
        out_path = fig_dir / "path_diagram"
        save_plot(fig, out_path)
        return out_path.with_suffix(".png")
    except Exception as exc:
        logger.warning(
            "Failed to generate SEM path diagram plot for %s: %s", target_name, exc
        )
        return None


def generate_sem_covariance_plot(
    sem_data: pd.DataFrame, target_name: str, out_dir: Path
) -> Path | None:
    try:
        if sem_data.empty or sem_data.shape[1] < 2:
            return None
        fig_dir = out_dir / "sem" / target_name
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 6), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")
        corr = sem_data.corr(numeric_only=True)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            linewidths=0.5,
            linecolor="#E5E9F0",
            cbar_kws={"shrink": 0.8, "label": "Correlation (r)"},
            ax=ax,
        )
        ax.set_title(
            f"SEM System Covariance & Inter-Correlation\nTarget: {target_name}",
            fontsize=11,
            fontweight="bold",
            color="#2E3440",
            pad=12,
        )
        fig.tight_layout()
        out_path = fig_dir / "sem_covariance_heatmap"
        save_plot(fig, out_path)
        return out_path.with_suffix(".png")
    except Exception as exc:
        logger.warning(
            "Failed to generate SEM covariance plot for %s: %s", target_name, exc
        )
        return None


def _latex_num(val: Any, precision: int = 3) -> str:
    if val is None or pd.isna(val):
        return r"\text{N/A}"
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return r"\text{N/A}"
        return f"{f:.{precision}f}"
    except (ValueError, TypeError):
        return r"\text{N/A}"


def generate_latex_tables(
    all_models_df: pd.DataFrame,
    out_dir: Path,
    sem_result_dict: dict[str, Any] | None = None,
) -> str:
    if all_models_df.empty:
        return ""

    task_type = str(all_models_df.iloc[0].get("task_type", "regression")).lower()
    rows: list[str] = []

    for _, r in all_models_df.iterrows():
        m = str(r.get("model_name", r.get("model_key", ""))).replace("_", r"\_")
        rank = int(r.get("rank", 1))
        n = int(r.get("n_subjects", 0))
        status = r.get("execution_status", "SUCCESS")
        rec = str(r.get("recommendation", "Secondary"))

        if status != "SUCCESS":
            ci = r"\text{N/A}"
            if task_type == "classification":
                rows.append(
                    f"{rank} & {m} & {n} & -- & -- & -- & -- & -- & {ci} & {rec} \\\\"
                )
            else:
                rows.append(
                    f"{rank} & {m} & {n} & -- & -- & -- & -- & -- & {ci} & {rec} \\\\"
                )
            continue

        ci = f"[{_latex_num(r.get('oof_ci_lower'))}, {_latex_num(r.get('oof_ci_upper'))}]"

        if task_type == "classification":
            acc_m = _latex_num(r.get("primary_metric_mean", r.get("accuracy_mean")))
            acc_s = _latex_num(r.get("primary_metric_std"))
            acc_str = f"{acc_m} \\pm {acc_s}"
            bacc_str = _latex_num(r.get("balanced_acc_mean"))
            f1_str = _latex_num(r.get("f1_macro_mean"))
            auc_str = _latex_num(r.get("roc_auc_mean"))
            mcc_str = _latex_num(r.get("mcc_mean"))
            rows.append(
                f"{rank} & {m} & {n} & {acc_str} & {bacc_str} & {f1_str} & {auc_str} & {mcc_str} & {ci} & {rec} \\\\"
            )
        else:
            r2_m = _latex_num(r.get("r2_mean", r.get("primary_metric_mean")))
            r2_s = _latex_num(r.get("r2_std", r.get("primary_metric_std")))
            r2_str = f"{r2_m} \\pm {r2_s}"
            rmse_str = _latex_num(r.get("rmse_mean"))
            mae_str = _latex_num(r.get("mae_mean"))
            pr_str = _latex_num(r.get("pearson_r_mean"))
            sr_str = _latex_num(r.get("spearman_rho_mean"))
            rows.append(
                f"{rank} & {m} & {n} & {r2_str} & {rmse_str} & {mae_str} & {pr_str} & {sr_str} & {ci} & {rec} \\\\"
            )

    if task_type == "classification":
        table_header = (
            "\\small\n\\begin{tabular}{clccccccccr}\n\\toprule\n"
            "\\textbf{Rank} & \\textbf{Estimator} & \\textbf{N} & \\textbf{Accuracy (Mean $\\pm$ SD)} & \\textbf{Bal. Acc} & \\textbf{Macro $F_1$} & \\textbf{ROC-AUC} & \\textbf{MCC} & \\textbf{95\\% Bootstrap CI} & \\textbf{Rec.} \\\\\n"
            "\\midrule\n"
        )
    else:
        table_header = (
            "\\small\n\\begin{tabular}{clccccccccr}\n\\toprule\n"
            "\\textbf{Rank} & \\textbf{Estimator} & \\textbf{N} & \\textbf{$R^2$ (Mean $\\pm$ SD)} & \\textbf{RMSE} & \\textbf{MAE} & \\textbf{Pearson $r$} & \\textbf{Spearman $\\rho$} & \\textbf{95\\% Bootstrap CI} & \\textbf{Rec.} \\\\\n"
            "\\midrule\n"
        )

    latex_code = (
        "% Table 1: Nested Cross-Validation Comprehensive Estimator Benchmark Leaderboard\n"
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Cross-Validated Model Generalization, Calibration, and Predictive Accuracy Benchmarks}\\label{tab:ml_tournament}\n"
        + table_header
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n"
    )

    if sem_result_dict:
        for target, s_res in sem_result_dict.items():
            if s_res.get("status") == "ok":
                t_clean = target.replace("_", r"\_")
                struct_df = s_res.get("structural_paths", pd.DataFrame())
                if not struct_df.empty:
                    s_rows = []
                    for _, sr in struct_df.iterrows():
                        p_str = f"{sr['Predictor']} \\to {sr['Outcome']}".replace(
                            "_", r"\_"
                        )
                        b_val = pd.to_numeric(sr.get("Beta_Std"), errors="coerce")
                        if sr.get("Significance") == "FIT_FAILED" or pd.isna(b_val):
                            s_rows.append(
                                f"{p_str} & \\text{{NaN}} & \\text{{NaN}} & \\text{{NaN}} & \\text{{NaN}} & \\text{{N/A}} & \\text{{FIT\\_FAILED}} \\\\"
                            )
                        else:
                            se_v = _safe_float(sr.get("SE"), 0.08)
                            z_v = _safe_float(sr.get("Test_Statistic", sr.get("z_value")), 0.0)
                            p_v = _safe_float(
                                sr.get("p_value_adjusted", sr.get("p_value")), 0.05
                            )
                            s_rows.append(
                                f"{p_str} & {b_val:.3f} & {se_v:.3f} & {z_v:.2f} & {p_v:.3f} & {sr.get('CI_95', 'N/A')} & {sr.get('Significance', 'ns')} \\\\"
                            )

                    corr_method = (
                        struct_df["Correction_Method"].iloc[0]
                        if "Correction_Method" in struct_df.columns
                        and not struct_df.empty
                        else "none"
                    )
                    latex_code += (
                        f"% Table 2: SEM Structural Path Estimates for {target}\n"
                        "\\begin{table}[h]\n\\centering\n"
                        f"\\caption{{Standardized Path Coefficients and Parameter Estimates for {t_clean}. "
                        f"$p$ is {corr_method}-adjusted for multiple comparisons across "
                        f"{len(struct_df)} paths; raw (unadjusted) $p$-values are available in the "
                        f"accompanying CSV output.}}\\label{{tab:sem_paths_{target}}}\n"
                        "\\small\n\\begin{tabular}{lcccccc}\n\\toprule\n"
                        "\\textbf{Hypothesized Path} & \\textbf{$\\beta$} & \\textbf{SE} & \\textbf{Stat.} & \\textbf{$p_{adj}$} & \\textbf{95\\% CI} & \\textbf{Sig.} \\\\\n"
                        "\\midrule\n"
                        + "\n".join(s_rows)
                        + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\n"
                    )

    (out_dir / "reports" / "manuscript_tables.tex").write_text(
        latex_code, encoding="utf-8"
    )
    return latex_code


def generate_html_report(
    all_models_df: pd.DataFrame,
    out_dir: Path,
    dataset_profile: DatasetProfile | None = None,
    dataset_df: pd.DataFrame | None = None,
    config: PipelineConfig | None = None,
    sem_result_dict: dict[str, dict[str, Any]] | None = None,
    execution_time: float = 0.0,
    confirmatory_df: pd.DataFrame | None = None,
    methodology: dict[str, Any] | None = None,
) -> None:
    sem_result_dict = sem_result_dict or {}
    methodology = methodology or {}
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    latex_code = generate_latex_tables(all_models_df, out_dir, sem_result_dict)

    targets = list(all_models_df["target"].unique()) if not all_models_df.empty else []

    selected_cols: list[str] | None = None
    if config:
        selected_cols = config.dataset.active_columns or (
            config.dataset.target_columns + config.dataset.feature_columns
        )

    dataset_eda_paths = (
        generate_dataset_eda_plots(
            dataset_df, dataset_profile, out_dir, selected_columns=selected_cols
        )
        if dataset_df is not None and dataset_profile is not None
        else {}
    )

    dataset_html = ""
    if dataset_profile:
        n_selected_vars = (
            len(selected_cols) if selected_cols else dataset_profile.total_columns
        )
        dataset_html = f"""
        <div class="panel">
          <div class="grid">
            <div class="card"><div class="card-label">Observations (N)</div><div class="card-val">{dataset_profile.total_rows:,}</div></div>
            <div class="card"><div class="card-label">Active Variables (P)</div><div class="card-val">{n_selected_vars:,}</div></div>
            <div class="card"><div class="card-label">Continuous Features</div><div class="card-val">{len(dataset_profile.numerical_columns)}</div></div>
          </div>
          <h3>Exploratory Data Visualizations (Selected Variables)</h3>
          <div class="img-grid">{"".join(encode_img_base64(p) for p in dataset_eda_paths.values())}</div>
        </div>
        """

    has_global_imputation = any(
        "[GLOBAL_PREFILTERING]" in msg
        or "missingness" in msg.lower()
        or "impute" in msg.lower()
        for msg in (config.transformation_log if config else [])
    )

    if has_global_imputation:
        methodology_note = (
            "<b>Methodology Disclosure / Pre-Filtering Notice:</b> In-memory missing value transformations "
            "were applied across the dataset prior to model partitioning. Out-of-fold evaluations reflect "
            "performance on globally pre-filtered distributions. Automated inner-fold preprocessing was preserved."
        )
        callout_div = f'<div class="callout" style="border-left-color: var(--nord13);"><p>{methodology_note}</p></div>'
    else:
        methodology_note = (
            "<b>Methodology Note:</b> All feature transformations, imputations, and hyperparameter "
            "searches were conducted strictly inside cross-validation folds (see the Confirmatory Model-Selection "
            "section below for how the headline performance figure specifically avoids selection-induced "
            "optimism). Attribution measures reflect multivariate, potentially non-linear predictive "
            "sensitivity; findings describe associative predictive contributions rather than causal mechanisms."
        )
        callout_div = f'<div class="callout"><p>{methodology_note}</p></div>'

    target_cards = ""
    for target in targets:
        target_df = all_models_df[all_models_df["target"] == target].sort_values("rank")
        is_clf = (
            str(target_df.iloc[0].get("task_type", "")).lower() == "classification"
            if not target_df.empty
            else False
        )

        # --- Fix 1: confirmatory model-selection panel -------------------
        # This is the PRIMARY, bias-corrected performance estimate for this
        # target -- computed by a nested procedure where model selection
        # and hyperparameter tuning never see the data they are ultimately
        # evaluated on (see the Methodology & Reproducibility section for
        # the full architecture explanation). The exploratory leaderboard
        # further below compares many candidates against each other and is
        # useful for understanding *which* algorithms tend to work well,
        # but its winner's own reported score is not bias-corrected for
        # having been selected from among many candidates.
        confirmatory_html = ""
        if confirmatory_df is not None and not confirmatory_df.empty:
            conf_rows = confirmatory_df[confirmatory_df["target"] == target]
            if not conf_rows.empty:
                cr = conf_rows.iloc[0]
                status = str(cr.get("execution_status", "UNKNOWN"))
                if status in ("SUCCESS", "PARTIAL_FAILURE"):
                    metric_name = str(cr.get("primary_metric_name", "")).upper()
                    point = _safe_float(cr.get("primary_metric_mean"))
                    ci_lo = _safe_float(cr.get("oof_ci_lower"))
                    ci_hi = _safe_float(cr.get("oof_ci_upper"))
                    n_failed = int(_safe_float(cr.get("n_failed_folds"), 0.0))
                    n_folds = int(_safe_float(cr.get("n_outer_folds"), 0.0))
                    n_cand = int(_safe_float(cr.get("n_candidates_compared"), 0.0))
                    top_model = html.escape(str(cr.get("most_frequently_selected_model", "—")))
                    warn_html = ""
                    if n_failed > 0:
                        warn_html = (
                            f'<p><span class="badge badge-warning">{n_failed}/{n_folds} '
                            f"outer folds failed</span> -- the estimate above is computed "
                            f"from the remaining successful folds only.</p>"
                        )
                    confirmatory_html = f"""
                    <div class="callout" style="border-left-color: var(--nord14);">
                      <h3 style="margin-top:0;">Confirmatory Result (Primary Estimate)</h3>
                      <div class="grid">
                        <div class="card"><div class="card-label">{metric_name} (bias-corrected)</div><div class="card-val">{point:.3f}</div></div>
                        <div class="card"><div class="card-label">95% Bootstrap CI</div><div class="card-val" style="font-size:1.1rem;">[{ci_lo:.3f}, {ci_hi:.3f}]</div></div>
                        <div class="card"><div class="card-label">Candidates Compared</div><div class="card-val">{n_cand}</div></div>
                      </div>
                      <p>Estimated via a nested model-selection procedure across {n_folds} outer folds
                      (most frequently selected model: <b>{top_model}</b> -- see
                      <code>{html.escape(target)}_confirmatory_selection_audit.json</code> for the full
                      per-fold selection record). {warn_html}
                      {BOOTSTRAP_CI_NOTE_HTML}</p>
                    </div>
                    """
                else:
                    confirmatory_html = (
                        '<div class="callout" style="border-left-color: var(--nord11);">'
                        "<p><b>Confirmatory Result:</b> the confirmatory model-selection "
                        "procedure did not complete successfully for this target "
                        f"(status: {html.escape(status)}). Only the exploratory "
                        "leaderboard below is available; treat its top-ranked model's "
                        "score as descriptive, not as a bias-corrected estimate.</p></div>"
                    )
        else:
            confirmatory_html = (
                '<div class="callout" style="border-left-color: var(--nord11);">'
                "<p><b>Confirmatory Result:</b> not available for this run.</p></div>"
            )

        if is_clf:
            table_head = (
                "<thead><tr><th>Rank</th><th>Estimator</th><th>N</th>"
                "<th>Accuracy (Mean ± SD)</th><th>Bal. Acc</th><th>Macro F1</th>"
                "<th>ROC-AUC</th><th>MCC</th><th>95% Bootstrap CI</th><th>Complexity</th><th>Recommendation</th></tr></thead>"
            )
        else:
            table_head = (
                "<thead><tr><th>Rank</th><th>Estimator</th><th>N</th>"
                "<th>R² (Mean ± SD)</th><th>RMSE</th><th>MAE</th><th>Pearson r</th>"
                "<th>Spearman rho</th><th>95% Bootstrap CI</th><th>Complexity</th><th>Recommendation</th></tr></thead>"
            )

        rows_html = ""
        for _, r in target_df.iterrows():
            status = r.get("execution_status", "SUCCESS")
            is_winner = bool(r.get("is_best", False)) and status == "SUCCESS"
            if status != "SUCCESS":
                rank_badge = f'<span class="badge badge-warning">{status}</span>'
            else:
                rank_badge = (
                    '<span class="badge badge-success" title="Best of the exploratory leaderboard comparison -- see the Confirmatory Model-Selection panel above for the bias-corrected estimate.">Exploratory Best</span>'
                    if is_winner
                    else f"#{r['rank']}"
                )

            if status != "SUCCESS":
                primary_text = "—"
                ci_text = "—"
                err_info = html.escape(
                    str(r.get("error_message") or "Model execution aborted")
                )
                recommendation_text = f"Failed ({err_info[:28]}...)"
                cols_html = "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>"
            else:
                ci_text = f"[{_safe_float(r.get('oof_ci_lower', 0.0)):.3f}, {_safe_float(r.get('oof_ci_upper', 0.0)):.3f}]"
                recommendation_text = str(r.get("recommendation", "Secondary"))
                if is_clf:
                    primary_text = f"{_safe_float(r.get('primary_metric_mean', r.get('accuracy_mean', 0.0))):.3f} ± {_safe_float(r.get('primary_metric_std', 0.0)):.3f}"
                    cols_html = (
                        f"<td>{_safe_float(r.get('balanced_acc_mean', float('nan'))):.3f}</td>"
                        f"<td>{_safe_float(r.get('f1_macro_mean', float('nan'))):.3f}</td>"
                        f"<td>{_safe_float(r.get('roc_auc_mean', float('nan'))):.3f}</td>"
                        f"<td>{_safe_float(r.get('mcc_mean', float('nan'))):.3f}</td>"
                    )
                else:
                    primary_text = f"{_safe_float(r.get('r2_mean', r.get('primary_metric_mean', 0.0))):.3f} ± {_safe_float(r.get('r2_std', r.get('primary_metric_std', 0.0))):.3f}"
                    cols_html = (
                        f"<td>{_safe_float(r.get('rmse_mean', float('nan'))):.3f}</td>"
                        f"<td>{_safe_float(r.get('mae_mean', float('nan'))):.3f}</td>"
                        f"<td>{_safe_float(r.get('pearson_r_mean', float('nan'))):.3f}</td>"
                        f"<td>{_safe_float(r.get('spearman_rho_mean', float('nan'))):.3f}</td>"
                    )

            complexity_text = str(r.get("complexity_tier", "Medium"))
            tuning_status = r.get("tuning_status", "OK")
            tuning_html = (
                ' <span class="badge badge-warning" title="Hyperparameter tuning failed for one or more folds.">Tuning Failed</span>'
                if tuning_status == "Failed" and status == "SUCCESS"
                else ""
            )

            rows_html += f"""
            <tr {'class="winner-row"' if is_winner else ""}>
              <td>{rank_badge}</td>
              <td><b>{html.escape(str(r.get("model_name", "")))}</b>{tuning_html}</td>
              <td>{int(r.get("n_subjects", 0))}</td>
              <td style="color:var(--nord8); font-weight:bold;">{primary_text}</td>
              {cols_html}
              <td>{ci_text}</td>
              <td><span class="badge badge-info">{complexity_text}</span></td>
              <td><b>{recommendation_text}</b></td>
            </tr>
            """

        perf_plot = encode_img_base64(
            out_dir / "figures" / "performance" / f"{target}__model_comparison.png",
            "Figure 1 — Algorithm Tournament Leaderboard",
        )
        if is_clf:
            cal_plot = ""
            diag_plot = encode_img_base64(
                out_dir
                / "figures"
                / "explainability"
                / f"{target}__classification_diagnostics.png",
                "Figure 2 — Classification Diagnostics Suite",
            )
        else:
            cal_plot = encode_img_base64(
                out_dir
                / "figures"
                / "explainability"
                / f"{target}__calibration_plot.png",
                "Figure 2 — Prediction Calibration (Observed vs Ŷ)",
            )
            diag_plot = encode_img_base64(
                out_dir
                / "figures"
                / "explainability"
                / f"{target}__regression_diagnostics.png",
                "Figure 3 — Residual Diagnostics Suite",
            )

        feat_plot = encode_img_base64(
            out_dir / "figures" / "explainability" / f"{target}__top_features_cv.png",
            "Figure 4 — Cross-Validation Feature Importance",
        )
        lc_plot = encode_img_base64(
            out_dir / "figures" / "performance" / f"{target}__learning_curve.png",
            "Figure 5 — Empirical Learning Curve",
        )
        shap_bee_path = (
            out_dir / "figures" / "explainability" / f"{target}__shap_beeswarm.png"
        )
        if shap_bee_path.is_file() and shap_bee_path.stat().st_size > 0:
            shap_bee = encode_img_base64(
                shap_bee_path,
                "Figure 6 — SHAP Beeswarm Distribution",
            )
        else:
            reason = (
                "skip_shap enabled"
                if (config and config.cv.skip_shap)
                else (
                    "shap not installed"
                    if not HAS_SHAP
                    else "non-tree model / insufficient data / skipped"
                )
            )
            shap_bee = f'<div class="img-card"><h4 style="margin-top:0;">Figure 6 — SHAP Beeswarm Distribution</h4><div style="padding: 24px; text-align: center; color: var(--nord13); font-style: italic; background: #fafafa; border-radius: 4px;">SHAP beeswarm unavailable: <span class="badge badge-warning">{reason}</span></div></div>'

        shap_dep = encode_img_base64(
            out_dir / "figures" / "explainability" / f"{target}__shap_dependence.png",
            "Figure 7 — SHAP Top Predictor Dependence Profile",
        )

        sem_html = ""
        s_res = sem_result_dict.get(target)
        if s_res and s_res.get("status") == "ok":
            diag = encode_img_base64(
                out_dir / "sem" / target / "path_diagram.png",
                "Figure 8 — SEM Path Network Diagram",
            )
            forest = encode_img_base64(
                out_dir / "sem" / target / "sem_forest_plot.png",
                "Figure 9 — Standardized Path Forest Plot",
            )
            effects_p = encode_img_base64(
                out_dir / "sem" / target / "sem_effects_plot.png",
                "Figure 10 — Mediation Decomposition",
            )
            cov_p = encode_img_base64(
                out_dir / "sem" / target / "sem_covariance_heatmap.png",
                "Figure 11 — SEM System Covariance Matrix",
            )

            struct_df = s_res.get("structural_paths", pd.DataFrame())
            fit_table_df = s_res.get("fit_table", pd.DataFrame())
            med_df = s_res.get("mediation_effects", pd.DataFrame())
            r2_df = s_res.get("variance_explained", pd.DataFrame())

            s_rows_html = []
            struct_correction_method = "none"
            for _, r in struct_df.iterrows():
                sig = r.get("Significance", "")
                b_num = pd.to_numeric(r.get("Beta_Std"), errors="coerce")
                struct_correction_method = r.get("Correction_Method", struct_correction_method)
                if sig == "FIT_FAILED" or pd.isna(b_num):
                    sig_html = f'<span class="badge badge-warning">{sig}</span>'
                    s_rows_html.append(
                        f"<tr><td><b>{r.get('Path', '')}</b></td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>N/A</td><td>{sig_html}</td></tr>"
                    )
                else:
                    stat_type = html.escape(str(r.get("Statistic_Type", "")))
                    s_rows_html.append(
                        "<tr>"
                        f"<td><b>{r.get('Path', '')}</b></td>"
                        f"<td>{b_num:.3f}</td>"
                        f"<td>{_safe_float(r.get('SE', 0.0)):.3f}</td>"
                        f"<td>{_safe_float(r.get('Test_Statistic', 0.0)):.2f} <span class='badge badge-info' style='font-size:0.7em;'>{stat_type}</span></td>"
                        f"<td>{_safe_float(r.get('p_value_raw', 0.0)):.3f}</td>"
                        f"<td><b>{_safe_float(r.get('p_value_adjusted', 0.0)):.3f}</b></td>"
                        f"<td>{r.get('CI_95', '')}</td>"
                        f"<td><b>{sig}</b></td>"
                        "</tr>"
                    )

            struct_table_html = (
                "".join(s_rows_html)
                if not struct_df.empty
                else "<tr><td colspan='8'>No structural paths estimated.</td></tr>"
            )
            struct_family_size = (
                int(struct_df["Family_Size"].iloc[0])
                if "Family_Size" in struct_df.columns and not struct_df.empty
                else 0
            )

            fit_table_html = (
                "".join(
                    f"<tr><td><b>{r.get('Fit Index', '')}</b></td><td>{r.get('Value', '')}</td><td>{r.get('Reference Cutoff', '')}</td><td>{r.get('Interpretation', '')}</td></tr>"
                    for _, r in fit_table_df.iterrows()
                )
                if not fit_table_df.empty
                else "<tr><td colspan='4'>Fit indices table unavailable.</td></tr>"
            )

            med_table_html = (
                "".join(
                    f"<tr><td><b>{r.get('Predictor', '')}</b></td><td>{r.get('Mediators', '')}</td><td>{_safe_float(r.get('Direct_Effect_c_prime')):.3f}</td><td>{_safe_float(r.get('Indirect_Effect_ab')):.3f}</td><td>{_safe_float(r.get('Total_Effect_c')):.3f}</td><td>{r.get('Indirect_95_CI', '')}</td>"
                    f"<td>{_safe_float(r.get('Indirect_p_value_raw')):.3f}</td><td><b>{_safe_float(r.get('Indirect_p_value_adjusted')):.3f}</b></td>"
                    f"<td>{r.get('Pct_Mediated', '')}</td><td><b>{r.get('Mediation_Classification', '')}</b></td></tr>"
                    for _, r in med_df.iterrows()
                )
                if not med_df.empty
                else "<tr><td colspan='10'>No mediation paths configured.</td></tr>"
            )
            med_correction_method = (
                med_df["Correction_Method"].iloc[0]
                if "Correction_Method" in med_df.columns and not med_df.empty
                else "none"
            )

            r2_table_html = (
                "".join(
                    f"<tr><td><b>{r.get('Endogenous_Variable', '')}</b></td><td>{_safe_float(r.get('R_Squared')):.4f}</td><td>{r.get('Variance_Explained_Pct', '')}</td></tr>"
                    for _, r in r2_df.iterrows()
                )
                if not r2_df.empty
                else "<tr><td colspan='3'>Variance table unavailable.</td></tr>"
            )

            sem_html = f"""
            <h2>Structural Equation Modeling (SEM) Statistical Dossier: {html.escape(target)}</h2>
            <div class="callout"><p>{html.escape(s_res.get("narrative", ""))}</p></div>

            <h3>1. Standardized Structural Path Estimates</h3>
            <div class="callout" style="border-left-color: var(--nord9);"><p><b>Multiplicity correction:</b> {html.escape(str(struct_correction_method))} applied across
            {struct_family_size} structural path(s) tested in this model. Both the raw (uncorrected)
            and multiplicity-adjusted p-values are shown; <b>Significance</b> reflects the adjusted p-value only.</p></div>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Hypothesized Path</th><th>β (Std)</th><th>SE</th><th>Test Stat.</th><th>p (raw)</th><th>p (adjusted)</th><th>95% CI</th><th>Sig.</th></tr></thead>
                <tbody>{struct_table_html}</tbody>
              </table>
            </div>

            <h3>2. Model Fit Benchmarks & Qualitative Evaluation</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Fit Metric</th><th>Observed Value</th><th>Reference Threshold</th><th>Qualitative Verdict</th></tr></thead>
                <tbody>{fit_table_html}</tbody>
              </table>
            </div>

            <h3>3. Mediation Analysis & Decomposition</h3>
            <div class="callout" style="border-left-color: var(--nord9);"><p><b>Multiplicity correction:</b> {html.escape(str(med_correction_method))} applied across
            the indirect-effect p-values tested for this target. Bootstrap CI method:
            bias-corrected (BC) percentile bootstrap, refitting the model on each resample (see per-row CI Method column in the CSV export).</p></div>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Predictor</th><th>Mediator(s)</th><th>Direct (c')</th><th>Indirect (a x b)</th><th>Total (c)</th><th>Indirect 95% CI</th><th>p (raw)</th><th>p (adjusted)</th><th>% Mediated</th><th>Classification</th></tr></thead>
                <tbody>{med_table_html}</tbody>
              </table>
            </div>

            <h3>4. Variance Explained (R²)</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Endogenous Variable</th><th>R²</th><th>Variance Explained</th></tr></thead>
                <tbody>{r2_table_html}</tbody>
              </table>
            </div>

            <h3>5. SEM Publication Visualizations</h3>
            <div class="img-grid">{diag}{forest}{effects_p}{cov_p}</div>
            """

        target_cards += f"""
        <details class="target-card" open>
          <summary><span>Target Outcome: <b>{html.escape(target)}</b></span></summary>
          <div class="details-body">
            {confirmatory_html}
            <h2>Exploratory Model Comparison (Descriptive): {html.escape(target)}</h2>
            <div class="callout" style="border-left-color: var(--nord9);"><p>The table below compares
            {len(target_df)} candidate models against each other using nested cross-validation; the
            "Exploratory Best" row and its <code>p (adjusted)</code> vs.-best column are useful for
            understanding relative algorithm performance, but selecting the best of many candidates
            this way is not bias-corrected the way the Confirmatory Result above is -- see the
            Methodology &amp; Reproducibility section for why.</p></div>
            <div class="table-wrap">
              <table>
                {table_head}
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            <h3>Core Diagnostic & Calibration Visualizations</h3>
            <div class="img-grid">{perf_plot}{cal_plot}{diag_plot}{feat_plot}{lc_plot}{shap_bee}{shap_dep}</div>
            {sem_html}
          </div>
        </details>
        """

    # --- Methodology & Reproducibility section -------------------------
    # Fix requirement: reports must explicitly display model-selection
    # strategy, CV configuration, scoring metric, multiplicity correction,
    # missing-data method, task-type determination, and warnings -- never
    # buried only in an obscure JSON field.
    warnings_list = methodology.get("warnings", [])
    warnings_html = (
        "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings_list)
        if warnings_list
        else "<li>No methodology warnings were recorded for this run.</li>"
    )
    task_type_rows_html = "".join(
        f"<tr><td><b>{html.escape(str(t))}</b></td>"
        f"<td>{html.escape(str(info.get('requested_task_type', '')))}</td>"
        f"<td>{html.escape(str(info.get('resolved_task_type', '')))}</td>"
        f"<td>{'Yes' if info.get('is_ambiguous') else 'No'}</td>"
        f"<td>{'Yes' if info.get('user_confirmed') else '—'}</td></tr>"
        for t, info in methodology.get("task_type_resolutions", {}).items()
    )
    sem_decisions_rows_html = "".join(
        f"<tr><td><b>{html.escape(str(t))}</b></td>"
        f"<td>{html.escape(str(info.get('missing_data_method_requested', '')))}</td>"
        f"<td>{html.escape(str(info.get('missing_data_method_used', '')))}</td>"
        f"<td>{html.escape(str(info.get('multiplicity_correction', '')))}</td>"
        f"<td>{html.escape(str(info.get('convergence_status', '')))}</td></tr>"
        for t, info in methodology.get("sem_decisions", {}).items()
    )
    cv_cfg = config.cv if config else None
    methodology_section_html = f"""
    <h2>Methodology &amp; Reproducibility</h2>
    <div class="panel">
      <h3>Model-Selection Architecture</h3>
      <div class="callout"><p>{html.escape(methodology.get("model_selection_architecture", ""))}</p></div>
      <h3>Cross-Validation Execution</h3>
      <div class="callout"><p>{html.escape(methodology.get("cv_execution", ""))}</p></div>
      {"<h3>Cross-Validation Configuration</h3><div class='grid'>"
        f"<div class='card'><div class='card-label'>Outer Folds</div><div class='card-val'>{cv_cfg.outer_folds}</div></div>"
        f"<div class='card'><div class='card-label'>Selection Folds</div><div class='card-val'>{cv_cfg.selection_folds}</div></div>"
        f"<div class='card'><div class='card-label'>Inner Folds</div><div class='card-val'>{cv_cfg.inner_folds}</div></div>"
        f"<div class='card'><div class='card-label'>Classification Scoring</div><div class='card-val' style='font-size:1rem;'>{html.escape(cv_cfg.classification_scoring)}</div></div>"
        f"<div class='card'><div class='card-label'>Multiplicity Correction</div><div class='card-val' style='font-size:1rem;'>{html.escape(cv_cfg.multiplicity_correction)}</div></div>"
        "</div>" if cv_cfg else ""}
      <h3>Task-Type Resolution (per target)</h3>
      <div class="table-wrap"><table>
        <thead><tr><th>Target</th><th>Requested</th><th>Resolved</th><th>Ambiguous?</th><th>User-Confirmed?</th></tr></thead>
        <tbody>{task_type_rows_html or "<tr><td colspan='5'>No targets recorded.</td></tr>"}</tbody>
      </table></div>
      {"<h3>SEM Methodology Decisions (per target)</h3><div class='table-wrap'><table>"
        "<thead><tr><th>Target</th><th>Missing-Data Requested</th><th>Missing-Data Used</th><th>Multiplicity Correction</th><th>Convergence Status</th></tr></thead>"
        f"<tbody>{sem_decisions_rows_html}</tbody></table></div>" if sem_decisions_rows_html else ""}
      <h3>Warnings Raised During This Run</h3>
      <ul>{warnings_html}</ul>
    </div>
    """

    full_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Scientific Machine Learning & SEM Research Report</title>{_REPORT_CSS}</head>
<body><div class="container">
  <div class="header"><div><h1>Academic Machine Learning & Structural Equation Modeling Report</h1><div class="sub-title">Nested CV Tournament, Confirmatory Model Selection, Calibration Analysis, Learning Curves & Covariance Structuring</div></div></div>
  {callout_div}
  {dataset_html}
  {target_cards}
  {methodology_section_html}
  <h2>Publication-Ready LaTeX Tables</h2>
  <div class="latex-box">{latex_code}</div>
</div></body></html>"""
    (reports_dir / "full_results.html").write_text(full_html, encoding="utf-8")
