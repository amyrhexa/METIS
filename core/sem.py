# core/sem.py
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config.exceptions import SEMConfigurationError
from config.schema import ALLOWED_MISSING_DATA_METHODS, SEMTargetConfig
from core.evaluation import apply_multiplicity_correction, significance_stars

try:
    from semopy import Model, calc_stats

    HAS_SEM = True
except ImportError:
    Model = None
    calc_stats = None
    HAS_SEM = False


def validate_missing_data_method(method: str | None) -> str:
    """Validate and normalize `SEMTargetConfig.missing_data_method`.

    Raises `config.exceptions.SEMConfigurationError` for anything outside
    `config.schema.ALLOWED_MISSING_DATA_METHODS` ("fiml" | "listwise").
    METIS never silently substitutes single-value (median) imputation for
    an unrecognized or misconfigured missing-data method request (Fix 5)
    -- a prior version of this function fell through to median imputation
    for any value it didn't specifically recognize, with no indication in
    the output that this had happened.
    """
    normalized = (method or "").strip().lower()
    if normalized not in ALLOWED_MISSING_DATA_METHODS:
        msg = (
            f"Unrecognized SEM missing_data_method: {method!r}. Must be one "
            f"of {sorted(ALLOWED_MISSING_DATA_METHODS)}. METIS does not "
            f"silently substitute single-value (e.g. median) imputation "
            f"for an unrecognized missing-data method -- set "
            f"`missing_data_method` to one of the allowed values."
        )
        raise SEMConfigurationError(msg)
    return normalized


def _safe_float(val: Any, default: float = float("nan")) -> float:
    """Safely cast numeric tokens to float without silent plausible-value substitution."""
    if val is None:
        return default
    try:
        s = str(val).strip()
        if not s or s in ("-", "--", "---", "NA", "nan", "NaN", "None", "."):
            return default
        f = float(val)
        return default if np.isnan(f) else f
    except (ValueError, TypeError):
        return default


STRUCTURAL_COLUMNS = [
    "Outcome",
    "Predictor",
    "Path",
    "Beta_Std",
    "SE",
    "Test_Statistic",
    # "z (ML/Wald)" for semopy's maximum-likelihood-estimated paths, or
    # "t (OLS)" for the OLS-fallback path -- Fix 11: a t-statistic
    # (df-dependent, OLS fallback) must never be labeled as a z-statistic
    # (asymptotic-normal, ML/Wald). Both are exposed under the neutral
    # "Test_Statistic" column name plus this explicit type label, rather
    # than silently reusing one distribution's name for the other's
    # statistic.
    "Statistic_Type",
    "DF",  # degrees of freedom for the t-statistic; NaN for z (ML/Wald)
    "p_value_raw",
    "p_value_adjusted",
    "Correction_Method",
    "Family_Size",
    "CI_95",
    "Significance",  # computed from p_value_adjusted, never p_value_raw
    # Deprecated: retained only for backward compatibility with anything
    # that reads "z_value" directly. Equal to Test_Statistic regardless of
    # whether that statistic is actually z- or t-distributed -- new code
    # should read Test_Statistic + Statistic_Type instead.
    "z_value",
]

MEDIATION_COLUMNS = [
    "Predictor",
    "Target",
    "Mediators",
    "Direct_Effect_c_prime",
    "Indirect_Effect_ab",
    "Total_Effect_c",
    "Indirect_95_CI",
    "Indirect_p_value_raw",
    "Indirect_p_value_adjusted",
    "Correction_Method",
    "Family_Size",
    "Pct_Mediated",
    "Mediation_Classification",
    "CI_Method",
    "Bootstrap_Resamples_Completed",
    "Bootstrap_Resamples_Requested",
    # Deprecated alias of Indirect_p_value_raw, retained for backward
    # compatibility.
    "Indirect_p_value",
]

R2_COLUMNS = [
    "Endogenous_Variable",
    "R_Squared",
    "Variance_Explained_Pct",
]

FIT_COLUMNS = [
    "Fit Index",
    "Value",
    "Reference Cutoff",
    "Interpretation",
]


@dataclass
class SEMFitIndices:
    chi2: float = float("nan")
    dof: int = -1
    p_val: float = float("nan")
    chi2_dof: float = float("nan")
    cfi: float = float("nan")
    tli: float = float("nan")
    rmsea: float = float("nan")
    rmsea_ci_lower: float = float("nan")
    rmsea_ci_upper: float = float("nan")
    srmr: float = float("nan")
    aic: float = float("nan")
    bic: float = float("nan")
    verdict: str = "Fit indices unavailable"
    # "converged" | "unstable" | "under_identified" | "fit_error" | "unknown"
    # -- Fix / SEM hardening: a failed or numerically unstable fit must
    # never be presented as an ordinary valid result merely because no
    # exception was raised. See `assess_sem_convergence`.
    convergence_status: str = "unknown"
    convergence_notes: str = ""

    def to_dataframe(self) -> pd.DataFrame:
        def fmt(v: float) -> str:
            return "N/A" if math.isnan(v) else f"{v:.3f}"

        rows = [
            {
                "Fit Index": "Convergence Status",
                "Value": self.convergence_status,
                "Reference Cutoff": "converged",
                "Interpretation": self.convergence_notes or "—",
            },
            {
                "Fit Index": "Chi-Square (χ²)",
                "Value": fmt(self.chi2),
                "Reference Cutoff": "p > .05 (Non-significant)",
                "Interpretation": "—"
                if math.isnan(self.chi2)
                else (
                    "Acceptable"
                    if self.p_val > 0.05
                    else "Significant (Sensitive to N)"
                ),
            },
            {
                "Fit Index": "Degrees of Freedom (df)",
                "Value": "N/A" if self.dof < 0 else str(self.dof),
                "Reference Cutoff": "≥ 0",
                "Interpretation": "—"
                if self.dof < 0
                else ("Over-identified" if self.dof > 0 else "Just-identified"),
            },
            {
                "Fit Index": "χ² / df Ratio",
                "Value": fmt(self.chi2_dof),
                "Reference Cutoff": "< 3.00 (Good: < 2.00)",
                "Interpretation": "—"
                if math.isnan(self.chi2_dof)
                else ("Good" if self.chi2_dof <= 3.0 else "Elevated"),
            },
            {
                "Fit Index": "Comparative Fit Index (CFI)",
                "Value": fmt(self.cfi),
                "Reference Cutoff": "≥ 0.95 (Acceptable: ≥ 0.90)",
                "Interpretation": "—"
                if math.isnan(self.cfi)
                else (
                    "Good"
                    if self.cfi >= 0.95
                    else ("Acceptable" if self.cfi >= 0.90 else "Poor")
                ),
            },
            {
                "Fit Index": "Tucker-Lewis Index (TLI)",
                "Value": fmt(self.tli),
                "Reference Cutoff": "≥ 0.95 (Acceptable: ≥ 0.90)",
                "Interpretation": "—"
                if math.isnan(self.tli)
                else (
                    "Good"
                    if self.tli >= 0.95
                    else ("Acceptable" if self.tli >= 0.90 else "Poor")
                ),
            },
            {
                "Fit Index": "Root Mean Square Error (RMSEA)",
                "Value": fmt(self.rmsea),
                "Reference Cutoff": "< 0.06 (Acceptable: < 0.08)",
                "Interpretation": "—"
                if math.isnan(self.rmsea)
                else (
                    "Good"
                    if self.rmsea <= 0.06
                    else ("Acceptable" if self.rmsea <= 0.08 else "Poor")
                ),
            },
            {
                "Fit Index": "Standardized Root Mean Residual (SRMR)",
                "Value": fmt(self.srmr),
                "Reference Cutoff": "< 0.08 (Good: < 0.05)",
                "Interpretation": "—"
                if math.isnan(self.srmr)
                else ("Good" if self.srmr <= 0.08 else "Poor"),
            },
            {
                "Fit Index": "Akaike Information Criterion (AIC)",
                "Value": "N/A" if math.isnan(self.aic) else f"{self.aic:.2f}",
                "Reference Cutoff": "Comparative",
                "Interpretation": "Baseline",
            },
            {
                "Fit Index": "Bayesian Information Criterion (BIC)",
                "Value": "N/A" if math.isnan(self.bic) else f"{self.bic:.2f}",
                "Reference Cutoff": "Comparative",
                "Interpretation": "Baseline",
            },
        ]
        return pd.DataFrame(rows, columns=FIT_COLUMNS)


def assess_sem_convergence(
    struct_df: pd.DataFrame, dof: int, implausible_beta_threshold: float = 5.0
) -> tuple[str, str]:
    """Heuristically flag a semopy fit that completed without raising an
    exception but shows signs of numerical instability or non-convergence
    (e.g. a Heywood-like implausible standardized coefficient, or a
    majority of paths with missing/zero standard errors) -- a failed or
    unstable SEM fit must never be presented as an ordinary valid result
    just because no exception occurred (SEM scientific hardening).

    Returns (status, notes). status is one of "converged", "unstable",
    "under_identified", or "no_paths" (nothing to assess).
    """
    if dof < 0:
        return (
            "under_identified",
            "Model has negative degrees of freedom (more free parameters "
            "than known covariances/variances) -- it is not identified, "
            "and fit indices (CFI/TLI/RMSEA/chi-square) are not "
            "meaningful and are not reported.",
        )
    if struct_df.empty:
        return "no_paths", "No structural paths were estimated."

    betas = pd.to_numeric(struct_df["Beta_Std"], errors="coerce")
    ses = pd.to_numeric(struct_df["SE"], errors="coerce")
    n_paths = len(struct_df)

    implausible = betas.abs() > implausible_beta_threshold
    n_implausible = int(implausible.sum())

    invalid_se = ses.isna() | (ses <= 0)
    n_invalid_se = int(invalid_se.sum())

    notes: list[str] = []
    status = "converged"
    if n_implausible > 0:
        status = "unstable"
        notes.append(
            f"{n_implausible}/{n_paths} standardized path coefficient(s) "
            f"exceed |{implausible_beta_threshold}| -- a common symptom of "
            f"a Heywood case or non-convergence in ML-estimated SEM."
        )
    if n_invalid_se > n_paths / 2:
        status = "unstable"
        notes.append(
            f"{n_invalid_se}/{n_paths} paths have a missing or "
            f"non-positive standard error -- standard errors could not be "
            f"reliably estimated for most of the model."
        )
    if not notes:
        notes.append("No numerical instability indicators detected.")
    return status, " ".join(notes)


def compute_srmr(sample_cov: np.ndarray, implied_cov: np.ndarray) -> float:
    """Calculate the Standardized Root Mean Squared Residual (SRMR) (Bentler 1995)."""
    p = sample_cov.shape[0]
    if p < 2 or sample_cov.shape != implied_cov.shape:
        return float("nan")

    sample_sd = np.sqrt(np.maximum(1e-12, np.diag(sample_cov)))
    implied_sd = np.sqrt(np.maximum(1e-12, np.diag(implied_cov)))

    d_sample = np.outer(sample_sd, sample_sd)
    d_implied = np.outer(implied_sd, implied_sd)

    r_sample = sample_cov / d_sample
    r_implied = implied_cov / d_implied

    res_corr = r_sample - r_implied
    triu_indices = np.triu_indices(p, k=1)
    residuals_sq = res_corr[triu_indices] ** 2
    n_pairs = max(1, len(triu_indices[0]))
    return float(np.sqrt(np.sum(residuals_sq) / n_pairs))


def sanitize_sem_identifier(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name.strip())
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"V_{cleaned}"
    return cleaned


def clean_sem_syntax(syntax: str) -> str:
    lines = []
    for raw_line in syntax.splitlines():
        line = re.sub(r"#.*$", "", raw_line).strip()
        if not line:
            continue
        line = re.sub(r"->", "~", line)
        lines.append(line)
    return "\n".join(lines)


def extract_variables_from_syntax(syntax: str) -> list[str]:
    cleaned = clean_sem_syntax(syntax)
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", cleaned)
    operators = {"Gf", "latent", "mean", "var", "cov", "exp", "log"}
    return [t for t in dict.fromkeys(tokens) if t not in operators]


def build_sem_variable_map(
    config: SEMTargetConfig, available_cols: list[str] | None = None
) -> dict[str, str]:
    raw_cols = [
        config.target,
        *config.feature_columns,
        *config.covariates,
        *config.mediators,
    ]
    for indicators in config.latent_variables.values():
        raw_cols.extend(indicators)

    if config.custom_model_syntax and config.custom_model_syntax.strip():
        syntax_vars = extract_variables_from_syntax(config.custom_model_syntax)
        if available_cols:
            syntax_vars = [v for v in syntax_vars if v in available_cols]
        raw_cols.extend(syntax_vars)

    if available_cols:
        raw_cols = [
            c for c in raw_cols if c in available_cols or c in config.latent_variables
        ]

    deduped_cols = list(dict.fromkeys(raw_cols))
    variable_map: dict[str, str] = {}
    gen_from: dict[str, str] = {}

    for raw in deduped_cols:
        sem_name = sanitize_sem_identifier(raw)
        if sem_name in gen_from and gen_from[sem_name] != raw:
            msg = f"SEM variable collision: '{raw}' and '{gen_from[sem_name]}' mapped to '{sem_name}'."
            raise ValueError(msg)
        gen_from[sem_name] = raw
        variable_map[raw] = sem_name
    return variable_map


def build_sem_model_syntax(
    config: SEMTargetConfig, variable_map: dict[str, str]
) -> str:
    if config.custom_model_syntax and config.custom_model_syntax.strip():
        cleaned = clean_sem_syntax(config.custom_model_syntax)
        for raw, sanitized in variable_map.items():
            if raw != sanitized:
                cleaned = re.sub(rf"\b{re.escape(raw)}\b", sanitized, cleaned)
        return cleaned

    lines: list[str] = []
    for lat_name, indicators in config.latent_variables.items():
        inds = " + ".join(variable_map[c] for c in indicators if c in variable_map)
        if inds:
            lines.append(f"{lat_name} =~ {inds}")

    target_name = variable_map.get(
        config.target, sanitize_sem_identifier(config.target)
    )
    direct_preds = [
        variable_map[c]
        for c in [*config.feature_columns, *config.covariates]
        if c in variable_map and c != config.target
    ]
    if direct_preds:
        lines.append(f"{target_name} ~ {' + '.join(direct_preds)}")

    med_names: list[str] = []
    for med in config.mediators:
        if med not in variable_map or med == config.target:
            continue
        med_name = variable_map[med]
        med_names.append(med_name)
        preds = [
            variable_map[c]
            for c in config.feature_columns
            if c in variable_map and c != med and c != config.target
        ]
        if preds:
            lines.append(f"{med_name} ~ {' + '.join(preds)}")
        lines.append(f"{target_name} ~ {med_name}")

    # Parallel mediators sharing common (unmodeled) causes are almost always
    # correlated in practice (e.g. brain-region volumes sharing overall brain
    # size). Fixing their residual covariance to zero -- semopy/lavaan's
    # default when it's left unspecified -- is a strong, usually false,
    # assumption that inflates chi-square and wrecks CFI/TLI/RMSEA. Free
    # these residual covariances explicitly so fit indices reflect genuine
    # model (mis)fit rather than this omission.
    for i, m_a in enumerate(med_names):
        for m_b in med_names[i + 1 :]:
            lines.append(f"{m_a} ~~ {m_b}")

    return "\n".join(lines)


def _prepare_sem_dataframe(
    dataset_df: pd.DataFrame,
    variable_map: dict[str, str],
    standardize: bool = True,
    preserve_nan: bool = False,
) -> pd.DataFrame:
    valid_cols = [
        c for c in dict.fromkeys(variable_map.keys()) if c in dataset_df.columns
    ]
    sem_data = pd.DataFrame()

    for col in valid_cols:
        target_name = variable_map[col]
        series = dataset_df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        num_series = pd.to_numeric(series, errors="coerce").astype(float)

        if not preserve_nan and num_series.isna().any():
            median_val = (
                float(num_series.dropna().median())
                if len(num_series.dropna()) > 0
                else 0.0
            )
            num_series = num_series.fillna(median_val)

        valid_vals = num_series.dropna()
        std_val = float(valid_vals.std(ddof=1)) if len(valid_vals) > 1 else 0.0
        if np.isnan(std_val):
            std_val = 0.0

        if standardize and std_val > 1e-6:
            mean_val = float(valid_vals.mean())
            sem_data[target_name] = (num_series - mean_val) / std_val
        else:
            sem_data[target_name] = num_series

    return sem_data


def extract_effects_from_sem_params(
    params_df: pd.DataFrame,
    target_clean: str,
    predictors: list[str],
    mediators: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """Extract structural regression coefficients directly from semopy inspect DataFrame."""
    direct_effects: dict[str, float] = {}
    med_to_target: dict[str, float] = {}
    pred_to_med: dict[str, dict[str, float]] = {m: {} for m in mediators}

    reg_paths = params_df[params_df["op"] == "~"].copy()
    for _, row in reg_paths.iterrows():
        outcome = str(row["lval_clean"])
        predictor = str(row["rval_clean"])
        beta = _safe_float(row.get("Est. Std", row.get("Estimate", float("nan"))))

        if outcome == target_clean:
            if predictor in predictors:
                direct_effects[predictor] = beta
            elif predictor in mediators:
                med_to_target[predictor] = beta
        elif outcome in mediators and predictor in predictors:
            pred_to_med[outcome][predictor] = beta

    return direct_effects, med_to_target, pred_to_med


def compute_sem_mediation_with_bootstrap(
    model_syntax: str,
    sem_data: pd.DataFrame,
    target: str,
    target_clean: str,
    predictors: list[str],
    mediators: list[str],
    rev_vmap: dict[str, str],
    point_params_df: pd.DataFrame,
    n_boot: int = 2000,
    random_state: int = 42,
    fit_obj: str = "MLW",
    logger: logging.Logger | None = None,
    multiplicity_correction: str = "holm",
    direct_p_adjusted_lookup: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Computes genuine non-parametric bootstrap confidence intervals refitting the exact SEM model.

    Fix 2 (multiplicity correction): when more than one predictor's
    indirect effect is tested, treating each at a raw/uncorrected p < .05
    threshold inflates the family-wise false-positive rate. This function
    now computes the raw indirect-effect p-value for every predictor
    first, applies `multiplicity_correction` (default: Holm step-down)
    across that whole family, and only then classifies each predictor's
    mediation pattern using the *adjusted* p-value -- raw values are
    retained alongside the adjusted ones (see MEDIATION_COLUMNS), never
    discarded.

    `direct_p_adjusted_lookup`, if provided, maps predictor -> the
    already-multiplicity-corrected p-value for that predictor's direct
    path to `target_clean` (as computed for the structural-paths table).
    When available it is used for the "dir_sig" half of the mediation
    classification instead of the raw direct-path p-value, so the
    mediation table's classification is never inconsistent with the
    (adjusted) significance already reported for that same path in the
    structural-paths table.
    """
    direct_pt, b_pt, a_pt = extract_effects_from_sem_params(
        point_params_df, target_clean, predictors, mediators
    )

    direct_p_dict: dict[str, float] = {}
    for _, row in point_params_df[point_params_df["op"] == "~"].iterrows():
        if str(row["lval_clean"]) == target_clean:
            p_val = _safe_float(row.get("p-value", row.get("p_value", float("nan"))))
            direct_p_dict[str(row["rval_clean"])] = p_val

    rng = np.random.default_rng(random_state)
    n_obs = len(sem_data)
    boot_indirect_records: dict[str, list[float]] = {p: [] for p in predictors}
    successful_resamples = 0

    if logger:
        logger.info(
            "Executing %d empirical bootstrap resamples for SEM mediation on %s...",
            n_boot,
            target_clean,
        )

    for _ in range(n_boot):
        boot_idx = rng.integers(0, n_obs, size=n_obs)
        boot_df = sem_data.iloc[boot_idx].copy()
        try:
            m_boot = Model(model_syntax)
            m_boot.fit(boot_df, obj=fit_obj)
            insp_b = m_boot.inspect(std_est=True)
            if insp_b is None or insp_b.empty:
                continue

            insp_b["lval_clean"] = insp_b["lval"].map(lambda x: rev_vmap.get(x, x))
            insp_b["rval_clean"] = insp_b["rval"].map(lambda x: rev_vmap.get(x, x))
            _, b_b, a_b = extract_effects_from_sem_params(
                insp_b, target_clean, predictors, mediators
            )

            for p in predictors:
                total_ind_b = 0.0
                for m in mediators:
                    a_val = a_b.get(m, {}).get(p, 0.0)
                    b_val = b_b.get(m, 0.0)
                    total_ind_b += a_val * b_val
                boot_indirect_records[p].append(total_ind_b)
            successful_resamples += 1
        except Exception:
            continue

    # --- Pass 1: compute every predictor's raw quantities (no classification yet) ---
    partial_rows: list[dict[str, Any]] = []
    for p in predictors:
        c_prime = direct_pt.get(p, float("nan"))
        c_prime_p_raw = direct_p_dict.get(p, float("nan"))
        c_prime_p_for_classification = (
            direct_p_adjusted_lookup.get(p, c_prime_p_raw)
            if direct_p_adjusted_lookup
            else c_prime_p_raw
        )
        total_indirect = 0.0
        meds_used = []
        for m in mediators:
            a_val = a_pt.get(m, {}).get(p, 0.0)
            b_val = b_pt.get(m, 0.0)
            total_indirect += a_val * b_val
            meds_used.append(m)

        total_effect = (
            c_prime + total_indirect
            if not math.isnan(c_prime) and not math.isnan(total_indirect)
            else float("nan")
        )
        boot_samples = np.array(boot_indirect_records.get(p, []), dtype=float)
        boot_samples = boot_samples[np.isfinite(boot_samples)]

        if len(boot_samples) >= 10:
            prop_less = np.mean(boot_samples < total_indirect)
            eps = 1.0 / len(boot_samples)
            z0 = float(stats.norm.ppf(np.clip(prop_less, eps, 1.0 - eps)))
            alpha_levels = [0.025, 0.975]
            pct_levels = [
                float(stats.norm.cdf(2.0 * z0 + stats.norm.ppf(a))) * 100.0
                for a in alpha_levels
            ]
            ci_l, ci_u = np.percentile(boot_samples, pct_levels)
            ci_str = f"[{ci_l:.3f}, {ci_u:.3f}]"
            p_ind_raw = float(
                2.0 * min(np.mean(boot_samples <= 0), np.mean(boot_samples >= 0))
            )
            p_ind_raw = min(1.0, max(eps, p_ind_raw))
            ci_method = "Bias-Corrected Bootstrap (BC)"
        else:
            ci_l, ci_u = float("nan"), float("nan")
            ci_str = "N/A"
            p_ind_raw = float("nan")
            ci_method = "Bootstrap Failed"

        pct_med = (
            (total_indirect / total_effect * 100.0)
            if not math.isnan(total_effect) and abs(total_effect) > 1e-4
            else 0.0
        )

        partial_rows.append(
            {
                "Predictor": p,
                "Target": target_clean,
                "Mediators": "; ".join(meds_used) if meds_used else "Direct Path",
                "Direct_Effect_c_prime": round(c_prime, 3)
                if not math.isnan(c_prime)
                else float("nan"),
                "Indirect_Effect_ab": round(total_indirect, 3)
                if not math.isnan(total_indirect)
                else float("nan"),
                "Total_Effect_c": round(total_effect, 3)
                if not math.isnan(total_effect)
                else float("nan"),
                "Indirect_95_CI": ci_str,
                "Pct_Mediated": "—"
                if math.isnan(total_effect) or abs(total_effect) <= 1e-4
                else f"{pct_med:.1f}%",
                "CI_Method": ci_method,
                "Bootstrap_Resamples_Completed": successful_resamples,
                "Bootstrap_Resamples_Requested": n_boot,
                "_p_ind_raw": p_ind_raw,
                "_ci_l": ci_l,
                "_ci_u": ci_u,
                "_dir_sig_p": c_prime_p_for_classification,
                "_total_indirect": total_indirect,
                "_c_prime": c_prime,
            }
        )

    return _finalize_mediation_rows(partial_rows, multiplicity_correction)


def _finalize_mediation_rows(
    partial_rows: list[dict[str, Any]], correction_method: str = "holm"
) -> pd.DataFrame:
    """Shared finalization step for both the semopy-bootstrap and OLS-fallback
    mediation paths: applies the multiplicity correction across the family
    of indirect-effect p-values, classifies each predictor's mediation
    pattern from the *adjusted* p-value, and assembles the final
    MEDIATION_COLUMNS-shaped DataFrame (Fix 2)."""
    if not partial_rows:
        return pd.DataFrame(columns=MEDIATION_COLUMNS)

    raw_p = np.array([r["_p_ind_raw"] for r in partial_rows], dtype=float)
    adjusted_p = apply_multiplicity_correction(raw_p, correction_method)
    family_size = int(np.sum(np.isfinite(raw_p)))

    final_rows: list[dict[str, Any]] = []
    for row, p_adj in zip(partial_rows, adjusted_p, strict=False):
        p_ind_raw = row["_p_ind_raw"]
        ci_l, ci_u = row["_ci_l"], row["_ci_u"]
        c_prime_p = row["_dir_sig_p"]
        total_indirect = row["_total_indirect"]
        c_prime = row["_c_prime"]

        ind_sig = (
            np.isfinite(p_adj)
            and p_adj < 0.05
            and np.isfinite(ci_l)
            and np.isfinite(ci_u)
            and ci_l * ci_u > 0
        )
        dir_sig = (not math.isnan(c_prime_p)) and (c_prime_p < 0.05)

        if math.isnan(p_ind_raw):
            verdict = "Bootstrap Failed"
        elif ind_sig and not dir_sig:
            verdict = "Indirect-only mediation"
        elif ind_sig and dir_sig:
            verdict = (
                "Complementary mediation"
                if total_indirect * c_prime > 0
                else "Competitive mediation"
            )
        elif not ind_sig and dir_sig:
            verdict = "Direct-only non-mediation"
        else:
            verdict = "No effect non-mediation"

        final_rows.append(
            {
                "Predictor": row["Predictor"],
                "Target": row["Target"],
                "Mediators": row["Mediators"],
                "Direct_Effect_c_prime": row["Direct_Effect_c_prime"],
                "Indirect_Effect_ab": row["Indirect_Effect_ab"],
                "Total_Effect_c": row["Total_Effect_c"],
                "Indirect_95_CI": row["Indirect_95_CI"],
                "Indirect_p_value_raw": round(p_ind_raw, 4)
                if np.isfinite(p_ind_raw)
                else float("nan"),
                "Indirect_p_value_adjusted": round(float(p_adj), 4)
                if np.isfinite(p_adj)
                else float("nan"),
                "Correction_Method": correction_method,
                "Family_Size": family_size,
                "Pct_Mediated": row["Pct_Mediated"],
                "Mediation_Classification": verdict,
                "CI_Method": row["CI_Method"],
                "Bootstrap_Resamples_Completed": row["Bootstrap_Resamples_Completed"],
                "Bootstrap_Resamples_Requested": row["Bootstrap_Resamples_Requested"],
                "Indirect_p_value": round(p_ind_raw, 4)
                if np.isfinite(p_ind_raw)
                else float("nan"),
            }
        )

    return pd.DataFrame(final_rows, columns=MEDIATION_COLUMNS)


def _finalize_structural_significance(
    struct_rows: list[dict[str, Any]], correction_method: str = "holm"
) -> pd.DataFrame:
    """Shared finalization step for structural-path rows from either the
    semopy (z, ML/Wald) or OLS-fallback (t, OLS) estimation path.

    Each input row must already contain 'p_value_raw', 'Test_Statistic',
    'Statistic_Type', and 'DF' (NaN for the semopy/z path). This function
    applies the requested multiplicity correction across the full family
    of paths in the model, and populates 'p_value_adjusted',
    'Correction_Method', 'Family_Size', 'Significance' (computed from the
    adjusted p-value via `core.evaluation.significance_stars`), and the
    deprecated 'z_value' alias column (Fix 2 / Fix 11).
    """
    if not struct_rows:
        return pd.DataFrame(columns=STRUCTURAL_COLUMNS)

    raw_p = np.array([r.get("p_value_raw", np.nan) for r in struct_rows], dtype=float)
    adjusted_p = apply_multiplicity_correction(raw_p, correction_method)
    family_size = int(np.sum(np.isfinite(raw_p)))

    final_rows: list[dict[str, Any]] = []
    for row, p_adj in zip(struct_rows, adjusted_p, strict=False):
        final_rows.append(
            {
                "Outcome": row["Outcome"],
                "Predictor": row["Predictor"],
                "Path": row["Path"],
                "Beta_Std": row["Beta_Std"],
                "SE": row["SE"],
                "Test_Statistic": row.get("Test_Statistic", float("nan")),
                "Statistic_Type": row.get("Statistic_Type", "unknown"),
                "DF": row.get("DF", float("nan")),
                "p_value_raw": row.get("p_value_raw", float("nan")),
                "p_value_adjusted": round(float(p_adj), 4)
                if np.isfinite(p_adj)
                else float("nan"),
                "Correction_Method": correction_method,
                "Family_Size": family_size,
                "CI_95": row["CI_95"],
                "Significance": significance_stars(p_adj),
                # Deprecated alias -- see STRUCTURAL_COLUMNS docstring note.
                "z_value": row.get("Test_Statistic", float("nan")),
            }
        )
    return pd.DataFrame(final_rows, columns=STRUCTURAL_COLUMNS)


def compute_comprehensive_ols_sem(
    dataset_df: pd.DataFrame,
    config: SEMTargetConfig,
    variable_map: dict[str, str],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, SEMFitIndices
]:
    target_col = config.target
    required_columns = list(
        dict.fromkeys(
            [target_col, *config.feature_columns, *config.covariates, *config.mediators]
        )
    )
    required_columns = [c for c in required_columns if c in dataset_df.columns]

    if target_col not in required_columns:
        return (
            pd.DataFrame(columns=STRUCTURAL_COLUMNS),
            pd.DataFrame(columns=MEDIATION_COLUMNS),
            pd.DataFrame(columns=R2_COLUMNS),
            pd.DataFrame(),
            pd.DataFrame(),
            SEMFitIndices(),
        )

    clean_data = pd.DataFrame()
    for col in required_columns:
        series = dataset_df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        s = pd.to_numeric(series, errors="coerce").astype(float)
        if col != target_col and s.isna().any():
            med_val = float(s.dropna().median()) if len(s.dropna()) > 0 else 0.0
            s = s.fillna(med_val)
        std_val = float(s.std(ddof=1)) if len(s.dropna()) > 1 else 0.0
        mean_val = float(s.mean()) if len(s.dropna()) > 0 else 0.0
        clean_data[col] = (s - mean_val) / std_val if std_val > 1e-6 else (s - mean_val)

    data = clean_data.dropna(subset=[target_col]).dropna()
    n = len(data)
    if n < 4:
        return (
            pd.DataFrame(columns=STRUCTURAL_COLUMNS),
            pd.DataFrame(columns=MEDIATION_COLUMNS),
            pd.DataFrame(columns=R2_COLUMNS),
            pd.DataFrame(),
            pd.DataFrame(),
            SEMFitIndices(),
        )

    struct_rows: list[dict[str, Any]] = []
    r2_rows: list[dict[str, Any]] = []
    med_paths: dict[str, dict[str, tuple[float, float, float, int]]] = {}
    system_loglik = 0.0
    total_estimated_params = 0

    def fit_equation(
        y_var: str, x_vars: list[str]
    ) -> tuple[dict[str, tuple[float, float, float, int]] | None, float, float]:
        xs = [x for x in dict.fromkeys(x_vars) if x in data.columns and x != y_var]
        if not xs:
            return {}, 0.0, 0.0
        y_vec = data[y_var].to_numpy(dtype=float)
        x_mat = np.column_stack([np.ones(n), data[xs].to_numpy(dtype=float)])
        try:
            beta, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
            df_e = max(1, n - x_mat.shape[1])
            res = y_vec - x_mat @ beta
            rss = float(np.sum(res**2))
            var_denom = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
            r2 = 1.0 - (rss / (var_denom if var_denom > 0 else 1.0))
            sigma2 = float(rss / df_e)
            cov_beta = sigma2 * np.linalg.pinv(x_mat.T @ x_mat)
            se_beta = np.sqrt(np.maximum(1e-12, np.diag(cov_beta)))
            out = {}
            for idx, x_name in enumerate(xs):
                b = float(beta[idx + 1])
                se = float(se_beta[idx + 1])
                t_stat = b / max(1e-8, se)
                p = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=df_e)))
                out[x_name] = (b, se, p, df_e)
            ll = -0.5 * n * (np.log(2.0 * np.pi) + 1.0 + np.log(max(1e-12, rss / n)))
            return out, max(0.0, float(r2)), float(ll)
        except Exception:
            return None, 0.0, 0.0

    for med in config.mediators:
        if med == target_col or med not in data.columns:
            continue
        m_fit, m_r2, m_ll = fit_equation(med, config.feature_columns)
        if m_fit is None:
            continue
        med_paths[med] = m_fit
        system_loglik += m_ll
        total_estimated_params += len(m_fit) + 1
        r2_rows.append(
            {
                "Endogenous_Variable": med,
                "R_Squared": round(m_r2, 4),
                "Variance_Explained_Pct": f"{m_r2:.1%}",
            }
        )
        for pred, (beta_value, se_value, p_value, df_e) in m_fit.items():
            t_val = float(beta_value) / max(1e-8, float(se_value))
            ci_l = float(beta_value) - 1.96 * float(se_value)
            ci_u = float(beta_value) + 1.96 * float(se_value)
            struct_rows.append(
                {
                    "Outcome": med,
                    "Predictor": pred,
                    "Path": f"{med} ← {pred}",
                    "Beta_Std": round(float(beta_value), 3),
                    "SE": round(float(se_value), 3),
                    "Test_Statistic": round(float(t_val), 3),
                    "Statistic_Type": "t (OLS)",
                    "DF": int(df_e),
                    "p_value_raw": round(float(p_value), 4),
                    "CI_95": f"[{ci_l:.3f}, {ci_u:.3f}]",
                }
            )

    all_target_preds = [
        p
        for p in [*config.feature_columns, *config.covariates, *config.mediators]
        if p != target_col
    ]
    target_fit_results, target_r2_score, t_ll = fit_equation(
        target_col, all_target_preds
    )
    if target_fit_results is not None:
        system_loglik += t_ll
        total_estimated_params += len(target_fit_results) + 1
        r2_rows.append(
            {
                "Endogenous_Variable": target_col,
                "R_Squared": round(target_r2_score, 4),
                "Variance_Explained_Pct": f"{target_r2_score:.1%}",
            }
        )
        for pred, (beta_value, se_value, p_value, df_e) in target_fit_results.items():
            t_val = float(beta_value) / max(1e-8, float(se_value))
            ci_l = float(beta_value) - 1.96 * float(se_value)
            ci_u = float(beta_value) + 1.96 * float(se_value)
            struct_rows.append(
                {
                    "Outcome": target_col,
                    "Predictor": pred,
                    "Path": f"{target_col} ← {pred}",
                    "Beta_Std": round(float(beta_value), 3),
                    "SE": round(float(se_value), 3),
                    "Test_Statistic": round(float(t_val), 3),
                    "Statistic_Type": "t (OLS)",
                    "DF": int(df_e),
                    "p_value_raw": round(float(p_value), 4),
                    "CI_95": f"[{ci_l:.3f}, {ci_u:.3f}]",
                }
            )

    partial_mediation_rows: list[dict[str, Any]] = []
    n_boot = config.bootstrap_n if config.bootstrap else 100
    rng = np.random.default_rng(config.random_state)

    for pred in config.feature_columns:
        if pred == target_col:
            continue
        direct_tuple = (
            target_fit_results.get(pred, (0.0, 0.1, 1.0))
            if target_fit_results
            else (0.0, 0.1, 1.0, 1)
        )
        c_prime = float(direct_tuple[0])
        c_prime_p_raw = float(direct_tuple[2])

        boot_ab = []
        for _ in range(n_boot):
            b_idx = rng.integers(0, n, size=n)
            d_b = data.iloc[b_idx]
            ab_sum = 0.0
            for med in config.mediators:
                if med not in data.columns or pred not in data.columns:
                    continue
                try:
                    preds_a = [p for p in config.feature_columns if p in d_b.columns]
                    x_a_mat = np.column_stack([np.ones(n), d_b[preds_a].to_numpy()])
                    coef_a = np.linalg.lstsq(x_a_mat, d_b[med].to_numpy(), rcond=None)[
                        0
                    ][preds_a.index(pred) + 1]

                    preds_all = [p for p in all_target_preds if p in d_b.columns]
                    x_b_mat = np.column_stack([np.ones(n), d_b[preds_all].to_numpy()])
                    coef_b = np.linalg.lstsq(
                        x_b_mat, d_b[target_col].to_numpy(), rcond=None
                    )[0][preds_all.index(med) + 1]
                    ab_sum += coef_a * coef_b
                except Exception:
                    pass
            boot_ab.append(ab_sum)

        total_indirect = float(np.mean(boot_ab)) if boot_ab else 0.0
        total_effect = c_prime + total_indirect
        ci_l, ci_u = (
            np.percentile(boot_ab, [2.5, 97.5])
            if boot_ab
            else (float("nan"), float("nan"))
        )
        p_ind_raw = (
            float(
                2.0
                * min(np.mean(np.array(boot_ab) <= 0), np.mean(np.array(boot_ab) >= 0))
            )
            if boot_ab
            else float("nan")
        )

        partial_mediation_rows.append(
            {
                "Predictor": pred,
                "Target": target_col,
                "Mediators": "; ".join(config.mediators)
                if config.mediators
                else "Direct Path",
                "Direct_Effect_c_prime": round(c_prime, 3),
                "Indirect_Effect_ab": round(total_indirect, 3),
                "Total_Effect_c": round(total_effect, 3),
                "Indirect_95_CI": f"[{ci_l:.3f}, {ci_u:.3f}]",
                "Pct_Mediated": f"{(total_indirect / total_effect * 100.0):.1f}%"
                if abs(total_effect) > 1e-4
                else "—",
                "CI_Method": "OLS Bootstrap Resampling (Fallback)",
                "Bootstrap_Resamples_Completed": len(boot_ab),
                "Bootstrap_Resamples_Requested": n_boot,
                "_p_ind_raw": p_ind_raw,
                "_ci_l": ci_l,
                "_ci_u": ci_u,
                "_dir_sig_p": c_prime_p_raw,
                "_total_indirect": total_indirect,
                "_c_prime": c_prime,
            }
        )

    aic_val = 2.0 * total_estimated_params - 2.0 * system_loglik
    bic_val = total_estimated_params * np.log(max(1, n)) - 2.0 * system_loglik

    fit_idx = SEMFitIndices(
        chi2=float("nan"),
        dof=-1,
        p_val=float("nan"),
        chi2_dof=float("nan"),
        cfi=float("nan"),
        tli=float("nan"),
        rmsea=float("nan"),
        srmr=float("nan"),
        aic=round(float(aic_val), 2),
        bic=round(float(bic_val), 2),
        verdict="OLS Regression Path Approximation (SEM Unfitted)",
        convergence_status="not_applicable_ols_fallback",
        convergence_notes=(
            "This is the OLS path-regression approximation, used because "
            "exact SEM (semopy) either is unavailable or failed to fit. "
            "It reports per-equation OLS coefficients and an approximate, "
            "per-equation-summed AIC/BIC -- it is not a joint SEM fit, and "
            "chi-square/CFI/TLI/RMSEA are not defined for it (reported as "
            "N/A rather than a placeholder value)."
        ),
    )

    correction_method = config.multiplicity_correction or "holm"
    struct_df = _finalize_structural_significance(struct_rows, correction_method)
    med_df = _finalize_mediation_rows(partial_mediation_rows, correction_method)
    r2_df = pd.DataFrame(r2_rows, columns=R2_COLUMNS)

    param_records = []
    for _, r in struct_df.iterrows():
        param_records.append(
            {
                "lval": variable_map.get(r["Outcome"], r["Outcome"]),
                "op": "~",
                "rval": variable_map.get(r["Predictor"], r["Predictor"]),
                "lval_clean": r["Outcome"],
                "rval_clean": r["Predictor"],
                "Estimate": r["Beta_Std"],
                "Est. Std": r["Beta_Std"],
                "Std. Err": r["SE"],
                "z-value": r["Test_Statistic"],
                "p-value": r["p_value_raw"],
            }
        )
    params_df = pd.DataFrame(param_records)

    return struct_df, med_df, r2_df, pd.DataFrame(), params_df, fit_idx


def generate_sem_narrative(
    target: str,
    struct_df: pd.DataFrame,
    med_df: pd.DataFrame,
    r2_df: pd.DataFrame,
    fit: SEMFitIndices,
    n_sample: int,
    fit_method: str = "semopy",
    missing_data_method: str = "fiml",
) -> str:
    r2_val = "an unavailable percentage"
    if not r2_df.empty:
        target_r2_score = r2_df[r2_df["Endogenous_Variable"] == target]
        if not target_r2_score.empty:
            pct = str(target_r2_score.iloc[0]["Variance_Explained_Pct"])
            r2_val = pct if pct != "N/A" else "an unavailable percentage"

    sig_paths = (
        struct_df[struct_df["p_value_adjusted"] < 0.05]
        if not struct_df.empty
        else pd.DataFrame()
    )
    path_summaries = []
    for _, r in sig_paths.head(6).iterrows():
        path_summaries.append(
            f"{r['Predictor']} was significantly associated with {r['Outcome']} "
            f"(β = {r['Beta_Std']:.2f}, p_adj = {r['p_value_adjusted']:.3f}, "
            f"{r['Correction_Method']}-corrected)"
        )

    path_text = (
        "; ".join(path_summaries) + "."
        if path_summaries
        else "No direct regression paths reached threshold statistical significance "
        "(after multiplicity correction)."
    )

    med_text = ""
    if not med_df.empty:
        sig_med = med_df[med_df["Indirect_p_value_adjusted"] < 0.05]
        if not sig_med.empty:
            top_m = sig_med.iloc[0]
            med_text = (
                f" Mediation path analysis ({top_m.get('CI_Method', 'Bootstrap')}) revealed that "
                f"{top_m['Mediators']} mediated the relationship between {top_m['Predictor']} and {target} "
                f"(Indirect β = {top_m['Indirect_Effect_ab']:.2f}, 95% CI {top_m['Indirect_95_CI']}, "
                f"p_adj = {top_m['Indirect_p_value_adjusted']:.3f}, {top_m['Correction_Method']}-corrected, "
                f"{top_m['Mediation_Classification']})."
            )

    if math.isnan(fit.cfi):
        fit_str = "Global covariance fit indices are not applicable or unavailable for this path model."
    else:
        srmr_str = (
            f"SRMR = {fit.srmr:.3f}" if not math.isnan(fit.srmr) else "SRMR = N/A"
        )
        aic_str = (
            f"AIC = {fit.aic:.1f}, BIC = {fit.bic:.1f}"
            if not math.isnan(fit.aic)
            else "AIC/BIC = N/A"
        )
        fit_str = (
            f"The empirical covariance structure demonstrated {fit.verdict.lower()} "
            f"(χ²({fit.dof}) = {fit.chi2:.2f}, p = {fit.p_val:.3f}, CFI = {fit.cfi:.3f}, "
            f"TLI = {fit.tli:.3f}, RMSEA = {fit.rmsea:.3f}, {srmr_str}, {aic_str})."
        )

    method_str = (
        "semopy maximum-likelihood covariance optimization"
        if fit_method == "semopy"
        else "OLS multivariate path approximation"
    )
    missing_str = (
        "Full-Information Maximum Likelihood (FIML), which uses all "
        "available data per case without deleting or imputing missing "
        "values,"
        if missing_data_method == "fiml"
        else "listwise deletion (complete cases only)"
    )
    convergence_note = (
        f" CAUTION: fit diagnostics flagged this model as '{fit.convergence_status}' "
        f"-- {fit.convergence_notes}"
        if fit.convergence_status not in ("converged", "not_applicable_ols_fallback", "unknown")
        else ""
    )

    narrative = (
        f"A Structural Equation Model (SEM) was estimated via {method_str} across N = {n_sample:,} observations "
        f"with target outcome '{target}', using {missing_str} to handle missing data. "
        f"{fit_str} The model accounted for {r2_val} of the total variance in {target}. Specifically, {path_text}{med_text} "
        f"Observational path coefficients reflect standardized associative effect magnitudes within the multivariate covariance manifold. "
        f"Significance labels and p-values reported above are multiplicity-corrected across the family of paths/effects tested in this "
        f"model (see Correction_Method); raw, uncorrected p-values remain available in the underlying tables.{convergence_note}"
    )
    return narrative


def run_sem_target(
    target: str,
    config: SEMTargetConfig,
    dataset_df: pd.DataFrame,
    out_dir: Path,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    if not config.enabled:
        return {"target": target, "status": "disabled"}

    available_cols = list(dataset_df.columns)
    try:
        variable_map = build_sem_variable_map(config, available_cols=available_cols)
    except Exception as exc:
        return {"target": target, "status": "invalid_config", "error": str(exc)}

    rev_vmap = {v: k for k, v in variable_map.items()}
    # Fix 5: fail loudly on an unrecognized missing_data_method rather than
    # silently falling back to single-value (median) imputation.
    missing_method = validate_missing_data_method(config.missing_data_method)

    # NaNs must be preserved for FIML (semopy needs the raw missingness pattern
    # to compute the full-information likelihood) and for listwise deletion
    # (so we can drop incomplete rows below). Both are the only supported
    # values (validated above), so no third "unknown method" branch exists.
    sem_data = _prepare_sem_dataframe(
        dataset_df,
        variable_map,
        standardize=config.standardize,
        preserve_nan=True,
    )

    if missing_method == "fiml":
        fit_obj = "FIML"
    else:  # "listwise" -- the only other value validate_missing_data_method allows
        sem_data = sem_data.dropna(how="any")
        fit_obj = "MLW"

    target_sem = variable_map.get(config.target, config.target)
    if target_sem in sem_data.columns:
        sem_data = sem_data.dropna(subset=[target_sem])

    n_sample = len(sem_data)
    target_sem_dir = out_dir / "sem" / target
    target_sem_dir.mkdir(parents=True, exist_ok=True)

    fit_method = "ols_path_approximation"
    struct_df = pd.DataFrame()
    med_df = pd.DataFrame()
    r2_df = pd.DataFrame()
    params_df = pd.DataFrame()
    fit_idx = SEMFitIndices()

    if HAS_SEM and Model is not None:
        try:
            syntax = build_sem_model_syntax(config, variable_map)
            model_inst = Model(syntax)
            model_inst.fit(sem_data, obj=fit_obj)
            inspect_df = model_inst.inspect(std_est=True)

            if inspect_df is not None and not inspect_df.empty:
                params_df = inspect_df.copy()
                params_df["lval_clean"] = params_df["lval"].map(
                    lambda x: rev_vmap.get(x, x)
                )
                params_df["rval_clean"] = params_df["rval"].map(
                    lambda x: rev_vmap.get(x, x)
                )

                reg_paths = params_df[params_df["op"] == "~"].copy()
                st_rows = []
                for _, pr in reg_paths.iterrows():
                    b_std = _safe_float(
                        pr.get("Est. Std", pr.get("Estimate", float("nan")))
                    )
                    se_value = _safe_float(pr.get("Std. Err", float("nan")))
                    pval = _safe_float(pr.get("p-value", float("nan")))
                    z_val = _safe_float(pr.get("z-value", float("nan")))
                    if (
                        math.isnan(z_val)
                        and not math.isnan(b_std)
                        and not math.isnan(se_value)
                        and se_value > 0
                    ):
                        z_val = b_std / se_value

                    ci_str = (
                        f"[{b_std - 1.96 * se_value:.3f}, {b_std + 1.96 * se_value:.3f}]"
                        if not math.isnan(b_std) and not math.isnan(se_value)
                        else "N/A"
                    )

                    st_rows.append(
                        {
                            "Outcome": pr["lval_clean"],
                            "Predictor": pr["rval_clean"],
                            "Path": f"{pr['lval_clean']} ← {pr['rval_clean']}",
                            "Beta_Std": round(b_std, 3)
                            if not math.isnan(b_std)
                            else float("nan"),
                            "SE": round(se_value, 3)
                            if not math.isnan(se_value)
                            else float("nan"),
                            "Test_Statistic": round(z_val, 3)
                            if not math.isnan(z_val)
                            else float("nan"),
                            "Statistic_Type": "z (ML/Wald)",
                            "DF": float("nan"),
                            "p_value_raw": round(pval, 4)
                            if not math.isnan(pval)
                            else float("nan"),
                            "CI_95": ci_str,
                        }
                    )
                # Fix 2: multiplicity correction is applied across the full
                # family of structural paths in this model (not per-path in
                # isolation), and 'Significance' is derived from the
                # *adjusted* p-value -- raw p-values are retained alongside.
                struct_df = _finalize_structural_significance(
                    st_rows, config.multiplicity_correction or "holm"
                )

                if calc_stats is not None:
                    try:
                        stats_s = calc_stats(model_inst)
                        if isinstance(stats_s, pd.DataFrame) and not stats_s.empty:
                            stat_dict = stats_s.iloc[0].to_dict()
                        elif isinstance(stats_s, pd.Series):
                            stat_dict = stats_s.to_dict()
                        elif isinstance(stats_s, dict):
                            stat_dict = stats_s
                        else:
                            stat_dict = {}

                        c_val = _safe_float(stat_dict.get("CFI"))
                        t_val = _safe_float(stat_dict.get("TLI"))
                        r_val = _safe_float(stat_dict.get("RMSEA"))
                        chi2_v = _safe_float(stat_dict.get("chi2"))
                        dof_v = int(_safe_float(stat_dict.get("DoF"), -1))
                        p_v = _safe_float(
                            stat_dict.get("chi2 p-value", stat_dict.get("p-value"))
                        )
                        aic_v = _safe_float(stat_dict.get("AIC"))
                        bic_v = _safe_float(stat_dict.get("BIC"))

                        srmr_v = float("nan")
                        try:
                            sigma_mat = None
                            if hasattr(model_inst, "calc_sigma"):
                                # calc_sigma() returns (sigma_matrix, (loadings, factor_cov));
                                # only the first element is the implied covariance matrix.
                                sigma_mat = model_inst.calc_sigma()[0]
                            elif hasattr(model_inst, "implied_cov"):
                                sigma_mat = model_inst.implied_cov

                            if sigma_mat is not None and hasattr(model_inst, "vars"):
                                obs_vars = [
                                    v
                                    for v in model_inst.vars.get("observed", [])
                                    if v in sem_data.columns
                                ]
                                if len(obs_vars) >= 2:
                                    s_cov = sem_data[obs_vars].cov().to_numpy()
                                    srmr_v = compute_srmr(s_cov, np.asarray(sigma_mat))
                        except Exception:
                            srmr_v = float("nan")

                        conv_status, conv_notes = assess_sem_convergence(
                            struct_df, dof_v
                        )
                        fit_idx = SEMFitIndices(
                            chi2=chi2_v,
                            dof=dof_v,
                            p_val=p_v,
                            chi2_dof=chi2_v / max(1, dof_v)
                            if dof_v > 0 and not math.isnan(chi2_v)
                            else float("nan"),
                            cfi=c_val,
                            tli=t_val,
                            rmsea=r_val,
                            srmr=srmr_v,
                            aic=aic_v,
                            bic=bic_v,
                            verdict=(
                                "Good Fit (semopy)"
                                if (not math.isnan(c_val) and c_val >= 0.95)
                                else "Acceptable / Poor Fit"
                            )
                            if conv_status == "converged"
                            else f"UNSTABLE FIT ({conv_status}) -- interpret with caution",
                            convergence_status=conv_status,
                            convergence_notes=conv_notes,
                        )
                    except Exception as exc:
                        if logger:
                            logger.warning(
                                "[%s] Error calculating fit statistics: %s", target, exc
                            )
                        fit_idx = SEMFitIndices(
                            verdict="Fit computation error",
                            convergence_status="fit_error",
                            convergence_notes=str(exc)[:300],
                        )

                if config.mediators:
                    preds = [
                        p
                        for p in config.feature_columns
                        if p in available_cols and p != target
                    ]
                    meds = [
                        m
                        for m in config.mediators
                        if m in available_cols and m != target
                    ]
                    n_resamples = config.bootstrap_n if config.bootstrap else 2000
                    direct_p_adjusted_lookup = {
                        str(r["Predictor"]): float(r["p_value_adjusted"])
                        for _, r in struct_df[struct_df["Outcome"] == config.target].iterrows()
                        if np.isfinite(r["p_value_adjusted"])
                    }
                    med_df = compute_sem_mediation_with_bootstrap(
                        syntax,
                        sem_data,
                        target,
                        config.target,
                        preds,
                        meds,
                        rev_vmap,
                        params_df,
                        n_boot=n_resamples,
                        random_state=config.random_state,
                        fit_obj=fit_obj,
                        logger=logger,
                        multiplicity_correction=config.multiplicity_correction or "holm",
                        direct_p_adjusted_lookup=direct_p_adjusted_lookup,
                    )

                endog_vars = list(dict.fromkeys(struct_df["Outcome"].tolist()))
                r2_records = []
                for ev in endog_vars:
                    ev_sem = variable_map.get(ev, ev)
                    if ev_sem in sem_data.columns:
                        y_true = sem_data[ev_sem].to_numpy()
                        pred_paths = struct_df[struct_df["Outcome"] == ev]
                        pred_cols = [
                            variable_map.get(p, p)
                            for p in pred_paths["Predictor"]
                            if variable_map.get(p, p) in sem_data.columns
                        ]
                        betas = [float(b) for b in pred_paths["Beta_Std"]]
                        if pred_cols and len(pred_cols) == len(betas):
                            x_mat = sem_data[pred_cols].to_numpy()
                            y_hat = x_mat @ np.array(betas)
                            resid = y_true - y_hat
                            # Under FIML (or any other setting that preserves
                            # missing values), some rows may be NaN. Use only
                            # complete rows for this equation's R^2 rather
                            # than letting a single NaN silently propagate
                            # through np.var() and get masked to a false 0.0
                            # by an unconditional max(0.0, ...) floor below.
                            finite_mask = np.isfinite(resid) & np.isfinite(y_true)
                            n_valid = int(finite_mask.sum())
                            if n_valid >= 2 and np.var(y_true[finite_mask]) > 1e-12:
                                r2_val = float(
                                    1.0
                                    - np.var(resid[finite_mask])
                                    / np.var(y_true[finite_mask])
                                )
                                r2_val = max(0.0, r2_val)
                                r2_str = f"{r2_val:.1%}"
                            else:
                                r2_val = float("nan")
                                r2_str = "N/A"
                            r2_records.append(
                                {
                                    "Endogenous_Variable": ev,
                                    "R_Squared": round(r2_val, 4)
                                    if np.isfinite(r2_val)
                                    else float("nan"),
                                    "Variance_Explained_Pct": r2_str,
                                }
                            )
                r2_df = pd.DataFrame(r2_records, columns=R2_COLUMNS)
                fit_method = "semopy"
                if logger:
                    logger.info(
                        "[%s] Successfully converged exact SEM via semopy.", target
                    )
        except Exception as exc:
            if logger:
                logger.warning(
                    "[%s] semopy fit failed (%s); reverting to OLS path approximation.",
                    target,
                    exc,
                )
            fit_method = "ols_path_approximation"

    if fit_method == "ols_path_approximation" or struct_df.empty:
        struct_df, med_df, r2_df, _, params_df, fit_idx = compute_comprehensive_ols_sem(
            dataset_df, config, variable_map
        )
        fit_method = "ols_path_approximation"

    fit_table_df = fit_idx.to_dataframe()
    narrative_text = generate_sem_narrative(
        target,
        struct_df,
        med_df,
        r2_df,
        fit_idx,
        n_sample,
        fit_method,
        missing_data_method=missing_method,
    )

    struct_df.to_csv(target_sem_dir / "sem_structural_paths.csv", index=False)
    med_df.to_csv(target_sem_dir / "sem_mediation_effects.csv", index=False)
    r2_df.to_csv(target_sem_dir / "sem_variance_explained.csv", index=False)
    fit_table_df.to_csv(target_sem_dir / "sem_fit_indices.csv", index=False)
    (target_sem_dir / "sem_narrative_report.txt").write_text(
        narrative_text, encoding="utf-8"
    )

    return {
        "target": target,
        "status": "ok",
        "parameters": params_df,
        "structural_paths": struct_df,
        "mediation_effects": med_df,
        "variance_explained": r2_df,
        "fit_indices": fit_idx,
        "fit_table": fit_table_df,
        "effects": med_df,
        "fit_method": fit_method,
        "n": n_sample,
        "variable_map": variable_map,
        "narrative": narrative_text,
        "sem_data": sem_data,
        # Audit trail for the run manifest (Fix 5 / manifest requirements):
        # the missing-data method actually used (validated, never a
        # silent substitution), the multiplicity-correction method applied
        # to structural-path and mediation p-values, and the convergence
        # assessment for this specific fit.
        "missing_data_method_requested": config.missing_data_method,
        "missing_data_method_used": missing_method,
        "multiplicity_correction": config.multiplicity_correction or "holm",
        "convergence_status": fit_idx.convergence_status,
        "convergence_notes": fit_idx.convergence_notes,
    }
