from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from config.exceptions import AmbiguousTaskTypeError
from config.schema import AnalysisTaskType

#: Numeric-target unique-value counts in this (inclusive) range cannot be
#: resolved to a task type from cardinality alone under AUTO: a variable
#: with, say, 4 unique numeric values could be a genuine 4-point ordinal
#: scale, a nominal group code, or a continuous measurement that happened
#: to be recorded/rounded to only 4 distinct values in this sample. See
#: `resolve_task_type_detailed` and `config.exceptions.AmbiguousTaskTypeError`
#: (Fix 3 in the METIS scientific remediation).
AMBIGUOUS_CARDINALITY_LOW = 3
AMBIGUOUS_CARDINALITY_HIGH = 10

ID_REGEX = re.compile(
    r"^(id|sub|subj|subject|patient|participant|record|sample|case)[_-]?(id)?$"
    r"|^[a-z0-9_-]*[_-](id|subject_id|participant_id|patient_id|sub_id|subj_id)$",
    re.IGNORECASE,
)


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    inferred_type: str  # 'numerical', 'binary_numeric', 'categorical', 'constant', 'id'
    missing_count: int
    missing_fraction: float
    unique_count: int
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    min: float | None = None
    max: float | None = None
    outlier_count_iqr: int = 0
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    total_rows: int
    total_columns: int
    columns: dict[str, ColumnProfile]
    numerical_columns: list[str]
    binary_numeric_columns: list[str]
    categorical_columns: list[str]
    constant_columns: list[str]
    suggested_id_columns: list[str]
    memory_usage_mb: float


def clean_dataframe(
    dataset_df: pd.DataFrame, logger: logging.Logger | None = None
) -> pd.DataFrame:
    """Sanitize tabular input, handling infinity, leading/trailing whitespace, and null tokens."""
    cleaned_df = dataset_df.copy()
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        cleaned_df[numeric_cols] = cleaned_df[numeric_cols].replace(
            [np.inf, -np.inf], np.nan
        )

    null_tokens = {
        "",
        "nan",
        "NaN",
        "NAN",
        "None",
        "none",
        "NONE",
        "NULL",
        "null",
        "Null",
        "NA",
        "na",
        "Na",
        "N/A",
        "n/a",
        "N/a",
        "-",
        "--",
        "---",
        ".",
        "?",
        "missing",
        "Missing",
    }

    for col in cleaned_df.select_dtypes(include=["object", "string"]).columns:
        series = cleaned_df[col]
        non_null_mask = series.notna()
        cleaned_series = series.copy()
        cleaned_series[non_null_mask] = (
            cleaned_series[non_null_mask].astype(str).str.strip()
        )
        cleaned_series = cleaned_series.replace(null_tokens, np.nan)
        try:
            converted = pd.to_numeric(cleaned_series, errors="coerce")
            if (
                converted.notna().sum() == cleaned_series.notna().sum()
                and converted.notna().sum() > 0
            ):
                cleaned_df[col] = converted
            else:
                cleaned_df[col] = cleaned_series
        except Exception:
            cleaned_df[col] = cleaned_series

    return cleaned_df


def inspect_dataset(df: pd.DataFrame) -> DatasetProfile:
    """Generate statistical summary and academically defensible schema inference."""
    total_rows, total_cols = df.shape
    columns_profile: dict[str, ColumnProfile] = {}
    numerical_cols: list[str] = []
    binary_numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    constant_cols: list[str] = []
    suggested_id_cols: list[str] = []

    mem_mb = (
        float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        if total_rows > 0 and total_cols > 0
        else 0.0
    )

    for col in df.columns:
        series = df[col]
        missing_cnt = int(series.isna().sum())
        missing_frac = float(missing_cnt / total_rows) if total_rows > 0 else 0.0
        unique_cnt = int(series.nunique(dropna=True))
        dtype_str = str(series.dtype)
        samples = series.dropna().head(5).tolist()

        c_mean, c_std, c_med, c_iqr, c_skew, c_kurt = None, None, None, None, None, None
        c_min, c_max = None, None
        outlier_cnt = 0

        if unique_cnt <= 1:
            inferred = "constant"
            constant_cols.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            valid_vals = (
                pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
            )
            if unique_cnt == 2 and set(np.unique(valid_vals)).issubset({0.0, 1.0}):
                inferred = "binary_numeric"
                binary_numeric_cols.append(col)
                numerical_cols.append(col)
                if len(valid_vals) > 0:
                    c_mean = float(np.mean(valid_vals))
                    c_std = (
                        float(np.std(valid_vals, ddof=1))
                        if len(valid_vals) > 1
                        else 0.0
                    )
                    c_med = float(np.median(valid_vals))
                    c_min = float(np.min(valid_vals))
                    c_max = float(np.max(valid_vals))
            elif (
                pd.api.types.is_integer_dtype(series)
                and unique_cnt == total_rows
                and total_rows > 1
                and bool(ID_REGEX.match(str(col).strip()))
            ):
                inferred = "id"
                suggested_id_cols.append(col)
            else:
                inferred = "numerical"
                numerical_cols.append(col)
                if len(valid_vals) > 0:
                    c_mean = float(np.mean(valid_vals))
                    c_std = (
                        float(np.std(valid_vals, ddof=1))
                        if len(valid_vals) > 1
                        else 0.0
                    )
                    c_med = float(np.median(valid_vals))
                    q25, q75 = np.percentile(valid_vals, [25, 75])
                    c_iqr = float(q75 - q25)
                    c_min = float(np.min(valid_vals))
                    c_max = float(np.max(valid_vals))
                    if len(valid_vals) > 3 and c_std > 1e-12:
                        sk_val = float(stats.skew(valid_vals))
                        kt_val = float(stats.kurtosis(valid_vals))
                        c_skew = sk_val if np.isfinite(sk_val) else None
                        c_kurt = kt_val if np.isfinite(kt_val) else None
                    if c_iqr > 0:
                        lower_bound = q25 - 1.5 * c_iqr
                        upper_bound = q75 + 1.5 * c_iqr
                        outlier_cnt = int(
                            np.sum(
                                (valid_vals < lower_bound) | (valid_vals > upper_bound)
                            )
                        )
        else:
            if bool(ID_REGEX.match(str(col).strip())) or (
                unique_cnt == total_rows and total_rows > 1
            ):
                inferred = "id"
                suggested_id_cols.append(col)
            else:
                inferred = "categorical"
                categorical_cols.append(col)

        columns_profile[col] = ColumnProfile(
            name=col,
            dtype=dtype_str,
            inferred_type=inferred,
            missing_count=missing_cnt,
            missing_fraction=missing_frac,
            unique_count=unique_cnt,
            mean=c_mean,
            std=c_std,
            median=c_med,
            iqr=c_iqr,
            skewness=c_skew,
            kurtosis=c_kurt,
            min=c_min,
            max=c_max,
            outlier_count_iqr=outlier_cnt,
            sample_values=samples,
        )

    return DatasetProfile(
        total_rows=total_rows,
        total_columns=total_cols,
        columns=columns_profile,
        numerical_columns=numerical_cols,
        binary_numeric_columns=binary_numeric_cols,
        categorical_columns=categorical_cols,
        constant_columns=constant_cols,
        suggested_id_columns=suggested_id_cols,
        memory_usage_mb=round(mem_mb, 2),
    )


def load_dataset(
    input_path: Path | str, logger: logging.Logger | None = None
) -> tuple[pd.DataFrame, DatasetProfile]:
    """Load tabular data into an inspected Pandas DataFrame."""
    path = Path(input_path)
    if not path.is_file():
        msg = f"Target data file not found at: {path}"
        raise FileNotFoundError(msg)

    if logger:
        logger.info("Loading dataset from %s", path)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in [".tsv", ".tab"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    df = clean_dataframe(df, logger)
    profile = inspect_dataset(df)

    if logger:
        logger.info(
            "Loaded %d rows x %d columns (%.2f MB).",
            profile.total_rows,
            profile.total_columns,
            profile.memory_usage_mb,
        )
    return df, profile


@dataclass
class TaskTypeResolution:
    """Full audit trail of a single task-type resolution decision.

    Distinguishes what was *requested* from what was *resolved*, and makes
    ambiguity an explicit, inspectable property rather than something that
    disappears the moment a bare `AnalysisTaskType` is returned. Intended to
    be recorded verbatim (via `to_dict`) into the run manifest so a reader
    never has to re-derive "why did METIS treat this column as regression?"
    from scratch.
    """

    column: str
    requested: AnalysisTaskType
    resolved: AnalysisTaskType
    unique_count: int
    is_ambiguous: bool
    reason: str
    unique_values_preview: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "requested_task_type": self.requested.value,
            "resolved_task_type": self.resolved.value,
            "unique_count": self.unique_count,
            "is_ambiguous": self.is_ambiguous,
            "reason": self.reason,
            "unique_values_preview": self.unique_values_preview,
        }


def resolve_task_type_detailed(
    series: pd.Series,
    requested: AnalysisTaskType = AnalysisTaskType.AUTO,
    column_name: str = "",
) -> TaskTypeResolution:
    """Infer task type (regression vs. classification) from target
    cardinality, returning full provenance rather than a bare enum.

    This function never raises: it *reports* ambiguity via
    `TaskTypeResolution.is_ambiguous` rather than deciding on the caller's
    behalf whether ambiguity is acceptable. Callers that need a "fail loudly
    unless explicitly confirmed" gate (e.g. `core.pipeline_runner`) should
    inspect `is_ambiguous` and consult their own configuration
    (`DatasetConfig.confirmed_ambiguous_targets`) before proceeding. The
    convenience wrapper `resolve_task_type` below always raises on
    unconfirmed ambiguity, for callers that have no such confirmation
    mechanism of their own.
    """
    valid = series.dropna()
    unique_cnt = int(valid.nunique())
    preview = sorted(valid.unique().tolist(), key=str)[:10] if unique_cnt else []

    if requested != AnalysisTaskType.AUTO:
        return TaskTypeResolution(
            column=column_name,
            requested=requested,
            resolved=requested,
            unique_count=unique_cnt,
            is_ambiguous=False,
            reason=(
                f"Explicit task_type='{requested.value}' was requested; "
                f"cardinality-based inference was not consulted."
            ),
            unique_values_preview=preview,
        )

    if not pd.api.types.is_numeric_dtype(valid):
        return TaskTypeResolution(
            column=column_name,
            requested=requested,
            resolved=AnalysisTaskType.CLASSIFICATION,
            unique_count=unique_cnt,
            is_ambiguous=False,
            reason="Non-numeric dtype resolves unambiguously to classification.",
            unique_values_preview=preview,
        )

    if unique_cnt <= 1:
        return TaskTypeResolution(
            column=column_name,
            requested=requested,
            resolved=AnalysisTaskType.REGRESSION,
            unique_count=unique_cnt,
            is_ambiguous=False,
            reason=(
                "Target has 0 or 1 unique value after dropping missing rows "
                "(degenerate/constant target); no meaningful model can be "
                "fit regardless of task type."
            ),
            unique_values_preview=preview,
        )

    if unique_cnt == 2:
        return TaskTypeResolution(
            column=column_name,
            requested=requested,
            resolved=AnalysisTaskType.CLASSIFICATION,
            unique_count=unique_cnt,
            is_ambiguous=False,
            reason="Exactly 2 unique numeric values resolves unambiguously to binary classification.",
            unique_values_preview=preview,
        )

    if unique_cnt > AMBIGUOUS_CARDINALITY_HIGH:
        return TaskTypeResolution(
            column=column_name,
            requested=requested,
            resolved=AnalysisTaskType.REGRESSION,
            unique_count=unique_cnt,
            is_ambiguous=False,
            reason=(
                f"{unique_cnt} unique numeric values (> {AMBIGUOUS_CARDINALITY_HIGH}) "
                f"resolves to regression under the AUTO heuristic."
            ),
            unique_values_preview=preview,
        )

    # AMBIGUOUS_CARDINALITY_LOW <= unique_cnt <= AMBIGUOUS_CARDINALITY_HIGH:
    # cardinality alone cannot distinguish a genuine ordinal/nominal
    # variable from a continuous measurement rounded to few distinct
    # values. METIS reports this as ambiguous rather than guessing.
    return TaskTypeResolution(
        column=column_name,
        requested=requested,
        resolved=AnalysisTaskType.REGRESSION,
        unique_count=unique_cnt,
        is_ambiguous=True,
        reason=(
            f"{unique_cnt} unique numeric values is in the scientifically "
            f"ambiguous range [{AMBIGUOUS_CARDINALITY_LOW}, "
            f"{AMBIGUOUS_CARDINALITY_HIGH}]: this could be an ordinal scale, "
            f"a nominal/grouping code, or a continuous variable rounded to "
            f"few distinct values. Cardinality alone does not determine "
            f"measurement scale. Default resolution if explicitly confirmed: "
            f"'regression' (unchanged from METIS's prior unconditional "
            f"default), but this now requires explicit acknowledgement."
        ),
        unique_values_preview=preview,
    )


def resolve_task_type(
    series: pd.Series,
    requested: AnalysisTaskType = AnalysisTaskType.AUTO,
    column_name: str = "",
    allow_ambiguous: bool = False,
) -> AnalysisTaskType:
    """Infer task type (regression vs. classification) from target cardinality.

    Thin convenience wrapper around `resolve_task_type_detailed` that raises
    `config.exceptions.AmbiguousTaskTypeError` when AUTO resolution lands in
    the scientifically ambiguous cardinality band (see
    AMBIGUOUS_CARDINALITY_LOW/HIGH), unless `allow_ambiguous=True` is passed
    explicitly by the caller (an opt-in escape hatch for callers, such as
    exploratory scripts or tests, that have already made a deliberate,
    informed choice to accept the default resolution).

    Callers that want to implement a proper per-target confirmation
    workflow against `DatasetConfig.confirmed_ambiguous_targets` (as
    `core.pipeline_runner.execute_pipeline` does) should call
    `resolve_task_type_detailed` directly instead, so they retain full
    control over how ambiguity is surfaced and confirmed.
    """
    detailed = resolve_task_type_detailed(series, requested, column_name=column_name)
    if detailed.is_ambiguous and not allow_ambiguous:
        raise AmbiguousTaskTypeError(
            column=column_name or "<unnamed column>",
            unique_count=detailed.unique_count,
            unique_values_preview=detailed.unique_values_preview,
            default_resolution=detailed.resolved.value,
        )
    return detailed.resolved
