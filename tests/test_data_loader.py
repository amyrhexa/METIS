# tests/test_data_loader.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.exceptions import AmbiguousTaskTypeError
from config.schema import AnalysisTaskType
from core.data_loader import (
    clean_dataframe,
    inspect_dataset,
    resolve_task_type,
    resolve_task_type_detailed,
)


def test_clean_dataframe_converts_null_tokens_and_inf_to_nan():
    df = pd.DataFrame(
        {
            "a": ["1", "2", "N/A", "--", "3"],
            "b": [1.0, np.inf, -np.inf, 4.0, 5.0],
            "c": [" x ", "none", "y", "NULL", " z "],
        }
    )
    cleaned = clean_dataframe(df)

    # "a" is fully numeric-coercible once null tokens are removed -> becomes numeric.
    assert pd.api.types.is_numeric_dtype(cleaned["a"])
    # "N/A" and "--" are both defined null tokens -> 2 missing values.
    assert cleaned["a"].isna().sum() == 2
    assert cleaned["a"].dropna().tolist() == [1.0, 2.0, 3.0]

    # +/-inf must never survive into a "clean" numeric column.
    assert not np.isinf(cleaned["b"].to_numpy()).any()
    assert cleaned["b"].isna().sum() == 2

    # Non-numeric text column: "none" and "NULL" are both null tokens -> 2 NaN,
    # surrounding whitespace stripped from the survivors.
    assert cleaned["c"].isna().sum() == 2
    assert cleaned["c"].dropna().tolist() == ["x", "y", "z"]


def test_clean_dataframe_does_not_mutate_input():
    df = pd.DataFrame({"a": ["1", "N/A", "3"]})
    original = df.copy()
    clean_dataframe(df)
    pd.testing.assert_frame_equal(df, original)


def test_inspect_dataset_classifies_column_types_correctly():
    n = 50
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "subject_id": [f"S{i:03d}" for i in range(n)],  # id
            "constant_col": [7] * n,  # constant
            "binary_flag": rng.integers(0, 2, size=n).astype(float),  # binary_numeric
            "continuous_val": rng.normal(size=n),  # numerical
            "group_label": rng.choice(["red", "green", "blue"], size=n),  # categorical
        }
    )
    profile = inspect_dataset(df)

    assert "subject_id" in profile.suggested_id_columns
    assert "constant_col" in profile.constant_columns
    assert "binary_flag" in profile.binary_numeric_columns
    assert "continuous_val" in profile.numerical_columns
    assert "group_label" in profile.categorical_columns
    assert profile.total_rows == n
    assert profile.total_columns == 5


def test_inspect_dataset_missingness_and_stats_are_correct():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan, 5.0]})
    profile = inspect_dataset(df)
    col = profile.columns["x"]
    assert col.missing_count == 1
    assert col.missing_fraction == pytest.approx(0.2)
    assert col.mean == pytest.approx(np.mean([1.0, 2.0, 3.0, 5.0]))
    assert col.min == 1.0
    assert col.max == 5.0


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.random.default_rng(1).normal(size=60), AnalysisTaskType.REGRESSION),
        ([0, 1] * 30, AnalysisTaskType.CLASSIFICATION),  # 2 unique values -> binary
        ([0] * 60, AnalysisTaskType.REGRESSION),  # degenerate/constant, not ambiguous
    ],
)
def test_resolve_task_type_auto_detection_unambiguous_cases(values, expected):
    """Cardinalities that resolve unambiguously (binary, constant, or clearly
    continuous >10-unique-value targets) must return directly without
    raising, exactly as before this remediation."""
    series = pd.Series(values)
    assert resolve_task_type(series, AnalysisTaskType.AUTO) == expected


def test_resolve_task_type_respects_explicit_request():
    series = pd.Series([0, 1, 0, 1, 1])
    # Even though this looks binary/classification-like, an explicit request wins,
    # and cardinality-based ambiguity detection is never consulted for it.
    assert (
        resolve_task_type(series, AnalysisTaskType.REGRESSION)
        == AnalysisTaskType.REGRESSION
    )
    detailed = resolve_task_type_detailed(series, AnalysisTaskType.REGRESSION)
    assert detailed.is_ambiguous is False
    assert "Explicit task_type" in detailed.reason


# --- Fix 3: ambiguous numeric-cardinality task-type resolution -------------
#
# A numeric target with ~3-10 unique values must never be silently resolved
# to regression under AUTO. `resolve_task_type` must raise
# `AmbiguousTaskTypeError`; `resolve_task_type_detailed` must never raise
# but must report `is_ambiguous=True` so a caller (e.g. the pipeline, which
# has access to `DatasetConfig.confirmed_ambiguous_targets`) can decide how
# to proceed and record that decision in the manifest.


@pytest.mark.parametrize("unique_count", [3, 5, 10])
def test_resolve_task_type_raises_for_ambiguous_cardinality(unique_count):
    series = pd.Series(list(range(unique_count)) * 10)
    with pytest.raises(AmbiguousTaskTypeError) as exc_info:
        resolve_task_type(series, AnalysisTaskType.AUTO, column_name="diagnosis_stage")
    assert "diagnosis_stage" in str(exc_info.value)
    assert str(unique_count) in str(exc_info.value)


@pytest.mark.parametrize("unique_count", [2, 11])
def test_resolve_task_type_does_not_raise_outside_ambiguous_band(unique_count):
    series = pd.Series(list(range(unique_count)) * 10)
    # Must not raise -- 2 is unambiguously binary classification, 11 is
    # unambiguously above the ambiguous band (treated as continuous).
    result = resolve_task_type(series, AnalysisTaskType.AUTO)
    expected = (
        AnalysisTaskType.CLASSIFICATION
        if unique_count == 2
        else AnalysisTaskType.REGRESSION
    )
    assert result == expected


def test_resolve_task_type_allow_ambiguous_escape_hatch():
    """An explicit, deliberate opt-in (allow_ambiguous=True) suppresses the
    raise and returns the same default resolution METIS used before this
    remediation -- but only when the caller asks for it by name."""
    series = pd.Series([0, 1, 2] * 20)
    result = resolve_task_type(series, AnalysisTaskType.AUTO, allow_ambiguous=True)
    assert result == AnalysisTaskType.REGRESSION


@pytest.mark.parametrize(
    ("unique_count", "expect_ambiguous"),
    [(2, False), (3, True), (5, True), (10, True), (11, False)],
)
def test_resolve_task_type_detailed_never_raises_and_flags_ambiguity(
    unique_count, expect_ambiguous
):
    series = pd.Series(list(range(unique_count)) * 10)
    detailed = resolve_task_type_detailed(
        series, AnalysisTaskType.AUTO, column_name="target"
    )
    assert detailed.is_ambiguous is expect_ambiguous
    assert detailed.unique_count == unique_count
    assert detailed.column == "target"
    assert detailed.requested == AnalysisTaskType.AUTO
    # Detailed resolution always still returns *some* resolved type, even
    # when ambiguous, so a caller with its own confirmation logic can use it.
    assert detailed.resolved in (
        AnalysisTaskType.REGRESSION,
        AnalysisTaskType.CLASSIFICATION,
    )


def test_resolve_task_type_detailed_to_dict_is_manifest_ready():
    series = pd.Series([0, 1, 2] * 20)
    detailed = resolve_task_type_detailed(
        series, AnalysisTaskType.AUTO, column_name="stage"
    )
    payload = detailed.to_dict()
    assert payload["column"] == "stage"
    assert payload["requested_task_type"] == "auto"
    assert payload["resolved_task_type"] == "regression"
    assert payload["is_ambiguous"] is True
    assert payload["unique_count"] == 3
    assert isinstance(payload["reason"], str) and len(payload["reason"]) > 0
