# tests/test_preprocessing.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.schema import PreprocessingConfig
from core.preprocessing import (
    ConstantColumnFilter,
    MissingnessFilter,
    build_preprocessor,
)


def test_missingness_filter_drops_columns_above_threshold_using_only_fit_data():
    # 3 columns: 0%, 30%, 60% missing.
    n = 100
    df = pd.DataFrame(
        {
            "keep_always": np.arange(n, dtype=float),
            "borderline": [np.nan] * 30 + list(range(70)),
            "mostly_missing": [np.nan] * 60 + list(range(40)),
        }
    )
    f = MissingnessFilter(threshold=0.25)
    f.fit(df)
    assert set(f.columns_to_keep_) == {"keep_always"}

    out = f.transform(df)
    assert list(out.columns) == ["keep_always"]


def test_missingness_filter_never_drops_every_column():
    # All columns exceed threshold -> must keep at least the least-missing one.
    df = pd.DataFrame(
        {
            "a": [np.nan] * 80 + list(range(20)),
            "b": [np.nan] * 90 + list(range(10)),
        }
    )
    f = MissingnessFilter(threshold=0.1)
    f.fit(df)
    assert len(f.columns_to_keep_) >= 1
    assert "a" in f.columns_to_keep_  # the less-missing column must survive


def test_missingness_filter_fit_is_scoped_to_the_data_it_sees():
    """The transformer must decide which columns to keep using ONLY the data
    passed to .fit() -- this is what makes it safe to use inside a per-fold
    Pipeline. A column that is fine in the full dataset but >threshold missing
    within a specific training fold must be dropped when fit on that fold,
    even though fitting on the full dataset would have kept it."""
    n = 100
    col = np.arange(n, dtype=float)
    col_with_gap = col.copy()
    # Missingness concentrated in what will become the "test" half only.
    col_with_gap[50:] = np.nan  # 50% missing in the second half
    df_full = pd.DataFrame({"x": col, "y": col_with_gap})

    train_df = df_full.iloc[:50]  # "y" has 0% missing here
    test_like_df = df_full.iloc[50:]  # "y" has 100% missing here

    f_train = MissingnessFilter(threshold=0.25)
    f_train.fit(train_df)
    assert "y" in f_train.columns_to_keep_  # fine within this fold's train split

    f_full = MissingnessFilter(threshold=0.25)
    f_full.fit(df_full)
    assert "y" not in f_full.columns_to_keep_  # 50% missing overall -> dropped

    # transform() must use whatever was decided at fit time, not recompute.
    out = f_train.transform(test_like_df)
    assert "y" in out.columns


def test_constant_column_filter_drops_columns_constant_within_fit_data():
    df = pd.DataFrame(
        {
            "varies": np.random.default_rng(0).normal(size=20),
            "constant": [1.0] * 20,
        }
    )
    f = ConstantColumnFilter(enabled=True)
    f.fit(df)
    assert "constant" not in f.columns_to_keep_
    assert "varies" in f.columns_to_keep_


def test_constant_column_filter_is_fold_local_not_global():
    """A column constant in one fold's training partition but *not* in the
    full dataset must be dropped when fit on that fold's data alone -- this
    is what makes constant-column detection safe to use inside a per-fold
    Pipeline (Fix 12), mirroring the equivalent MissingnessFilter guarantee."""
    n = 100
    varying_globally = np.concatenate([np.zeros(50), np.arange(1, 51, dtype=float)])
    # A second, always-varying column so the "never drop every column" safety
    # net doesn't mask the fold-local constancy of "x" being asserted here.
    df_full = pd.DataFrame(
        {"x": varying_globally, "other": np.arange(n, dtype=float)}
    )
    fold_train = df_full.iloc[:50]  # "x" is all zeros -> constant within this fold only

    f_fold = ConstantColumnFilter(enabled=True)
    f_fold.fit(fold_train)
    assert "x" not in f_fold.columns_to_keep_  # constant within this fold's train data

    f_full = ConstantColumnFilter(enabled=True)
    f_full.fit(df_full)
    assert "x" in f_full.columns_to_keep_  # varies across the full dataset


def test_constant_column_filter_never_drops_every_column():
    df = pd.DataFrame({"only_col": [1, 1, 1, 1]})
    f = ConstantColumnFilter(enabled=True)
    f.fit(df)
    assert "only_col" in f.columns_to_keep_


def test_build_preprocessor_drops_constant_columns_after_fit():
    df = pd.DataFrame(
        {
            "varies": np.random.default_rng(0).normal(size=20),
            "constant": [1.0] * 20,
        }
    )
    ct = build_preprocessor(df, categorical_cols=[], config=PreprocessingConfig())
    # Structural routing includes every numeric column -- the constancy
    # decision is made by the fold-local ConstantColumnFilter at fit time,
    # not by build_preprocessor before any data has been split.
    num_cols = next(t[2] for t in ct.transformers if t[0] == "num")
    assert set(num_cols) == {"varies", "constant"}

    ct.fit(df)
    out_names = list(ct.get_feature_names_out())
    assert "constant" not in out_names
    assert "varies" in out_names


def test_build_preprocessor_keeps_constant_columns_when_disabled():
    df = pd.DataFrame({"varies": range(20), "constant": [1] * 20})
    cfg = PreprocessingConfig(drop_constant_columns=False)
    ct = build_preprocessor(df, categorical_cols=[], config=cfg)
    ct.fit(df)
    out_names = list(ct.get_feature_names_out())
    assert "constant" in out_names


def test_build_preprocessor_onehot_drops_single_dummy_for_binary_category():
    n = 40
    df = pd.DataFrame(
        {
            "num_feat": np.random.default_rng(0).normal(size=n),
            "sex": np.random.default_rng(0).choice(["M", "F"], size=n),
        }
    )
    ct = build_preprocessor(df, categorical_cols=["sex"], config=PreprocessingConfig())
    ct.fit(df)
    out_names = list(ct.get_feature_names_out())
    # drop='if_binary' -> exactly one dummy column for a 2-category feature.
    sex_cols = [c for c in out_names if c.startswith("sex")]
    assert len(sex_cols) == 1


def test_build_preprocessor_raises_when_no_columns_available():
    df = pd.DataFrame(index=range(5))  # zero columns
    with pytest.raises(ValueError):
        build_preprocessor(df, categorical_cols=[], config=PreprocessingConfig())


def test_build_preprocessor_keeps_a_constant_column_if_it_is_the_only_one():
    """The constant-column guard must never empty the transformer list entirely
    -- if every column is constant, build_preprocessor should still produce a
    usable (if uninformative) preprocessor rather than raising."""
    df = pd.DataFrame({"only_col": [1, 1, 1, 1]})
    ct = build_preprocessor(df, categorical_cols=[], config=PreprocessingConfig())
    num_cols = next(t[2] for t in ct.transformers if t[0] == "num")
    assert "only_col" in num_cols
    ct.fit(df)
    assert "only_col" in list(ct.get_feature_names_out())
