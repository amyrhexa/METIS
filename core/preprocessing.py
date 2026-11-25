from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from config.schema import PreprocessingConfig


class MissingnessFilter(BaseEstimator, TransformerMixin):
    """Prunes features exceeding a missingness threshold evaluated strictly on training folds."""

    def __init__(self, threshold: float = 0.25) -> None:
        self.threshold = threshold
        self.columns_to_keep_: list[str] = []
        self.indices_to_keep_: np.ndarray = np.array([], dtype=int)
        self.feature_names_in_: np.ndarray = np.array([], dtype=str)
        self.n_features_in_: int = 0

    def fit(self, X: Any, y: Any = None) -> MissingnessFilter:
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.asarray(X.columns, dtype=str)
            missing_frac = X.isna().mean().fillna(0.0)
            # pandas 3.x may return a read-only array from .to_numpy(); copy so
            # the fallback assignment below doesn't crash when every column in
            # this fold exceeds the missingness threshold.
            keep_mask = (missing_frac <= self.threshold).to_numpy().copy()
            if not np.any(keep_mask) and len(X.columns) > 0:
                keep_mask[int(missing_frac.to_numpy().argmin())] = True
            self.indices_to_keep_ = np.where(keep_mask)[0]
            self.columns_to_keep_ = [str(X.columns[i]) for i in self.indices_to_keep_]
        else:
            arr = np.asarray(X)
            self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
            if arr.ndim > 1 and arr.shape[0] > 0:
                missing_frac = np.isnan(arr).mean(axis=0)
                keep_mask = missing_frac <= self.threshold
                if not np.any(keep_mask) and arr.shape[1] > 0:
                    keep_mask[int(np.nanargmin(missing_frac))] = True
                self.indices_to_keep_ = np.where(keep_mask)[0]
            else:
                self.indices_to_keep_ = np.arange(self.n_features_in_)
            self.columns_to_keep_ = [f"x{i}" for i in self.indices_to_keep_]
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=str
            )
        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            valid = [c for c in self.columns_to_keep_ if c in X.columns]
            return X[valid]
        arr = np.asarray(X)
        if (
            len(self.indices_to_keep_) > 0
            and arr.ndim > 1
            and arr.shape[1] >= len(self.indices_to_keep_)
        ):
            return arr[:, self.indices_to_keep_]
        return arr

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        if input_features is not None:
            inp = [str(f) for f in input_features]
            if len(self.indices_to_keep_) > 0:
                valid_idx = [i for i in self.indices_to_keep_ if i < len(inp)]
                if valid_idx:
                    return np.asarray([inp[i] for i in valid_idx], dtype=str)
            matched = [c for c in self.columns_to_keep_ if c in inp]
            if matched:
                return np.asarray(matched, dtype=str)
        return np.asarray(self.columns_to_keep_, dtype=str)


class ConstantColumnFilter(BaseEstimator, TransformerMixin):
    """Drops columns that are constant (<=1 unique value, NaN counted as its
    own value) as observed strictly within the data passed to `.fit()`.

    This is deliberately fold-local (Fix 12 in the METIS scientific
    remediation): a column that is constant across the *entire* dataset is
    trivially constant within every fold too, so dropping it structurally
    before cross-validation is harmless. But the converse is not
    guaranteed — a column with variance in the full dataset could, in
    principle, be constant within a specific training fold's subset — and
    the previous implementation decided constancy once, on the full
    pre-split feature matrix, rather than per fold. Moving the decision
    into a transformer that only ever sees `.fit()`-time data (i.e., data
    already isolated to one fold's training partition when used inside a
    per-fold `Pipeline`/`clone()`) closes that gap and matches the same
    "fit only inside the fold" guarantee already provided by
    `MissingnessFilter`.

    Mirrors `MissingnessFilter`'s safety net: if every column in the fitted
    data is constant, none are dropped (a preprocessor must never be left
    with zero columns).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.columns_to_keep_: list[str] = []
        self.indices_to_keep_: np.ndarray = np.array([], dtype=int)
        self.feature_names_in_: np.ndarray = np.array([], dtype=str)
        self.n_features_in_: int = 0

    def fit(self, X: Any, y: Any = None) -> ConstantColumnFilter:
        if isinstance(X, pd.DataFrame):
            self.n_features_in_ = X.shape[1]
            self.feature_names_in_ = np.asarray(X.columns, dtype=str)
            if not self.enabled or X.shape[1] <= 1:
                self.indices_to_keep_ = np.arange(X.shape[1])
            else:
                nunique = X.nunique(dropna=False).to_numpy()
                keep_mask = nunique > 1
                if not np.any(keep_mask):
                    keep_mask[:] = True
                self.indices_to_keep_ = np.where(keep_mask)[0]
            self.columns_to_keep_ = [str(X.columns[i]) for i in self.indices_to_keep_]
        else:
            arr = np.asarray(X)
            self.n_features_in_ = arr.shape[1] if arr.ndim > 1 else 1
            if not self.enabled or arr.ndim <= 1 or arr.shape[1] <= 1:
                self.indices_to_keep_ = np.arange(self.n_features_in_)
            else:
                keep_mask = np.array(
                    [
                        pd.Series(arr[:, i]).nunique(dropna=False) > 1
                        for i in range(arr.shape[1])
                    ]
                )
                if not np.any(keep_mask):
                    keep_mask[:] = True
                self.indices_to_keep_ = np.where(keep_mask)[0]
            self.columns_to_keep_ = [f"x{i}" for i in self.indices_to_keep_]
            self.feature_names_in_ = np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)], dtype=str
            )
        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            valid = [c for c in self.columns_to_keep_ if c in X.columns]
            return X[valid]
        arr = np.asarray(X)
        if (
            len(self.indices_to_keep_) > 0
            and arr.ndim > 1
            and arr.shape[1] >= len(self.indices_to_keep_)
        ):
            return arr[:, self.indices_to_keep_]
        return arr

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> np.ndarray:
        if input_features is not None:
            inp = [str(f) for f in input_features]
            if len(self.indices_to_keep_) > 0:
                valid_idx = [i for i in self.indices_to_keep_ if i < len(inp)]
                if valid_idx:
                    return np.asarray([inp[i] for i in valid_idx], dtype=str)
            matched = [c for c in self.columns_to_keep_ if c in inp]
            if matched:
                return np.asarray(matched, dtype=str)
        return np.asarray(self.columns_to_keep_, dtype=str)


def build_preprocessor(
    X: pd.DataFrame,
    categorical_cols: list[str],
    config: PreprocessingConfig | None = None,
) -> ColumnTransformer:
    """Build a leakage-safe ColumnTransformer for numeric and categorical pipelines.

    Categorical one-hot encoding uses drop='if_binary' to ensure that two-category
    features yield a single dummy, preventing redundant representation and collinearity.

    Column-existence routing (which columns go to the numeric branch vs. the
    categorical branch) is structural information derived from dtype/column
    naming and is decided here, at construction time, from the full feature
    matrix `X` — this is safe because it never depends on a fitted
    statistic (see `ConstantColumnFilter`'s docstring for why constant-
    column *dropping* specifically was moved inside the fold-local
    transformer chain instead of being decided here).
    """
    cfg = config or PreprocessingConfig()
    cat_set = set(categorical_cols)
    numeric_cols = [c for c in X.columns if c not in cat_set]
    active_cat_cols = [c for c in categorical_cols if c in X.columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_cols:
        scaler = StandardScaler()
        if cfg.scaler_type == "robust":
            scaler = RobustScaler()
        elif cfg.scaler_type == "minmax":
            scaler = MinMaxScaler()

        num_steps: list[tuple[str, Any]] = [
            (
                "constant_filter",
                ConstantColumnFilter(enabled=cfg.drop_constant_columns),
            ),
            (
                "missingness_filter",
                MissingnessFilter(threshold=cfg.missingness_threshold),
            ),
            ("imputer", SimpleImputer(strategy=cfg.numeric_imputer_strategy)),
        ]
        if cfg.scale_numeric:
            num_steps.append(("scaler", scaler))

        transformers.append(("num", Pipeline(num_steps), numeric_cols))

    if active_cat_cols:
        cat_steps: list[tuple[str, Any]] = [
            (
                "constant_filter",
                ConstantColumnFilter(enabled=cfg.drop_constant_columns),
            ),
            (
                "missingness_filter",
                MissingnessFilter(threshold=cfg.missingness_threshold),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy=cfg.categorical_imputer_strategy, fill_value="missing"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown=cast(Any, cfg.handle_unknown_categories),
                    sparse_output=False,
                ),
            ),
        ]
        transformers.append(("cat", Pipeline(cat_steps), active_cat_cols))

    if not transformers:
        msg = "Cannot construct preprocessor: No input feature columns available."
        raise ValueError(msg)

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
