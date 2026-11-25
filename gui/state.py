# gui/state.py
"""Central state repository with undoable in-memory DataFrame transformation tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, Signal

from config.schema import (
    CVConfig,
    DatasetConfig,
    ModelZooConfig,
    OutputConfig,
    PipelineConfig,
    ReportingConfig,
    SEMTargetConfig,
)
from core.data_loader import DatasetProfile, inspect_dataset


@dataclass
class AnalysisSession:
    original_df: pd.DataFrame | None = None
    dataset_df: pd.DataFrame | None = None
    dataset_profile: DatasetProfile | None = None
    transformation_log: list[str] = field(default_factory=list)
    active_columns: list[str] = field(default_factory=list)
    dataset_config: DatasetConfig | None = None
    model_config: ModelZooConfig | None = None
    cv_config: CVConfig | None = None
    sem_configs: dict[str, SEMTargetConfig] = field(default_factory=dict)
    sem_ui_state: dict[str, Any] = field(
        default_factory=lambda: {
            "enabled": True,
            "target": "",
            "predictors": [],
            "mediators": [],
            "covariates": [],
            "latent_defs": {},
            "estimator": "ML",
            "missing": "fiml",
            "bootstrap": True,
            "boot_iter": 2000,
            "standardize": True,
            "robust_se": False,
            "custom_syntax": False,
            "syntax": "",
        }
    )
    reporting_config: ReportingConfig | None = None
    output_config: OutputConfig | None = None
    pipeline_config: PipelineConfig | None = None
    per_model_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    undo_stack: list[pd.DataFrame] = field(default_factory=list)
    max_undo_depth: int = 15


class SessionStateBridge(QObject):
    dataset_df_updated = Signal(object)
    profile_updated = Signal(object)
    active_columns_updated = Signal(object)
    transformation_added = Signal(str)

    def __init__(self, session: AnalysisSession | None = None) -> None:
        super().__init__()
        self.session = session or AnalysisSession()

    def set_loaded_data(
        self, dataset_df: pd.DataFrame, profile: DatasetProfile
    ) -> None:
        self.session.original_df = dataset_df.copy()
        self.session.dataset_df = dataset_df.copy()
        self.session.dataset_profile = profile
        self.session.active_columns = list(dataset_df.columns)
        self.session.transformation_log.clear()
        self.session.undo_stack.clear()
        self.dataset_df_updated.emit(self.session.dataset_df)
        self.profile_updated.emit(self.session.dataset_profile)
        self.active_columns_updated.emit(self.session.active_columns)

    def update_dataset_df(self, dataset_df: pd.DataFrame, log_msg: str) -> None:
        if self.session.dataset_df is not None:
            if len(self.session.undo_stack) >= self.session.max_undo_depth:
                self.session.undo_stack.pop(0)
            self.session.undo_stack.append(self.session.dataset_df.copy())
            old_cols = list(self.session.dataset_df.columns)
        else:
            old_cols = []

        self.session.dataset_df = dataset_df.copy()
        self.session.dataset_profile = inspect_dataset(dataset_df)

        current_active = [
            c for c in self.session.active_columns if c in dataset_df.columns
        ]
        new_cols = [c for c in dataset_df.columns if c not in old_cols]
        for nc in new_cols:
            if nc not in current_active:
                current_active.append(nc)

        self.session.active_columns = (
            current_active if current_active else list(dataset_df.columns)
        )
        self.session.transformation_log.append(log_msg)

        self.dataset_df_updated.emit(self.session.dataset_df)
        self.profile_updated.emit(self.session.dataset_profile)
        self.active_columns_updated.emit(self.session.active_columns)
        self.transformation_added.emit(log_msg)

    def undo_last_transformation(self) -> bool:
        if not self.session.undo_stack:
            return False
        prev_df = self.session.undo_stack.pop()
        self.session.dataset_df = prev_df
        self.session.dataset_profile = inspect_dataset(prev_df)
        self.session.active_columns = [
            c for c in self.session.active_columns if c in prev_df.columns
        ] or list(prev_df.columns)
        msg = f"Undid last transformation (Reverted to {len(prev_df.columns)} columns)."
        self.session.transformation_log.append(msg)
        self.dataset_df_updated.emit(self.session.dataset_df)
        self.profile_updated.emit(self.session.dataset_profile)
        self.active_columns_updated.emit(self.session.active_columns)
        self.transformation_added.emit(msg)
        return True

    def reset_dataset_df(self) -> None:
        if self.session.original_df is not None:
            if self.session.dataset_df is not None:
                if len(self.session.undo_stack) >= self.session.max_undo_depth:
                    self.session.undo_stack.pop(0)
                self.session.undo_stack.append(self.session.dataset_df.copy())
            self.session.dataset_df = self.session.original_df.copy()
            self.session.dataset_profile = inspect_dataset(self.session.dataset_df)
            self.session.active_columns = list(self.session.dataset_df.columns)
            msg = "Reset in-memory dataset to original source state."
            self.session.transformation_log.append(msg)
            self.dataset_df_updated.emit(self.session.dataset_df)
            self.profile_updated.emit(self.session.dataset_profile)
            self.active_columns_updated.emit(self.session.active_columns)
            self.transformation_added.emit(msg)
