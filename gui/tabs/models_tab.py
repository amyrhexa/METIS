# gui/tabs/models_tab.py
"""Tab 3: Machine Learning Tournament Matrix Partitioning, Model Zoo, and Curated Hyperparameters."""

from __future__ import annotations

import contextlib
import os
from typing import Any

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.schema import (
    ALLOWED_CLASSIFICATION_SCORING,
    ALLOWED_MULTIPLICITY_METHODS,
    AnalysisTaskType,
    CVConfig,
    DatasetConfig,
    ModelZooConfig,
    PipelineConfig,
)
from core.modeling import (
    HAS_CATBOOST,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    get_model_zoo,
)
from gui.state import SessionStateBridge
from gui.theme import NORD_PALETTE
from gui.widgets import ColumnSearchSelector


class ModelsTab(QWidget):
    def __init__(
        self, state_bridge: SessionStateBridge, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.state_bridge = state_bridge
        self._is_syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        part_box = QGroupBox("Target (Y) & Predictor Feature (X) Selection")
        part_layout = QVBoxLayout(part_box)

        self.target_selector = ColumnSearchSelector(
            title="Target Outcomes (Y)", badge_color=NORD_PALETTE.NORD13
        )
        self.target_selector.selection_changed.connect(self._on_targets_changed)
        part_layout.addWidget(self.target_selector)

        self.cardinality_notice = QLabel("")
        self.cardinality_notice.setStyleSheet(
            f"color: {NORD_PALETTE.NORD13}; font-size: 11px; font-weight: bold;"
        )
        self.cardinality_notice.setWordWrap(True)
        self.cardinality_notice.hide()
        part_layout.addWidget(self.cardinality_notice)

        self.feature_selector = ColumnSearchSelector(
            title="Predictor Matrix (X)", badge_color=NORD_PALETTE.NORD7
        )
        self.feature_selector.selection_changed.connect(self._on_features_changed)
        part_layout.addWidget(self.feature_selector)

        splitter.addWidget(part_box)

        zoo_box = QGroupBox("Algorithm Zoo & Per-Model Parameter Overrides")
        zoo_layout = QVBoxLayout(zoo_box)

        self.use_all_models_cb = QCheckBox("Run Complete Model Zoo Tournament")
        self.use_all_models_cb.toggled.connect(self._toggle_all_models)
        zoo_layout.addWidget(self.use_all_models_cb)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(
            ["Enabled", "Model", "Complexity", "Interpretability"]
        )
        self.model_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.model_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.model_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.model_table.itemSelectionChanged.connect(self._on_model_table_selection)
        zoo_layout.addWidget(self.model_table)

        param_group = QGroupBox("Curated Hyperparameter Settings for Selected Model")
        param_layout = QVBoxLayout(param_group)
        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_container = QWidget()
        self.param_form = QFormLayout(self.param_container)
        self.param_scroll.setWidget(self.param_container)
        param_layout.addWidget(self.param_scroll)
        zoo_layout.addWidget(param_group)

        stack_box = QGroupBox("Stacking Meta-Learner")
        stack_form = QFormLayout(stack_box)
        self.enable_stacking_cb = QCheckBox("Build Stacking Super-Learner Ensemble")
        self.enable_stacking_cb.setChecked(True)
        self.stacking_k_spin = QSpinBox()
        self.stacking_k_spin.setRange(2, 8)
        self.stacking_k_spin.setValue(3)
        stack_form.addRow("", self.enable_stacking_cb)
        stack_form.addRow("Base Learners (Top K):", self.stacking_k_spin)
        zoo_layout.addWidget(stack_box)
        splitter.addWidget(zoo_box)

        cv_box = QGroupBox("Nested CV Architecture")
        cv_form = QFormLayout(cv_box)

        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(
            [
                AnalysisTaskType.REGRESSION.value,
                AnalysisTaskType.CLASSIFICATION.value,
            ]
        )
        self.task_type_combo.currentTextChanged.connect(self._on_task_type_changed)
        cv_form.addRow("Task Type:", self.task_type_combo)

        self.outer_folds_spin = QSpinBox()
        self.outer_folds_spin.setRange(2, 20)
        self.outer_folds_spin.setValue(5)
        cv_form.addRow("Outer CV Folds:", self.outer_folds_spin)

        self.inner_folds_spin = QSpinBox()
        self.inner_folds_spin.setRange(2, 20)
        self.inner_folds_spin.setValue(3)
        cv_form.addRow("Inner CV Folds:", self.inner_folds_spin)

        self.n_iter_spin = QSpinBox()
        self.n_iter_spin.setRange(1, 500)
        self.n_iter_spin.setValue(25)
        cv_form.addRow("Search Budget (n_iter):", self.n_iter_spin)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(-1, max(1, os.cpu_count() or 1))
        self.n_jobs_spin.setValue(-1)
        cv_form.addRow("CPU Workers (n_jobs):", self.n_jobs_spin)

        self.parallel_outer_folds_cb = QCheckBox(
            "Run outer folds in parallel (uses n_jobs workers)"
        )
        self.parallel_outer_folds_cb.setToolTip(
            "Off by default: outer folds run sequentially, single-threaded, "
            "for deterministic execution. When enabled, outer folds run "
            "concurrently via n_jobs worker threads -- each fold is an "
            "independent computation, so this changes wall-clock time only, "
            "never results."
        )
        cv_form.addRow("", self.parallel_outer_folds_cb)

        self.selection_folds_spin = QSpinBox()
        self.selection_folds_spin.setRange(2, 20)
        self.selection_folds_spin.setValue(3)
        self.selection_folds_spin.setToolTip(
            "Folds used, within each outer fold's training partition, to "
            "compare candidate models before the winner is tuned and "
            "evaluated on that fold's held-out test partition (confirmatory "
            "model-selection procedure)."
        )
        cv_form.addRow("Model-Selection Folds:", self.selection_folds_spin)

        self.classification_scoring_combo = QComboBox()
        self.classification_scoring_combo.addItems(
            sorted(ALLOWED_CLASSIFICATION_SCORING)
        )
        self.classification_scoring_combo.setCurrentText("auto")
        self.classification_scoring_combo.setToolTip(
            "'auto' switches from accuracy to balanced_accuracy for "
            "hyperparameter search when a fold's training class balance "
            "exceeds the imbalance ratio threshold."
        )
        cv_form.addRow(
            "Classification Scoring:", self.classification_scoring_combo
        )

        self.imbalance_ratio_spin = QDoubleSpinBox()
        self.imbalance_ratio_spin.setRange(1.01, 100.0)
        self.imbalance_ratio_spin.setSingleStep(0.1)
        self.imbalance_ratio_spin.setValue(1.5)
        cv_form.addRow("Imbalance Ratio Threshold:", self.imbalance_ratio_spin)

        self.multiplicity_correction_combo = QComboBox()
        self.multiplicity_correction_combo.addItems(
            sorted(ALLOWED_MULTIPLICITY_METHODS)
        )
        self.multiplicity_correction_combo.setCurrentText("holm")
        self.multiplicity_correction_combo.setToolTip(
            "Multiple-comparison correction applied to pairwise model-vs-best "
            "p-values (and, separately, to SEM structural-path/mediation "
            "p-values)."
        )
        cv_form.addRow(
            "Multiplicity Correction:", self.multiplicity_correction_combo
        )

        self.bootstrap_spin = QSpinBox()
        self.bootstrap_spin.setRange(100, 10000)
        self.bootstrap_spin.setValue(1000)
        cv_form.addRow("Bootstrap Resamples:", self.bootstrap_spin)

        self.perm_repeats_spin = QSpinBox()
        self.perm_repeats_spin.setRange(5, 100)
        self.perm_repeats_spin.setValue(20)
        cv_form.addRow("Permutation Repeats:", self.perm_repeats_spin)

        self.use_group_cv_cb = QCheckBox("Enforce Group-Aware Splitting")
        self.group_id_combo = QComboBox()
        self.group_id_combo.addItem("None")
        cv_form.addRow("", self.use_group_cv_cb)
        cv_form.addRow("Cluster Identifier:", self.group_id_combo)

        self.skip_shap_cb = QCheckBox("Skip SHAP (Use Fast Permutation)")
        self.fast_mode_cb = QCheckBox("Fast Mode")
        cv_form.addRow("", self.skip_shap_cb)
        cv_form.addRow("", self.fast_mode_cb)

        splitter.addWidget(cv_box)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        layout.addWidget(splitter)

        self.per_model_widgets: dict[str, dict[str, Any]] = {}
        self._populate_zoo_table()

    def _on_task_type_changed(self, task_val: str) -> None:
        self._populate_zoo_table()

    def set_interactive(self, enabled: bool) -> None:
        self.target_selector.setEnabled(enabled)
        self.feature_selector.setEnabled(enabled)
        self.use_all_models_cb.setEnabled(enabled)
        self.model_table.setEnabled(enabled and not self.use_all_models_cb.isChecked())
        self.param_scroll.setEnabled(enabled)
        self.enable_stacking_cb.setEnabled(enabled)
        self.stacking_k_spin.setEnabled(enabled)
        self.task_type_combo.setEnabled(enabled)
        self.outer_folds_spin.setEnabled(enabled)
        self.inner_folds_spin.setEnabled(enabled)
        self.n_iter_spin.setEnabled(enabled)
        self.n_jobs_spin.setEnabled(enabled)
        self.bootstrap_spin.setEnabled(enabled)
        self.perm_repeats_spin.setEnabled(enabled)
        self.use_group_cv_cb.setEnabled(enabled)
        self.group_id_combo.setEnabled(enabled)
        self.skip_shap_cb.setEnabled(enabled)
        self.fast_mode_cb.setEnabled(enabled)

    def sync_columns(self, columns: list[str]) -> None:
        if self._is_syncing:
            return
        self._is_syncing = True

        dataset_df = self.state_bridge.session.dataset_df
        all_cols = list(dataset_df.columns) if dataset_df is not None else list(columns)
        active_cols = [c for c in columns if c in all_cols] or all_cols

        dtypes = (
            {c: str(dataset_df[c].dtype) for c in all_cols}
            if dataset_df is not None
            else {}
        )

        prev_targets = self.target_selector.get_selected_columns()
        prev_features = self.feature_selector.get_selected_columns()

        new_targets = [t for t in prev_targets if t in active_cols]

        if prev_features:
            new_features = [f for f in prev_features if f in active_cols]
            old_set = set(prev_features) | set(prev_targets)
            for c in active_cols:
                if c not in old_set and c not in new_targets:
                    new_features.append(c)
        else:
            new_features = [c for c in active_cols if c not in new_targets]

        self.target_selector.set_columns(
            active_cols, selected=new_targets, dtypes=dtypes
        )
        self.feature_selector.set_columns(
            active_cols, selected=new_features, dtypes=dtypes
        )
        self.feature_selector.set_excluded_columns(new_targets)

        prev_group = self.group_id_combo.currentText()
        self.group_id_combo.clear()
        self.group_id_combo.addItem("None")
        for c in active_cols:
            self.group_id_combo.addItem(c)
        if prev_group in active_cols:
            self.group_id_combo.setCurrentText(prev_group)

        self._is_syncing = False

    def set_from_config(self, config: PipelineConfig) -> None:
        """Restores complete model tournament and CV configuration from a PipelineConfig."""
        self._is_syncing = True
        dataset_df = self.state_bridge.session.dataset_df
        all_cols = list(dataset_df.columns) if dataset_df is not None else []
        active_cols = (
            [c for c in config.dataset.active_columns if c in all_cols]
            if all_cols
            else config.dataset.active_columns
        ) or all_cols

        dtypes = (
            {c: str(dataset_df[c].dtype) for c in all_cols}
            if dataset_df is not None
            else {}
        )

        targets = [
            t for t in config.dataset.target_columns if t in active_cols
        ] or config.dataset.target_columns
        features = [
            f for f in config.dataset.feature_columns if f in active_cols
        ] or config.dataset.feature_columns

        self.target_selector.set_columns(active_cols, selected=targets, dtypes=dtypes)
        self.feature_selector.set_columns(active_cols, selected=features, dtypes=dtypes)
        self.feature_selector.set_excluded_columns(targets)

        raw_task = (
            config.dataset.task_type.value
            if hasattr(config.dataset.task_type, "value")
            else str(config.dataset.task_type)
        )
        t_idx = self.task_type_combo.findText(raw_task)
        if t_idx >= 0:
            self.task_type_combo.setCurrentIndex(t_idx)

        self.outer_folds_spin.setValue(config.cv.outer_folds)
        self.inner_folds_spin.setValue(config.cv.inner_folds)
        self.n_iter_spin.setValue(config.cv.n_iter)
        self.n_jobs_spin.setValue(config.cv.n_jobs)
        self.parallel_outer_folds_cb.setChecked(config.cv.parallel_outer_folds)
        self.selection_folds_spin.setValue(config.cv.selection_folds)
        idx = self.classification_scoring_combo.findText(
            config.cv.classification_scoring
        )
        if idx >= 0:
            self.classification_scoring_combo.setCurrentIndex(idx)
        self.imbalance_ratio_spin.setValue(config.cv.imbalance_ratio_threshold)
        idx = self.multiplicity_correction_combo.findText(
            config.cv.multiplicity_correction
        )
        if idx >= 0:
            self.multiplicity_correction_combo.setCurrentIndex(idx)
        self.bootstrap_spin.setValue(config.cv.bootstrap_n)
        self.perm_repeats_spin.setValue(config.cv.permutation_repeats)
        self.use_group_cv_cb.setChecked(config.cv.use_group_cv)

        self.group_id_combo.clear()
        self.group_id_combo.addItem("None")
        for c in active_cols:
            self.group_id_combo.addItem(c)
        if config.dataset.id_column:
            self.group_id_combo.setCurrentText(config.dataset.id_column)

        self.skip_shap_cb.setChecked(config.cv.skip_shap)
        self.fast_mode_cb.setChecked(config.cv.fast_mode)

        self.use_all_models_cb.setChecked(config.models.use_all_models)
        self.enable_stacking_cb.setChecked(config.models.enable_stacking)
        self.stacking_k_spin.setValue(config.models.stacking_top_k)

        selected_set = set(config.models.selected_models)
        for r in range(self.model_table.rowCount()):
            item = self.model_table.item(r, 1)
            if item is not None:
                k = item.data(Qt.ItemDataRole.UserRole)
                chk = self.model_table.item(r, 0)
                if chk and bool(chk.flags() & Qt.ItemFlag.ItemIsUserCheckable):
                    chk.setCheckState(
                        Qt.CheckState.Checked
                        if k in selected_set
                        else Qt.CheckState.Unchecked
                    )

        self.state_bridge.session.per_model_overrides = dict(
            config.models.per_model_overrides
        )
        self._on_model_table_selection()
        self._is_syncing = False

    def _populate_zoo_table(self) -> None:
        task = AnalysisTaskType(self.task_type_combo.currentText())
        zoo = get_model_zoo(task, include_full=True)
        all_keys = list(zoo.keys())

        prev_checked = set()
        for r in range(self.model_table.rowCount()):
            chk = self.model_table.item(r, 0)
            item = self.model_table.item(r, 1)
            if chk and item and chk.checkState() == Qt.CheckState.Checked:
                prev_checked.add(item.data(Qt.ItemDataRole.UserRole))

        self.model_table.setRowCount(len(all_keys))
        for r_idx, k in enumerate(all_keys):
            spec = zoo[k]
            chk = QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )

            is_avail = True
            tooltip = ""
            if k == "xgboost" and not HAS_XGBOOST:
                is_avail = False
                tooltip = "xgboost not installed"
            elif k == "lightgbm" and not HAS_LIGHTGBM:
                is_avail = False
                tooltip = "lightgbm not installed"
            elif k == "catboost" and not HAS_CATBOOST:
                is_avail = False
                tooltip = "catboost not installed"

            if not is_avail:
                chk.setCheckState(Qt.CheckState.Unchecked)
                chk.setFlags(Qt.ItemFlag.NoItemFlags)
            else:
                default_active = (
                    ["ridge", "lasso", "random_forest", "hist_gradient_boosting"]
                    if task == AnalysisTaskType.REGRESSION
                    else ["logistic", "random_forest", "hist_gradient_boosting"]
                )
                should_check = (
                    k in prev_checked if prev_checked else k in default_active
                )
                chk.setCheckState(
                    Qt.CheckState.Checked if should_check else Qt.CheckState.Unchecked
                )

            self.model_table.setItem(r_idx, 0, chk)
            name_item = QTableWidgetItem(spec.display_name)
            name_item.setData(Qt.ItemDataRole.UserRole, k)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            if tooltip:
                name_item.setToolTip(tooltip)
                name_item.setForeground(QColor(NORD_PALETTE.NORD3))
            self.model_table.setItem(r_idx, 1, name_item)

            c_item = QTableWidgetItem(spec.complexity_tier)
            c_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.model_table.setItem(r_idx, 2, c_item)

            i_item = QTableWidgetItem(spec.interpretability_tier)
            i_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.model_table.setItem(r_idx, 3, i_item)

        if self.model_table.rowCount() > 0:
            self.model_table.selectRow(0)

    def _on_model_table_selection(self) -> None:
        row = self.model_table.currentRow()
        if row < 0:
            return
        item = self.model_table.item(row, 1)
        if item is not None:
            model_key = item.data(Qt.ItemDataRole.UserRole)
            if model_key:
                self._build_curated_param_form(model_key)

    def _build_curated_param_form(self, model_key: str) -> None:
        while self.param_form.count():
            item = self.param_form.takeAt(0)
            if item is not None:
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        self.per_model_widgets[model_key] = {}
        saved_overrides = self.state_bridge.session.per_model_overrides.get(
            model_key, {}
        )

        if model_key in ["random_forest", "extra_trees"]:
            n_est = QSpinBox()
            n_est.setRange(10, 1000)
            n_est.setValue(saved_overrides.get("n_estimators", 100))
            m_depth = QSpinBox()
            m_depth.setRange(0, 100)
            m_depth.setValue(saved_overrides.get("max_depth") or 0)
            m_depth.setSpecialValueText("None (Unlimited)")
            min_leaf = QSpinBox()
            min_leaf.setRange(1, 50)
            min_leaf.setValue(saved_overrides.get("min_samples_leaf", 1))
            self.param_form.addRow("n_estimators:", n_est)
            self.param_form.addRow("max_depth:", m_depth)
            self.param_form.addRow("min_samples_leaf:", min_leaf)
            self.per_model_widgets[model_key] = {
                "n_estimators": n_est,
                "max_depth": m_depth,
                "min_samples_leaf": min_leaf,
            }

        elif model_key in ["ridge", "lasso", "elasticnet"]:
            alpha = QDoubleSpinBox()
            alpha.setRange(0.0001, 10000.0)
            alpha.setValue(saved_overrides.get("alpha", 1.0))
            alpha.setDecimals(4)
            self.param_form.addRow("alpha (Regularization):", alpha)
            self.per_model_widgets[model_key] = {"alpha": alpha}
            if model_key == "elasticnet":
                l1 = QDoubleSpinBox()
                l1.setRange(0.01, 0.99)
                l1.setValue(saved_overrides.get("l1_ratio", 0.5))
                self.param_form.addRow("l1_ratio:", l1)
                self.per_model_widgets[model_key]["l1_ratio"] = l1

        elif model_key in ["hist_gradient_boosting", "xgboost", "lightgbm", "catboost"]:
            lr = QDoubleSpinBox()
            lr.setRange(0.001, 1.0)
            lr.setValue(saved_overrides.get("learning_rate", 0.1))
            lr.setDecimals(3)
            max_iter = QSpinBox()
            max_iter.setRange(20, 1000)
            max_iter.setValue(
                saved_overrides.get(
                    "max_iter", saved_overrides.get("n_estimators", 100)
                )
            )
            self.param_form.addRow("learning_rate:", lr)
            self.param_form.addRow("max_iter / n_estimators:", max_iter)
            self.per_model_widgets[model_key] = {
                "learning_rate": lr,
                "max_iter": max_iter,
            }

        elif model_key in ["svr", "svc"]:
            c_val = QDoubleSpinBox()
            c_val.setRange(0.01, 100.0)
            c_val.setValue(saved_overrides.get("C", 1.0))
            kernel = QComboBox()
            kernel.addItems(["rbf", "linear", "poly", "sigmoid"])
            if "kernel" in saved_overrides:
                kernel.setCurrentText(str(saved_overrides["kernel"]))
            self.param_form.addRow("C (Penalty):", c_val)
            self.param_form.addRow("Kernel:", kernel)
            self.per_model_widgets[model_key] = {"C": c_val, "kernel": kernel}

        elif model_key == "mlp":
            hidden_val = saved_overrides.get("hidden_layer_sizes", (64, 32))
            hidden_str = (
                ", ".join(map(str, hidden_val))
                if isinstance(hidden_val, (list, tuple))
                else str(hidden_val)
            )
            hidden = QLineEdit(hidden_str)
            alpha = QDoubleSpinBox()
            alpha.setRange(0.0001, 10.0)
            alpha.setValue(saved_overrides.get("alpha", 0.0001))
            alpha.setDecimals(4)
            self.param_form.addRow("hidden_layer_sizes:", hidden)
            self.param_form.addRow("alpha (L2 Penalty):", alpha)
            self.per_model_widgets[model_key] = {
                "hidden_layer_sizes": hidden,
                "alpha": alpha,
            }
        else:
            self.param_form.addRow(
                QLabel(f"Default curated tuning parameters active for {model_key}.")
            )

    def _on_targets_changed(self, targets: list[str]) -> None:
        if self._is_syncing:
            return
        self.feature_selector.set_excluded_columns(targets)
        session = self.state_bridge.session
        if session.dataset_df is None or not targets:
            self.cardinality_notice.hide()
            return

        warnings_found = []
        for t in targets:
            if t in session.dataset_df.columns:
                s = session.dataset_df[t].dropna()
                u_cnt = s.nunique()
                if 3 <= u_cnt <= 10 and pd.api.types.is_numeric_dtype(s):
                    warnings_found.append(f"'{t}' has {u_cnt} unique values")

        if warnings_found:
            self.cardinality_notice.setText(
                f"Low target cardinality notice: {', '.join(warnings_found)}. "
                f"Verify if Task Type should be Classification or Regression."
            )
            self.cardinality_notice.show()
        else:
            self.cardinality_notice.hide()

    def _on_features_changed(self, features: list[str]) -> None:
        if self._is_syncing:
            return
        self.target_selector.set_excluded_columns(features)

    def _toggle_all_models(self, checked: bool) -> None:
        self.model_table.setEnabled(not checked)

    def get_dataset_config(self) -> DatasetConfig:
        targets = self.target_selector.get_selected_columns()
        features = self.feature_selector.get_selected_columns()
        id_col = self.group_id_combo.currentText()
        if id_col == "None":
            id_col = None

        raw_task = self.task_type_combo.currentText()
        task_type = (
            AnalysisTaskType(raw_task)
            if raw_task in AnalysisTaskType._value2member_map_
            else AnalysisTaskType.REGRESSION
        )

        return DatasetConfig(
            target_columns=targets,
            feature_columns=features,
            active_columns=list(self.state_bridge.session.active_columns),
            id_column=id_col,
            task_type=task_type,
        )

    def get_cv_config(self) -> CVConfig:
        return CVConfig(
            outer_folds=self.outer_folds_spin.value(),
            inner_folds=self.inner_folds_spin.value(),
            n_iter=self.n_iter_spin.value(),
            n_jobs=self.n_jobs_spin.value(),
            parallel_outer_folds=self.parallel_outer_folds_cb.isChecked(),
            selection_folds=self.selection_folds_spin.value(),
            classification_scoring=self.classification_scoring_combo.currentText(),
            imbalance_ratio_threshold=self.imbalance_ratio_spin.value(),
            multiplicity_correction=self.multiplicity_correction_combo.currentText(),
            bootstrap_n=self.bootstrap_spin.value(),
            permutation_repeats=self.perm_repeats_spin.value(),
            skip_shap=self.skip_shap_cb.isChecked(),
            use_group_cv=self.use_group_cv_cb.isChecked(),
            fast_mode=self.fast_mode_cb.isChecked(),
        )

    def get_model_config(self) -> ModelZooConfig:
        selected = []
        for r in range(self.model_table.rowCount()):
            chk = self.model_table.item(r, 0)
            if chk and chk.checkState() == Qt.CheckState.Checked:
                item = self.model_table.item(r, 1)
                if item is not None:
                    key = item.data(Qt.ItemDataRole.UserRole)
                    if key:
                        selected.append(key)

        return ModelZooConfig(
            selected_models=selected,
            use_all_models=self.use_all_models_cb.isChecked(),
            enable_stacking=self.enable_stacking_cb.isChecked(),
            stacking_top_k=self.stacking_k_spin.value(),
            per_model_overrides=self.get_per_model_overrides(),
        )

    def get_per_model_overrides(self) -> dict[str, dict[str, Any]]:
        overrides = {}
        for m_key, w_dict in self.per_model_widgets.items():
            overrides[m_key] = {}
            for p_name, widget in w_dict.items():
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    with contextlib.suppress(RuntimeError):
                        val = widget.value()
                        if p_name == "max_depth" and val == 0:
                            val = None
                        overrides[m_key][p_name] = val
                elif isinstance(widget, QComboBox):
                    with contextlib.suppress(RuntimeError):
                        overrides[m_key][p_name] = widget.currentText()
                elif isinstance(widget, QLineEdit) and p_name == "hidden_layer_sizes":
                    with contextlib.suppress(Exception):
                        overrides[m_key][p_name] = tuple(
                            int(x.strip())
                            for x in widget.text().split(",")
                            if x.strip()
                        )
        return overrides
