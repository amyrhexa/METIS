# gui/tabs/transform_tab.py
"""Tab 2: In-memory data transformation workspace using candidate columns from Load Data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from config.schema import PipelineConfig
from gui.state import SessionStateBridge
from gui.theme import NORD_PALETTE


class TransformTab(QWidget):
    def __init__(
        self, state_bridge: SessionStateBridge, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.state_bridge = state_bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        op_box = QGroupBox("Transformation Studio")
        op_layout = QVBoxLayout(op_box)
        op_layout.setSpacing(10)

        cat_form = QFormLayout()
        self.op_category_combo = QComboBox()
        self.op_category_combo.addItems(
            [
                "Arithmetic (Create Derived Variable)",
                "Rename Column",
                "One-Hot Encode Categoricals",
                "Normalize / Scale Features",
                "Handle Missing Values (Warning: Leakage Risk)",
                "Data Type Conversion",
                "Mathematical Transforms",
                "Drop Columns",
            ]
        )
        self.op_category_combo.currentIndexChanged.connect(self._on_category_changed)
        cat_form.addRow("Operation:", self.op_category_combo)
        op_layout.addLayout(cat_form)

        self.scope_status_lbl = QLabel(
            "Active Scope: 0 candidate column(s) selected in Load Data."
        )
        self.scope_status_lbl.setStyleSheet(
            f"color: {NORD_PALETTE.NORD8}; font-weight: bold; font-size: 11.5px;"
        )
        op_layout.addWidget(self.scope_status_lbl)

        self.stack = QStackedWidget()

        # 0: Arithmetic
        p_arith = QWidget()
        f_arith = QFormLayout(p_arith)
        self.arith_name = QLineEdit()
        self.arith_name.setPlaceholderText("e.g., ratio_x_y")
        self.arith_left = QComboBox()
        self.arith_op = QComboBox()
        self.arith_op.addItems(
            ["+ (Add)", "- (Subtract)", "* (Multiply)", "/ (Divide)"]
        )
        self.arith_right = QComboBox()
        f_arith.addRow("New Variable Name:", self.arith_name)
        f_arith.addRow("Left Column (X):", self.arith_left)
        f_arith.addRow("Operator:", self.arith_op)
        f_arith.addRow("Right Column (Y):", self.arith_right)
        self.stack.addWidget(p_arith)

        # 1: Rename
        p_ren = QWidget()
        f_ren = QFormLayout(p_ren)
        self.ren_old = QComboBox()
        self.ren_new = QLineEdit()
        self.ren_new.setPlaceholderText("New unique identifier...")
        f_ren.addRow("Target Column:", self.ren_old)
        f_ren.addRow("New Name:", self.ren_new)
        self.stack.addWidget(p_ren)

        # 2: One-Hot Encode
        p_ohe = QWidget()
        f_ohe = QFormLayout(p_ohe)
        self.ohe_drop_first = QCheckBox("Drop First Indicator (Avoid Collinearity)")
        self.ohe_keep_orig = QCheckBox("Keep Original Column(s)")
        f_ohe.addRow(
            "", QLabel("Encodes all active categorical/object candidate columns.")
        )
        f_ohe.addRow("", self.ohe_drop_first)
        f_ohe.addRow("", self.ohe_keep_orig)
        self.stack.addWidget(p_ohe)

        # 3: Scaling
        p_scale = QWidget()
        f_scale = QFormLayout(p_scale)
        self.scale_type = QComboBox()
        self.scale_type.addItems(
            ["StandardScaler (Z-score)", "MinMaxScaler [0, 1]", "RobustScaler (IQR)"]
        )
        self.scale_prefix = QLineEdit("scaled_")
        self.scale_replace = QCheckBox("Overwrite Original Columns")
        f_scale.addRow("Method:", self.scale_type)
        f_scale.addRow("Output Prefix:", self.scale_prefix)
        f_scale.addRow("", self.scale_replace)
        self.stack.addWidget(p_scale)

        # 4: Missing Values
        p_miss = QWidget()
        f_miss = QFormLayout(p_miss)
        self.miss_action = QComboBox()
        self.miss_action.addItems(
            [
                "Impute Numeric with Median",
                "Impute Numeric with Mean",
                "Impute with Zero/Constant",
                "Impute Categorical with Most Frequent",
                "Drop Rows with Missing Values in Candidate Scope",
                "Drop Candidate Columns Exceeding NA Threshold",
            ]
        )
        self.miss_thresh = QDoubleSpinBox()
        self.miss_thresh.setRange(0.01, 0.99)
        self.miss_thresh.setValue(0.25)
        self.miss_thresh.setSingleStep(0.05)
        f_miss.addRow("Strategy:", self.miss_action)
        f_miss.addRow("Max NA Threshold:", self.miss_thresh)
        self.stack.addWidget(p_miss)

        # 5: Type Conversion
        p_type = QWidget()
        f_type = QFormLayout(p_type)
        self.type_target = QComboBox()
        self.type_target.addItems(
            [
                "Numeric (Coerce Errors to NaN)",
                "String / Categorical",
                "Integer (Fill NA with 0)",
            ]
        )
        f_type.addRow("Target Type:", self.type_target)
        self.stack.addWidget(p_type)

        # 6: Mathematical Transforms
        p_math = QWidget()
        f_math = QFormLayout(p_math)
        self.math_op = QComboBox()
        self.math_op.addItems(
            [
                "Log(1 + x)",
                "Square Root (sqrt)",
                "Square (x²)",
                "Clip Outliers (1.5x IQR)",
                "Remove Exact Duplicate Rows",
            ]
        )
        f_math.addRow("Transformation:", self.math_op)
        self.stack.addWidget(p_math)

        # 7: Drop Columns
        p_drop = QWidget()
        f_drop = QFormLayout(p_drop)
        self.drop_warn_lbl = QLabel(
            "Permanently removes all active candidate columns from the matrix."
        )
        self.drop_warn_lbl.setStyleSheet(
            f"color: {NORD_PALETTE.NORD11}; font-weight: 600;"
        )
        f_drop.addRow("", self.drop_warn_lbl)
        self.stack.addWidget(p_drop)

        op_layout.addWidget(self.stack)

        btn_bar = QHBoxLayout()
        self.apply_btn = QPushButton("Apply Transformation")
        self.apply_btn.setProperty("role", "primary")
        self.apply_btn.clicked.connect(self._execute_transformation)
        self.undo_btn = QPushButton("Undo Last")
        self.undo_btn.clicked.connect(self._execute_undo)
        self.reset_btn = QPushButton("Reset to Original")
        self.reset_btn.setProperty("role", "danger")
        self.reset_btn.clicked.connect(self._execute_reset)

        btn_bar.addWidget(self.apply_btn)
        btn_bar.addWidget(self.undo_btn)
        btn_bar.addWidget(self.reset_btn)
        op_layout.addLayout(btn_bar)

        main_splitter.addWidget(op_box)

        right_splitter = QSplitter(Qt.Orientation.Vertical)

        log_box = QGroupBox("Transformation Audit Log")
        log_layout = QVBoxLayout(log_box)
        self.log_list = QListWidget()
        log_layout.addWidget(self.log_list)
        right_splitter.addWidget(log_box)

        preview_box = QGroupBox("Candidate Columns Preview (First 10 Rows)")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setDragEnabled(False)
        preview_layout.addWidget(self.preview_table)
        right_splitter.addWidget(preview_box)

        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 3)
        main_splitter.addWidget(right_splitter)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        layout.addWidget(main_splitter)

    def set_interactive(self, enabled: bool) -> None:
        self.op_category_combo.setEnabled(enabled)
        self.stack.setEnabled(enabled)
        self.apply_btn.setEnabled(enabled)
        self.undo_btn.setEnabled(enabled)
        self.reset_btn.setEnabled(enabled)

    def _get_valid_active_columns(self) -> list[str]:
        dataset_df = self.state_bridge.session.dataset_df
        if dataset_df is None:
            return []
        active = self.state_bridge.session.active_columns
        valid = [c for c in active if c in dataset_df.columns]
        return valid if valid else list(dataset_df.columns)

    def sync_data(self, dataset_df: pd.DataFrame | None) -> None:
        self.sync_columns(self._get_valid_active_columns())

    def sync_columns(self, cols: list[str]) -> None:
        active_cols = self._get_valid_active_columns()
        self.scope_status_lbl.setText(
            f"Active Scope: {len(active_cols)} candidate column(s) selected in Load Data."
        )

        for combo in [self.arith_left, self.arith_right, self.ren_old]:
            combo.blockSignals(True)
            prev_txt = combo.currentText()
            combo.clear()
            combo.addItems(active_cols)
            if prev_txt in active_cols:
                combo.setCurrentText(prev_txt)
            combo.blockSignals(False)

        self._update_preview(active_cols)
        self._refresh_log()

    def set_from_config(self, config: PipelineConfig) -> None:
        """Restores transformation log and active column scopes from a PipelineConfig."""
        if config.transformation_log:
            self.state_bridge.session.transformation_log = list(
                config.transformation_log
            )
        self._refresh_log()
        self.sync_columns(self._get_valid_active_columns())

    def _on_category_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)

    def _update_preview(self, columns: list[str] | None = None) -> None:
        dataset_df = self.state_bridge.session.dataset_df
        if dataset_df is None or dataset_df.empty:
            self.preview_table.setRowCount(0)
            self.preview_table.setColumnCount(0)
            return

        cols_to_show = [
            c
            for c in (
                columns if columns is not None else self._get_valid_active_columns()
            )
            if c in dataset_df.columns
        ]
        if not cols_to_show:
            cols_to_show = list(dataset_df.columns[:8])

        n_rows = min(10, len(dataset_df))
        self.preview_table.setRowCount(n_rows)
        self.preview_table.setColumnCount(len(cols_to_show))
        self.preview_table.setHorizontalHeaderLabels(cols_to_show)

        for r_idx in range(n_rows):
            for c_idx, col in enumerate(cols_to_show):
                val = dataset_df[col].iloc[r_idx]
                item = QTableWidgetItem("" if pd.isna(val) else str(val))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.preview_table.setItem(r_idx, c_idx, item)

    def _execute_transformation(self) -> None:
        dataset_df = self.state_bridge.session.dataset_df
        if dataset_df is None:
            QMessageBox.warning(
                self, "No Data", "Load a dataset in the Load Data tab first."
            )
            return

        active_cols = self._get_valid_active_columns()
        idx = self.stack.currentIndex()
        updated_df = dataset_df.copy()
        log_msg = ""

        try:
            if idx == 0:  # Arithmetic
                name = self.arith_name.text().strip()
                left = self.arith_left.currentText()
                right = self.arith_right.currentText()
                op_raw = self.arith_op.currentText()
                if not name:
                    raise ValueError("New variable name cannot be empty.")
                if name in updated_df.columns:
                    raise ValueError(f"Variable '{name}' already exists.")
                if not left or not right:
                    raise ValueError(
                        "Please select both left and right operand columns from candidate pool."
                    )
                if not pd.api.types.is_numeric_dtype(
                    updated_df[left]
                ) or not pd.api.types.is_numeric_dtype(updated_df[right]):
                    raise ValueError(
                        f"Both '{left}' and '{right}' must be numeric columns."
                    )

                if op_raw.startswith("+"):
                    updated_df[name] = updated_df[left] + updated_df[right]
                elif op_raw.startswith("-"):
                    updated_df[name] = updated_df[left] - updated_df[right]
                elif op_raw.startswith("*"):
                    updated_df[name] = updated_df[left] * updated_df[right]
                elif op_raw.startswith("/"):
                    updated_df[name] = updated_df[left] / updated_df[right].replace(
                        0, np.nan
                    )
                    updated_df[name] = updated_df[name].replace(
                        [np.inf, -np.inf], np.nan
                    )

                log_msg = f"Created derived variable '{name}' = '{left}' {op_raw[0]} '{right}'"
                self.arith_name.clear()
                if name not in self.state_bridge.session.active_columns:
                    self.state_bridge.session.active_columns.append(name)

            elif idx == 1:  # Rename
                old_col = self.ren_old.currentText()
                new_col = self.ren_new.text().strip()
                if not old_col:
                    raise ValueError("Please select a candidate column to rename.")
                if not new_col:
                    raise ValueError("New identifier cannot be empty.")
                if new_col in updated_df.columns:
                    raise ValueError(f"Column '{new_col}' already exists.")
                updated_df = updated_df.rename(columns={old_col: new_col})
                log_msg = f"Renamed column '{old_col}' to '{new_col}'"
                self.ren_new.clear()
                self.state_bridge.session.active_columns = [
                    new_col if c == old_col else c
                    for c in self.state_bridge.session.active_columns
                ]

            elif idx == 2:  # One-Hot Encode
                if not active_cols:
                    raise ValueError("No candidate columns selected in Load Data tab.")
                drop_f = self.ohe_drop_first.isChecked()
                keep_orig = self.ohe_keep_orig.isChecked()
                dummies = pd.get_dummies(
                    updated_df[active_cols],
                    prefix=active_cols,
                    drop_first=drop_f,
                    dtype=float,
                )
                updated_df = pd.concat([updated_df, dummies], axis=1)
                if not keep_orig:
                    updated_df = updated_df.drop(columns=active_cols)
                log_msg = f"One-Hot Encoded {len(active_cols)} column(s) (drop_first={drop_f})"

            elif idx == 3:  # Normalize / Scale
                if not active_cols:
                    raise ValueError("No candidate columns selected in Load Data tab.")
                num_targets = [
                    c
                    for c in active_cols
                    if pd.api.types.is_numeric_dtype(updated_df[c])
                ]
                if not num_targets:
                    raise ValueError(
                        "None of the selected candidate columns are numeric."
                    )
                stype = self.scale_type.currentText()
                scaler = (
                    StandardScaler()
                    if "Standard" in stype
                    else (MinMaxScaler() if "MinMax" in stype else RobustScaler())
                )
                scaled_mat = scaler.fit_transform(
                    updated_df[num_targets].select_dtypes(include=[np.number])
                )
                prefix = (
                    ""
                    if self.scale_replace.isChecked()
                    else (self.scale_prefix.text().strip() or "scaled_")
                )
                for c_idx, col in enumerate(num_targets):
                    updated_df[f"{prefix}{col}"] = scaled_mat[:, c_idx]
                log_msg = f"Scaled {len(num_targets)} variable(s) using {scaler.__class__.__name__}"

            elif idx == 4:  # Missing Values
                if not active_cols:
                    raise ValueError("No candidate columns selected in Load Data tab.")
                action = self.miss_action.currentText()
                if "Median" in action:
                    for col in active_cols:
                        if pd.api.types.is_numeric_dtype(updated_df[col]):
                            updated_df[col] = updated_df[col].fillna(
                                updated_df[col].median()
                            )
                elif "Mean" in action:
                    for col in active_cols:
                        if pd.api.types.is_numeric_dtype(updated_df[col]):
                            updated_df[col] = updated_df[col].fillna(
                                updated_df[col].mean()
                            )
                elif "Zero/Constant" in action:
                    for col in active_cols:
                        updated_df[col] = updated_df[col].fillna(0)
                elif "Most Frequent" in action:
                    for col in active_cols:
                        mode_vals = updated_df[col].mode()
                        if not mode_vals.empty:
                            updated_df[col] = updated_df[col].fillna(mode_vals.iloc[0])
                elif "Drop Rows" in action:
                    updated_df = updated_df.dropna(subset=active_cols)
                elif "Drop Candidate Columns" in action:
                    th = self.miss_thresh.value()
                    drop_cols = [
                        c for c in active_cols if updated_df[c].isna().mean() > th
                    ]
                    if drop_cols:
                        if len(drop_cols) >= len(updated_df.columns):
                            raise ValueError("Cannot drop all columns from the matrix.")
                        updated_df = updated_df.drop(columns=drop_cols)
                log_msg = f"[GLOBAL_PREFILTERING] Applied missingness operation: '{action}' on {len(active_cols)} column(s)"

            elif idx == 5:  # Type Conversion
                if not active_cols:
                    raise ValueError("No candidate columns selected in Load Data tab.")
                ctype = self.type_target.currentText()
                for col in active_cols:
                    if "Numeric" in ctype:
                        updated_df[col] = pd.to_numeric(
                            updated_df[col], errors="coerce"
                        )
                    elif "String" in ctype:
                        updated_df[col] = updated_df[col].astype(str)
                    elif "Integer" in ctype:
                        updated_df[col] = (
                            pd.to_numeric(updated_df[col], errors="coerce")
                            .fillna(0)
                            .astype(int)
                        )
                log_msg = f"Converted {len(active_cols)} column(s) to {ctype}"

            elif idx == 6:  # Mathematical Transforms
                mop = self.math_op.currentText()
                if "Duplicate" in mop:
                    updated_df = updated_df.drop_duplicates()
                    log_msg = "Removed exact duplicate rows"
                else:
                    if not active_cols:
                        raise ValueError(
                            "No candidate columns selected in Load Data tab."
                        )
                    num_targets = [
                        c
                        for c in active_cols
                        if pd.api.types.is_numeric_dtype(updated_df[c])
                    ]
                    if not num_targets:
                        raise ValueError(
                            "None of the selected candidate columns are numeric."
                        )
                    for col in num_targets:
                        if "Log" in mop:
                            updated_df[col] = np.log1p(
                                np.maximum(
                                    0,
                                    pd.to_numeric(
                                        updated_df[col], errors="coerce"
                                    ).fillna(0),
                                )
                            )
                        elif "Square Root" in mop:
                            updated_df[col] = np.sqrt(
                                np.maximum(
                                    0,
                                    pd.to_numeric(
                                        updated_df[col], errors="coerce"
                                    ).fillna(0),
                                )
                            )
                        elif "Square" in mop:
                            updated_df[col] = (
                                pd.to_numeric(updated_df[col], errors="coerce") ** 2
                            )
                        elif "Clip" in mop:
                            q25, q75 = updated_df[col].quantile([0.25, 0.75])
                            iqr = q75 - q25
                            updated_df[col] = updated_df[col].clip(
                                q25 - 1.5 * iqr, q75 + 1.5 * iqr
                            )
                    log_msg = f"Applied '{mop}' on {len(num_targets)} numeric column(s)"

            elif idx == 7:  # Drop Columns
                if not active_cols:
                    raise ValueError("No candidate columns selected in Load Data tab.")
                if len(active_cols) >= len(updated_df.columns):
                    raise ValueError("Cannot drop all columns from the dataset.")
                updated_df = updated_df.drop(columns=active_cols)
                log_msg = f"Dropped {len(active_cols)} column(s): {', '.join(active_cols[:3])}..."

            self.state_bridge.update_dataset_df(updated_df, log_msg)
            self.sync_columns(self._get_valid_active_columns())
        except Exception as exc:
            QMessageBox.critical(
                self, "Transformation Error", f"Operation failed:\n{exc}"
            )

    def _execute_undo(self) -> None:
        if not self.state_bridge.undo_last_transformation():
            QMessageBox.information(self, "Undo", "No previous transformation to undo.")
        else:
            self.sync_columns(self._get_valid_active_columns())

    def _execute_reset(self) -> None:
        self.state_bridge.reset_dataset_df()
        self.sync_columns(self._get_valid_active_columns())

    def _refresh_log(self) -> None:
        self.log_list.clear()
        for item in self.state_bridge.session.transformation_log:
            self.log_list.addItem(item)
