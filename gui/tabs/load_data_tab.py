# gui/tabs/load_data_tab.py
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from config.schema import PipelineConfig
from core.data_loader import load_dataset
from gui.state import SessionStateBridge
from gui.widgets import ColumnSearchSelector, StatCard


class LoadDataTab(QWidget):
    dataset_df_changed = Signal(object)
    dataset_profile_changed = Signal(object)
    active_columns_changed = Signal(object)
    manifest_loaded = Signal(object, dict)

    def __init__(
        self, state_bridge: SessionStateBridge, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.state_bridge = state_bridge

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        file_box = QGroupBox("Dataset / Run Manifest Source")
        file_form = QFormLayout(file_box)
        file_form.setSpacing(8)

        input_row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText(
            "Select tabular data (*.csv, *.tsv, *.tab, *.parquet) or run_manifest.json..."
        )
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_file)
        self.load_btn = QPushButton("Load & Profile")
        self.load_btn.setProperty("role", "primary")
        self.load_btn.clicked.connect(self._load_file)

        input_row.addWidget(self.path_edit, stretch=1)
        input_row.addWidget(self.browse_btn)
        input_row.addWidget(self.load_btn)
        file_form.addRow("Source Path:", input_row)
        layout.addWidget(file_box)

        self.cards_layout = QHBoxLayout()
        self.card_rows = StatCard("Rows (N)", "—", "Observations")
        self.card_cols = StatCard("Columns (P)", "—", "Attributes")
        self.card_mem = StatCard("Memory", "—", "RAM Allocation")
        self.card_miss = StatCard("Missing", "—", "Global NA %")
        self.card_num = StatCard("Numeric", "—", "Continuous")
        self.card_cat = StatCard("Categorical", "—", "Discrete")

        for c in [
            self.card_rows,
            self.card_cols,
            self.card_mem,
            self.card_miss,
            self.card_num,
            self.card_cat,
        ]:
            self.cards_layout.addWidget(c)
        layout.addLayout(self.cards_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        preview_box = QGroupBox("Dataset Preview (First 15 Rows)")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setDragEnabled(False)
        self.preview_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Interactive
        )
        preview_layout.addWidget(self.preview_table)
        splitter.addWidget(preview_box)

        col_box = QGroupBox("Active Downstream Column Pool")
        col_layout = QVBoxLayout(col_box)
        self.col_selector = ColumnSearchSelector(title="Candidate Columns")
        self.col_selector.selection_changed.connect(self._on_columns_selected)
        col_layout.addWidget(self.col_selector)
        splitter.addWidget(col_box)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

    def set_interactive(self, enabled: bool) -> None:
        self.path_edit.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.load_btn.setEnabled(enabled)
        self.col_selector.setEnabled(enabled)
        self.preview_table.setEnabled(enabled)

    def _browse_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Open Tabular Dataset or Run Manifest",
            "",
            "All Supported (*.csv *.tsv *.tab *.parquet *.json);;Run Manifest (*.json);;Delimited (*.csv *.tsv *.tab);;Parquet (*.parquet);;All Files (*)",
        )
        if p:
            self.path_edit.setText(p)
            self._load_file()

    def _load_file(self) -> None:
        path_str = self.path_edit.text().strip()
        if not path_str or not Path(path_str).is_file():
            QMessageBox.critical(
                self,
                "Invalid File",
                "Please select a valid tabular data or JSON manifest file.",
            )
            return

        target_path = Path(path_str)
        if target_path.suffix.lower() == ".json":
            self._load_manifest_file(target_path)
        else:
            self._load_tabular_file(target_path)

    def _load_tabular_file(self, path: Path) -> None:
        try:
            dataset_df, profile = load_dataset(path)
            self.state_bridge.set_loaded_data(dataset_df, profile)
            self._refresh_views()
            self.dataset_df_changed.emit(self.state_bridge.session.dataset_df)
            self.dataset_profile_changed.emit(self.state_bridge.session.dataset_profile)
            self.active_columns_changed.emit(self.state_bridge.session.active_columns)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Failure", f"Could not load tabular dataset:\n{exc}"
            )

    def _resolve_manifest_data_path(
        self, manifest_path: Path, config: PipelineConfig
    ) -> Path | None:
        """Finds the dataset associated with a run_manifest.json with multiple fallback strategies."""
        orig_path = (
            Path(config.dataset.input_path) if config.dataset.input_path else None
        )
        if orig_path and orig_path.is_file():
            return orig_path

        derived_in_manifest_dir = manifest_path.parent / "derived_dataset.csv"
        if derived_in_manifest_dir.is_file():
            return derived_in_manifest_dir

        if orig_path:
            same_dir_named = manifest_path.parent / orig_path.name
            if same_dir_named.is_file():
                return same_dir_named

        QMessageBox.warning(
            self,
            "Dataset Path Resolution Required",
            f"The manifest references a dataset path that was not found:\n{config.dataset.input_path}\n\nPlease locate the dataset file.",
        )
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Locate Dataset for Imported Manifest",
            str(manifest_path.parent),
            "Tabular Files (*.csv *.tsv *.tab *.parquet);;All Files (*)",
        )
        return Path(p) if p and Path(p).is_file() else None

    def _load_manifest_file(self, manifest_path: Path) -> None:
        try:
            config, manifest_meta = PipelineConfig.load_manifest(manifest_path)
            data_file = self._resolve_manifest_data_path(manifest_path, config)
            if not data_file:
                QMessageBox.critical(
                    self,
                    "Import Cancelled",
                    "Cannot import run manifest without an associated dataset file.",
                )
                return

            dataset_df, profile = load_dataset(data_file)

            active_cols = [
                c for c in config.dataset.active_columns if c in dataset_df.columns
            ] or list(dataset_df.columns)
            config.dataset.active_columns = active_cols
            config.dataset.input_path = str(data_file.resolve())

            self.state_bridge.set_loaded_data(dataset_df, profile)
            self.state_bridge.session.active_columns = list(active_cols)
            if config.transformation_log:
                self.state_bridge.session.transformation_log = list(
                    config.transformation_log
                )

            self._refresh_views()

            run_id = manifest_meta.get("run_id", "N/A")
            start_t = manifest_meta.get("start_time", "N/A")
            status = manifest_meta.get("status", "N/A")

            self.card_rows.set_value(
                f"{len(dataset_df):,}", subtitle=f"Manifest: {run_id[:14]}..."
            )

            self.dataset_df_changed.emit(self.state_bridge.session.dataset_df)
            self.dataset_profile_changed.emit(self.state_bridge.session.dataset_profile)
            self.active_columns_changed.emit(self.state_bridge.session.active_columns)
            self.manifest_loaded.emit(config, manifest_meta)

            QMessageBox.information(
                self,
                "Run Manifest Imported Successfully",
                f"Reproducible run configuration imported:\n\n"
                f"• Run ID: {run_id}\n"
                f"• Timestamp: {start_t}\n"
                f"• Original Status: {status}\n"
                f"• Targets: {', '.join(config.dataset.target_columns) or 'None'}\n"
                f"• Predictors: {len(config.dataset.feature_columns)} features\n\n"
                f"All tabs (Transform, Models, SEM, Output, Execute) have been reconfigured.",
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Manifest Import Error",
                f"Failed to parse or apply manifest:\n{exc}",
            )

    def _refresh_views(self) -> None:
        dataset_df = self.state_bridge.session.dataset_df
        profile = self.state_bridge.session.dataset_profile
        if dataset_df is None or profile is None:
            return

        self.card_rows.set_value(f"{len(dataset_df):,}", "Observations")
        self.card_cols.set_value(f"{len(dataset_df.columns):,}", "Attributes")
        self.card_mem.set_value(f"{profile.memory_usage_mb:.2f} MB", "RAM Allocation")
        total_cells = max(1, len(dataset_df) * len(dataset_df.columns))
        miss_count = int(dataset_df.isna().sum().sum())
        self.card_miss.set_value(f"{(miss_count / total_cells):.1%}", "Global NA %")
        self.card_num.set_value(f"{len(profile.numerical_columns)}", "Continuous")
        self.card_cat.set_value(f"{len(profile.categorical_columns)}", "Discrete")

        n_preview = min(15, len(dataset_df))
        self.preview_table.setRowCount(n_preview)
        self.preview_table.setColumnCount(len(dataset_df.columns))
        self.preview_table.setHorizontalHeaderLabels(
            [str(c) for c in dataset_df.columns]
        )
        for r_idx in range(n_preview):
            for c_idx, _col in enumerate(dataset_df.columns):
                val = dataset_df.iloc[r_idx, c_idx]
                item = QTableWidgetItem("" if val is None else str(val))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.preview_table.setItem(r_idx, c_idx, item)

        dtypes_map = {c: str(dataset_df[c].dtype) for c in dataset_df.columns}
        self.col_selector.set_columns(
            list(dataset_df.columns),
            selected=self.state_bridge.session.active_columns,
            dtypes=dtypes_map,
        )

    def _on_columns_selected(self, cols: list[str]) -> None:
        self.state_bridge.session.active_columns = list(cols)
        self.active_columns_changed.emit(cols)
