# gui/main_window.py
"""Main application window coordinating exactly 6 primary scientific workflow tabs."""

from __future__ import annotations

import os
import sys
from typing import Any

os.environ.setdefault("NO_ATK_BRIDGE", "1")

from PySide6.QtGui import QCloseEvent, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config.schema import PipelineConfig
from gui.state import AnalysisSession, SessionStateBridge
from gui.tabs.execute_tab import ExecuteTab
from gui.tabs.load_data_tab import LoadDataTab
from gui.tabs.models_tab import ModelsTab
from gui.tabs.output_tab import OutputTab
from gui.tabs.sem_tab import SEMTab
from gui.tabs.transform_tab import TransformTab
from gui.theme import get_application_stylesheet


class MainWindow(QMainWindow):
    """Refactored GUI scientific platform with 6 coordinated workflow tabs."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            "METIS | Machine Evaluation & Tabular Inference System v1.0"
        )
        self.setMinimumSize(1200, 800)
        self.resize(1560, 960)
        self.session_bridge = SessionStateBridge(AnalysisSession())

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(6)

        self.tabs = QTabWidget()
        self.load_data_tab = LoadDataTab(self.session_bridge)
        self.transform_tab = TransformTab(self.session_bridge)
        self.models_tab = ModelsTab(self.session_bridge)
        self.sem_tab = SEMTab(self.session_bridge)
        self.output_tab = OutputTab(self.session_bridge)
        self.execute_tab = ExecuteTab(self.session_bridge)

        self.tabs.addTab(self.load_data_tab, "1. Load Data")
        self.tabs.addTab(self.transform_tab, "2. Transform")
        self.tabs.addTab(self.models_tab, "3. Models")
        self.tabs.addTab(self.sem_tab, "4. SEM")
        self.tabs.addTab(self.output_tab, "5. Output")
        self.tabs.addTab(self.execute_tab, "6. Execute")
        central_layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # Cross-Tab Signal Coordination
        self.load_data_tab.active_columns_changed.connect(self._on_columns_updated)
        self.load_data_tab.dataset_df_changed.connect(self._on_data_updated)
        self.load_data_tab.manifest_loaded.connect(self._on_manifest_loaded)
        self.session_bridge.dataset_df_updated.connect(self._on_data_updated)
        self.session_bridge.active_columns_updated.connect(self._on_columns_updated)
        self.tabs.currentChanged.connect(self._on_tab_switched)
        self.execute_tab.request_pipeline_run.connect(self._run_pipeline)
        self.execute_tab.execution_started.connect(
            lambda: self.set_execution_lock(True)
        )
        self.execute_tab.execution_finished.connect(
            lambda ok, _: self.set_execution_lock(False)
        )

    def _get_valid_active_columns(self) -> list[str]:
        session = self.session_bridge.session
        if session.dataset_df is None:
            return []
        pool = [c for c in session.active_columns if c in session.dataset_df.columns]
        return pool or list(session.dataset_df.columns)

    def set_execution_lock(self, locked: bool) -> None:
        if locked:
            self.tabs.setCurrentIndex(5)
        self.tabs.tabBar().setEnabled(not locked)
        self.load_data_tab.set_interactive(not locked)
        self.transform_tab.set_interactive(not locked)
        self.models_tab.set_interactive(not locked)
        self.sem_tab.set_interactive(not locked)
        self.output_tab.set_interactive(not locked)
        if not locked:
            cfg = self._build_pipeline_config()
            self.execute_tab.refresh_review(cfg)

    def _on_columns_updated(self, cols: list[str]) -> None:
        active_cols = self._get_valid_active_columns() or list(cols)
        self.transform_tab.sync_columns(active_cols)
        self.models_tab.sync_columns(active_cols)
        self.sem_tab.sync_variables(targets=active_cols, features=active_cols)

    def _on_data_updated(self, dataset_df: Any) -> None:
        if dataset_df is not None:
            self.transform_tab.sync_data(dataset_df)
            active_cols = self._get_valid_active_columns()
            self._on_columns_updated(active_cols)

    def _on_tab_switched(self, index: int) -> None:
        if index == 1:
            self.transform_tab.sync_data(self.session_bridge.session.dataset_df)
        elif index == 5:
            cfg = self._build_pipeline_config()
            self.execute_tab.refresh_review(cfg)

    def _on_manifest_loaded(
        self, config: PipelineConfig, manifest_meta: dict[str, Any]
    ) -> None:
        """Restores complete multi-tab METIS configuration from an imported manifest."""
        self.models_tab.set_from_config(config)
        self.sem_tab.set_from_config(config)
        self.output_tab.set_from_config(config)
        self.transform_tab.set_from_config(config)
        self.execute_tab.refresh_review(config)

    def _build_pipeline_config(self) -> PipelineConfig:
        ds_cfg = self.models_tab.get_dataset_config()
        ds_cfg.input_path = self.load_data_tab.path_edit.text().strip()
        ds_cfg.active_columns = list(self.session_bridge.session.active_columns)
        cv_cfg = self.models_tab.get_cv_config()
        model_cfg = self.models_tab.get_model_config()
        sem_configs = self.sem_tab.get_sem_configs()
        out_cfg = self.output_tab.get_output_config()
        rep_cfg = self.output_tab.get_reporting_config()
        return PipelineConfig(
            dataset=ds_cfg,
            cv=cv_cfg,
            models=model_cfg,
            sem_configs=sem_configs,
            reporting=rep_cfg,
            output_config=out_cfg,
            run_sem=bool(sem_configs),
            output_dir=out_cfg.output_dir,
            mode=out_cfg.mode,
            transformation_log=list(self.session_bridge.session.transformation_log),
        )

    def _run_pipeline(self) -> None:
        cfg = self._build_pipeline_config()
        self.execute_tab.start_execution(cfg)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.execute_tab.worker and self.execute_tab.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Pipeline Execution in Progress",
                "A modeling tournament is currently running in the background.\n\n"
                "Do you want to terminate the background process and exit METIS?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.execute_tab.worker.stop_event.set()
                self.execute_tab.worker.quit()
                self.execute_tab.worker.wait(3000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    families = QFontDatabase.families()
    inter_family = next(
        (f for f in ("Inter", "Inter Variable", "Inter UI") if f in families), "Inter"
    )
    app.setFont(QFont(inter_family, 10))
    app.setStyleSheet(get_application_stylesheet())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
