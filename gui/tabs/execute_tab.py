# gui/tabs/execute_tab.py
"""Execute tab: config review, pre-flight validation, and a fully logged live execution window."""

from __future__ import annotations

import html
import logging
import threading
import traceback
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config.schema import PipelineConfig
from core.pipeline_runner import execute_pipeline
from core.progress import ProgressUpdate
from gui.state import SessionStateBridge
from gui.theme import NORD_PALETTE
from gui.widgets import StatusBadge


class QtLogForwarder(logging.Handler):
    """Thread-safe logging handler forwarding messages via Qt signals."""

    def __init__(self, signal_emitter: Signal) -> None:
        super().__init__()
        self.signal_emitter = signal_emitter

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.signal_emitter.emit(msg, record.levelno)


class PipelineWorkerThread(QThread):
    progress_signal = Signal(object)
    log_signal = Signal(str, int)
    finished_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(
        self,
        config: PipelineConfig,
        dataset_df: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.dataset_df = dataset_df
        self.stop_event = threading.Event()

    def run(self) -> None:
        try:
            log_handler = QtLogForwarder(self.log_signal)
            execute_pipeline(
                self.config,
                dataset_df=self.dataset_df,
                progress_callback=self.progress_signal.emit,
                log_handler=log_handler,
                should_stop=self.stop_event.is_set,
            )
            self.finished_signal.emit(self.config.output_dir)
        except Exception as exc:
            self.error_signal.emit(f"{exc}\n\n{traceback.format_exc()}")


class ExecuteTab(QWidget):
    request_pipeline_run = Signal()
    execution_started = Signal()
    execution_finished = Signal(bool, str)

    def __init__(
        self, state_bridge: SessionStateBridge, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.state_bridge = state_bridge
        self.worker: PipelineWorkerThread | None = None
        self.last_output_dir: str | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        rev_box = QGroupBox("Configuration Dossier Review")
        rev_layout = QVBoxLayout(rev_box)
        self.review_text = QTextEdit()
        self.review_text.setReadOnly(True)
        rev_layout.addWidget(self.review_text)
        top_splitter.addWidget(rev_box)
        val_box = QGroupBox("Pre-Flight Validation")
        val_layout = QVBoxLayout(val_box)
        self.val_list = QListWidget()
        self.val_list.setMaximumHeight(160)
        val_layout.addWidget(self.val_list)
        top_splitter.addWidget(val_box)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        main_splitter.addWidget(top_splitter)

        log_box = QGroupBox("Live Execution Log")
        log_layout = QVBoxLayout(log_box)
        self.log_window = QPlainTextEdit()
        self.log_window.setReadOnly(True)
        self.log_window.setMaximumBlockCount(10000)
        self.log_window.setStyleSheet(
            f"font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11.5px; background-color: {NORD_PALETTE.NORD0}; color: {NORD_PALETTE.NORD6}; border: 1px solid {NORD_PALETTE.NORD2}; border-radius: 5px;"
        )
        log_layout.addWidget(self.log_window)
        log_btn_row = QHBoxLayout()
        clear_log_btn = QPushButton("Clear Log")
        clear_log_btn.clicked.connect(self.log_window.clear)
        copy_log_btn = QPushButton("Copy Log")
        copy_log_btn.clicked.connect(self.log_window.selectAll)
        log_btn_row.addStretch(1)
        log_btn_row.addWidget(clear_log_btn)
        log_btn_row.addWidget(copy_log_btn)
        log_layout.addLayout(log_btn_row)
        main_splitter.addWidget(log_box)

        bottom_box = QGroupBox("Execution Control & Telemetry")
        bottom_layout = QVBoxLayout(bottom_box)
        bottom_layout.setSpacing(8)
        ctrl_row = QHBoxLayout()
        self.run_btn = QPushButton("Accept & Execute Pipeline")
        self.run_btn.setProperty("role", "primary")
        self.run_btn.clicked.connect(self._on_run_clicked)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setProperty("role", "danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        self.open_html_btn = QPushButton("Open HTML Report")
        self.open_html_btn.setEnabled(False)
        self.open_html_btn.clicked.connect(self._open_html)
        self.open_dir_btn = QPushButton("Open Output Folder")
        self.open_dir_btn.setEnabled(False)
        self.open_dir_btn.clicked.connect(self._open_dir)
        self.status_badge = StatusBadge(
            "Status: Idle",
            bg_color="rgba(76, 86, 106, 0.3)",
            text_color=NORD_PALETTE.NORD4,
        )
        self.speed_badge = StatusBadge(
            "Speed: —",
            bg_color="rgba(94, 129, 172, 0.2)",
            text_color=NORD_PALETTE.NORD9,
        )
        self.eta_badge = StatusBadge(
            "ETA: —",
            bg_color="rgba(235, 203, 139, 0.2)",
            text_color=NORD_PALETTE.NORD13,
        )
        self.ram_badge = StatusBadge(
            "RAM: —", bg_color="rgba(143, 188, 187, 0.2)", text_color=NORD_PALETTE.NORD7
        )
        ctrl_row.addWidget(self.run_btn)
        ctrl_row.addWidget(self.stop_btn)
        ctrl_row.addWidget(self.open_html_btn)
        ctrl_row.addWidget(self.open_dir_btn)
        ctrl_row.addStretch(1)
        ctrl_row.addWidget(self.status_badge)
        ctrl_row.addWidget(self.speed_badge)
        ctrl_row.addWidget(self.eta_badge)
        ctrl_row.addWidget(self.ram_badge)
        bottom_layout.addLayout(ctrl_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFormat("Idle (0.0%)")
        bottom_layout.addWidget(self.progress)
        main_splitter.addWidget(bottom_box)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 2)
        layout.addWidget(main_splitter)

    def refresh_review(self, config: PipelineConfig) -> bool:
        session = self.state_bridge.session
        dataset_df = session.dataset_df
        errors = []
        warnings = []
        if dataset_df is None:
            errors.append("Critical: No dataset loaded.")
        else:
            if not config.dataset.target_columns:
                errors.append("Critical: No target outcomes (Y) selected.")
            if not config.dataset.feature_columns:
                errors.append("Critical: No predictor features (X) selected.")
            if not config.models.use_all_models and not config.models.selected_models:
                errors.append("Critical: No models selected for tournament.")
            for t in config.dataset.target_columns:
                if t in config.dataset.feature_columns:
                    errors.append(
                        f"Invalid: Target '{t}' is present in feature matrix."
                    )
            if config.run_sem and config.sem_configs:
                for _target_k, sc in config.sem_configs.items():
                    if sc.target and sc.target not in dataset_df.columns:
                        errors.append(
                            f"SEM Target '{sc.target}' not found in loaded dataset."
                        )

        self.val_list.clear()
        is_valid = len(errors) == 0
        for e in errors:
            self.val_list.addItem(f"❌ {e}")
        for w in warnings:
            self.val_list.addItem(f"⚠️ {w}")
        if is_valid:
            self.val_list.addItem("✅ Pre-flight verification passed successfully.")
        self.run_btn.setEnabled(
            is_valid and (self.worker is None or not self.worker.isRunning())
        )
        n_rows = len(dataset_df) if dataset_df is not None else 0
        n_cols = len(dataset_df.columns) if dataset_df is not None else 0
        tx_cnt = len(session.transformation_log)
        summary_md = f"""### Dataset State
- **Working Matrix**: {n_rows:,} observations x {n_cols:,} variables
- **Transformation Steps**: {tx_cnt} applied
- **Active Column Pool**: {len(session.active_columns)} columns
### ML Tournament & SEM
- **Targets**: {", ".join(config.dataset.target_columns) or "None"}
- **Predictors**: {len(config.dataset.feature_columns)} features
- **SEM**: {"Enabled" if config.run_sem else "Disabled"} | **Output Directory**: `{config.output_dir}`
"""
        self.review_text.setMarkdown(summary_md)
        return is_valid

    def _on_run_clicked(self) -> None:
        self.request_pipeline_run.emit()

    def start_execution(self, config: PipelineConfig) -> None:
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_html_btn.setEnabled(False)
        self.open_dir_btn.setEnabled(False)
        self.progress.setValue(0)
        self.status_badge.setText("Status: Running")
        self.log_window.clear()
        self.last_output_dir = config.output_dir
        self.worker = PipelineWorkerThread(
            config, dataset_df=self.state_bridge.session.dataset_df, parent=self
        )
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.log_signal.connect(self._on_log_message)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)
        self.execution_started.emit()
        self.worker.start()

    def _on_stop_clicked(self) -> None:
        if self.worker and self.worker.isRunning():
            self.status_badge.setText("Status: Stopping...")
            self.worker.stop_event.set()
            self.stop_btn.setEnabled(False)

    @Slot(str, int)
    def _on_log_message(self, message: str, level: int) -> None:
        color = "#ECEFF4"
        if level >= logging.ERROR:
            color = "#BF616A"
        elif level >= logging.WARNING:
            color = "#EBCB8B"
        elif level <= logging.DEBUG:
            color = "#4C566A"
        html_msg = f'<span style="color: {color};">{html.escape(message)}</span>'
        self.log_window.appendHtml(html_msg)
        scrollbar = self.log_window.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(object)
    def _on_progress(self, update: ProgressUpdate) -> None:
        scaled = int(update.percentage * 10)
        self.progress.setValue(scaled)
        self.progress.setFormat(
            f"{update.percentage:.1f}% ({update.current_step}/{update.total_steps} steps)"
        )
        context: list[str] = []
        if update.target_name:
            context.append(f"Target: {update.target_name}")
        if update.model_name:
            fold_str = (
                f" | Fold {update.fold}/{update.total_folds}" if update.fold else ""
            )
            context.append(f"Model: {update.model_name}{fold_str}")
        ctx = f" | {' | '.join(context)}" if context else ""
        line = (
            f"[Step {update.current_step}/{update.total_steps} | {update.percentage:5.1f}%] "
            f"Stage: {update.stage.value}{ctx} | {update.message}"
        )
        self._on_log_message(line, logging.INFO)
        self.speed_badge.setText(f"{update.steps_per_second:.1f} it/s")
        if update.eta_seconds is not None:
            m, s = divmod(int(update.eta_seconds), 60)
            self.eta_badge.setText(f"ETA: {m:02d}:{s:02d}")
        self.ram_badge.setText(f"RAM: {update.memory_usage_mb:.0f} MB")

    @Slot(str)
    def _on_finished(self, out_dir: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_html_btn.setEnabled(True)
        self.open_dir_btn.setEnabled(True)
        self.progress.setValue(1000)
        self.progress.setFormat("Completed (100.0%)")
        self.status_badge.setText("Status: Completed")
        self.last_output_dir = out_dir
        self.execution_finished.emit(True, out_dir)

    @Slot(str)
    def _on_error(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_badge.setText("Status: Failed")
        self._on_log_message(f"Execution Error:\n{err}", logging.ERROR)
        QMessageBox.critical(
            self,
            "Execution Error",
            f"The pipeline encountered an unhandled exception:\n\n{err}",
        )
        self.execution_finished.emit(False, err)

    def _open_html(self) -> None:
        if self.last_output_dir:
            p = Path(self.last_output_dir).resolve() / "reports" / "full_results.html"
            if p.is_file():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))

    def _open_dir(self) -> None:
        if self.last_output_dir and Path(self.last_output_dir).is_dir():
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(Path(self.last_output_dir).resolve()))
            )
