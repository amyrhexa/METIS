# gui/tabs/output_tab.py
from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config.schema import OutputConfig, PipelineConfig, ReportingConfig
from gui.state import SessionStateBridge


class OutputTab(QWidget):
    def __init__(
        self, state_bridge: SessionStateBridge, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.state_bridge = state_bridge

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)

        dest_box = QGroupBox("1. Destination Directory & Metadata")
        dest_form = QFormLayout(dest_box)
        out_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit("results")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_dir)
        out_row.addWidget(self.output_dir_edit, stretch=1)
        out_row.addWidget(self.browse_btn)
        dest_form.addRow("Destination Folder:", out_row)

        self.title_edit = QLineEdit(
            "Academic Machine Learning & Structural Equation Modeling Report"
        )
        dest_form.addRow("Report Title:", self.title_edit)

        self.diagnostic_covariate_edit = QLineEdit()
        self.diagnostic_covariate_edit.setPlaceholderText(
            "Optional column name (e.g. age) -- leave blank to disable"
        )
        dest_form.addRow(
            "Diagnostic Plot Covariate:", self.diagnostic_covariate_edit
        )

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["exploratory", "confirmatory", "benchmark", "smoke_test", "production"]
        )
        dest_form.addRow("Execution Mode:", self.mode_combo)
        layout.addWidget(dest_box)

        style_box = QGroupBox("2. Table Formatting & Visual Appearance")
        style_form = QFormLayout(style_box)
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(
            ["nord", "default", "grayscale", "colorblind-safe", "journal blue"]
        )
        style_form.addRow("Color Palette:", self.palette_combo)

        self.style_combo = QComboBox()
        self.style_combo.addItems(["publication", "compact", "striped"])
        style_form.addRow("Table Style:", self.style_combo)

        self.prec_spin = QSpinBox()
        self.prec_spin.setRange(2, 4)
        self.prec_spin.setValue(3)
        style_form.addRow("Decimal Precision:", self.prec_spin)
        layout.addWidget(style_box)

        art_box = QGroupBox("3. Report Artifacts & Figure Formats")
        art_form = QFormLayout(art_box)
        self.gen_html_cb = QCheckBox("Interactive HTML Technical Report Dossier")
        self.gen_html_cb.setChecked(True)
        self.gen_latex_cb = QCheckBox("Booktabs LaTeX Tables (.tex)")
        self.gen_latex_cb.setChecked(True)
        self.gen_csv_cb = QCheckBox("CSV Benchmark Tables")
        self.gen_csv_cb.setChecked(True)
        self.gen_manifest_cb = QCheckBox("Reproducibility JSON Manifest")
        self.gen_manifest_cb.setChecked(True)

        fmt_row = QHBoxLayout()
        self.fmt_png = QCheckBox("PNG")
        self.fmt_png.setChecked(True)
        self.fmt_pdf = QCheckBox("PDF")
        self.fmt_pdf.setChecked(True)
        self.fmt_svg = QCheckBox("SVG")
        fmt_row.addWidget(self.fmt_png)
        fmt_row.addWidget(self.fmt_pdf)
        fmt_row.addWidget(self.fmt_svg)
        fmt_row.addStretch(1)

        art_form.addRow("", self.gen_html_cb)
        art_form.addRow("", self.gen_latex_cb)
        art_form.addRow("", self.gen_csv_cb)
        art_form.addRow("", self.gen_manifest_cb)
        art_form.addRow("Figure Export Formats:", fmt_row)

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(100, 600)
        self.dpi_spin.setValue(300)
        art_form.addRow("Figure Resolution (DPI):", self.dpi_spin)
        layout.addWidget(art_box)

        met_box = QGroupBox("4. Target Evaluation Metrics")
        met_layout = QHBoxLayout(met_box)

        reg_box = QGroupBox("Regression Metrics")
        reg_v = QVBoxLayout(reg_box)
        self.m_r2 = QCheckBox("R² Score")
        self.m_rmse = QCheckBox("RMSE")
        self.m_mae = QCheckBox("MAE")
        self.m_pearson = QCheckBox("Pearson r")
        self.m_spearman = QCheckBox("Spearman rho")
        for cb in [self.m_r2, self.m_rmse, self.m_mae, self.m_pearson, self.m_spearman]:
            cb.setChecked(True)
            reg_v.addWidget(cb)
        met_layout.addWidget(reg_box)

        clf_box = QGroupBox("Classification Metrics")
        clf_v = QVBoxLayout(clf_box)
        self.m_acc = QCheckBox("Accuracy")
        self.m_bal_acc = QCheckBox("Balanced Accuracy")
        self.m_f1 = QCheckBox("F1 Macro")
        self.m_roc = QCheckBox("ROC AUC")
        self.m_mcc = QCheckBox("Matthews Corr (MCC)")
        for cb in [self.m_acc, self.m_bal_acc, self.m_f1, self.m_roc, self.m_mcc]:
            cb.setChecked(True)
            clf_v.addWidget(cb)
        met_layout.addWidget(clf_box)

        stat_box = QGroupBox("Statistical Inference")
        stat_v = QVBoxLayout(stat_box)
        self.m_boot_ci = QCheckBox("95% Non-parametric Bootstrap CI")
        self.m_paired_ci = QCheckBox("Paired Difference CI (vs #1)")
        self.m_ttest = QCheckBox("Nadeau-Bengio Resampled t-test")
        for cb in [self.m_boot_ci, self.m_paired_ci, self.m_ttest]:
            cb.setChecked(True)
            stat_v.addWidget(cb)
        met_layout.addWidget(stat_box)
        layout.addWidget(met_box)

        sec_box = QGroupBox("5. Section Inclusions")
        sec_layout = QHBoxLayout(sec_box)

        ml_sec_box = QGroupBox("ML Sections")
        ml_sec_v = QVBoxLayout(ml_sec_box)
        self.sec_leaderboard = QCheckBox("Leaderboard Table")
        self.sec_comparison = QCheckBox("Model Comparison Barplot")
        self.sec_calibration = QCheckBox("Calibration Plot")
        self.sec_residual = QCheckBox("Residual Diagnostics")
        self.sec_importance = QCheckBox("Feature Importance")
        self.sec_shap = QCheckBox("SHAP Visualizations")
        self.sec_learning = QCheckBox("Learning Curve")
        for cb in [
            self.sec_leaderboard,
            self.sec_comparison,
            self.sec_calibration,
            self.sec_residual,
            self.sec_importance,
            self.sec_shap,
            self.sec_learning,
        ]:
            cb.setChecked(True)
            ml_sec_v.addWidget(cb)
        sec_layout.addWidget(ml_sec_box)

        sem_sec_box = QGroupBox("SEM Sections")
        sem_sec_v = QVBoxLayout(sem_sec_box)
        self.sec_sem_paths = QCheckBox("Structural Paths Table")
        self.sec_sem_med = QCheckBox("Mediation Effects Table")
        self.sec_sem_fit = QCheckBox("Model Fit Benchmarks")
        self.sec_sem_diagram = QCheckBox("SEM Path Diagram")
        self.sec_sem_forest = QCheckBox("Path Forest Plot")
        self.sec_sem_cov = QCheckBox("Covariance Heatmap")
        for cb in [
            self.sec_sem_paths,
            self.sec_sem_med,
            self.sec_sem_fit,
            self.sec_sem_diagram,
            self.sec_sem_forest,
            self.sec_sem_cov,
        ]:
            cb.setChecked(True)
            sem_sec_v.addWidget(cb)
        sec_layout.addWidget(sem_sec_box)
        layout.addWidget(sec_box)

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def set_interactive(self, enabled: bool) -> None:
        self.output_dir_edit.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.title_edit.setEnabled(enabled)
        self.mode_combo.setEnabled(enabled)
        self.palette_combo.setEnabled(enabled)
        self.style_combo.setEnabled(enabled)
        self.prec_spin.setEnabled(enabled)
        self.gen_html_cb.setEnabled(enabled)
        self.gen_latex_cb.setEnabled(enabled)
        self.gen_csv_cb.setEnabled(enabled)
        self.gen_manifest_cb.setEnabled(enabled)
        self.fmt_png.setEnabled(enabled)
        self.fmt_pdf.setEnabled(enabled)
        self.fmt_svg.setEnabled(enabled)
        self.dpi_spin.setEnabled(enabled)

        for cb in [
            self.m_r2,
            self.m_rmse,
            self.m_mae,
            self.m_pearson,
            self.m_spearman,
            self.m_acc,
            self.m_bal_acc,
            self.m_f1,
            self.m_roc,
            self.m_mcc,
            self.m_boot_ci,
            self.m_paired_ci,
            self.m_ttest,
            self.sec_leaderboard,
            self.sec_comparison,
            self.sec_calibration,
            self.sec_residual,
            self.sec_importance,
            self.sec_shap,
            self.sec_learning,
            self.sec_sem_paths,
            self.sec_sem_med,
            self.sec_sem_fit,
            self.sec_sem_diagram,
            self.sec_sem_forest,
            self.sec_sem_cov,
        ]:
            cb.setEnabled(enabled)

    def _browse_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_dir_edit.text()
        )
        if d:
            self.output_dir_edit.setText(d)

    def set_from_config(self, config: PipelineConfig) -> None:
        """Restores output format, directory, metadata, and section inclusions from a PipelineConfig."""
        out_cfg = config.output_config
        rep_cfg = config.reporting

        dest_dir = config.output_dir or out_cfg.output_dir or "results"
        self.output_dir_edit.setText(dest_dir)
        self.title_edit.setText(out_cfg.report_title or rep_cfg.title)
        self.diagnostic_covariate_edit.setText(
            out_cfg.diagnostic_covariate_column or ""
        )

        mode_val = config.mode or out_cfg.mode
        m_idx = self.mode_combo.findText(mode_val)
        if m_idx >= 0:
            self.mode_combo.setCurrentIndex(m_idx)

        pal_val = out_cfg.table_palette or rep_cfg.color_palette
        p_idx = self.palette_combo.findText(pal_val)
        if p_idx >= 0:
            self.palette_combo.setCurrentIndex(p_idx)

        s_idx = self.style_combo.findText(out_cfg.table_style)
        if s_idx >= 0:
            self.style_combo.setCurrentIndex(s_idx)

        self.prec_spin.setValue(out_cfg.decimal_precision)

        self.gen_html_cb.setChecked(out_cfg.generate_html)
        self.gen_latex_cb.setChecked(out_cfg.generate_latex)
        self.gen_csv_cb.setChecked(out_cfg.generate_csv)
        self.gen_manifest_cb.setChecked(out_cfg.generate_json_manifest)

        formats = out_cfg.figure_formats or ["png", "pdf"]
        self.fmt_png.setChecked("png" in formats)
        self.fmt_pdf.setChecked("pdf" in formats)
        self.fmt_svg.setChecked("svg" in formats)
        self.dpi_spin.setValue(out_cfg.figure_dpi)

        m = out_cfg.include_metrics or {}
        self.m_r2.setChecked(m.get("r2", True))
        self.m_rmse.setChecked(m.get("rmse", True))
        self.m_mae.setChecked(m.get("mae", True))
        self.m_pearson.setChecked(m.get("pearson_r", True))
        self.m_spearman.setChecked(m.get("spearman_rho", True))
        self.m_acc.setChecked(m.get("accuracy", True))
        self.m_bal_acc.setChecked(m.get("balanced_accuracy", True))
        self.m_f1.setChecked(m.get("f1_macro", True))
        self.m_roc.setChecked(m.get("roc_auc", True))
        self.m_mcc.setChecked(m.get("mcc", True))
        self.m_boot_ci.setChecked(m.get("bootstrap_ci", True))
        self.m_paired_ci.setChecked(m.get("paired_difference", True))
        self.m_ttest.setChecked(m.get("corrected_ttest", True))

        ml_s = out_cfg.include_ml_sections or {}
        self.sec_leaderboard.setChecked(ml_s.get("leaderboard", True))
        self.sec_comparison.setChecked(ml_s.get("model_comparison", True))
        self.sec_calibration.setChecked(ml_s.get("calibration", True))
        self.sec_residual.setChecked(ml_s.get("regression_diagnostics", True))
        self.sec_importance.setChecked(ml_s.get("feature_importance", True))
        self.sec_shap.setChecked(ml_s.get("shap_beeswarm", True))
        self.sec_learning.setChecked(ml_s.get("learning_curve", True))

        sem_s = out_cfg.include_sem_sections or {}
        self.sec_sem_paths.setChecked(sem_s.get("structural_paths", True))
        self.sec_sem_med.setChecked(sem_s.get("mediation_effects", True))
        self.sec_sem_fit.setChecked(sem_s.get("fit_indices", True))
        self.sec_sem_diagram.setChecked(sem_s.get("path_diagram", True))
        self.sec_sem_forest.setChecked(sem_s.get("forest_plot", True))
        self.sec_sem_cov.setChecked(sem_s.get("covariance_heatmap", True))

    def get_output_config(self) -> OutputConfig:
        formats = []
        if self.fmt_png.isChecked():
            formats.append("png")
        if self.fmt_pdf.isChecked():
            formats.append("pdf")
        if self.fmt_svg.isChecked():
            formats.append("svg")

        clean_out_dir = self.output_dir_edit.text().strip() or "results"

        return OutputConfig(
            output_dir=str(Path(clean_out_dir)),
            report_title=self.title_edit.text().strip(),
            diagnostic_covariate_column=(
                self.diagnostic_covariate_edit.text().strip() or None
            ),
            mode=self.mode_combo.currentText(),
            table_palette=self.palette_combo.currentText(),
            table_style=self.style_combo.currentText(),
            decimal_precision=self.prec_spin.value(),
            generate_html=self.gen_html_cb.isChecked(),
            generate_latex=self.gen_latex_cb.isChecked(),
            generate_csv=self.gen_csv_cb.isChecked(),
            generate_json_manifest=self.gen_manifest_cb.isChecked(),
            figure_dpi=self.dpi_spin.value(),
            figure_formats=formats or ["png", "pdf"],
            include_metrics={
                "r2": self.m_r2.isChecked(),
                "rmse": self.m_rmse.isChecked(),
                "mae": self.m_mae.isChecked(),
                "pearson_r": self.m_pearson.isChecked(),
                "spearman_rho": self.m_spearman.isChecked(),
                "accuracy": self.m_acc.isChecked(),
                "balanced_accuracy": self.m_bal_acc.isChecked(),
                "f1_macro": self.m_f1.isChecked(),
                "roc_auc": self.m_roc.isChecked(),
                "mcc": self.m_mcc.isChecked(),
                "bootstrap_ci": self.m_boot_ci.isChecked(),
                "paired_difference": self.m_paired_ci.isChecked(),
                "corrected_ttest": self.m_ttest.isChecked(),
            },
            include_ml_sections={
                "leaderboard": self.sec_leaderboard.isChecked(),
                "model_comparison": self.sec_comparison.isChecked(),
                "calibration": self.sec_calibration.isChecked(),
                "regression_diagnostics": self.sec_residual.isChecked(),
                "feature_importance": self.sec_importance.isChecked(),
                "shap_beeswarm": self.sec_shap.isChecked(),
                "learning_curve": self.sec_learning.isChecked(),
            },
            include_sem_sections={
                "structural_paths": self.sec_sem_paths.isChecked(),
                "mediation_effects": self.sec_sem_med.isChecked(),
                "fit_indices": self.sec_sem_fit.isChecked(),
                "path_diagram": self.sec_sem_diagram.isChecked(),
                "forest_plot": self.sec_sem_forest.isChecked(),
                "covariance_heatmap": self.sec_sem_cov.isChecked(),
            },
        )

    def get_reporting_config(self) -> ReportingConfig:
        return ReportingConfig(
            generate_html=self.gen_html_cb.isChecked(),
            generate_latex=self.gen_latex_cb.isChecked(),
            generate_csv_tables=self.gen_csv_cb.isChecked(),
            generate_json_manifest=self.gen_manifest_cb.isChecked(),
            title=self.title_edit.text().strip(),
            dpi=self.dpi_spin.value(),
            color_palette=self.palette_combo.currentText(),
        )
