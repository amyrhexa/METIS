# gui/tabs/sem_tab.py
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config.schema import PipelineConfig, SEMTargetConfig
from core.sem import build_sem_model_syntax, build_sem_variable_map
from gui.state import SessionStateBridge
from gui.theme import NORD_PALETTE
from gui.widgets import ColumnSearchSelector


class SEMTab(QWidget):
    def __init__(
        self, state_bridge: SessionStateBridge, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.state_bridge = state_bridge
        self.latent_defs: dict[str, list[str]] = {}
        self._is_syncing = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.enable_sem_cb = QCheckBox(
            "Enable Structural Equation Modeling (SEM) & Mediation Path Analysis"
        )
        self.enable_sem_cb.setChecked(True)
        self.enable_sem_cb.toggled.connect(self._toggle_sem_enabled)
        layout.addWidget(self.enable_sem_cb)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.var_box = QGroupBox("Covariance Architecture & Variable Topology")
        var_layout = QVBoxLayout(self.var_box)

        form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        form.addRow("Endogenous Target (Y):", self.target_combo)
        var_layout.addLayout(form)

        self.pred_selector = ColumnSearchSelector(
            title="Direct Predictors (X -> Y)", badge_color=NORD_PALETTE.NORD7
        )
        self.pred_selector.selection_changed.connect(self._on_selector_changed)
        var_layout.addWidget(self.pred_selector)

        self.med_selector = ColumnSearchSelector(
            title="Mediators (X -> M -> Y)", badge_color=NORD_PALETTE.NORD12
        )
        self.med_selector.selection_changed.connect(self._on_selector_changed)
        var_layout.addWidget(self.med_selector)

        self.cov_selector = ColumnSearchSelector(
            title="Covariates (Confounders)", badge_color=NORD_PALETTE.NORD15
        )
        self.cov_selector.selection_changed.connect(self._on_selector_changed)
        var_layout.addWidget(self.cov_selector)

        lat_box = QGroupBox("Latent Construct Builder")
        lat_layout = QVBoxLayout(lat_box)
        lat_form = QFormLayout()
        self.lat_name_edit = QLineEdit()
        self.lat_name_edit.setPlaceholderText("e.g., LatentCognition")
        lat_form.addRow("Construct Name:", self.lat_name_edit)
        lat_layout.addLayout(lat_form)

        self.lat_ind_selector = ColumnSearchSelector(title="Indicator Variables")
        self.lat_ind_selector.selection_changed.connect(self._save_state_to_session)
        lat_layout.addWidget(self.lat_ind_selector)

        lat_btn_row = QHBoxLayout()
        self.add_latent_btn = QPushButton("Add Construct")
        self.add_latent_btn.clicked.connect(self._add_latent_construct)
        self.remove_latent_btn = QPushButton("Remove Selected")
        self.remove_latent_btn.clicked.connect(self._remove_latent_construct)
        lat_btn_row.addWidget(self.add_latent_btn)
        lat_btn_row.addWidget(self.remove_latent_btn)
        lat_layout.addLayout(lat_btn_row)

        self.lat_list = QListWidget()
        lat_layout.addWidget(self.lat_list)
        var_layout.addWidget(lat_box)

        splitter.addWidget(self.var_box)

        self.opt_box = QGroupBox("Estimation Parameters & Lavaan Model Syntax")
        opt_layout = QVBoxLayout(self.opt_box)

        opt_form = QFormLayout()
        self.estimator_combo = QComboBox()
        self.estimator_combo.addItems(["ML", "GLS", "WLS", "DWLS", "MLW"])
        self.estimator_combo.currentIndexChanged.connect(self._save_state_to_session)
        opt_form.addRow("Estimator Engine:", self.estimator_combo)

        self.missing_combo = QComboBox()
        self.missing_combo.addItems(["fiml", "listwise"])
        self.missing_combo.currentIndexChanged.connect(self._save_state_to_session)
        opt_form.addRow("Missing Data Method:", self.missing_combo)

        self.bootstrap_cb = QCheckBox("Bootstrap Indirect Effects")
        self.bootstrap_cb.setChecked(True)
        self.bootstrap_cb.toggled.connect(self._save_state_to_session)
        self.boot_iter_spin = QSpinBox()
        self.boot_iter_spin.setRange(100, 10000)
        self.boot_iter_spin.setValue(2000)
        self.boot_iter_spin.valueChanged.connect(self._save_state_to_session)
        opt_form.addRow("", self.bootstrap_cb)
        opt_form.addRow("Bootstrap Iterations:", self.boot_iter_spin)

        self.standardize_cb = QCheckBox("Standardize Continuous Variates")
        self.standardize_cb.setChecked(True)
        self.standardize_cb.toggled.connect(self._save_state_to_session)
        self.robust_se_cb = QCheckBox("Compute Robust Standard Errors")
        self.robust_se_cb.toggled.connect(self._save_state_to_session)
        opt_form.addRow("", self.standardize_cb)
        opt_form.addRow("", self.robust_se_cb)
        opt_layout.addLayout(opt_form)

        syn_hdr = QHBoxLayout()
        syn_hdr.addWidget(QLabel("Lavaan Syntax Specification:"))
        self.custom_syntax_cb = QCheckBox("Manual Syntax Mode")
        self.custom_syntax_cb.toggled.connect(self._toggle_custom_syntax)
        syn_hdr.addStretch(1)
        syn_hdr.addWidget(self.custom_syntax_cb)
        opt_layout.addLayout(syn_hdr)

        self.syntax_edit = QPlainTextEdit()
        self.syntax_edit.setReadOnly(True)
        self.syntax_edit.textChanged.connect(self._save_state_to_session)
        self.syntax_edit.setStyleSheet(
            f"font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background-color: {NORD_PALETTE.NORD0}; border: 1px solid {NORD_PALETTE.NORD2}; border-radius: 5px; padding: 6px;"
        )
        opt_layout.addWidget(self.syntax_edit)
        splitter.addWidget(self.opt_box)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def set_interactive(self, enabled: bool) -> None:
        self.enable_sem_cb.setEnabled(True)
        is_active = enabled and self.enable_sem_cb.isChecked()

        self.var_box.setEnabled(is_active)
        self.opt_box.setEnabled(is_active)
        self.target_combo.setEnabled(is_active)
        self.pred_selector.setEnabled(is_active)
        self.med_selector.setEnabled(is_active)
        self.cov_selector.setEnabled(is_active)
        self.lat_name_edit.setEnabled(is_active)
        self.lat_ind_selector.setEnabled(is_active)
        self.add_latent_btn.setEnabled(is_active)
        self.remove_latent_btn.setEnabled(is_active)
        self.lat_list.setEnabled(is_active)
        self.estimator_combo.setEnabled(is_active)
        self.missing_combo.setEnabled(is_active)
        self.bootstrap_cb.setEnabled(is_active)
        self.boot_iter_spin.setEnabled(is_active)
        self.standardize_cb.setEnabled(is_active)
        self.robust_se_cb.setEnabled(is_active)
        self.custom_syntax_cb.setEnabled(is_active)
        self.syntax_edit.setEnabled(is_active and self.custom_syntax_cb.isChecked())

    def sync_variables(self, targets: list[str], features: list[str]) -> None:
        if self._is_syncing:
            return
        self._is_syncing = True

        st = self.state_bridge.session.sem_ui_state

        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(targets)
        saved_target = st.get("target", "")
        if saved_target in targets:
            self.target_combo.setCurrentText(saved_target)
        elif targets:
            self.target_combo.setCurrentIndex(0)
        self.target_combo.blockSignals(False)

        dtypes = {}
        if self.state_bridge.session.dataset_df is not None:
            dtypes = {
                c: str(self.state_bridge.session.dataset_df[c].dtype) for c in features
            }

        saved_preds = st.get("predictors", [])
        saved_meds = st.get("mediators", [])
        saved_covs = st.get("covariates", [])

        preds_to_use = (
            [c for c in saved_preds if c in features]
            if saved_preds
            else (features[:4] if len(features) >= 4 else features)
        )
        meds_to_use = (
            [c for c in saved_meds if c in features]
            if saved_meds
            else (features[4:5] if len(features) >= 5 else [])
        )
        covs_to_use = [c for c in saved_covs if c in features] if saved_covs else []

        self.pred_selector.set_columns(features, selected=preds_to_use, dtypes=dtypes)
        self.med_selector.set_columns(features, selected=meds_to_use, dtypes=dtypes)
        self.cov_selector.set_columns(features, selected=covs_to_use, dtypes=dtypes)
        self.lat_ind_selector.set_columns(features, selected=[], dtypes=dtypes)

        self.estimator_combo.setCurrentText(st.get("estimator", "ML"))
        self.missing_combo.setCurrentText(st.get("missing", "fiml"))
        self.bootstrap_cb.setChecked(st.get("bootstrap", True))
        self.boot_iter_spin.setValue(st.get("boot_iter", 2000))
        self.standardize_cb.setChecked(st.get("standardize", True))
        self.robust_se_cb.setChecked(st.get("robust_se", False))

        self.latent_defs = dict(st.get("latent_defs", {}))
        self.lat_list.clear()
        for l_name, l_inds in self.latent_defs.items():
            self.lat_list.addItem(f"{l_name} =~ {' + '.join(l_inds)}")

        is_custom = st.get("custom_syntax", False)
        self.custom_syntax_cb.setChecked(is_custom)
        if is_custom:
            self.syntax_edit.setPlainText(st.get("syntax", ""))
        else:
            self._update_syntax()

        self._is_syncing = False

    def set_from_config(self, config: PipelineConfig) -> None:
        """Restores complete SEM model topology and estimation settings from a PipelineConfig."""
        self._is_syncing = True
        self.enable_sem_cb.setChecked(config.run_sem or bool(config.sem_configs))

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

        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems(active_cols)

        if config.sem_configs:
            target_key = next(iter(config.sem_configs.keys()))
            sem_cfg = config.sem_configs[target_key]

            if sem_cfg.target in active_cols:
                self.target_combo.setCurrentText(sem_cfg.target)

            self.pred_selector.set_columns(
                active_cols, selected=sem_cfg.feature_columns, dtypes=dtypes
            )
            self.med_selector.set_columns(
                active_cols, selected=sem_cfg.mediators, dtypes=dtypes
            )
            self.cov_selector.set_columns(
                active_cols, selected=sem_cfg.covariates, dtypes=dtypes
            )
            self.lat_ind_selector.set_columns(active_cols, selected=[], dtypes=dtypes)

            self.latent_defs = dict(sem_cfg.latent_variables)
            self.lat_list.clear()
            for l_name, l_inds in self.latent_defs.items():
                self.lat_list.addItem(f"{l_name} =~ {' + '.join(l_inds)}")

            self.estimator_combo.setCurrentText(sem_cfg.estimator)
            self.missing_combo.setCurrentText(sem_cfg.missing_data_method)
            self.bootstrap_cb.setChecked(sem_cfg.bootstrap)
            self.boot_iter_spin.setValue(sem_cfg.bootstrap_n)
            self.standardize_cb.setChecked(sem_cfg.standardize)
            self.robust_se_cb.setChecked(sem_cfg.robust_se)

            is_custom = bool(
                sem_cfg.custom_model_syntax and sem_cfg.custom_model_syntax.strip()
            )
            self.custom_syntax_cb.setChecked(is_custom)
            if is_custom:
                self.syntax_edit.setPlainText(sem_cfg.custom_model_syntax)
            else:
                self._is_syncing = False
                self._update_syntax()
                self._is_syncing = True

        self.target_combo.blockSignals(False)
        self._is_syncing = False
        self._save_state_to_session()

    def _on_target_changed(self) -> None:
        if not self._is_syncing:
            self._update_syntax()

    def _on_selector_changed(self) -> None:
        if not self._is_syncing:
            self._update_syntax()
            self._save_state_to_session()

    def _toggle_custom_syntax(self, checked: bool) -> None:
        self.syntax_edit.setReadOnly(not checked)
        if not checked:
            self._update_syntax()
        self._save_state_to_session()

    def _toggle_sem_enabled(self, checked: bool) -> None:
        self.set_interactive(checked)
        self._save_state_to_session()

    def _add_latent_construct(self) -> None:
        name = self.lat_name_edit.text().strip()
        inds = self.lat_ind_selector.get_selected_columns()
        if not name or len(inds) < 2:
            QMessageBox.warning(
                self,
                "Invalid Latent Construct",
                "Enter a name and select at least 2 indicators.",
            )
            return
        self.latent_defs[name] = inds
        self.lat_list.addItem(f"{name} =~ {' + '.join(inds)}")
        self.lat_name_edit.clear()
        self._update_syntax()

    def _remove_latent_construct(self) -> None:
        row = self.lat_list.currentRow()
        if row >= 0:
            item = self.lat_list.takeItem(row)
            if item is not None:
                name = item.text().split("=~")[0].strip()
                self.latent_defs.pop(name, None)
                self._update_syntax()

    def _update_syntax(self) -> None:
        if self._is_syncing or self.custom_syntax_cb.isChecked():
            return
        target = self.target_combo.currentText()
        if not target:
            self.syntax_edit.setPlainText(
                "# Select a target outcome to generate SEM structural paths."
            )
            return

        preds = self.pred_selector.get_selected_columns()
        meds = self.med_selector.get_selected_columns()
        covs = self.cov_selector.get_selected_columns()

        config = SEMTargetConfig(
            target=target,
            feature_columns=preds,
            mediators=meds,
            covariates=covs,
            latent_variables=self.latent_defs,
        )
        try:
            variable_map = build_sem_variable_map(config)
            syntax = build_sem_model_syntax(config, variable_map)
            self.syntax_edit.setPlainText(syntax)
        except Exception as exc:
            self.syntax_edit.setPlainText(f"# Syntax warning: {exc}")

        self._save_state_to_session()

    def _save_state_to_session(self) -> None:
        if self._is_syncing:
            return
        st = self.state_bridge.session.sem_ui_state
        st["enabled"] = self.enable_sem_cb.isChecked()
        st["target"] = self.target_combo.currentText()
        st["predictors"] = self.pred_selector.get_selected_columns()
        st["mediators"] = self.med_selector.get_selected_columns()
        st["covariates"] = self.cov_selector.get_selected_columns()
        st["latent_defs"] = dict(self.latent_defs)
        st["estimator"] = self.estimator_combo.currentText()
        st["missing"] = self.missing_combo.currentText()
        st["bootstrap"] = self.bootstrap_cb.isChecked()
        st["boot_iter"] = self.boot_iter_spin.value()
        st["standardize"] = self.standardize_cb.isChecked()
        st["robust_se"] = self.robust_se_cb.isChecked()
        st["custom_syntax"] = self.custom_syntax_cb.isChecked()
        st["syntax"] = self.syntax_edit.toPlainText()

    def get_sem_configs(self) -> dict[str, SEMTargetConfig]:
        if not self.enable_sem_cb.isChecked():
            return {}
        target = self.target_combo.currentText()
        if not target:
            return {}

        cfg = SEMTargetConfig(
            target=target,
            enabled=True,
            feature_columns=self.pred_selector.get_selected_columns(),
            mediators=self.med_selector.get_selected_columns(),
            covariates=self.cov_selector.get_selected_columns(),
            latent_variables=dict(self.latent_defs),
            custom_model_syntax=self.syntax_edit.toPlainText().strip()
            if self.custom_syntax_cb.isChecked()
            else None,
            missing_data_method=self.missing_combo.currentText(),
            estimator=self.estimator_combo.currentText(),
            bootstrap=self.bootstrap_cb.isChecked(),
            bootstrap_n=self.boot_iter_spin.value(),
            standardize=self.standardize_cb.isChecked(),
            robust_se=self.robust_se_cb.isChecked(),
        )
        return {target: cfg}
