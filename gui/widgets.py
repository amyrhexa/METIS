# gui/widgets.py
"""Standardized reusable UI widgets including the universal ColumnSearchSelector."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.theme import NORD_PALETTE


class StatusBadge(QLabel):
    """Visual status pill badge."""

    def __init__(
        self,
        text: str,
        bg_color: str = "rgba(136, 192, 208, 0.2)",
        text_color: str = NORD_PALETTE.NORD8,
        border_color: str = NORD_PALETTE.NORD8,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 9px;
                padding: 2px 8px;
                font-size: 11px;
                font-weight: 700;
            }}
            """
        )


class StatCard(QFrame):
    """Metric KPI card displaying a value, title, and descriptive footnote."""

    def __init__(
        self,
        title: str,
        value: str = "—",
        subtitle: str = "",
        accent_color: str = NORD_PALETTE.NORD8,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            f"""
            StatCard {{
                background-color: {NORD_PALETTE.NORD1};
                border: 1px solid {NORD_PALETTE.NORD2};
                border-radius: 6px;
            }}
            StatCard:hover {{
                border-color: {NORD_PALETTE.NORD9};
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(
            f"color: {NORD_PALETTE.NORD4}; font-size: 10.5px; font-weight: 700; letter-spacing: 0.5px;"
        )

        self.value_lbl = QLabel(value)
        self.value_lbl.setStyleSheet(
            f"color: {accent_color}; font-size: 18px; font-weight: 800; padding: 1px 0;"
        )

        self.sub_lbl = QLabel(subtitle)
        self.sub_lbl.setStyleSheet(f"color: {NORD_PALETTE.NORD4}; font-size: 11px;")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        if subtitle:
            layout.addWidget(self.sub_lbl)

    def set_value(self, value: str, subtitle: str | None = None) -> None:
        self.value_lbl.setText(value)
        if subtitle is not None:
            self.sub_lbl.setText(subtitle)


class SearchLineEdit(QLineEdit):
    """Instant search input field with clear button."""

    def __init__(
        self, placeholder: str = "Search...", parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setPlaceholderText(placeholder)
        self.setClearButtonEnabled(True)
        self.setStyleSheet(
            f"""
            QLineEdit {{
                background-color: {NORD_PALETTE.NORD0};
                border: 1px solid {NORD_PALETTE.NORD2};
                border-radius: 5px;
                padding: 4px 8px;
                color: {NORD_PALETTE.NORD6};
            }}
            QLineEdit:focus {{
                border-color: {NORD_PALETTE.NORD8};
            }}
            """
        )


class ColumnSearchSelector(QWidget):
    """Universal searchable checklist column selector component supporting click, Shift/Ctrl, and batch actions."""

    selection_changed = Signal(list)

    def __init__(
        self,
        title: str = "Select Columns",
        badge_color: str = NORD_PALETTE.NORD8,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.all_columns: list[str] = []
        self.excluded_columns: set[str] = set()
        self.column_dtypes: dict[str, str] = {}
        self._updating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        hdr_row = QHBoxLayout()
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            f"font-weight: 700; color: {NORD_PALETTE.NORD5}; font-size: 12px;"
        )
        self.badge = StatusBadge(
            "0 Selected", text_color=badge_color, border_color=badge_color
        )
        hdr_row.addWidget(self.title_lbl)
        hdr_row.addWidget(self.badge)
        hdr_row.addStretch(1)
        layout.addLayout(hdr_row)

        self.search_edit = SearchLineEdit("🔍 Filter columns by name...")
        self.search_edit.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_edit)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._select_all)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self._clear_all)
        self.num_only_btn = QPushButton("Select Numeric Only")
        self.num_only_btn.clicked.connect(self._select_numeric_only)

        btn_row.addWidget(self.select_all_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addWidget(self.num_only_btn)
        layout.addLayout(btn_row)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_widget.itemChanged.connect(self._on_item_changed)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def set_columns(
        self,
        columns: list[str],
        selected: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
    ) -> None:
        self.all_columns = list(columns)
        self.column_dtypes = dtypes or {}
        selected_set = set(selected if selected is not None else columns)

        self._updating = True
        self.list_widget.clear()
        for col in self.all_columns:
            if col in self.excluded_columns:
                continue
            item = QListWidgetItem()
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            dtype_str = (
                f" [{self.column_dtypes[col]}]" if col in self.column_dtypes else ""
            )
            item.setText(f"{col}{dtype_str}")
            item.setData(Qt.ItemDataRole.UserRole, col)
            is_sel = col in selected_set
            item.setCheckState(
                Qt.CheckState.Checked if is_sel else Qt.CheckState.Unchecked
            )
            item.setSelected(is_sel)
            self.list_widget.addItem(item)
        self._updating = False

        self._update_badge()
        self._filter_list(self.search_edit.text())

    def set_excluded_columns(self, excluded: list[str]) -> None:
        self.excluded_columns = set(excluded)
        current_sel = set(self.get_selected_columns())
        self.set_columns(
            self.all_columns, selected=list(current_sel), dtypes=self.column_dtypes
        )

    def get_selected_columns(self) -> list[str]:
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                col_name = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
                selected.append(col_name)
        return selected

    def set_selected_columns(self, selected: list[str]) -> None:
        sel_set = set(selected)
        self._updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            col_name = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            is_sel = col_name in sel_set
            item.setCheckState(
                Qt.CheckState.Checked if is_sel else Qt.CheckState.Unchecked
            )
            item.setSelected(is_sel)
        self._updating = False
        self._update_badge()

    def _filter_list(self, query: str) -> None:
        q = query.strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            col_name = str(item.data(Qt.ItemDataRole.UserRole) or item.text()).lower()
            item.setHidden(bool(q and q not in col_name))

    def _select_all(self) -> None:
        self._updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)
                item.setSelected(True)
        self._updating = False
        self._update_badge()
        self.selection_changed.emit(self.get_selected_columns())

    def _clear_all(self) -> None:
        self._updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)
                item.setSelected(False)
        self._updating = False
        self._update_badge()
        self.selection_changed.emit(self.get_selected_columns())

    def _select_numeric_only(self) -> None:
        self._updating = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            col = str(item.data(Qt.ItemDataRole.UserRole) or item.text())
            dt = self.column_dtypes.get(col, "").lower()
            is_num = any(t in dt for t in ["int", "float", "double", "num"])
            if not item.isHidden():
                item.setCheckState(
                    Qt.CheckState.Checked if is_num else Qt.CheckState.Unchecked
                )
                item.setSelected(is_num)
        self._updating = False
        self._update_badge()
        self.selection_changed.emit(self.get_selected_columns())

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        self._updating = True
        item.setSelected(item.checkState() == Qt.CheckState.Checked)
        self._updating = False
        self._update_badge()
        self.selection_changed.emit(self.get_selected_columns())

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if self._updating:
            return
        self._updating = True
        new_state = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new_state)
        item.setSelected(new_state == Qt.CheckState.Checked)
        self._updating = False
        self._update_badge()
        self.selection_changed.emit(self.get_selected_columns())

    def _update_badge(self) -> None:
        sel_cnt = len(self.get_selected_columns())
        total = self.list_widget.count()
        self.badge.setText(f"{sel_cnt}/{total} Selected")

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        self.search_edit.setEnabled(enabled)
        self.select_all_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)
        self.num_only_btn.setEnabled(enabled)
        self.list_widget.setEnabled(enabled)
