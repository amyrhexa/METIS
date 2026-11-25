# gui/theme.py
"""Centralized Nord Dark Design System and Stylesheet for PySide6."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NordColors:
    NORD0: str = "#2E3440"
    NORD1: str = "#3B4252"
    NORD2: str = "#434C5E"
    NORD3: str = "#4C566A"
    NORD4: str = "#D8DEE9"
    NORD5: str = "#E5E9F0"
    NORD6: str = "#ECEFF4"
    NORD7: str = "#8FBCBB"
    NORD8: str = "#88C0D0"
    NORD9: str = "#81A1C1"
    NORD10: str = "#5E81AC"
    NORD11: str = "#BF616A"
    NORD12: str = "#D08770"
    NORD13: str = "#EBCB8B"
    NORD14: str = "#A3BE8C"
    NORD15: str = "#B48EAD"


NORD_PALETTE = NordColors()


def get_application_stylesheet() -> str:
    """Return the complete application-wide QSS stylesheet using Inter."""
    return f"""
    * {{
        font-family: "Inter", "Inter Variable", "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 13px;
        color: {NORD_PALETTE.NORD6};
        outline: none;
    }}

    QMainWindow, QWidget {{
        background-color: {NORD_PALETTE.NORD0};
    }}

    QGroupBox {{
        background-color: {NORD_PALETTE.NORD1};
        border: 1px solid {NORD_PALETTE.NORD2};
        border-radius: 6px;
        margin-top: 14px;
        padding: 14px 10px 10px 10px;
        font-weight: 700;
        font-size: 12.5px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 4px;
        color: {NORD_PALETTE.NORD8};
    }}

    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background-color: {NORD_PALETTE.NORD0};
        border: 1px solid {NORD_PALETTE.NORD2};
        border-radius: 5px;
        padding: 5px 8px;
        color: {NORD_PALETTE.NORD6};
        selection-background-color: {NORD_PALETTE.NORD10};
        selection-color: {NORD_PALETTE.NORD6};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border: 1px solid {NORD_PALETTE.NORD8};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD3};
        border-color: {NORD_PALETTE.NORD2};
    }}

    QComboBox {{
        padding-right: 18px;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 0px;
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {NORD_PALETTE.NORD1};
        border: 1px solid {NORD_PALETTE.NORD2};
        selection-background-color: {NORD_PALETTE.NORD10};
        selection-color: {NORD_PALETTE.NORD6};
        padding: 3px;
        border-radius: 5px;
    }}

    QPushButton {{
        background-color: {NORD_PALETTE.NORD2};
        border: 1px solid {NORD_PALETTE.NORD3};
        border-radius: 5px;
        padding: 6px 14px;
        font-weight: 600;
        color: {NORD_PALETTE.NORD5};
    }}
    QPushButton:hover {{
        background-color: {NORD_PALETTE.NORD3};
        border-color: {NORD_PALETTE.NORD8};
        color: {NORD_PALETTE.NORD6};
    }}
    QPushButton:pressed {{
        background-color: {NORD_PALETTE.NORD1};
    }}
    QPushButton:disabled {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD3};
        border-color: {NORD_PALETTE.NORD2};
    }}

    QPushButton[role="primary"] {{
        background-color: {NORD_PALETTE.NORD10};
        border: 1px solid {NORD_PALETTE.NORD9};
        color: {NORD_PALETTE.NORD6};
        font-weight: 700;
    }}
    QPushButton[role="primary"]:hover {{
        background-color: {NORD_PALETTE.NORD9};
        border-color: {NORD_PALETTE.NORD8};
    }}
    QPushButton[role="primary"]:pressed {{
        background-color: {NORD_PALETTE.NORD10};
    }}

    QPushButton[role="danger"] {{
        background-color: {NORD_PALETTE.NORD11};
        border: 1px solid {NORD_PALETTE.NORD12};
        color: {NORD_PALETTE.NORD6};
        font-weight: 700;
    }}
    QPushButton[role="danger"]:hover {{
        background-color: {NORD_PALETTE.NORD12};
    }}

    QCheckBox {{
        spacing: 7px;
        color: {NORD_PALETTE.NORD5};
        font-weight: 500;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border-radius: 4px;
        border: 1px solid {NORD_PALETTE.NORD3};
        background-color: {NORD_PALETTE.NORD0};
    }}
    QCheckBox::indicator:hover {{
        border-color: {NORD_PALETTE.NORD8};
    }}
    QCheckBox::indicator:checked {{
        background-color: {NORD_PALETTE.NORD10};
        border-color: {NORD_PALETTE.NORD8};
    }}
    QCheckBox::indicator:checked:disabled {{
        background-color: {NORD_PALETTE.NORD3};
        border-color: {NORD_PALETTE.NORD2};
    }}
    QCheckBox:disabled {{
        color: {NORD_PALETTE.NORD3};
    }}

    QTabWidget::pane {{
        border: 1px solid {NORD_PALETTE.NORD2};
        border-radius: 6px;
        background-color: {NORD_PALETTE.NORD0};
        top: -1px;
    }}
    QTabBar::tab {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD4};
        padding: 8px 16px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
        margin-right: 2px;
        font-weight: 600;
        border: 1px solid {NORD_PALETTE.NORD2};
        border-bottom: none;
    }}
    QTabBar::tab:selected {{
        background-color: {NORD_PALETTE.NORD0};
        color: {NORD_PALETTE.NORD8};
        border-top: 2px solid {NORD_PALETTE.NORD8};
        border-bottom: 1px solid {NORD_PALETTE.NORD0};
    }}
    QTabBar::tab:hover:!selected {{
        background-color: {NORD_PALETTE.NORD2};
        color: {NORD_PALETTE.NORD6};
    }}
    QTabBar::tab:disabled {{
        color: {NORD_PALETTE.NORD3};
        background-color: {NORD_PALETTE.NORD0};
    }}

    QTableWidget, QTableView {{
        background-color: {NORD_PALETTE.NORD0};
        alternate-background-color: #353C4A;
        border: 1px solid {NORD_PALETTE.NORD2};
        border-radius: 5px;
        gridline-color: {NORD_PALETTE.NORD1};
        selection-background-color: rgba(94, 129, 172, 0.45);
        selection-color: {NORD_PALETTE.NORD6};
        color: {NORD_PALETTE.NORD6};
    }}
    QTableWidget::item, QTableView::item {{
        padding: 4px 6px;
        border: none;
        color: {NORD_PALETTE.NORD6};
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background-color: rgba(94, 129, 172, 0.45);
        color: {NORD_PALETTE.NORD6};
    }}
    QHeaderView {{
        background-color: {NORD_PALETTE.NORD1};
        border: none;
    }}
    QHeaderView::section {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD8};
        padding: 6px 8px;
        border: none;
        border-right: 1px solid {NORD_PALETTE.NORD2};
        border-bottom: 1px solid {NORD_PALETTE.NORD2};
        font-weight: 700;
        font-size: 11.5px;
        text-transform: uppercase;
    }}
    QHeaderView::section:last {{
        border-right: none;
    }}
    QTableCornerButton::section {{
        background-color: {NORD_PALETTE.NORD1};
        border: none;
        border-bottom: 1px solid {NORD_PALETTE.NORD2};
        border-right: 1px solid {NORD_PALETTE.NORD2};
    }}

    QListWidget {{
        background-color: {NORD_PALETTE.NORD0};
        border: 1px solid {NORD_PALETTE.NORD2};
        border-radius: 5px;
        padding: 3px;
    }}
    QListWidget::item {{
        padding: 4px 6px;
        border-radius: 4px;
        color: {NORD_PALETTE.NORD5};
    }}
    QListWidget::item:hover {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD6};
    }}
    QListWidget::item:selected {{
        background-color: {NORD_PALETTE.NORD10};
        color: {NORD_PALETTE.NORD6};
    }}
    QListWidget:disabled {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD3};
    }}

    QScrollArea {{
        border: none;
        background-color: transparent;
    }}
    QScrollBar:vertical {{
        background: {NORD_PALETTE.NORD0};
        width: 7px;
        margin: 0px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical {{
        background: {NORD_PALETTE.NORD3};
        min-height: 20px;
        border-radius: 3px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {NORD_PALETTE.NORD8};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {NORD_PALETTE.NORD0};
        height: 7px;
        margin: 0px;
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal {{
        background: {NORD_PALETTE.NORD3};
        min-width: 20px;
        border-radius: 3px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {NORD_PALETTE.NORD8};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0px;
    }}

    QProgressBar {{
        background-color: {NORD_PALETTE.NORD0};
        border: 1px solid {NORD_PALETTE.NORD2};
        border-radius: 5px;
        text-align: center;
        color: {NORD_PALETTE.NORD6};
        font-weight: 700;
        height: 20px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {NORD_PALETTE.NORD10}, stop:1 {NORD_PALETTE.NORD8});
        border-radius: 4px;
    }}

    QSplitter::handle {{
        background-color: {NORD_PALETTE.NORD2};
    }}
    QSplitter::handle:horizontal {{
        width: 3px;
    }}
    QSplitter::handle:vertical {{
        height: 3px;
    }}

    QToolTip {{
        background-color: {NORD_PALETTE.NORD1};
        color: {NORD_PALETTE.NORD6};
        border: 1px solid {NORD_PALETTE.NORD8};
        border-radius: 4px;
        padding: 5px 8px;
        font-size: 12px;
    }}
    """
