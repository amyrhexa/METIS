# gui/tabs/__init__.py
"""Tab widgets comprising the six-stage scientific workflow in METIS."""

from __future__ import annotations

from gui.tabs.execute_tab import ExecuteTab
from gui.tabs.load_data_tab import LoadDataTab
from gui.tabs.models_tab import ModelsTab
from gui.tabs.output_tab import OutputTab
from gui.tabs.sem_tab import SEMTab
from gui.tabs.transform_tab import TransformTab

__all__ = [
    "ExecuteTab",
    "LoadDataTab",
    "ModelsTab",
    "OutputTab",
    "SEMTab",
    "TransformTab",
]
