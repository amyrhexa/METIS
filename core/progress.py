from __future__ import annotations

import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import psutil


class PipelineStage(StrEnum):
    INITIALIZING = "Initializing Environment"
    LOADING_DATA = "Loading & Profiling Data"
    PREPROCESSING = "Building Preprocessing Pipelines"
    SEM_ANALYSIS = "Structural Equation Modeling"
    NESTED_CV = "Nested Cross-Validation Tournament"
    EXPLAINABILITY = "Attribution & Explainability"
    REPORTING = "Compiling Technical Dossier & Reports"
    COMPLETED = "Workflow Finished"
    CANCELLED = "Execution Cancelled"
    FAILED = "Pipeline Failed"


@dataclass
class ProgressUpdate:
    stage: PipelineStage
    current_step: int
    total_steps: int
    percentage: float
    message: str
    target_name: str | None = None
    model_name: str | None = None
    fold: int | None = None
    total_folds: int | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None
    steps_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    memory_percent: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


ProgressCallback = Callable[[ProgressUpdate], None]


class TerminalProgressBar:
    """High-performance ANSI live terminal progress bar with memory and ETA telemetry."""

    def __init__(self, width: int = 28, enabled: bool = True) -> None:
        self.width = width
        self.enabled = enabled and sys.stdout.isatty()
        self._last_line_len = 0
        self._lock = threading.Lock()

    def render(self, update: ProgressUpdate) -> None:
        if not self.enabled:
            return

        with self._lock:
            pct = max(0.0, min(100.0, update.percentage))
            filled = int(self.width * (pct / 100.0))
            bar = "█" * filled + "░" * (self.width - filled)

            if (
                update.eta_seconds is not None
                and math.isfinite(update.eta_seconds)
                and update.eta_seconds >= 0
            ):
                eta_m = int(update.eta_seconds // 60)
                eta_s = int(update.eta_seconds % 60)
                eta_str = f"{eta_m:02d}:{eta_s:02d}"
            else:
                eta_str = "--:--"

            elapsed = max(0.0, update.elapsed_seconds)
            el_m = int(elapsed // 60)
            el_s = int(elapsed % 60)
            elapsed_str = f"{el_m:02d}:{el_s:02d}"

            context_parts = []
            if update.target_name:
                context_parts.append(f"Target: {update.target_name}")
            if update.model_name:
                context_parts.append(f"Model: {update.model_name}")
            if update.fold is not None and update.total_folds:
                context_parts.append(f"Fold: {update.fold}/{update.total_folds}")

            ctx = f" | {', '.join(context_parts)}" if context_parts else ""
            line = (
                f"\r\033[K[\033[38;2;136;192;208m{bar}\033[0m] {pct:5.1f}% | "
                f"{update.current_step}/{update.total_steps} steps | "
                f"⏱ {elapsed_str}<{eta_str} | "
                f"⚡ {update.steps_per_second:.1f} it/s | "
                f"RAM: {update.memory_usage_mb:.0f}MB{ctx} - {update.message}"
            )

            terminal_width = (
                os.get_terminal_size().columns
                if hasattr(os, "get_terminal_size")
                else 120
            )
            if len(line) > terminal_width:
                line = line[: terminal_width - 3] + "..."

            sys.stdout.write(line)
            sys.stdout.flush()
            self._last_line_len = len(line)

    def clear(self) -> None:
        if self.enabled and self._last_line_len > 0:
            with self._lock:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
                self._last_line_len = 0


class ProgressTracker:
    """Thread-safe hierarchical progress engine with Exponential Moving Average smoothing."""

    def __init__(
        self,
        total_steps: int,
        callback: ProgressCallback | None = None,
        alpha: float = 0.2,
        enable_cli_bar: bool = True,
    ) -> None:
        self.total_steps = max(1, total_steps)
        self.callback = callback
        self.current_step = 0
        self.start_time = time.monotonic()
        self.last_update_time = self.start_time
        self.smoothed_step_duration: float | None = None
        self.alpha = alpha
        self.process = psutil.Process()
        self._lock = threading.Lock()
        self.terminal_bar = TerminalProgressBar(enabled=enable_cli_bar)

    def set_total_steps(self, total: int) -> None:
        with self._lock:
            self.total_steps = max(1, total)

    def advance(
        self,
        step_increment: int = 1,
        stage: PipelineStage = PipelineStage.NESTED_CV,
        message: str = "",
        target_name: str | None = None,
        model_name: str | None = None,
        fold: int | None = None,
        total_folds: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ProgressUpdate:
        with self._lock:
            now = time.monotonic()
            self.current_step = min(
                self.total_steps, self.current_step + step_increment
            )
            elapsed = now - self.start_time

            duration = (now - self.last_update_time) / max(1, step_increment)
            self.last_update_time = now

            if self.smoothed_step_duration is None:
                self.smoothed_step_duration = duration
            else:
                self.smoothed_step_duration = (
                    self.alpha * duration
                    + (1.0 - self.alpha) * self.smoothed_step_duration
                )

            remaining_steps = max(0, self.total_steps - self.current_step)
            eta = (
                remaining_steps * self.smoothed_step_duration
                if self.smoothed_step_duration is not None and remaining_steps > 0
                else 0.0
            )

            speed = (
                1.0 / self.smoothed_step_duration
                if self.smoothed_step_duration and self.smoothed_step_duration > 0
                else 0.0
            )
            pct = (self.current_step / self.total_steps) * 100.0

            try:
                mem_info = self.process.memory_info()
                mem_mb = mem_info.rss / (1024 * 1024)
                mem_pct = self.process.memory_percent()
            except Exception:
                mem_mb, mem_pct = 0.0, 0.0

            update_obj = ProgressUpdate(
                stage=stage,
                current_step=self.current_step,
                total_steps=self.total_steps,
                percentage=pct,
                message=message,
                target_name=target_name,
                model_name=model_name,
                fold=fold,
                total_folds=total_folds,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                steps_per_second=speed,
                memory_usage_mb=round(mem_mb, 1),
                memory_percent=round(mem_pct, 1),
                metadata=metadata or {},
            )

            self.terminal_bar.render(update_obj)

            if self.callback:
                self.callback(update_obj)

            return update_obj

    def finish(self, message: str = "Pipeline completed successfully.") -> None:
        self.advance(
            step_increment=max(0, self.total_steps - self.current_step),
            stage=PipelineStage.COMPLETED,
            message=message,
        )
        if self.terminal_bar.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()
