# config/exceptions.py
"""Shared exception types used to make scientifically meaningful failures explicit.

Per the METIS remediation principles, a configuration error or an ambiguous
methodological situation must never silently degrade into a scientifically
weaker fallback. These exceptions are raised instead of allowing that to
happen, and they carry enough structured information for the caller (CLI,
GUI, or another script) to explain the problem to the researcher and offer
a concrete resolution.

This module is intentionally placed under `config/` (rather than `core/`)
and has no dependency on the rest of METIS, specifically so that both
`config/schema.py` and every `core/*.py` module can import it without risk
of circular imports (several `core/*.py` modules import from
`config.schema`, so `config.schema` itself must not import anything from
`core`, which is a package whose `__init__.py` eagerly imports
`core.pipeline_runner`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class MetisError(Exception):
    """Base class for all METIS-specific errors."""


class MetisConfigError(MetisError):
    """Raised when a configuration value is invalid or scientifically unsafe.

    Deliberately distinct from a bare ``ValueError`` so callers (GUI/CLI) can
    catch configuration problems specifically and surface them to the user
    with actionable guidance, rather than treating them as unexpected bugs.
    """


@dataclass
class AmbiguousTaskTypeError(MetisConfigError):
    """Raised when AUTO task-type resolution encounters a numeric target
    whose cardinality (typically 3-10 unique values) does not, by itself,
    tell us whether the variable is continuous, ordinal, or nominal.

    METIS intentionally does not guess in this situation (cardinality alone
    is not sufficient evidence of measurement scale), and refuses to run
    the analysis until the researcher either:
      1. Sets an explicit `task_type` for the column
         (`dataset.task_type_overrides`), or
      2. Explicitly acknowledges the ambiguity (adds the column to
         `dataset.confirmed_ambiguous_targets`) and accepts the default
         resolution.
    """

    column: str = ""
    unique_count: int = 0
    unique_values_preview: list[Any] = field(default_factory=list)
    default_resolution: str = ""

    def __post_init__(self) -> None:
        Exception.__init__(self, self.__str__())

    def __str__(self) -> str:
        preview = ", ".join(str(v) for v in self.unique_values_preview[:10])
        return (
            f"Column '{self.column}' has {self.unique_count} unique numeric "
            f"values ({preview}{'...' if self.unique_count > 10 else ''}) under "
            f"task_type='auto'. Cardinality alone cannot determine whether this "
            f"is a continuous, ordinal, or nominal variable — METIS will not "
            f"silently assume '{self.default_resolution}'. Either set an "
            f"explicit task_type for '{self.column}' (regression/classification) "
            f"in `dataset.task_type_overrides`, or add '{self.column}' to "
            f"`dataset.confirmed_ambiguous_targets` to proceed with the default "
            f"resolution ('{self.default_resolution}') and record that "
            f"acknowledgement in the run manifest."
        )


class SEMConfigurationError(MetisConfigError):
    """Raised for invalid or unsupported SEM configuration values (e.g. an
    unrecognized `missing_data_method`). METIS never silently substitutes a
    weaker missing-data method for an unrecognized request."""


class PipelineValidationError(MetisConfigError):
    """Raised when `execute_pipeline` cannot proceed because the requested
    configuration is scientifically ambiguous or unsafe and has not been
    explicitly resolved/confirmed by the caller."""
