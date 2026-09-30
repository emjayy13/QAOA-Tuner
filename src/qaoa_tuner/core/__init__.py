"""Core types and custom exceptions for QAOA-Tuner."""

from qaoa_tuner.core.exceptions import (
    GraphColoringError,
    InvalidGraphError,
    UnsatisfiableColoringError,
)
from qaoa_tuner.core.types import (
    ClassicalColoringResult,
    Coloring,
    ColoringValidationResult,
    Edge,
    GraphDict,
)

__all__ = [
    "ClassicalColoringResult",
    "Coloring",
    "ColoringValidationResult",
    "Edge",
    "GraphColoringError",
    "GraphDict",
    "InvalidGraphError",
    "UnsatisfiableColoringError",
]
