"""Metrics containers for solution quality, classical comparisons, and execution resources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SolutionMetrics:
    """Quantitative metrics evaluating quantum graph coloring solution quality.

    Attributes:
        valid_coloring_rate: Total probability mass concentrated on valid (zero-conflict) colorings.
        best_coloring: Best coloring assignment observed {node: color}.
        best_coloring_conflicts: Number of edge conflicts in the best coloring.
        is_valid_solution_found: True if best_coloring has 0 conflicts and colors all nodes.
        expected_conflicts: Weighted average of conflicts across all sampled bitstrings.
        classical_chromatic_number: Exact chromatic number chi(G) computed classically.
        classical_is_satisfiable: Whether k colors is classically sufficient to color G.
        ground_state_overlap: Total probability mass on the exact degenerate ground states.
    """

    valid_coloring_rate: float
    best_coloring: dict[int, int]
    best_coloring_conflicts: int
    is_valid_solution_found: bool
    expected_conflicts: float
    classical_chromatic_number: int
    classical_is_satisfiable: bool
    ground_state_overlap: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        data = asdict(self)
        # Ensure integer keys in coloring dict are converted to strings for JSON compliance
        data["best_coloring"] = {str(k): v for k, v in self.best_coloring.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SolutionMetrics:
        """Construct SolutionMetrics from dictionary."""
        clean = data.copy()
        clean["best_coloring"] = {int(k): v for k, v in clean["best_coloring"].items()}
        return cls(**clean)


@dataclass(frozen=True)
class ExecutionMetrics:
    """Execution resources and runtime profiling metrics.

    Attributes:
        wall_clock_time_seconds: Total duration for optimization and final evaluation.
        optimization_evaluations: Number of circuit evaluation steps during optimization.
        final_objective_value: Final objective cost returned by the classical optimizer.
        qiskit_version: Runtime Qiskit package version.
        qiskit_aer_version: Runtime Qiskit Aer package version.
        timestamp: ISO format timestamp of execution start.
    """

    wall_clock_time_seconds: float
    optimization_evaluations: int
    final_objective_value: float
    qiskit_version: str
    qiskit_aer_version: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionMetrics:
        """Construct ExecutionMetrics from dictionary."""
        return cls(**data)
