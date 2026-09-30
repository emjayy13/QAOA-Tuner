"""Serializable experiment result container encapsulating all outcome metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qaoa_tuner.compilation.profiler import CircuitResourceMetrics
from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.experiment.metrics import ExecutionMetrics, SolutionMetrics


@dataclass(frozen=True)
class ExperimentResult:
    """Consolidated outcome of an executed QAOA experiment.

    Attributes:
        config: Original configuration used for the experiment.
        solution_metrics: Domain metrics on graph coloring validity and conflicts.
        execution_metrics: Wall-clock timing, iterations, and platform versions.
        optimal_gamma: Optimized gamma parameter vector [gamma_0, ..., gamma_{p-1}].
        optimal_beta: Optimized beta parameter vector [beta_0, ..., beta_{p-1}].
        optimization_history: Trace of objective values across optimization steps.
        measurement_counts: Raw final measurement histogram {bitstring: count}.
        measurement_probabilities: Normalized probabilities {bitstring: probability}.
        circuit_metrics: Compiled hardware resource metrics (depth, 2Q gates, SWAPs).
        logical_metrics: Original uncompiled circuit resource metrics.
    """

    config: ExperimentConfig
    solution_metrics: SolutionMetrics
    execution_metrics: ExecutionMetrics
    optimal_gamma: list[float]
    optimal_beta: list[float]
    optimization_history: list[float]
    measurement_counts: dict[str, int]
    measurement_probabilities: dict[str, float]
    circuit_metrics: CircuitResourceMetrics | None = None
    logical_metrics: CircuitResourceMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert entire experiment result to a nested JSON-compatible dictionary."""
        return {
            "config": self.config.to_dict(),
            "solution_metrics": self.solution_metrics.to_dict(),
            "execution_metrics": self.execution_metrics.to_dict(),
            "optimal_gamma": self.optimal_gamma,
            "optimal_beta": self.optimal_beta,
            "optimization_history": self.optimization_history,
            "measurement_counts": self.measurement_counts,
            "measurement_probabilities": self.measurement_probabilities,
            "circuit_metrics": self.circuit_metrics.to_dict() if self.circuit_metrics else None,
            "logical_metrics": self.logical_metrics.to_dict() if self.logical_metrics else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Construct an ExperimentResult from a nested dictionary."""
        circ_m = (
            CircuitResourceMetrics.from_dict(data["circuit_metrics"])
            if data.get("circuit_metrics")
            else None
        )
        log_m = (
            CircuitResourceMetrics.from_dict(data["logical_metrics"])
            if data.get("logical_metrics")
            else None
        )
        return cls(
            config=ExperimentConfig.from_dict(data["config"]),
            solution_metrics=SolutionMetrics.from_dict(data["solution_metrics"]),
            execution_metrics=ExecutionMetrics.from_dict(data["execution_metrics"]),
            optimal_gamma=data["optimal_gamma"],
            optimal_beta=data["optimal_beta"],
            optimization_history=data["optimization_history"],
            measurement_counts=data["measurement_counts"],
            measurement_probabilities=data["measurement_probabilities"],
            circuit_metrics=circ_m,
            logical_metrics=log_m,
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize result to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentResult:
        """Construct an ExperimentResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: str | Path) -> Path:
        """Save result to a JSON file on disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, filepath: str | Path) -> ExperimentResult:
        """Load an ExperimentResult from a JSON file on disk."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Result file not found: {path}")
        content = path.read_text(encoding="utf-8")
        return cls.from_json(content)
