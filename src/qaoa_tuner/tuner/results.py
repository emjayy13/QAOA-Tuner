"""Serializable records of a tuning run (JSON and CSV)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qaoa_tuner.tuner.pareto import Objective, group_equivalent


@dataclass(frozen=True)
class TunerRecord:
    """Measured outcome of ONE configuration. Every number comes from an executed simulation.

    Quality / robustness
        valid_coloring_rate: Probability mass on valid colorings (after mitigation, if any).
        valid_rate_std_error: Binomial shot-noise error (propagated through the fit for ZNE);
            None for readout mitigation, where the matrix inversion changes the statistics.
        expected_conflicts: Expected number of constraint violations.
        ideal_valid_coloring_rate: Same parameters evaluated without noise.
        quality_degradation: ideal_valid_coloring_rate - valid_coloring_rate (smaller = more
            robust; can be negative if mitigation overshoots or shot noise dominates).

    Hardware cost (of the circuit actually executed, after transpilation)
        circuit_depth, two_qubit_gates, one_qubit_gates: single transpiled circuit.
        logical_two_qubit_gates / two_qubit_overhead_ratio: before vs after transpilation.
        two_qubit_gate_budget: two-qubit gates summed over every circuit executed for the
            estimate (ZNE runs several folded circuits; readout calibration circuits contain
            no two-qubit gates). Shots are equal across circuits, so they cancel.
        total_shots: all shots executed, including calibration for readout mitigation.

    Bookkeeping
        uncovered_gates: Gates in the executed circuit with no error in the noise model
            (non-empty means noise is under-applied; should be empty).
        is_pareto_optimal / num_dominators / dominated_by: filled in after analysis;
            ``dominated_by`` lists up to 3 dominating configurations.
    """

    label: str
    qaoa_p: int
    optimizer: str
    optimization_level: int
    mitigation: str
    valid_coloring_rate: float
    valid_rate_std_error: float | None
    expected_conflicts: float
    ideal_valid_coloring_rate: float
    quality_degradation: float
    circuit_depth: int
    two_qubit_gates: int
    one_qubit_gates: int
    logical_two_qubit_gates: int
    two_qubit_overhead_ratio: float
    two_qubit_gate_budget: int
    total_shots: int
    uncovered_gates: list[str] = field(default_factory=list)
    is_pareto_optimal: bool = False
    num_dominators: int = 0
    dominated_by: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParetoGroup:
    """One distinct trade-off on the frontier.

    ``representative`` is the lowest transpiler level (then first label) among all frontier
    configurations with exactly identical objective values; ``equivalent_labels`` lists the
    others (typically the same circuit reached at a higher transpiler level).
    """

    representative: TunerRecord
    equivalent_labels: list[str]


@dataclass(frozen=True)
class TunerResult:
    """All records of a tuning run plus everything needed to reproduce it."""

    settings: dict[str, Any]
    noise_profile: dict[str, Any]
    backend: dict[str, Any]
    objectives: list[dict[str, Any]]
    tolerance: dict[str, float]
    training: list[dict[str, Any]]
    records: list[TunerRecord]
    wall_time_seconds: float
    software: dict[str, str]

    def pareto_records(self) -> list[TunerRecord]:
        return [r for r in self.records if r.is_pareto_optimal]

    def pareto_groups(self) -> list[ParetoGroup]:
        """Frontier configurations, with exact ties collapsed into single entries."""
        front = self.pareto_records()
        objectives = [Objective(**o) for o in self.objectives]
        groups = group_equivalent([r.to_dict() for r in front], objectives)
        result = []
        for indices in groups:
            members = sorted(
                (front[i] for i in indices), key=lambda r: (r.optimization_level, r.label)
            )
            result.append(ParetoGroup(members[0], [m.label for m in members[1:]]))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "noise_profile": self.noise_profile,
            "backend": self.backend,
            "objectives": self.objectives,
            "tolerance": self.tolerance,
            "training": self.training,
            "num_configurations": len(self.records),
            "num_pareto_optimal": len(self.pareto_records()),
            "pareto_groups": [
                {
                    "representative": g.representative.label,
                    "equivalent_labels": g.equivalent_labels,
                }
                for g in self.pareto_groups()
            ],
            "wall_time_seconds": self.wall_time_seconds,
            "software": self.software,
            "records": [r.to_dict() for r in self.records],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    def save_csv(self, filepath: str | Path) -> Path:
        """One row per configuration; list-valued fields are joined with ';'."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for record in self.records:
            row = record.to_dict()
            row["uncovered_gates"] = ";".join(row["uncovered_gates"])
            row["dominated_by"] = ";".join(row["dominated_by"])
            rows.append(row)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        return path
