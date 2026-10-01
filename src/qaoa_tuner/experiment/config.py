"""Structured, immutable experiment configuration specification."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from qaoa_tuner.core.types import GraphDict


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable, fully serializable configuration defining a reproducible experiment.

    Attributes:
        graph_dict: Full serializable graph structure {name, num_nodes, edges}.
        num_colors: Target number of colors (k >= 2).
        qaoa_p: QAOA circuit depth (p >= 1).
        optimizer: Classical optimizer name ("COBYLA" or "SPSA").
        max_iter: Max iterations for classical optimizer.
        shots: Measurement shots per evaluation.
        experiment_id: Unique identifier for this experiment run.
        name: Optional human-readable experiment name.
        backend_name: Target backend identifier (default: "aer_simulator_ideal").
        transpiler_optimization_level: Transpiler pass intensity (0, 1, 2, 3).
        noise_model_name: Optional noise profile identifier.
        mitigation_method: Error mitigation technique ("none", "readout", "zne").
        seed: Random seed for deterministic simulation and reproducibility.
        initial_point: Optional initial parameter values [gamma_0..p-1, beta_0..p-1].
        metadata: Arbitrary user-defined tags and notes.
    """

    graph_dict: GraphDict
    num_colors: int = 2
    qaoa_p: int = 1
    optimizer: str = "COBYLA"
    max_iter: int = 30
    shots: int = 1024
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "qaoa_experiment"
    backend_name: str = "aer_simulator_ideal"
    transpiler_optimization_level: int = 1
    noise_model_name: str | None = None
    mitigation_method: str = "none"
    seed: int | None = 42
    initial_point: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_colors < 2:
            raise ValueError(f"num_colors must be >= 2, got {self.num_colors}")
        if self.qaoa_p < 1:
            raise ValueError(f"qaoa_p must be >= 1, got {self.qaoa_p}")
        if self.optimizer.upper() not in ["COBYLA", "SPSA"]:
            raise ValueError(f"Unsupported optimizer: '{self.optimizer}'")
        if self.mitigation_method.lower() not in ["none", "readout", "zne"]:
            raise ValueError(
                f"mitigation_method must be one of 'none', 'readout', 'zne', "
                f"got '{self.mitigation_method}'"
            )
        if self.transpiler_optimization_level not in [0, 1, 2, 3]:
            raise ValueError(
                f"transpiler_optimization_level must be in [0, 1, 2, 3], got {self.transpiler_optimization_level}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Construct an ExperimentConfig from a dictionary."""
        clean_data = data.copy()
        return cls(**clean_data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ExperimentConfig:
        """Construct an ExperimentConfig from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
