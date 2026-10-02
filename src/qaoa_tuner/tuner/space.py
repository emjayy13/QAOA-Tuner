"""Configuration space: what the tuner is allowed to vary, and what stays fixed.

The search space is deliberately a small, explicit grid (no learned or adaptive search):
every configuration that will be run can be listed up front, and the reader can see exactly
why each one exists. Default grid: 3 depths x 2 optimizers x 4 transpiler levels x
3 mitigation modes = 72 configurations.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from typing import Any

from qaoa_tuner.core.types import GraphDict

VALID_OPTIMIZERS = ("COBYLA", "SPSA")
VALID_MITIGATIONS = ("none", "zne", "readout")
VALID_OPT_LEVELS = (0, 1, 2, 3)


@dataclass(frozen=True)
class TunerConfig:
    """One point in the search space (the variables the tuner changes)."""

    qaoa_p: int
    optimizer: str
    optimization_level: int
    mitigation: str

    def __post_init__(self) -> None:
        if self.qaoa_p < 1:
            raise ValueError(f"qaoa_p must be >= 1, got {self.qaoa_p}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {VALID_OPTIMIZERS}, got '{self.optimizer}'")
        if self.optimization_level not in VALID_OPT_LEVELS:
            raise ValueError(
                f"optimization_level must be one of {VALID_OPT_LEVELS}, got {self.optimization_level}"
            )
        if self.mitigation not in VALID_MITIGATIONS:
            raise ValueError(
                f"mitigation must be one of {VALID_MITIGATIONS}, got '{self.mitigation}'"
            )

    @property
    def label(self) -> str:
        return f"p{self.qaoa_p}-{self.optimizer}-O{self.optimization_level}-{self.mitigation}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfigurationSpace:
    """Grid of values for each tuned variable."""

    depths: tuple[int, ...] = (1, 2, 3)
    optimizers: tuple[str, ...] = ("COBYLA", "SPSA")
    optimization_levels: tuple[int, ...] = (0, 1, 2, 3)
    mitigations: tuple[str, ...] = ("none", "zne", "readout")

    def generate(self) -> list[TunerConfig]:
        """All configurations in a fixed, deterministic order (p, optimizer, level, mitigation)."""
        axes = (self.depths, self.optimizers, self.optimization_levels, self.mitigations)
        if any(len(axis) == 0 for axis in axes):
            raise ValueError("Every axis of the configuration space needs at least one value.")
        configs = [
            TunerConfig(p, str(opt).upper(), level, str(mit).lower())
            for p, opt, level, mit in itertools.product(*axes)
        ]
        if len({c.label for c in configs}) != len(configs):
            raise ValueError("Configuration space contains duplicate values.")
        return configs

    @property
    def size(self) -> int:
        return len(self.generate())


def generate_configurations(space: ConfigurationSpace | None = None) -> list[TunerConfig]:
    """Convenience wrapper: the default grid, or the grid of a custom space."""
    return (space or ConfigurationSpace()).generate()


@dataclass(frozen=True)
class TunerSettings:
    """Everything that stays FIXED across the whole tuning run.

    Attributes:
        graph_dict: Serializable graph {name, num_nodes, edges}.
        num_colors: Number of colors k.
        noise_profile: Name of a preset in ``qaoa_tuner.noise.models`` (must contain noise).
        backend_name: Backend providing topology and basis gates (see BackendProvider).
            Noise always comes from ``noise_profile``, not from the backend's own calibration.
        shots: Shots per circuit execution.
        seed: Seed for training, transpilation and simulation.
        max_iter: Optimizer iterations during (ideal) training.
        zne_scale_factors: Odd integers used when mitigation == "zne"; must include 1.
        zne_model: "linear" or "polynomial".
        calibration_shots: Shots per readout-calibration circuit (default: ``shots``).
    """

    graph_dict: GraphDict
    num_colors: int = 2
    noise_profile: str = "realistic_superconducting"
    backend_name: str = "fake_linear_5q"
    shots: int = 1024
    seed: int = 42
    max_iter: int = 25
    zne_scale_factors: tuple[int, ...] = (1, 3, 5)
    zne_model: str = "linear"
    calibration_shots: int | None = None

    def __post_init__(self) -> None:
        if self.num_colors < 2:
            raise ValueError(f"num_colors must be >= 2, got {self.num_colors}")
        if self.shots < 1 or self.max_iter < 1:
            raise ValueError("shots and max_iter must be >= 1.")
        if 1 not in self.zne_scale_factors:
            raise ValueError("zne_scale_factors must include 1 (the unscaled circuit).")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["zne_scale_factors"] = list(self.zne_scale_factors)
        return data
