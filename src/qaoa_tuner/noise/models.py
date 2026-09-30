"""Parameterized Qiskit Aer noise models and physical decoherence channels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration parameters for constructing a realistic physical noise model.

    Attributes:
        name: Identifier name for the noise model.
        p1_error: Single-qubit gate depolarizing error probability.
        p2_error: Two-qubit gate depolarizing error probability.
        readout_error_0to1: Probability of measuring 1 when true state is 0.
        readout_error_1to0: Probability of measuring 0 when true state is 1.
        enable_thermal_relaxation: Whether to model T1 and T2 decoherence.
        t1: Longitudinal relaxation time in seconds (default: 50 microseconds).
        t2: Dephasing time in seconds (default: 70 microseconds, constraint: T2 <= 2*T1).
        gate_time_1q: Duration of single-qubit gate in seconds (default: 35 nanoseconds).
        gate_time_2q: Duration of two-qubit gate in seconds (default: 300 nanoseconds).
    """

    name: str = "custom_noise"
    p1_error: float = 0.001
    p2_error: float = 0.01
    readout_error_0to1: float = 0.015
    readout_error_1to0: float = 0.015
    enable_thermal_relaxation: bool = False
    t1: float = 50e-6
    t2: float = 70e-6
    gate_time_1q: float = 35e-9
    gate_time_2q: float = 300e-9

    def __post_init__(self) -> None:
        if self.enable_thermal_relaxation and self.t2 > 2 * self.t1:
            raise ValueError(
                f"Physical constraint violated: T2 ({self.t2}s) cannot exceed 2 * T1 ({2 * self.t1}s)."
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NoiseConfig:
        """Construct configuration from dictionary."""
        return cls(**data)


# Documented physical presets representing typical superconducting device regimes
PRESET_NOISE_PROFILES: dict[str, NoiseConfig] = {
    "ideal": NoiseConfig(
        name="ideal",
        p1_error=0.0,
        p2_error=0.0,
        readout_error_0to1=0.0,
        readout_error_1to0=0.0,
        enable_thermal_relaxation=False,
    ),
    "depolarizing_mild": NoiseConfig(
        name="depolarizing_mild",
        p1_error=0.0005,
        p2_error=0.005,
        readout_error_0to1=0.0,
        readout_error_1to0=0.0,
        enable_thermal_relaxation=False,
    ),
    "depolarizing_heavy": NoiseConfig(
        name="depolarizing_heavy",
        p1_error=0.002,
        p2_error=0.02,
        readout_error_0to1=0.0,
        readout_error_1to0=0.0,
        enable_thermal_relaxation=False,
    ),
    "readout_only": NoiseConfig(
        name="readout_only",
        p1_error=0.0,
        p2_error=0.0,
        readout_error_0to1=0.025,
        readout_error_1to0=0.025,
        enable_thermal_relaxation=False,
    ),
    "realistic_superconducting": NoiseConfig(
        name="realistic_superconducting",
        p1_error=0.0008,
        p2_error=0.01,
        readout_error_0to1=0.015,
        readout_error_1to0=0.015,
        enable_thermal_relaxation=True,
        t1=50e-6,
        t2=70e-6,
        gate_time_1q=35e-9,
        gate_time_2q=300e-9,
    ),
}


def list_available_noise_profiles() -> list[str]:
    """Return list of supported standard noise presets."""
    return list(PRESET_NOISE_PROFILES.keys())


def get_preset_noise_config(name: str) -> NoiseConfig:
    """Retrieve predefined NoiseConfig by name."""
    clean = name.lower()
    if clean in PRESET_NOISE_PROFILES:
        return PRESET_NOISE_PROFILES[clean]
    raise ValueError(
        f"Unknown noise profile '{name}'. Available: {list_available_noise_profiles()}"
    )


def build_noise_model(config: NoiseConfig | str) -> NoiseModel | None:
    """Construct a Qiskit Aer NoiseModel from configuration parameters.

    Args:
        config: NoiseConfig object or string identifier of a preset profile.

    Returns:
        NoiseModel instance, or None if the configuration represents an ideal backend.
    """
    if isinstance(config, str):
        config = get_preset_noise_config(config)

    # If all errors are zero and thermal relaxation is disabled, backend is ideal
    if (
        config.p1_error == 0.0
        and config.p2_error == 0.0
        and config.readout_error_0to1 == 0.0
        and config.readout_error_1to0 == 0.0
        and not config.enable_thermal_relaxation
    ):
        return None

    noise_model = NoiseModel()

    # 1. Single-Qubit Gate Errors
    q1_ops = ["rz", "sx", "x", "id"]
    error_1q = None

    if config.enable_thermal_relaxation:
        error_1q = thermal_relaxation_error(
            t1=config.t1,
            t2=config.t2,
            time=config.gate_time_1q,
        )

    if config.p1_error > 0.0:
        depol_1q = depolarizing_error(config.p1_error, 1)
        error_1q = depol_1q if error_1q is None else error_1q.compose(depol_1q)

    if error_1q is not None:
        noise_model.add_all_qubit_quantum_error(error_1q, q1_ops)

    # 2. Two-Qubit Gate Errors
    q2_ops = ["cx", "cz", "ecr", "rzz"]
    error_2q = None

    if config.enable_thermal_relaxation:
        # 2-qubit thermal relaxation tensor product
        t_relax_1 = thermal_relaxation_error(
            t1=config.t1, t2=config.t2, time=config.gate_time_2q
        )
        t_relax_2 = thermal_relaxation_error(
            t1=config.t1, t2=config.t2, time=config.gate_time_2q
        )
        error_2q = t_relax_1.tensor(t_relax_2)

    if config.p2_error > 0.0:
        depol_2q = depolarizing_error(config.p2_error, 2)
        error_2q = depol_2q if error_2q is None else error_2q.compose(depol_2q)

    if error_2q is not None:
        noise_model.add_all_qubit_quantum_error(error_2q, q2_ops)

    # 3. Measurement Readout Errors
    if config.readout_error_0to1 > 0.0 or config.readout_error_1to0 > 0.0:
        p00 = 1.0 - config.readout_error_0to1
        p01 = config.readout_error_0to1
        p10 = config.readout_error_1to0
        p11 = 1.0 - config.readout_error_1to0

        ro_error = ReadoutError([[p00, p01], [p10, p11]])
        noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model
