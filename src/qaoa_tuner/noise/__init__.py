"""Noise modeling, Aer noise channels, and ideal vs noisy comparative benchmarks."""

from qaoa_tuner.noise.benchmark import (
    NoiseComparisonResult,
    compare_ideal_vs_noisy,
    plot_ideal_vs_noisy_distribution,
)
from qaoa_tuner.noise.models import (
    NoiseConfig,
    build_noise_model,
    get_preset_noise_config,
    list_available_noise_profiles,
)

__all__ = [
    "NoiseComparisonResult",
    "NoiseConfig",
    "build_noise_model",
    "compare_ideal_vs_noisy",
    "get_preset_noise_config",
    "list_available_noise_profiles",
    "plot_ideal_vs_noisy_distribution",
]
