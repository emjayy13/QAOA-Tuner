"""Comparative benchmark analyzing ideal vs noisy QAOA execution and degradation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.experiment.engine import ExperimentEngine
from qaoa_tuner.experiment.result import ExperimentResult
from qaoa_tuner.noise.models import NoiseConfig


@dataclass(frozen=True)
class NoiseComparisonResult:
    """Quantitative comparison between ideal and noisy QAOA executions.

    Attributes:
        ideal_result: Outcome under zero-noise simulation.
        noisy_result: Outcome under physical decoherence and error channels.
        quality_degradation_rate: Relative loss in valid coloring probability (Q_ideal - Q_noisy) / Q_ideal.
        conflict_increase: Absolute increase in expected edge conflicts under noise.
        noise_profile_name: Identifier of the noise model evaluated.
    """

    ideal_result: ExperimentResult
    noisy_result: ExperimentResult
    quality_degradation_rate: float
    conflict_increase: float
    noise_profile_name: str

    def to_dict(self) -> dict[str, Any]:
        """Convert comparison to dictionary."""
        return {
            "ideal_result": self.ideal_result.to_dict(),
            "noisy_result": self.noisy_result.to_dict(),
            "quality_degradation_rate": self.quality_degradation_rate,
            "conflict_increase": self.conflict_increase,
            "noise_profile_name": self.noise_profile_name,
        }


def compare_ideal_vs_noisy(
    config: ExperimentConfig,
    noise_profile: str | NoiseConfig = "depolarizing_mild",
) -> NoiseComparisonResult:
    """Execute an experiment under ideal and noisy conditions to quantify physical degradation.

    Args:
        config: Base ExperimentConfig.
        noise_profile: NoiseConfig or string preset name (e.g. 'depolarizing_mild').

    Returns:
        NoiseComparisonResult containing both results and relative degradation metrics.
    """
    if isinstance(noise_profile, str):
        p_name = noise_profile
    else:
        p_name = noise_profile.name

    engine = ExperimentEngine()

    # 1. Ideal Execution
    ideal_cfg = ExperimentConfig(
        graph_dict=config.graph_dict,
        num_colors=config.num_colors,
        qaoa_p=config.qaoa_p,
        optimizer=config.optimizer,
        max_iter=config.max_iter,
        shots=config.shots,
        backend_name="aer_simulator_ideal",
        noise_model_name=None,
        seed=config.seed,
        name=f"{config.name}_ideal",
    )
    ideal_res = engine.run(ideal_cfg)

    # 2. Noisy Execution
    noisy_cfg = ExperimentConfig(
        graph_dict=config.graph_dict,
        num_colors=config.num_colors,
        qaoa_p=config.qaoa_p,
        optimizer=config.optimizer,
        max_iter=config.max_iter,
        shots=config.shots,
        backend_name=config.backend_name,
        noise_model_name=p_name,
        seed=config.seed,
        name=f"{config.name}_noisy_{p_name}",
    )
    noisy_res = engine.run(noisy_cfg)

    # 3. Calculate degradation
    ideal_q = ideal_res.solution_metrics.valid_coloring_rate
    noisy_q = noisy_res.solution_metrics.valid_coloring_rate
    degradation = (ideal_q - noisy_q) / max(1e-6, ideal_q)

    conflict_delta = (
        noisy_res.solution_metrics.expected_conflicts
        - ideal_res.solution_metrics.expected_conflicts
    )

    return NoiseComparisonResult(
        ideal_result=ideal_res,
        noisy_result=noisy_res,
        quality_degradation_rate=round(float(degradation), 4),
        conflict_increase=round(float(conflict_delta), 4),
        noise_profile_name=p_name,
    )


def plot_ideal_vs_noisy_distribution(
    comparison: NoiseComparisonResult,
    top_k: int = 10,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Generate a scientific comparison plot of ideal vs noisy measurement distributions.

    Adheres strictly to the warm scientific visual identity:
        - Canvas background: #FAF8F5 (warm cream)
        - Ideal series: #2D6A4F (forest sage)
        - Noisy series: #C85A32 (warm terracotta)
        - Text & axes: #262626 (charcoal)

    Args:
        comparison: NoiseComparisonResult from benchmark.
        top_k: Number of most probable bitstrings to display.
        output_path: Optional path to save the figure image.

    Returns:
        matplotlib Figure instance.
    """
    ideal_probs = comparison.ideal_result.measurement_probabilities
    noisy_probs = comparison.noisy_result.measurement_probabilities

    # Select top-k bitstrings from the union of top probabilities
    all_keys = set(ideal_probs.keys()).union(set(noisy_probs.keys()))
    sorted_keys = sorted(
        all_keys,
        key=lambda k: max(ideal_probs.get(k, 0.0), noisy_probs.get(k, 0.0)),
        reverse=True,
    )[:top_k]

    ideal_vals = [ideal_probs.get(k, 0.0) for k in sorted_keys]
    noisy_vals = [noisy_probs.get(k, 0.0) for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#FAF8F5")
    ax.set_facecolor("#FFFFFF")

    x_indices = range(len(sorted_keys))
    width = 0.35

    ax.bar(
        [x - width / 2 for x in x_indices],
        ideal_vals,
        width,
        label=f"Ideal Simulation (Success: {comparison.ideal_result.solution_metrics.valid_coloring_rate:.1%})",
        color="#2D6A4F",
        edgecolor="#1E4735",
        alpha=0.9,
    )
    ax.bar(
        [x + width / 2 for x in x_indices],
        noisy_vals,
        width,
        label=f"Noisy [{comparison.noise_profile_name}] (Success: {comparison.noisy_result.solution_metrics.valid_coloring_rate:.1%})",
        color="#C85A32",
        edgecolor="#963F20",
        alpha=0.9,
    )

    ax.set_xlabel("Measured Bitstrings (Qubit Register)", fontsize=11, color="#262626", labelpad=8)
    ax.set_ylabel("Observed Probability", fontsize=11, color="#262626", labelpad=8)
    ax.set_title(
        f"QAOA Hardware Noise Degradation: Ideal vs. {comparison.noise_profile_name}\n"
        f"Quality Degradation: {comparison.quality_degradation_rate:.1%} | Conflict Increase: +{comparison.conflict_increase:.2f}",
        fontsize=12,
        fontweight="bold",
        color="#262626",
        pad=12,
    )

    ax.set_xticks(list(x_indices))
    ax.set_xticklabels(sorted_keys, rotation=45, ha="right", fontsize=9, color="#262626")
    ax.tick_params(colors="#262626")

    # Subtle gridlines
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#8C827A")
    for spine in ax.spines.values():
        spine.set_color("#D4CCC5")

    ax.legend(frameon=True, facecolor="#FAF8F5", edgecolor="#D4CCC5", fontsize=10)
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")

    return fig
