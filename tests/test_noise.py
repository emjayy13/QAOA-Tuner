"""Unit and integration tests for Qiskit Aer noise models and degradation benchmarking."""

from pathlib import Path

import pytest
from qiskit_aer.noise import NoiseModel

from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.noise.benchmark import (
    compare_ideal_vs_noisy,
    plot_ideal_vs_noisy_distribution,
)
from qaoa_tuner.noise.models import (
    NoiseConfig,
    build_noise_model,
    get_preset_noise_config,
    list_available_noise_profiles,
)
from qaoa_tuner.qaoa.runner import QaoaRunner


class TestNoiseModels:
    """Test noise model configurations and Qiskit Aer instantiation."""

    def test_list_and_get_presets(self):
        profiles = list_available_noise_profiles()
        assert "ideal" in profiles
        assert "depolarizing_mild" in profiles
        assert "readout_only" in profiles
        assert "realistic_superconducting" in profiles

        cfg = get_preset_noise_config("depolarizing_mild")
        assert cfg.p1_error == 0.0005
        assert cfg.p2_error == 0.005

    def test_ideal_returns_none(self):
        nm = build_noise_model("ideal")
        assert nm is None

    def test_depolarizing_model_creation(self):
        nm = build_noise_model("depolarizing_mild")
        assert isinstance(nm, NoiseModel)
        # Check that basis gates include quantum errors
        assert len(nm.basis_gates) > 0

    def test_readout_only_model(self):
        nm = build_noise_model("readout_only")
        assert isinstance(nm, NoiseModel)

    def test_realistic_superconducting_model(self):
        nm = build_noise_model("realistic_superconducting")
        assert isinstance(nm, NoiseModel)

    def test_thermal_t2_constraint(self):
        with pytest.raises(ValueError, match="Physical constraint violated"):
            NoiseConfig(
                enable_thermal_relaxation=True,
                t1=50e-6,
                t2=120e-6,  # T2 > 2*T1 is unphysical
            )


class TestNoisyExecution:
    """Test physical degradation in QAOA execution under noise."""

    def test_noise_degrades_solution_quality(self, cycle_4_graph):
        """Verify that noise reduces ground-state concentration and increases conflicts."""
        opt_point = [2.37, 1.96]  # Valid ground-state preparing angles for C4 p=1

        # Ideal runner
        ideal_runner = QaoaRunner(
            graph=cycle_4_graph,
            num_colors=2,
            p=1,
            shots=1000,
            seed=42,
            noise_model=None,
        )
        ideal_eval = ideal_runner.evaluate_point(opt_point, shots=1000)

        # Noisy runner with heavy depolarizing noise
        noisy_model = build_noise_model("depolarizing_heavy")
        noisy_runner = QaoaRunner(
            graph=cycle_4_graph,
            num_colors=2,
            p=1,
            shots=1000,
            seed=42,
            noise_model=noisy_model,
        )
        noisy_eval = noisy_runner.evaluate_point(opt_point, shots=1000)

        # Noise should increase expected conflicts and decrease valid state probability
        assert noisy_eval.valid_coloring_probability < ideal_eval.valid_coloring_probability
        assert noisy_eval.expected_conflicts > ideal_eval.expected_conflicts


class TestNoiseBenchmark:
    """Test ideal vs noisy comparative benchmark and plotting."""

    def test_compare_ideal_vs_noisy(self, cycle_4_graph):
        config = ExperimentConfig(
            graph_dict=cycle_4_graph.to_dict(),
            num_colors=2,
            qaoa_p=1,
            optimizer="COBYLA",
            max_iter=10,
            shots=512,
            seed=42,
        )

        comparison = compare_ideal_vs_noisy(config, noise_profile="depolarizing_mild")

        assert comparison.ideal_result.solution_metrics.is_valid_solution_found is True
        assert comparison.noisy_result.solution_metrics.is_valid_solution_found is True
        assert comparison.noise_profile_name == "depolarizing_mild"
        assert isinstance(comparison.quality_degradation_rate, float)
        assert isinstance(comparison.conflict_increase, float)

    def test_plot_generation(self, cycle_4_graph, tmp_path):
        config = ExperimentConfig(
            graph_dict=cycle_4_graph.to_dict(),
            num_colors=2,
            qaoa_p=1,
            optimizer="COBYLA",
            max_iter=5,
            shots=256,
            seed=42,
        )

        comparison = compare_ideal_vs_noisy(config, noise_profile="depolarizing_mild")
        out_img = Path(tmp_path) / "noise_plot.png"
        fig = plot_ideal_vs_noisy_distribution(comparison, top_k=5, output_path=out_img)

        assert out_img.exists()
        assert out_img.stat().st_size > 1000  # Valid non-empty PNG image
        assert fig is not None
