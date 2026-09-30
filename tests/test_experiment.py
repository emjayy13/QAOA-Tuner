"""Unit and integration tests for the Experiment Engine, Config, Metrics, and Serialization."""

import pytest

from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.experiment.engine import ExperimentEngine
from qaoa_tuner.experiment.metrics import SolutionMetrics
from qaoa_tuner.experiment.result import ExperimentResult


class TestExperimentConfig:
    """Test configuration validation and serialization."""

    def test_valid_config_serialization(self, cycle_4_graph):
        config = ExperimentConfig(
            graph_dict=cycle_4_graph.to_dict(),
            num_colors=2,
            qaoa_p=1,
            optimizer="COBYLA",
            max_iter=20,
            shots=512,
            seed=42,
        )

        # Dictionary roundtrip
        d = config.to_dict()
        assert d["num_colors"] == 2
        assert d["qaoa_p"] == 1
        assert d["optimizer"] == "COBYLA"

        config_from_d = ExperimentConfig.from_dict(d)
        assert config_from_d == config

        # JSON roundtrip
        json_str = config.to_json()
        config_from_json = ExperimentConfig.from_json(json_str)
        assert config_from_json == config

    def test_config_validation_rejections(self, cycle_4_graph):
        g_dict = cycle_4_graph.to_dict()

        with pytest.raises(ValueError, match="num_colors"):
            ExperimentConfig(graph_dict=g_dict, num_colors=1)

        with pytest.raises(ValueError, match="qaoa_p"):
            ExperimentConfig(graph_dict=g_dict, qaoa_p=0)

        with pytest.raises(ValueError, match="optimizer"):
            ExperimentConfig(graph_dict=g_dict, optimizer="ADAM")

        with pytest.raises(ValueError, match="transpiler_optimization_level"):
            ExperimentConfig(graph_dict=g_dict, transpiler_optimization_level=5)


class TestMetricsSerialization:
    """Test metrics container serialization."""

    def test_solution_metrics_serialization(self):
        sm = SolutionMetrics(
            valid_coloring_rate=0.75,
            best_coloring={0: 1, 1: 0},
            best_coloring_conflicts=0,
            is_valid_solution_found=True,
            expected_conflicts=0.3,
            classical_chromatic_number=2,
            classical_is_satisfiable=True,
            ground_state_overlap=0.75,
        )
        d = sm.to_dict()
        sm2 = SolutionMetrics.from_dict(d)
        assert sm == sm2
        assert sm2.best_coloring == {0: 1, 1: 0}


class TestExperimentEngine:
    """Test end-to-end execution of experiments through ExperimentEngine."""

    def test_engine_run_c4(self, cycle_4_graph):
        config = ExperimentConfig(
            graph_dict=cycle_4_graph.to_dict(),
            num_colors=2,
            qaoa_p=1,
            optimizer="COBYLA",
            max_iter=15,
            shots=512,
            seed=100,
        )

        engine = ExperimentEngine()
        result = engine.run(config)

        assert isinstance(result, ExperimentResult)
        assert result.config == config
        assert result.solution_metrics.classical_chromatic_number == 2
        assert result.solution_metrics.classical_is_satisfiable is True
        assert result.solution_metrics.is_valid_solution_found is True
        assert result.solution_metrics.best_coloring_conflicts == 0
        assert result.execution_metrics.optimization_evaluations > 0
        assert result.execution_metrics.wall_clock_time_seconds > 0
        assert len(result.optimal_gamma) == 1
        assert len(result.optimal_beta) == 1

    def test_engine_run_and_save(self, cycle_4_graph, tmp_path):
        config = ExperimentConfig(
            graph_dict=cycle_4_graph.to_dict(),
            num_colors=2,
            qaoa_p=1,
            optimizer="COBYLA",
            max_iter=10,
            shots=256,
            seed=42,
            name="test_save_run",
        )

        engine = ExperimentEngine()
        result, saved_path = engine.run_and_save(config, output_dir=tmp_path)

        assert saved_path.exists()
        loaded = ExperimentResult.load(saved_path)
        assert loaded.config.name == "test_save_run"
        assert loaded.solution_metrics.is_valid_solution_found is True
        assert loaded.optimal_gamma == result.optimal_gamma
