"""Reusable experiment execution engine orchestrating classical verification and QAOA optimization."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import qiskit
import qiskit_aer

from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.experiment.metrics import ExecutionMetrics, SolutionMetrics
from qaoa_tuner.experiment.result import ExperimentResult
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.problem.verifier import (
    find_all_valid_colorings,
    find_chromatic_number,
    solve_classical_exact,
)
from qaoa_tuner.qaoa.decoder import decode_bitstring
from qaoa_tuner.qaoa.runner import QaoaRunner


class ExperimentEngine:
    """Orchestrates reproducible QAOA experiments, metrics collection, and serialization.

    Designed to be called uniformly from CLI, scripts, test suites, or the Streamlit dashboard.
    """

    def __init__(self) -> None:
        pass

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute an experiment defined by an ExperimentConfig.

        Workflow:
            1. Reconstruct and validate graph topology.
            2. Compute classical ground truth (chromatic number, exact solvability).
            3. Execute QAOA circuit generation, transpilation, and parameter optimization.
            4. Extract solution quality, overlap with classical ground states, and execution resources.
            5. Return structured, immutable ExperimentResult.

        Args:
            config: Immutable ExperimentConfig.

        Returns:
            ExperimentResult containing all evaluated metrics.
        """
        start_time = time.time()
        start_iso = datetime.now().isoformat()

        # 1. Reconstruct graph
        graph = ColoringGraph.from_dict(config.graph_dict)

        # 2. Classical reference verification
        classical_chromatic = find_chromatic_number(graph)
        classical_res = solve_classical_exact(graph, config.num_colors)
        classical_satisfiable = classical_res.is_satisfiable

        classical_ground_colorings = []
        if graph.num_nodes <= 12 and classical_satisfiable:
            classical_ground_colorings = find_all_valid_colorings(graph, config.num_colors)

        # 3. QAOA Optimization
        runner = QaoaRunner(
            graph=graph,
            num_colors=config.num_colors,
            p=config.qaoa_p,
            shots=config.shots,
            seed=config.seed,
        )

        exec_res = runner.optimize(
            optimizer=config.optimizer,
            max_iter=config.max_iter,
            initial_point=config.initial_point,
            final_shots=config.shots,
        )

        elapsed_time = time.time() - start_time
        decoded = exec_res.decoded_result
        opt_res = exec_res.optimizer_result

        # 4. Compute ground-state overlap
        ground_overlap = 0.0
        if classical_ground_colorings:
            for bs, prob in decoded.probabilities.items():
                bs_coloring = decode_bitstring(
                    bs, graph, config.num_colors, runner.cost_hamiltonian.encoding
                )
                if bs_coloring in classical_ground_colorings:
                    ground_overlap += prob

        # 5. Build structured metrics
        sol_metrics = SolutionMetrics(
            valid_coloring_rate=decoded.valid_coloring_probability,
            best_coloring=decoded.best_coloring,
            best_coloring_conflicts=decoded.best_validation.num_conflicts,
            is_valid_solution_found=decoded.best_validation.is_valid,
            expected_conflicts=decoded.expected_conflicts,
            classical_chromatic_number=classical_chromatic,
            classical_is_satisfiable=classical_satisfiable,
            ground_state_overlap=ground_overlap,
        )

        exec_metrics = ExecutionMetrics(
            wall_clock_time_seconds=elapsed_time,
            optimization_evaluations=opt_res.num_evaluations,
            final_objective_value=opt_res.optimal_value,
            qiskit_version=qiskit.__version__,
            qiskit_aer_version=qiskit_aer.__version__,
            timestamp=start_iso,
        )

        p = config.qaoa_p
        optimal_gamma = opt_res.optimal_point[:p]
        optimal_beta = opt_res.optimal_point[p:]

        return ExperimentResult(
            config=config,
            solution_metrics=sol_metrics,
            execution_metrics=exec_metrics,
            optimal_gamma=optimal_gamma,
            optimal_beta=optimal_beta,
            optimization_history=opt_res.history,
            measurement_counts=decoded.counts,
            measurement_probabilities=decoded.probabilities,
        )

    def run_and_save(
        self,
        config: ExperimentConfig,
        output_dir: str | Path = "data/experiments",
    ) -> tuple[ExperimentResult, Path]:
        """Run experiment and save results to disk as JSON."""
        result = self.run(config)
        out_path = Path(output_dir) / f"{config.experiment_id}_{config.name}.json"
        saved_path = result.save(out_path)
        return result, saved_path
