"""Execution runner orchestrating QAOA circuit construction, simulation, and parameter optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

from qaoa_tuner.core.exceptions import InvalidGraphError
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.qaoa.ansatz import build_qaoa_circuit
from qaoa_tuner.qaoa.decoder import DecodedQaoaResult, decode_counts
from qaoa_tuner.qaoa.hamiltonian import CostHamiltonian, build_cost_hamiltonian
from qaoa_tuner.qaoa.optimizers import (
    OptimizerResult,
    minimize_cobyla,
    minimize_spsa,
)


@dataclass(frozen=True)
class QaoaExecutionResult:
    """Complete summary of a QAOA run.

    Attributes:
        graph: Target ColoringGraph instance.
        num_colors: Number of colors (k).
        p: QAOA depth.
        cost_hamiltonian: Constructed CostHamiltonian.
        optimizer_result: Detailed outcome of classical parameter optimization.
        decoded_result: Decoded measurement statistics and validation results.
        optimal_circuit: Final QuantumCircuit bound with optimal parameters.
    """

    graph: ColoringGraph
    num_colors: int
    p: int
    cost_hamiltonian: CostHamiltonian
    optimizer_result: OptimizerResult
    decoded_result: DecodedQaoaResult
    optimal_circuit: QuantumCircuit


class QaoaRunner:
    """Manages the full lifecycle of QAOA execution on ideal Aer simulation."""

    def __init__(
        self,
        graph: ColoringGraph,
        num_colors: int = 2,
        p: int = 1,
        shots: int = 1024,
        seed: int | None = 42,
        penalty_vertex: float = 2.0,
        penalty_edge: float = 1.0,
    ) -> None:
        """Initialize QAOA runner for a given graph and depth.

        Args:
            graph: Target ColoringGraph.
            num_colors: Number of colors (k).
            p: QAOA depth (number of layers).
            shots: Measurement shots per evaluation.
            seed: Random seed for simulation reproducibility.
            penalty_vertex: Vertex constraint weight A.
            penalty_edge: Edge conflict constraint weight B.
        """
        if p < 1:
            raise ValueError(f"QAOA depth p must be >= 1, got {p}.")
        if num_colors < 2:
            raise InvalidGraphError(f"Coloring requires >= 2 colors, got {num_colors}.")

        self.graph = graph
        self.num_colors = num_colors
        self.p = p
        self.shots = shots
        self.seed = seed

        self.cost_hamiltonian = build_cost_hamiltonian(
            graph=graph,
            num_colors=num_colors,
            penalty_vertex=penalty_vertex,
            penalty_edge=penalty_edge,
        )

        self.circuit, self.gammas, self.betas = build_qaoa_circuit(
            cost_hamiltonian=self.cost_hamiltonian,
            p=p,
            measure=True,
        )

        self.simulator = AerSimulator(seed_simulator=seed)
        # Transpile once to the simulator target to ensure optimal gate set execution
        self._compiled_template = transpile(self.circuit, self.simulator)

    def _bind_circuit(self, params: list[float] | np.ndarray) -> QuantumCircuit:
        """Bind gamma and beta parameters to the parameterized circuit."""
        if len(params) != 2 * self.p:
            raise ValueError(f"Expected {2 * self.p} parameters, got {len(params)}.")

        param_dict = {}
        for layer_idx in range(self.p):
            param_dict[self.gammas[layer_idx]] = float(params[layer_idx])
            param_dict[self.betas[layer_idx]] = float(params[self.p + layer_idx])

        return self._compiled_template.assign_parameters(param_dict)

    def evaluate_point(
        self,
        params: list[float] | np.ndarray,
        shots: int | None = None,
    ) -> DecodedQaoaResult:
        """Execute the QAOA circuit at fixed parameters and decode results."""
        run_shots = shots if shots is not None else self.shots
        bound_qc = self._bind_circuit(params)

        job = self.simulator.run(bound_qc, shots=run_shots)
        result = job.result()
        counts = result.get_counts()

        return decode_counts(
            counts=counts,
            graph=self.graph,
            num_colors=self.num_colors,
            encoding=self.cost_hamiltonian.encoding,
        )

    def objective_function(self, params: np.ndarray) -> float:
        """Scalar objective function for classical optimizers (minimizes expected conflicts)."""
        decoded = self.evaluate_point(params, shots=self.shots)
        return decoded.expected_conflicts

    def optimize(
        self,
        optimizer: str = "COBYLA",
        max_iter: int = 40,
        initial_point: list[float] | None = None,
        final_shots: int = 2048,
    ) -> QaoaExecutionResult:
        """Train variational parameters using the specified classical optimizer.

        Args:
            optimizer: "COBYLA" or "SPSA".
            max_iter: Maximum optimization iterations.
            initial_point: Optional initial parameter guess [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}].
                           Defaults to standard heuristic values (e.g. gamma ~ 0.5, beta ~ 0.5).
            final_shots: Shot count for evaluating the final optimized state.
        """
        if initial_point is None:
            # Heuristic initialization: gammas ~ 0.4, betas ~ 0.4
            initial_point = [0.4] * self.p + [0.4] * self.p

        opt_upper = optimizer.upper()
        if opt_upper == "COBYLA":
            opt_res = minimize_cobyla(
                objective_fn=self.objective_function,
                initial_point=initial_point,
                max_iter=max_iter,
            )
        elif opt_upper == "SPSA":
            opt_res = minimize_spsa(
                objective_fn=self.objective_function,
                initial_point=initial_point,
                max_iter=max_iter,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unsupported optimizer: '{optimizer}'. Supported: 'COBYLA', 'SPSA'.")

        # Evaluate final state with higher statistics
        final_decoded = self.evaluate_point(opt_res.optimal_point, shots=final_shots)
        optimal_qc = self._bind_circuit(opt_res.optimal_point)

        return QaoaExecutionResult(
            graph=self.graph,
            num_colors=self.num_colors,
            p=self.p,
            cost_hamiltonian=self.cost_hamiltonian,
            optimizer_result=opt_res,
            decoded_result=final_decoded,
            optimal_circuit=optimal_qc,
        )
