"""Evaluate one configuration: train, transpile, execute under noise, optionally mitigate.

Protocol (same logic as the phase 6 mitigation benchmark):

1. TRAIN once per (depth p, optimizer) on an IDEAL simulator. Training depends only on p and
   the optimizer, so the 72 grid points need just 6 trainings. LIMITATION: parameters are
   transferred from an ideal run; training under noise could find different parameters.
2. TRANSPILE the bound circuit for the chosen backend at the chosen optimization level
   (this fixes depth, routing and the number of two-qubit gates).
3. EXECUTE the transpiled circuit on Aer with the documented noise model. Because the circuit
   that runs IS the transpiled one, the transpiler level and backend topology affect quality
   (extra routed gates pick up extra noise), not only cost.
4. MITIGATE: none, readout (calibrated once, reused), or ZNE (fold the transpiled circuit,
   run without re-optimizing, extrapolate).

Shared work is cached, so e.g. the "none" and "zne" records of one (p, optimizer, level) reuse
the same noisy run at scale factor 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from qiskit_aer import AerSimulator

from qaoa_tuner.compilation.backends import BackendProvider
from qaoa_tuner.compilation.profiler import profile_circuit
from qaoa_tuner.compilation.transpiler import TranspilationResult, transpile_qaoa_circuit
from qaoa_tuner.mitigation.readout import (
    TensoredReadoutCalibrator,
    calibrate_tensored_readout_mitigation,
)
from qaoa_tuner.mitigation.zne import extrapolate_zero_noise, fold_circuit_gates
from qaoa_tuner.noise.models import build_noise_model, get_preset_noise_config
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.qaoa.decoder import DecodedQaoaResult, decode_counts
from qaoa_tuner.qaoa.hamiltonian import build_cost_hamiltonian
from qaoa_tuner.qaoa.runner import QaoaRunner
from qaoa_tuner.tuner.results import TunerRecord
from qaoa_tuner.tuner.space import TunerConfig, TunerSettings

logger = logging.getLogger(__name__)

_NON_GATE_OPS = {"measure", "barrier", "delay", "reset"}


@dataclass
class _Trained:
    runner: QaoaRunner
    point: list[float]
    ideal_valid_rate: float
    evaluations: int
    final_objective: float


@dataclass
class _Compiled:
    transpilation: TranspilationResult
    decoded: DecodedQaoaResult


def _binomial_std_error(p: float, shots: int) -> float:
    return float((max(p * (1.0 - p), 0.0) / shots) ** 0.5)


class ConfigurationEvaluator:
    """Runs TunerConfig points against fixed TunerSettings, caching shared computation."""

    def __init__(self, settings: TunerSettings) -> None:
        self.settings = settings
        self.graph = ColoringGraph.from_dict(settings.graph_dict)
        self.noise_cfg = get_preset_noise_config(settings.noise_profile)
        self.noise_model = build_noise_model(self.noise_cfg)
        if self.noise_model is None:
            raise ValueError(
                f"Noise profile '{settings.noise_profile}' has no noise; the tuner needs a "
                "noisy profile to compare mitigation and compilation choices."
            )

        cost = build_cost_hamiltonian(self.graph, settings.num_colors)
        self.num_qubits = cost.num_qubits
        self.encoding = cost.encoding

        backend = BackendProvider.get_backend(
            settings.backend_name, num_qubits=max(self.num_qubits, 5), seed=settings.seed
        )
        is_aer = isinstance(backend, AerSimulator)
        limit = getattr(backend, "num_qubits", None)
        if not is_aer and isinstance(limit, int) and 0 < limit < self.num_qubits:
            raise ValueError(
                f"Backend '{settings.backend_name}' has {limit} qubits but this problem needs "
                f"{self.num_qubits} ({self.encoding}, k={settings.num_colors}). Choose a larger "
                "backend (e.g. fake_generic_12q) or a smaller problem."
            )

        # Execution always uses a noisy Aer simulator. For an unconstrained Aer target we also
        # transpile against the NOISY simulator, so only gates that carry noise are emitted.
        self.simulator = AerSimulator(noise_model=self.noise_model, seed_simulator=settings.seed)
        self.target_backend = self.simulator if is_aer else backend
        self.backend_info = BackendProvider.query_backend_info(self.target_backend)

        self._trained: dict[tuple[int, str], _Trained] = {}
        self._compiled: dict[tuple[int, str, int], _Compiled] = {}
        self._zne: dict[tuple[int, str, int], tuple[list[float], list[float], int]] = {}
        self._calibrator: TensoredReadoutCalibrator | None = None

    # ------------------------------------------------------------------ stages ----
    def _train(self, p: int, optimizer: str) -> _Trained:
        key = (p, optimizer)
        if key not in self._trained:
            s = self.settings
            logger.info("Training p=%d optimizer=%s on ideal simulator", p, optimizer)
            runner = QaoaRunner(
                graph=self.graph, num_colors=s.num_colors, p=p, shots=s.shots,
                seed=s.seed, noise_model=None,
            )
            result = runner.optimize(
                optimizer=optimizer, max_iter=s.max_iter, initial_point=None, final_shots=s.shots
            )
            self._trained[key] = _Trained(
                runner=runner,
                point=[float(x) for x in result.optimizer_result.optimal_point],
                ideal_valid_rate=result.decoded_result.valid_coloring_probability,
                evaluations=result.optimizer_result.num_evaluations,
                final_objective=float(result.optimizer_result.optimal_value),
            )
        return self._trained[key]

    def _uncovered_gates(self, circuit) -> list[str]:
        covered = set(self.noise_model.noise_instructions)
        return sorted(
            name for name in circuit.count_ops() if name not in covered and name not in _NON_GATE_OPS
        )

    def _compile_and_run(self, p: int, optimizer: str, level: int) -> _Compiled:
        key = (p, optimizer, level)
        if key not in self._compiled:
            s = self.settings
            trained = self._train(p, optimizer)
            runner = trained.runner
            params = {}
            for i in range(p):
                params[runner.gammas[i]] = trained.point[i]
                params[runner.betas[i]] = trained.point[p + i]
            bound_logical = runner.circuit.assign_parameters(params)

            tr = transpile_qaoa_circuit(
                bound_logical, self.target_backend, optimization_level=level,
                seed_transpiler=s.seed,
            )
            uncovered = self._uncovered_gates(tr.transpiled_circuit)
            if uncovered:
                logger.warning("Gates without noise in executed circuit %s: %s", key, uncovered)

            counts = self.simulator.run(tr.transpiled_circuit, shots=s.shots).result().get_counts()
            decoded = decode_counts(counts, self.graph, s.num_colors, self.encoding)
            self._compiled[key] = _Compiled(tr, decoded)
        return self._compiled[key]

    def _readout_calibrator(self) -> TensoredReadoutCalibrator:
        if self._calibrator is None:
            s = self.settings
            self._calibrator = calibrate_tensored_readout_mitigation(
                self.num_qubits, shots=s.calibration_shots or s.shots,
                noise_model=self.noise_model, seed=s.seed,
            )
        return self._calibrator

    def _zne_points(self, p: int, optimizer: str, level: int):
        key = (p, optimizer, level)
        if key not in self._zne:
            s = self.settings
            compiled = self._compile_and_run(p, optimizer, level)
            circuit = compiled.transpilation.transpiled_circuit
            scales = sorted({int(x) for x in s.zne_scale_factors})
            valid_vals, conflict_vals, budget = [], [], 0
            for scale in scales:
                folded = fold_circuit_gates(circuit, scale)
                budget += profile_circuit(folded).two_qubit_gate_count
                if scale == 1:
                    decoded = compiled.decoded  # identical circuit and seed: reuse the run
                else:
                    counts = self.simulator.run(folded, shots=s.shots).result().get_counts()
                    decoded = decode_counts(counts, self.graph, s.num_colors, self.encoding)
                valid_vals.append(decoded.valid_coloring_probability)
                conflict_vals.append(decoded.expected_conflicts)
            self._zne[key] = (valid_vals, conflict_vals, budget)
        return self._zne[key]

    # ---------------------------------------------------------------- public ----
    def training_summary(self) -> list[dict]:
        return [
            {
                "qaoa_p": p, "optimizer": opt, "trained_on": "ideal simulator",
                "evaluations": t.evaluations, "final_objective_value": t.final_objective,
                "ideal_valid_coloring_rate": t.ideal_valid_rate,
            }
            for (p, opt), t in sorted(self._trained.items())
        ]

    def counts_for(self, config: TunerConfig) -> dict[str, int] | None:
        """Measured counts behind a configuration's estimate (None for ZNE: scalar estimate only)."""
        base = self._compile_and_run(
            config.qaoa_p, config.optimizer, config.optimization_level
        ).decoded.counts
        if config.mitigation == "none":
            return dict(base)
        if config.mitigation == "readout":
            return self._readout_calibrator().mitigate_counts(base)
        return None

    def evaluate(self, config: TunerConfig) -> TunerRecord:
        """Run one configuration and return its measured record."""
        s = self.settings
        trained = self._train(config.qaoa_p, config.optimizer)
        compiled = self._compile_and_run(config.qaoa_p, config.optimizer, config.optimization_level)
        tr = compiled.transpilation
        base = compiled.decoded
        base_2q = tr.transpiled_metrics.two_qubit_gate_count

        if config.mitigation == "none":
            valid = base.valid_coloring_probability
            conflicts = base.expected_conflicts
            std_err: float | None = _binomial_std_error(valid, s.shots)
            budget, total_shots = base_2q, s.shots

        elif config.mitigation == "readout":
            calibrator = self._readout_calibrator()
            mitigated = calibrator.mitigate_counts(base.counts)
            dec = decode_counts(mitigated, self.graph, s.num_colors, self.encoding)
            valid, conflicts, std_err = dec.valid_coloring_probability, dec.expected_conflicts, None
            budget = base_2q  # calibration circuits contain no two-qubit gates
            total_shots = s.shots + calibrator.calibration_overhead_shots

        else:  # "zne"
            valid_vals, conflict_vals, budget = self._zne_points(
                config.qaoa_p, config.optimizer, config.optimization_level
            )
            scales = sorted({int(x) for x in s.zne_scale_factors})
            vfit = extrapolate_zero_noise(
                scales, valid_vals, s.zne_model,
                std_errs=[_binomial_std_error(v, s.shots) for v in valid_vals],
            )
            cfit = extrapolate_zero_noise(scales, conflict_vals, s.zne_model)
            valid = min(max(vfit.extrapolated_zero_noise_value, 0.0), 1.0)  # probabilities
            conflicts = max(cfit.extrapolated_zero_noise_value, 0.0)
            std_err = vfit.extrapolated_std_error
            total_shots = s.shots * len(scales)

        return TunerRecord(
            label=config.label,
            qaoa_p=config.qaoa_p,
            optimizer=config.optimizer,
            optimization_level=config.optimization_level,
            mitigation=config.mitigation,
            valid_coloring_rate=valid,
            valid_rate_std_error=std_err,
            expected_conflicts=conflicts,
            ideal_valid_coloring_rate=trained.ideal_valid_rate,
            quality_degradation=trained.ideal_valid_rate - valid,
            circuit_depth=tr.transpiled_metrics.depth,
            two_qubit_gates=base_2q,
            one_qubit_gates=tr.transpiled_metrics.one_qubit_gate_count,
            logical_two_qubit_gates=tr.logical_metrics.two_qubit_gate_count,
            two_qubit_overhead_ratio=tr.two_qubit_overhead_ratio,
            two_qubit_gate_budget=int(budget),
            total_shots=int(total_shots),
            uncovered_gates=self._uncovered_gates(tr.transpiled_circuit),
        )
