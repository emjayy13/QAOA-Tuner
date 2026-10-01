"""Ideal -> noisy -> mitigated comparison with measured improvement and overhead.

Protocol (the same optimal parameters are evaluated four ways):
    1. TRAIN   QAOA parameters once on an IDEAL simulator (noise-free optimization).
    2. IDEAL   evaluate those parameters on the ideal simulator.
    3. NOISY   evaluate the same parameters under the chosen noise model.
    4. READOUT reuse the noisy counts and apply readout mitigation (extra cost: calibration).
    5. ZNE     run folded circuits (scale 1, 3, 5, ...) under noise, extrapolate to zero noise.

Why train ideal and then evaluate everywhere: it isolates the effect of noise and of
mitigation from the effect of training under noise. (``compare_ideal_vs_noisy`` in
``qaoa_tuner.noise.benchmark`` instead trains inside the noisy simulator; both are valid
but answer slightly different questions.)

Everything reported here comes from executed simulations. Nothing is estimated or hard-coded.
Single runs are subject to shot noise: look at ``valid_rate_std_error`` before reading
meaning into small differences.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import qiskit
import qiskit_aer

from qaoa_tuner.compilation.profiler import profile_circuit
from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.mitigation.readout import (
    calibrate_readout_mitigation,
    calibrate_tensored_readout_mitigation,
)
from qaoa_tuner.mitigation.zne import ZNEResult, extrapolate_zero_noise, fold_circuit_gates
from qaoa_tuner.noise.models import NoiseConfig, build_noise_model, get_preset_noise_config
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.qaoa.decoder import decode_counts
from qaoa_tuner.qaoa.runner import QaoaRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageMetrics:
    """Quality and cost of one stage of the comparison.

    Attributes:
        label: Stage name ("ideal", "noisy", "noisy+readout", "noisy+zne").
        valid_coloring_rate: Probability mass on valid colorings.
        expected_conflicts: Expected number of constraint violations.
        valid_rate_std_error: Binomial shot-noise standard error of valid_coloring_rate
            where meaningful (raw and ZNE stages); None for readout-mitigated results
            because the matrix inversion changes the statistics.
        shots_per_circuit: Shots used for each execution of the main circuit.
        extra_circuits: Circuits executed beyond the single unmitigated noisy evaluation.
        extra_shots: Shots executed beyond the single unmitigated noisy evaluation.
        max_two_qubit_gates_per_circuit: Two-qubit gates in the deepest executed circuit.
        wall_time_seconds: Time spent in this stage.
        notes: Plain-language caveats for this stage.
    """

    label: str
    valid_coloring_rate: float
    expected_conflicts: float
    valid_rate_std_error: float | None
    shots_per_circuit: int
    extra_circuits: int
    extra_shots: int
    max_two_qubit_gates_per_circuit: int
    wall_time_seconds: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MitigationComparisonResult:
    """Full, serializable record of an ideal / noisy / mitigated comparison."""

    config: dict[str, Any]
    noise_profile: dict[str, Any]
    optimal_point: list[float]
    training: dict[str, Any]
    ideal: StageMetrics
    noisy: StageMetrics
    readout: StageMetrics
    zne: StageMetrics
    readout_calibration: dict[str, Any]
    zne_valid_rate_fit: dict[str, Any]
    zne_conflicts_fit: dict[str, Any]
    zne_valid_rate_unclipped: float
    derived: dict[str, Any]
    software: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "noise_profile": self.noise_profile,
            "optimal_point": self.optimal_point,
            "training": self.training,
            "stages": {
                "ideal": self.ideal.to_dict(),
                "noisy": self.noisy.to_dict(),
                "readout": self.readout.to_dict(),
                "zne": self.zne.to_dict(),
            },
            "readout_calibration": self.readout_calibration,
            "zne_valid_rate_fit": self.zne_valid_rate_fit,
            "zne_conflicts_fit": self.zne_conflicts_fit,
            "zne_valid_rate_unclipped": self.zne_valid_rate_unclipped,
            "derived": self.derived,
            "software": self.software,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _binomial_std_error(p: float, shots: int) -> float:
    return float((max(p * (1.0 - p), 0.0) / shots) ** 0.5)


def _recovered_fraction(ideal: float, noisy: float, mitigated: float) -> float | None:
    """Fraction of the ideal-vs-noisy gap closed by mitigation (None if there is no gap)."""
    gap = ideal - noisy
    if gap < 1e-9:
        return None
    return float((mitigated - noisy) / gap)


def compare_mitigation_performance(
    config: ExperimentConfig,
    noise_profile: str | NoiseConfig = "realistic_superconducting",
    zne_scale_factors: tuple[int, ...] = (1, 3, 5),
    zne_model: str = "linear",
    readout_method: str = "tensored",
    calibration_shots: int | None = None,
) -> MitigationComparisonResult:
    """Run the ideal / noisy / readout-mitigated / ZNE-mitigated comparison.

    Args:
        config: Graph, colors, depth, optimizer, shots, seed, initial point.
            (Its noise/mitigation fields are ignored; ``noise_profile`` decides the noise.)
        noise_profile: Preset name or NoiseConfig. Must contain some noise.
        zne_scale_factors: Odd integers; must include 1 for a meaningful extrapolation.
        zne_model: "linear" or "polynomial".
        readout_method: "tensored" (2 calibration circuits) or "full" (2^n, n <= 8).
        calibration_shots: Shots per calibration circuit (default: config.shots).
    """
    if readout_method not in ("tensored", "full"):
        raise ValueError(f"readout_method must be 'tensored' or 'full', got '{readout_method}'.")
    if 1 not in zne_scale_factors:
        raise ValueError("zne_scale_factors must include 1 (the unscaled circuit).")

    if isinstance(noise_profile, str):
        noise_cfg = get_preset_noise_config(noise_profile)
    else:
        noise_cfg = noise_profile
    noise_model = build_noise_model(noise_cfg)
    if noise_model is None:
        raise ValueError("Mitigation comparison needs a noisy profile; got an ideal one.")

    shots = config.shots
    cal_shots = calibration_shots if calibration_shots is not None else shots
    graph = ColoringGraph.from_dict(config.graph_dict)
    k = config.num_colors
    logger.info(
        "Mitigation comparison: graph=%s k=%d p=%d optimizer=%s noise=%s shots=%d seed=%s",
        graph.name, k, config.qaoa_p, config.optimizer, noise_cfg.name, shots, config.seed,
    )

    def make_runner(nm: Any) -> QaoaRunner:
        return QaoaRunner(
            graph=graph, num_colors=k, p=config.qaoa_p, shots=shots,
            seed=config.seed, noise_model=nm,
        )

    # --- 1 & 2. Train on the ideal simulator, evaluate ideal -----------------------------
    t0 = time.perf_counter()
    ideal_runner = make_runner(None)
    ideal_exec = ideal_runner.optimize(
        optimizer=config.optimizer,
        max_iter=config.max_iter,
        initial_point=config.initial_point,
        final_shots=shots,
    )
    point = [float(x) for x in ideal_exec.optimizer_result.optimal_point]
    ideal_dec = ideal_exec.decoded_result
    t_ideal = time.perf_counter() - t0

    # --- 3. Noisy evaluation of the SAME parameters --------------------------------------
    t0 = time.perf_counter()
    noisy_runner = make_runner(noise_model)
    bound = noisy_runner._bind_circuit(point)  # transpiled to the Aer target, parameters bound
    base_2q = profile_circuit(bound).two_qubit_gate_count
    noisy_dec = noisy_runner.evaluate_point(point, shots=shots)
    t_noisy = time.perf_counter() - t0
    n_qubits = noisy_runner.cost_hamiltonian.num_qubits

    # --- 4. Readout mitigation (calibrate, then correct the noisy counts) ----------------
    t0 = time.perf_counter()
    if readout_method == "tensored":
        calibrator = calibrate_tensored_readout_mitigation(
            n_qubits, shots=cal_shots, noise_model=noise_model, seed=config.seed or 42
        )
    else:
        calibrator = calibrate_readout_mitigation(
            num_qubits=n_qubits, shots=cal_shots, noise_model=noise_model, seed=config.seed or 42
        )
    readout_counts = calibrator.mitigate_counts(noisy_dec.counts)
    readout_dec = decode_counts(readout_counts, graph, k, noisy_runner.cost_hamiltonian.encoding)
    t_readout = time.perf_counter() - t0

    # --- 5. ZNE: fold, run under noise, extrapolate --------------------------------------
    t0 = time.perf_counter()
    scales = sorted(set(int(s) for s in zne_scale_factors))
    valid_vals: list[float] = []
    conflict_vals: list[float] = []
    max_2q = base_2q
    for scale in scales:
        folded = fold_circuit_gates(bound, scale)
        max_2q = max(max_2q, profile_circuit(folded).two_qubit_gate_count)
        # Run the folded circuit directly (no transpile) so the G^dagger G pairs are kept.
        counts = noisy_runner.simulator.run(folded, shots=shots).result().get_counts()
        dec = decode_counts(counts, graph, k, noisy_runner.cost_hamiltonian.encoding)
        valid_vals.append(dec.valid_coloring_probability)
        conflict_vals.append(dec.expected_conflicts)

    valid_fit: ZNEResult = extrapolate_zero_noise(
        scales, valid_vals, model=zne_model,
        std_errs=[_binomial_std_error(v, shots) for v in valid_vals],
    )
    conflict_fit: ZNEResult = extrapolate_zero_noise(scales, conflict_vals, model=zne_model)
    raw_valid = valid_fit.extrapolated_zero_noise_value
    zne_valid = min(max(raw_valid, 0.0), 1.0)
    zne_conflicts = max(conflict_fit.extrapolated_zero_noise_value, 0.0)
    t_zne = time.perf_counter() - t0

    zne_notes = (
        f"Extrapolated scalar metrics only (no mitigated distribution). Folds two-qubit gates "
        f"only; scales {scales}, model={zne_model}."
    )
    if zne_valid != raw_valid:
        zne_notes += f" Valid rate clipped from {raw_valid:.4f} into [0, 1]."

    # --- Assemble stage records ------------------------------------------------------------
    ideal_stage = StageMetrics(
        "ideal", ideal_dec.valid_coloring_probability, ideal_dec.expected_conflicts,
        _binomial_std_error(ideal_dec.valid_coloring_probability, shots),
        shots, 0, 0, base_2q, t_ideal, "Noise-free simulator; also where parameters were trained.",
    )
    noisy_stage = StageMetrics(
        "noisy", noisy_dec.valid_coloring_probability, noisy_dec.expected_conflicts,
        _binomial_std_error(noisy_dec.valid_coloring_probability, shots),
        shots, 0, 0, base_2q, t_noisy, f"Noise profile '{noise_cfg.name}'; unmitigated baseline.",
    )
    readout_stage = StageMetrics(
        "noisy+readout", readout_dec.valid_coloring_probability, readout_dec.expected_conflicts,
        None, shots, calibrator.num_calibration_circuits, calibrator.calibration_overhead_shots,
        base_2q, t_readout,
        f"Reuses the noisy counts; only calibration is extra ({readout_method} calibrator, "
        f"condition number {calibrator.condition_number:.3f}). Gate noise is not corrected.",
    )
    zne_stage = StageMetrics(
        "noisy+zne", zne_valid, zne_conflicts, valid_fit.extrapolated_std_error, shots,
        len(scales) - 1, (len(scales) - 1) * shots, max_2q, t_zne, zne_notes,
    )

    gap = ideal_stage.valid_coloring_rate - noisy_stage.valid_coloring_rate
    noisy_se = noisy_stage.valid_rate_std_error or 0.0
    derived = {
        "degradation_abs": gap,
        "degradation_rel": gap / ideal_stage.valid_coloring_rate
        if ideal_stage.valid_coloring_rate > 0 else None,
        "degradation_in_noisy_std_errors": gap / noisy_se if noisy_se > 0 else None,
        "readout_improvement_abs": readout_stage.valid_coloring_rate - noisy_stage.valid_coloring_rate,
        "zne_improvement_abs": zne_stage.valid_coloring_rate - noisy_stage.valid_coloring_rate,
        "readout_recovered_fraction": _recovered_fraction(
            ideal_stage.valid_coloring_rate, noisy_stage.valid_coloring_rate,
            readout_stage.valid_coloring_rate),
        "zne_recovered_fraction": _recovered_fraction(
            ideal_stage.valid_coloring_rate, noisy_stage.valid_coloring_rate,
            zne_stage.valid_coloring_rate),
    }

    return MitigationComparisonResult(
        config=config.to_dict(),
        noise_profile=noise_cfg.to_dict(),
        optimal_point=point,
        training={
            "trained_on": "ideal simulator",
            "optimizer": config.optimizer,
            "evaluations": ideal_exec.optimizer_result.num_evaluations,
            "final_objective_value": ideal_exec.optimizer_result.optimal_value,
        },
        ideal=ideal_stage,
        noisy=noisy_stage,
        readout=readout_stage,
        zne=zne_stage,
        readout_calibration={
            "method": readout_method,
            "num_qubits": n_qubits,
            "calibration_circuits": calibrator.num_calibration_circuits,
            "calibration_shots_total": calibrator.calibration_overhead_shots,
            "condition_number": calibrator.condition_number,
        },
        zne_valid_rate_fit=valid_fit.to_dict(),
        zne_conflicts_fit=conflict_fit.to_dict(),
        zne_valid_rate_unclipped=raw_valid,
        derived=derived,
        software={
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "qiskit_aer": qiskit_aer.__version__,
        },
    )


def plot_mitigation_comparison(
    result: MitigationComparisonResult,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Bar chart of valid-coloring probability across the four stages (warm palette)."""
    stages = [result.ideal, result.noisy, result.readout, result.zne]
    colors = ["#2D6A4F", "#C85A32", "#D49A3A", "#A67C52"]
    labels = [s.label for s in stages]
    values = [s.valid_coloring_rate for s in stages]
    errors = [s.valid_rate_std_error or 0.0 for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#FAF8F5")
    ax.set_facecolor("#FFFFFF")
    ax.bar(labels, values, yerr=errors, capsize=5, color=colors, edgecolor="#262626", alpha=0.9)
    ax.set_ylabel("Valid-coloring probability", color="#262626")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Mitigation comparison ({result.noise_profile['name']}, "
        f"p={result.config['qaoa_p']}, shots={result.config['shots']})\n"
        "Error bars: binomial shot noise (not shown for readout stage)",
        fontsize=11, color="#262626",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3, color="#8C827A")
    for spine in ax.spines.values():
        spine.set_color("#D4CCC5")
    ax.tick_params(colors="#262626")
    plt.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, facecolor=fig.get_facecolor())
    return fig


def main() -> int:
    """Command line entry: python -m qaoa_tuner.mitigation.benchmark"""
    from qaoa_tuner.problem.generators import (
        create_complete_graph,
        create_cycle_graph,
        create_path_graph,
    )

    graphs = {
        "c4": lambda: create_cycle_graph(4),
        "c5": lambda: create_cycle_graph(5),
        "k3": lambda: create_complete_graph(3),
        "p4": lambda: create_path_graph(4),
    }
    parser = argparse.ArgumentParser(description="Ideal vs noisy vs mitigated QAOA comparison.")
    parser.add_argument("--graph", choices=sorted(graphs), default="c4")
    parser.add_argument("--colors", type=int, default=2)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--optimizer", choices=["COBYLA", "SPSA"], default="COBYLA")
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-profile", default="realistic_superconducting")
    parser.add_argument("--readout-method", choices=["tensored", "full"], default="tensored")
    parser.add_argument("--zne-scales", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--zne-model", choices=["linear", "polynomial"], default="linear")
    parser.add_argument("--output-dir", default="data/mitigation")
    parser.add_argument("--plot", action="store_true", help="Also save a PNG next to the JSON.")
    args = parser.parse_args()

    graph = graphs[args.graph]()
    name = f"{graph.name}_k{args.colors}_p{args.p}_{args.optimizer.lower()}_{args.noise_profile}"
    config = ExperimentConfig(
        graph_dict=graph.to_dict(), num_colors=args.colors, qaoa_p=args.p,
        optimizer=args.optimizer, max_iter=args.max_iter, shots=args.shots,
        seed=args.seed, name=name,
    )
    result = compare_mitigation_performance(
        config, args.noise_profile, tuple(args.zne_scales), args.zne_model, args.readout_method
    )

    print("=" * 72)
    print(f" {name}")
    print("=" * 72)
    print(f"{'stage':<16}{'valid rate':>12}{'+/- se':>10}{'exp. conflicts':>16}{'extra shots':>13}")
    for s in (result.ideal, result.noisy, result.readout, result.zne):
        se = f"{s.valid_rate_std_error:.4f}" if s.valid_rate_std_error is not None else "n/a"
        print(f"{s.label:<16}{s.valid_coloring_rate:>12.4f}{se:>10}"
              f"{s.expected_conflicts:>16.4f}{s.extra_shots:>13}")
    print("-" * 72)
    for key, val in result.derived.items():
        print(f"{key:<36}{'n/a' if val is None else f'{val:.4f}'}")

    out_dir = Path(args.output_dir)
    saved = result.save(out_dir / f"{name}.json")
    print(f"Saved: {saved}")
    if args.plot:
        png = out_dir / f"{name}.png"
        plot_mitigation_comparison(result, png)
        print(f"Plot:  {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
