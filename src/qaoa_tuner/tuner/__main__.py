"""Command line: python -m qaoa_tuner.tuner"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qaoa_tuner.problem.generators import (
    create_complete_graph,
    create_cycle_graph,
    create_path_graph,
)
from qaoa_tuner.tuner.pareto import CORE_OBJECTIVES, DEFAULT_OBJECTIVES
from qaoa_tuner.tuner.space import ConfigurationSpace, TunerSettings
from qaoa_tuner.tuner.tuner import ConfigurationTuner

GRAPHS = {
    "c4": lambda: create_cycle_graph(4),
    "c5": lambda: create_cycle_graph(5),
    "k3": lambda: create_complete_graph(3),
    "p4": lambda: create_path_graph(4),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="QAOA-Tuner: grid search + Pareto analysis.")
    parser.add_argument("--graph", choices=sorted(GRAPHS), default="c4")
    parser.add_argument("--colors", type=int, default=2)
    parser.add_argument("--noise-profile", default="realistic_superconducting")
    parser.add_argument("--backend", default="fake_linear_5q")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--optimizers", nargs="+", default=["COBYLA", "SPSA"])
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--mitigations", nargs="+", default=["none", "zne", "readout"])
    parser.add_argument("--zne-scales", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument(
        "--quality-tolerance", type=float, default=0.0,
        help="Treat valid-rate differences below this as ties in the Pareto analysis.",
    )
    parser.add_argument(
        "--no-shots-objective", action="store_true",
        help="Use only quality, two-qubit gate budget and degradation (ignores shot overhead).",
    )
    parser.add_argument("--output-dir", default="data/tuner")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Root stays at WARNING so Qiskit's very chatty INFO logs are hidden; only our own
    # package logs progress when --verbose is given.
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if args.verbose:
        logging.getLogger("qaoa_tuner").setLevel(logging.INFO)

    graph = GRAPHS[args.graph]()
    settings = TunerSettings(
        graph_dict=graph.to_dict(), num_colors=args.colors, noise_profile=args.noise_profile,
        backend_name=args.backend, shots=args.shots, seed=args.seed, max_iter=args.max_iter,
        zne_scale_factors=tuple(args.zne_scales),
    )
    space = ConfigurationSpace(
        tuple(args.depths), tuple(args.optimizers), tuple(args.levels), tuple(args.mitigations)
    )
    tolerance = {"valid_coloring_rate": args.quality_tolerance} if args.quality_tolerance else None
    objectives = CORE_OBJECTIVES if args.no_shots_objective else DEFAULT_OBJECTIVES
    tuner = ConfigurationTuner(settings, space, objectives=objectives, tolerance=tolerance)

    total = space.size
    print(f"Evaluating {total} configurations (time depends on qubit count, shots and iterations)...")

    def progress(done: int, count: int, label: str) -> None:
        if done == count or done % 12 == 0:
            print(f"  {done}/{count} done (last: {label})")

    result = tuner.run(progress)

    groups = sorted(result.pareto_groups(), key=lambda g: -g.representative.valid_coloring_rate)
    names = ", ".join(o["name"] for o in result.objectives)
    print("=" * 108)
    print(f" {graph.name}, k={args.colors}, noise={args.noise_profile}, backend={args.backend}")
    print(f" Objectives: {names}")
    print(f" {len(result.pareto_records())} of {len(result.records)} configurations are "
          f"Pareto-efficient, forming {len(groups)} distinct trade-offs")
    print("=" * 108)
    print(f"{'configuration':<26}{'valid':>8}{'+/-se':>8}{'degrad.':>9}"
          f"{'2q gates':>10}{'2q budget':>11}{'depth':>7}{'shots':>8}  also reached by")
    for g in groups:
        r = g.representative
        se = f"{r.valid_rate_std_error:.3f}" if r.valid_rate_std_error is not None else "n/a"
        also = ", ".join(g.equivalent_labels)
        print(f"{r.label:<26}{r.valid_coloring_rate:>8.3f}{se:>8}{r.quality_degradation:>9.3f}"
              f"{r.two_qubit_gates:>10}{r.two_qubit_gate_budget:>11}{r.circuit_depth:>7}"
              f"{r.total_shots:>8}  {also}")

    stem = f"{graph.name}_k{args.colors}_{args.noise_profile}_{args.backend}_seed{args.seed}"
    out = Path(args.output_dir)
    json_path = result.save_json(out / (stem + ".json"))
    print(f"Saved: {json_path}")
    print(f"Saved: {result.save_csv(out / (stem + '.csv'))}")
    print(f"Next:  python -m qaoa_tuner.recommendation {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
