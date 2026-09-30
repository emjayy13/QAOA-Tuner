"""Command-line interface for running reproducible QAOA experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src is in sys.path when invoked directly as a script
_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from qaoa_tuner.experiment.config import ExperimentConfig  # noqa: E402
from qaoa_tuner.experiment.engine import ExperimentEngine  # noqa: E402
from qaoa_tuner.problem.generators import (  # noqa: E402
    create_bipartite_graph,
    create_complete_graph,
    create_cycle_graph,
    create_path_graph,
    create_random_erdos_renyi,
)


def get_graph_by_name(name: str):
    """Factory creating benchmark graphs from CLI names."""
    clean = name.lower()
    if clean in ["cycle4", "c4"]:
        return create_cycle_graph(4)
    elif clean in ["cycle5", "c5"]:
        return create_cycle_graph(5)
    elif clean in ["triangle", "k3"]:
        return create_complete_graph(3)
    elif clean in ["k4"]:
        return create_complete_graph(4)
    elif clean in ["path4", "p4"]:
        return create_path_graph(4)
    elif clean.startswith("bipartite"):
        return create_bipartite_graph(3, 3, seed=42)
    elif clean.startswith("er"):
        return create_random_erdos_renyi(5, 0.5, seed=42)
    else:
        # Default fallback: 4-cycle
        return create_cycle_graph(4)


def main() -> int:
    """CLI entry point for QAOA-Tuner."""
    parser = argparse.ArgumentParser(
        description="QAOA-Tuner: Run quantum graph-coloring experiments."
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="c4",
        help="Graph name: c4, c5, k3, k4, p4, bipartite, er (default: c4)",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=2,
        help="Number of colors (k >= 2, default: 2)",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=1,
        help="QAOA depth p (default: 1)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="COBYLA",
        choices=["COBYLA", "SPSA"],
        help="Classical optimizer (default: COBYLA)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=25,
        help="Maximum optimizer iterations (default: 25)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Measurement shots (default: 1024)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments",
        help="Directory to save JSON experiment output",
    )

    args = parser.parse_args()

    graph = get_graph_by_name(args.graph)
    config = ExperimentConfig(
        graph_dict=graph.to_dict(),
        num_colors=args.colors,
        qaoa_p=args.p,
        optimizer=args.optimizer,
        max_iter=args.max_iter,
        shots=args.shots,
        seed=args.seed,
        name=f"{graph.name}_k{args.colors}_p{args.p}_{args.optimizer.lower()}",
    )

    print("=" * 60)
    print(f" QAOA-Tuner Experiment: {config.name}")
    print(f" Graph: {graph.name} (|V|={graph.num_nodes}, |E|={graph.num_edges})")
    print(f" Colors: {config.num_colors} | QAOA Depth p: {config.qaoa_p} | Optimizer: {config.optimizer}")
    print("=" * 60)

    engine = ExperimentEngine()
    result, path = engine.run_and_save(config, output_dir=args.output_dir)

    print("\n--- RESULTS ---")
    print(f"Classical Chromatic Number chi(G): {result.solution_metrics.classical_chromatic_number}")
    print(f"Valid Coloring Found:             {result.solution_metrics.is_valid_solution_found}")
    print(f"Best Observed Coloring:           {result.solution_metrics.best_coloring}")
    print(f"Best Coloring Conflicts:          {result.solution_metrics.best_coloring_conflicts}")
    print(f"Valid Coloring Rate:              {result.solution_metrics.valid_coloring_rate:.2%}")
    print(f"Optimal Objective Value:          {result.execution_metrics.final_objective_value:.4f}")
    print(f"Execution Wall Time:              {result.execution_metrics.wall_clock_time_seconds:.2f}s")
    print(f"Saved Results to:                 {path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
