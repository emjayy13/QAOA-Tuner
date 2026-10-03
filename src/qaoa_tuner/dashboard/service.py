"""Backend glue for the dashboard. No Streamlit imports, so it is testable on its own."""

from __future__ import annotations

import csv
import io
import logging
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx

from qaoa_tuner.problem.generators import (
    create_complete_graph,
    create_cycle_graph,
    create_path_graph,
)
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.tuner.evaluator import ConfigurationEvaluator
from qaoa_tuner.tuner.results import TunerRecord, TunerResult
from qaoa_tuner.tuner.space import ConfigurationSpace, TunerConfig, TunerSettings
from qaoa_tuner.tuner.tuner import ConfigurationTuner

logger = logging.getLogger(__name__)

GRAPH_KINDS = ("cycle", "path", "complete", "random")
NOISE_PROFILES = ("realistic_superconducting", "depolarizing_mild", "readout_only")
BACKENDS = ("fake_linear_5q", "aer_simulator_ideal")
MITIGATIONS = ("none", "readout", "zne")
MAX_QUBITS = 12  # beyond this the statevector simulations become too slow for a dashboard


def qubits_needed(num_nodes: int, num_colors: int) -> int:
    """k=2 uses 1 qubit per vertex; k>=3 uses the one-hot encoding with k qubits per vertex."""
    return num_nodes if num_colors == 2 else num_nodes * num_colors


def build_graph(kind: str, num_nodes: int, density: float = 0.5, seed: int = 7) -> ColoringGraph:
    """Create a graph from simple controls. Raises ValueError with a readable message."""
    if kind == "cycle":
        if num_nodes < 3:
            raise ValueError("A cycle needs at least 3 vertices.")
        return create_cycle_graph(num_nodes)
    if kind == "path":
        if num_nodes < 2:
            raise ValueError("A path needs at least 2 vertices.")
        return create_path_graph(num_nodes)
    if kind == "complete":
        if num_nodes < 2:
            raise ValueError("A complete graph needs at least 2 vertices.")
        return create_complete_graph(num_nodes)
    if kind == "random":
        if num_nodes < 2:
            raise ValueError("A random graph needs at least 2 vertices.")
        if not 0.0 < density <= 1.0:
            raise ValueError("Edge density must be in (0, 1].")
        g = nx.gnp_random_graph(num_nodes, density, seed=seed)
        edges = [(int(u), int(v)) for u, v in g.edges()]
        if not edges:
            raise ValueError("This random graph has no edges; raise the density or change the seed.")
        name = f"random_n{num_nodes}_d{density:.2f}_s{seed}"
        return ColoringGraph.from_dict({"name": name, "num_nodes": num_nodes, "edges": edges})
    raise ValueError(f"Unknown graph type '{kind}'. Choose one of {GRAPH_KINDS}.")


def validate_problem(graph: ColoringGraph, num_colors: int) -> list[str]:
    """Return human-readable reasons this problem cannot be run (empty list = fine)."""
    errors: list[str] = []
    if num_colors < 2:
        errors.append("At least 2 colors are required.")
        return errors
    if not graph.edges:
        errors.append("The graph has no edges, so every coloring is valid.")
        return errors

    nxg = nx.Graph()
    nxg.add_nodes_from(range(graph.num_nodes))
    nxg.add_edges_from(graph.edges)

    omega = max(len(c) for c in nx.find_cliques(nxg))
    if num_colors < omega:
        errors.append(f"This graph contains a clique of size {omega}, so it needs at least {omega} colors.")
    elif num_colors == 2 and not nx.is_bipartite(nxg):
        errors.append("With 2 colors the graph must be bipartite (no odd cycle); this one is not.")

    qubits = qubits_needed(graph.num_nodes, num_colors)
    if qubits > MAX_QUBITS:
        errors.append(
            f"This problem needs {qubits} qubits (limit {MAX_QUBITS} for interactive simulation). "
            "Use fewer vertices or colors."
        )
    return errors


def make_settings(
    graph: ColoringGraph, num_colors: int, noise_profile: str, backend_name: str,
    shots: int, seed: int, max_iter: int, zne_scale_factors: list[int],
) -> TunerSettings:
    return TunerSettings(
        graph_dict=graph.to_dict(), num_colors=num_colors, noise_profile=noise_profile,
        backend_name=backend_name, shots=shots, seed=seed, max_iter=max_iter,
        zne_scale_factors=tuple(sorted(set(zne_scale_factors))),
    )


@dataclass(frozen=True)
class LadderResult:
    """One circuit evaluated three ways. ``counts[mitigation]`` is None for ZNE."""

    records: list[TunerRecord]
    counts: dict[str, dict[str, int] | None]


def run_ladder(
    settings: TunerSettings, qaoa_p: int, optimizer: str, optimization_level: int
) -> LadderResult:
    """One circuit with no mitigation, readout mitigation and ZNE (shared work is reused)."""
    evaluator = ConfigurationEvaluator(settings)
    configs = [TunerConfig(qaoa_p, optimizer, optimization_level, m) for m in MITIGATIONS]
    records = [evaluator.evaluate(c) for c in configs]
    counts = {c.mitigation: evaluator.counts_for(c) for c in configs}
    return LadderResult(records, counts)


def run_tuner(
    settings: TunerSettings,
    space: ConfigurationSpace,
    progress: Callable[[int, int, str], None] | None = None,
) -> TunerResult:
    return ConfigurationTuner(settings, space).run(progress)


def result_to_csv(result: TunerResult) -> str:
    """CSV text of all records (same layout as TunerResult.save_csv)."""
    rows = []
    for record in result.records:
        row = record.to_dict()
        row["uncovered_gates"] = ";".join(row["uncovered_gates"])
        row["dominated_by"] = ";".join(row["dominated_by"])
        rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]) if rows else [])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def graph_from_dict(data: dict) -> ColoringGraph:
    return ColoringGraph.from_dict(data)


@dataclass(frozen=True)
class Outcome:
    """One measured bitstring, decoded into a coloring."""

    bitstring: str
    probability: float
    colors: list[int | None]  # color per vertex; None = vertex without exactly one color
    violations: int  # same-color edges + vertices that do not have exactly one color
    valid: bool


def decode_bitstring(graph: ColoringGraph, num_colors: int, bitstring: str):
    """Decode a measured bitstring (Qiskit order: rightmost bit = qubit 0).

    k = 2: qubit v is the color of vertex v.
    k >= 3: one-hot, qubit v*k + c set means vertex v has color c.
    Returns (colors, violations, valid).
    """
    bits = bitstring.replace(" ", "")
    n = graph.num_nodes
    width = qubits_needed(n, num_colors)
    if len(bits) != width:
        raise ValueError(f"Expected a bitstring of length {width}, got {len(bits)}.")

    colors: list[int | None] = []
    invalid_vertices = 0
    if num_colors == 2:
        colors = [int(bits[width - 1 - v]) for v in range(n)]
    else:
        for v in range(n):
            chosen = [c for c in range(num_colors) if bits[width - 1 - (v * num_colors + c)] == "1"]
            if len(chosen) == 1:
                colors.append(chosen[0])
            else:
                colors.append(None)
                invalid_vertices += 1
    monochromatic = sum(1 for u, w in graph.edges if colors[u] is not None and colors[u] == colors[w])
    violations = monochromatic + invalid_vertices
    return colors, violations, violations == 0


def describe_outcomes(
    graph: ColoringGraph, num_colors: int, counts: dict[str, int], top: int | None = 10
) -> list[Outcome]:
    """Most probable outcomes first (ties by bitstring). ``top=None`` returns all of them."""
    total = sum(counts.values())
    if total <= 0:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if top is not None:
        ranked = ranked[:top]
    outcomes = []
    for bitstring, count in ranked:
        colors, violations, valid = decode_bitstring(graph, num_colors, bitstring)
        outcomes.append(Outcome(bitstring.replace(" ", ""), count / total, colors, violations, valid))
    return outcomes
