"""Decoding of sampled quantum bitstrings into graph coloring assignments and validation metrics."""

from __future__ import annotations

from dataclasses import dataclass

from qaoa_tuner.core.types import Coloring, ColoringValidationResult
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.problem.verifier import validate_coloring


@dataclass(frozen=True)
class DecodedQaoaResult:
    """Comprehensive analysis of measured quantum bitstring counts.

    Attributes:
        counts: Raw measurement dictionary {bitstring: count}.
        probabilities: Normalized probabilities {bitstring: probability}.
        best_bitstring: Bitstring achieving minimum constraint violations.
        best_coloring: Decoded coloring corresponding to best_bitstring.
        best_validation: Full validation analysis of best_coloring.
        most_frequent_bitstring: Bitstring sampled most frequently.
        most_frequent_coloring: Decoded coloring corresponding to most frequent bitstring.
        most_frequent_validation: Validation analysis of most frequent coloring.
        valid_coloring_probability: Total probability mass concentrated on strictly valid colorings.
        expected_conflicts: Average number of edge conflicts under the sampled distribution.
        total_shots: Total number of sampled shots.
    """

    counts: dict[str, int]
    probabilities: dict[str, float]
    best_bitstring: str
    best_coloring: Coloring
    best_validation: ColoringValidationResult
    most_frequent_bitstring: str
    most_frequent_coloring: Coloring
    most_frequent_validation: ColoringValidationResult
    valid_coloring_probability: float
    expected_conflicts: float
    total_shots: int


def decode_bitstring(
    bitstring: str,
    graph: ColoringGraph,
    num_colors: int = 2,
    encoding: str = "binary_2color",
) -> Coloring:
    """Decode a raw Qiskit bitstring into a node-to-color assignment.

    Qiskit bitstrings follow little-endian ordering:
        bitstring[len - 1 - q] corresponds to qubit q.

    Args:
        bitstring: Binary string of length equal to number of qubits.
        graph: Target ColoringGraph instance.
        num_colors: Number of colors (k).
        encoding: "binary_2color" (1 qubit/node) or "one_hot_kcolor" (k qubits/node).

    Returns:
        Coloring dict mapping node index to color index.
    """
    total_qubits = len(bitstring)
    coloring: Coloring = {}

    if encoding == "binary_2color":
        for node in graph.nodes:
            # Qubit node corresponds to position from right
            qubit_idx = node
            bit_char = bitstring[total_qubits - 1 - qubit_idx]
            coloring[node] = 1 if bit_char == "1" else 0
        return coloring

    elif encoding == "one_hot_kcolor":
        for node in graph.nodes:
            active_colors: list[int] = []
            for c in range(num_colors):
                q = node * num_colors + c
                bit_char = bitstring[total_qubits - 1 - q]
                if bit_char == "1":
                    active_colors.append(c)

            if len(active_colors) == 1:
                coloring[node] = active_colors[0]
            elif len(active_colors) > 1:
                # Ambiguous: pick the first active color, validation will flag 1-hot penalty
                coloring[node] = active_colors[0]
            else:
                # No color bit was set: mark with -1 so validator flags uncolored node
                coloring[node] = -1

        return coloring

    else:
        raise ValueError(f"Unknown encoding: '{encoding}'")


def decode_counts(
    counts: dict[str, int],
    graph: ColoringGraph,
    num_colors: int = 2,
    encoding: str = "binary_2color",
) -> DecodedQaoaResult:
    """Analyze measurement counts, evaluate solution validity, and compute quantum metrics.

    Args:
        counts: Qiskit measurement dictionary {bitstring: count}.
        graph: Target ColoringGraph instance.
        num_colors: Number of colors (k).
        encoding: "binary_2color" or "one_hot_kcolor".

    Returns:
        DecodedQaoaResult with statistical analysis and validity rates.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        raise ValueError("Cannot decode empty counts dictionary.")

    probabilities = {bs: cnt / total_shots for bs, cnt in counts.items()}

    valid_mass = 0.0
    weighted_conflicts = 0.0

    best_bs: str | None = None
    best_conflicts = float("inf")
    best_coloring: Coloring = {}
    best_val: ColoringValidationResult | None = None

    most_freq_bs = max(counts, key=counts.get)
    most_freq_coloring = decode_bitstring(most_freq_bs, graph, num_colors, encoding)
    most_freq_val = validate_coloring(graph, most_freq_coloring, num_colors)

    for bs, cnt in counts.items():
        prob = cnt / total_shots
        c = decode_bitstring(bs, graph, num_colors, encoding)
        val = validate_coloring(graph, c, num_colors)

        # Total conflict metric includes edge conflicts + uncolored/invalid nodes
        effective_conflicts = (
            val.num_conflicts + len(val.uncolored_nodes) + len(val.invalid_color_nodes)
        )
        weighted_conflicts += prob * effective_conflicts

        if val.is_valid:
            valid_mass += prob

        # Best bitstring minimizes conflicts; ties broken by higher frequency
        if effective_conflicts < best_conflicts or (
            effective_conflicts == best_conflicts
            and (best_bs is None or counts[bs] > counts[best_bs])
        ):
            best_bs = bs
            best_conflicts = effective_conflicts
            best_coloring = c
            best_val = val

    assert best_bs is not None and best_val is not None

    return DecodedQaoaResult(
        counts=counts,
        probabilities=probabilities,
        best_bitstring=best_bs,
        best_coloring=best_coloring,
        best_validation=best_val,
        most_frequent_bitstring=most_freq_bs,
        most_frequent_coloring=most_freq_coloring,
        most_frequent_validation=most_freq_val,
        valid_coloring_probability=valid_mass,
        expected_conflicts=weighted_conflicts,
        total_shots=total_shots,
    )
