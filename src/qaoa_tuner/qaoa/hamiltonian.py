"""Cost and Mixer Hamiltonian formulations for graph coloring."""

from __future__ import annotations

from dataclasses import dataclass

from qaoa_tuner.core.exceptions import InvalidGraphError
from qaoa_tuner.problem.graph import ColoringGraph


@dataclass(frozen=True)
class CostHamiltonian:
    """Container for the Ising/QUBO Cost Hamiltonian of a graph coloring problem.

    Attributes:
        pauli_terms: List of (pauli_string, coefficient) representing the Hamiltonian.
        num_qubits: Total number of qubits required.
        num_colors: Number of colors (k).
        encoding: "binary_2color" (1 qubit/node) or "one_hot_kcolor" (k qubits/node).
        constant_offset: Energy constant shift from identity terms.
    """

    pauli_terms: list[tuple[str, float]]
    num_qubits: int
    num_colors: int
    encoding: str
    constant_offset: float

    def evaluate_bitstring(self, bitstring: str) -> float:
        """Evaluate the classical energy of a measured bitstring under this Hamiltonian.

        Qiskit convention: bitstring is little-endian where bitstring[num_qubits - 1 - q]
        corresponds to qubit q.
        """
        if len(bitstring) != self.num_qubits:
            raise ValueError(f"Expected bitstring of length {self.num_qubits}, got {len(bitstring)}")

        energy = self.constant_offset
        for pauli_str, coeff in self.pauli_terms:
            val = 1.0
            for q_idx, char in enumerate(pauli_str):
                # Qiskit pauli_str convention: rightmost char is qubit 0
                qubit_pos = self.num_qubits - 1 - q_idx
                bit_val = bitstring[qubit_pos]
                if char == "Z":
                    # Z eigenval: '0' -> +1, '1' -> -1
                    val *= 1.0 if bit_val == "0" else -1.0
                elif char == "I":
                    pass
                else:
                    raise ValueError(f"Non-diagonal Pauli {char} encountered in Cost Hamiltonian.")
            energy += coeff * val
        return energy


def build_cost_hamiltonian(
    graph: ColoringGraph,
    num_colors: int = 2,
    penalty_vertex: float = 2.0,
    penalty_edge: float = 1.0,
) -> CostHamiltonian:
    """Construct the Ising Cost Hamiltonian H_C for graph coloring.

    Supported Encodings:
    1. Binary 2-Coloring (k=2):
       - Uses 1 qubit per vertex (N qubits total).
       - Hamiltonian: H_C = sum_{(u,v) in E} (I + Z_u Z_v) / 2.
       - A conflicting edge (same spin) incurs +1 penalty; valid edge incurs 0 penalty.
       - Ground state energy is exactly 0 if the graph is 2-colorable.

    2. One-Hot k-Coloring (k >= 3):
       - Uses k qubits per vertex (N * k qubits total).
       - Qubit index for node v, color c: q = v * k + c.
       - Vertex penalty: A * (sum_c x_{v,c} - 1)^2.
       - Edge penalty: B * sum_{(u,v) in E} sum_c x_{u,c} x_{v,c}.
       - Expanded using x_i = (I - Z_i) / 2.

    Args:
        graph: Target ColoringGraph instance.
        num_colors: Number of colors (k).
        penalty_vertex: Weight A for vertex 1-hot constraint (default: 2.0).
        penalty_edge: Weight B for edge conflict constraint (default: 1.0).

    Returns:
        CostHamiltonian instance with Pauli terms and offset.
    """
    if num_colors < 2:
        raise InvalidGraphError(f"Graph coloring requires at least 2 colors, got {num_colors}.")

    if num_colors == 2:
        # Binary 2-coloring: 1 qubit per node
        n = graph.num_nodes
        pauli_dict: dict[str, float] = {}
        offset = 0.5 * graph.num_edges

        for u, v in graph.edges:
            # Construct Pauli string with Z at u and v, I elsewhere
            # In Qiskit string: char 0 is qubit (n - 1), char (n - 1) is qubit 0
            chars = ["I"] * n
            chars[n - 1 - u] = "Z"
            chars[n - 1 - v] = "Z"
            p_str = "".join(chars)
            pauli_dict[p_str] = pauli_dict.get(p_str, 0.0) + 0.5

        terms = [(k, v) for k, v in pauli_dict.items() if abs(v) > 1e-9]
        return CostHamiltonian(
            pauli_terms=terms,
            num_qubits=n,
            num_colors=2,
            encoding="binary_2color",
            constant_offset=offset,
        )

    # General one-hot k-coloring (k >= 3): N * k qubits
    n_qubits = graph.num_nodes * num_colors
    pauli_dict = {}
    total_offset = 0.0

    def qubit_idx(node: int, color: int) -> int:
        return node * num_colors + color

    def add_term(term_str: str, weight: float) -> None:
        pauli_dict[term_str] = pauli_dict.get(term_str, 0.0) + weight

    def make_pauli_str(z_indices: list[int]) -> str:
        chars = ["I"] * n_qubits
        for idx in z_indices:
            chars[n_qubits - 1 - idx] = "Z"
        return "".join(chars)

    # 1. Vertex constraint: A * (sum_c x_{v,c} - 1)^2
    # = A * [ 1 - sum_c x_{v,c} + 2 sum_{c < c'} x_{v,c} x_{v,c'} ]
    for v in graph.nodes:
        # Constant +1 from 1
        total_offset += penalty_vertex * 1.0

        # - sum_c x_{v,c} where x = (I - Z)/2 => -1/2 * I + 1/2 * Z
        for c in range(num_colors):
            q = qubit_idx(v, c)
            total_offset -= penalty_vertex * 0.5
            add_term(make_pauli_str([q]), penalty_vertex * 0.5)

        # + 2 sum_{c < c'} x_i x_j where x_i x_j = 1/4 (I - Z_i - Z_j + Z_i Z_j)
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                q1 = qubit_idx(v, c1)
                q2 = qubit_idx(v, c2)
                # 2 * 1/4 = 0.5
                factor = penalty_vertex * 0.5
                total_offset += factor * 1.0
                add_term(make_pauli_str([q1]), -factor * 1.0)
                add_term(make_pauli_str([q2]), -factor * 1.0)
                add_term(make_pauli_str([q1, q2]), factor * 1.0)

    # 2. Edge constraint: B * sum_{(u,v) in E} sum_c x_{u,c} x_{v,c}
    for u, v in graph.edges:
        for c in range(num_colors):
            qu = qubit_idx(u, c)
            qv = qubit_idx(v, c)
            # x_u x_v = 1/4 (I - Z_u - Z_v + Z_u Z_v)
            factor = penalty_edge * 0.25
            total_offset += factor * 1.0
            add_term(make_pauli_str([qu]), -factor * 1.0)
            add_term(make_pauli_str([qv]), -factor * 1.0)
            add_term(make_pauli_str([qu, qv]), factor * 1.0)

    terms = [(k, v) for k, v in pauli_dict.items() if abs(v) > 1e-9]
    return CostHamiltonian(
        pauli_terms=terms,
        num_qubits=n_qubits,
        num_colors=num_colors,
        encoding="one_hot_kcolor",
        constant_offset=total_offset,
    )


def build_mixer_hamiltonian(num_qubits: int) -> list[tuple[str, float]]:
    """Construct the standard transverse field Mixer Hamiltonian H_M = sum_j X_j."""
    terms = []
    for j in range(num_qubits):
        chars = ["I"] * num_qubits
        chars[num_qubits - 1 - j] = "X"
        terms.append(("".join(chars), 1.0))
    return terms
