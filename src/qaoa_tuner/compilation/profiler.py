"""Quantum circuit resource profiling and gate decomposition analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qiskit.circuit import QuantumCircuit


@dataclass(frozen=True)
class CircuitResourceMetrics:
    """Detailed hardware resource metrics extracted from a quantum circuit.

    Attributes:
        depth: Critical path circuit depth.
        total_gate_count: Total number of quantum gates.
        one_qubit_gate_count: Count of 1-qubit single-pulse operations (rz, sx, x, id, etc.).
        two_qubit_gate_count: Count of 2-qubit entangling operations (cx, cz, ecr, rzz, etc.).
        swap_count: Number of SWAP gates inserted for hardware connectivity routing.
        num_qubits: Number of quantum bits allocated in the circuit register.
        gate_breakdown: Exact histogram of gate types {gate_name: count}.
    """

    depth: int
    total_gate_count: int
    one_qubit_gate_count: int
    two_qubit_gate_count: int
    swap_count: int
    num_qubits: int
    gate_breakdown: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CircuitResourceMetrics:
        """Construct metrics container from a dictionary."""
        return cls(**data)


def profile_circuit(circuit: QuantumCircuit) -> CircuitResourceMetrics:
    """Analyze a quantum circuit and extract all hardware resource metrics.

    Args:
        circuit: A compiled or uncompiled QuantumCircuit instance.

    Returns:
        CircuitResourceMetrics containing depth, gate counts, and 2-qubit decomposition.
    """
    depth = circuit.depth()
    num_qubits = circuit.num_qubits

    # Qiskit circuit.count_ops() returns a dict of {op_name: count}
    ops_count = circuit.count_ops()
    gate_breakdown = dict(ops_count)

    # Exclude non-unitary auxiliary operations (barriers, measurements, delays) from gate counts
    non_gate_ops = {"barrier", "measure", "delay", "reset"}

    one_qubit_count = 0
    two_qubit_count = 0
    swap_count = gate_breakdown.get("swap", 0)

    # Inspect instruction signatures in the circuit
    for instruction in circuit.data:
        op_name = instruction.operation.name
        if op_name in non_gate_ops:
            continue

        qargs_len = len(instruction.qubits)
        if qargs_len == 1:
            one_qubit_count += 1
        elif qargs_len == 2:
            two_qubit_count += 1
            # In Qiskit, some transpilers decompose SWAP into 3 CX gates.
            # If explicit 'swap' is present, it's counted in swap_count.

    total_gates = one_qubit_count + two_qubit_count

    return CircuitResourceMetrics(
        depth=depth,
        total_gate_count=total_gates,
        one_qubit_gate_count=one_qubit_count,
        two_qubit_gate_count=two_qubit_count,
        swap_count=swap_count,
        num_qubits=num_qubits,
        gate_breakdown=gate_breakdown,
    )
