"""Transpilation pipeline and multi-level compilation comparison engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qiskit import transpile
from qiskit.circuit import QuantumCircuit

from qaoa_tuner.compilation.profiler import CircuitResourceMetrics, profile_circuit


@dataclass(frozen=True)
class TranspilationResult:
    """Hardware compilation result comparing logical design vs transpiled implementation.

    Attributes:
        optimization_level: Transpiler optimization pass level (0, 1, 2, 3).
        backend_name: Name of target hardware/simulator backend.
        logical_metrics: Resource profile of original uncompiled circuit.
        transpiled_metrics: Resource profile after hardware basis decomposition and routing.
        depth_overhead_ratio: transpiled_depth / max(1, logical_depth).
        two_qubit_overhead_ratio: transpiled_2q / max(1, logical_2q).
        swaps_introduced: Estimated or explicit SWAP gates added during layout/routing.
        transpiled_circuit: The final compiled QuantumCircuit.
    """

    optimization_level: int
    backend_name: str
    logical_metrics: CircuitResourceMetrics
    transpiled_metrics: CircuitResourceMetrics
    depth_overhead_ratio: float
    two_qubit_overhead_ratio: float
    swaps_introduced: int
    transpiled_circuit: QuantumCircuit

    def to_dict(self) -> dict[str, Any]:
        """Convert result to a serializable dictionary (omits raw circuit object)."""
        data = asdict(self)
        # QuantumCircuit is not JSON-serializable; remove or serialize its QPY/qasm if needed
        data.pop("transpiled_circuit", None)
        return data


def transpile_qaoa_circuit(
    circuit: QuantumCircuit,
    backend: Any,
    optimization_level: int = 1,
    seed_transpiler: int = 42,
) -> TranspilationResult:
    """Transpile a QAOA circuit targeting hardware topology and basis gate constraints.

    Transpilation Stages in Qiskit:
        - Level 0: Fast mapping, no optimization.
        - Level 1: Light optimization (cancels adjacent inverse gates).
        - Level 2: Medium optimization (noise-aware layout, CommutationAnalysis).
        - Level 3: Heavy optimization (resynthesis of 2-qubit blocks, extensive routing search).

    Args:
        circuit: Logical QuantumCircuit.
        backend: Target backend (AerSimulator or GenericBackendV2).
        optimization_level: Optimization level in [0, 1, 2, 3].
        seed_transpiler: Seed for deterministic stochastic routing.

    Returns:
        TranspilationResult with full logical vs transpiled comparison.
    """
    if optimization_level not in [0, 1, 2, 3]:
        raise ValueError(
            f"Optimization level must be in [0, 1, 2, 3], got {optimization_level}"
        )

    backend_name = getattr(backend, "name", str(type(backend).__name__))
    logical_metrics = profile_circuit(circuit)

    transpiled_qc = transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )

    transpiled_metrics = profile_circuit(transpiled_qc)

    # Compute overhead ratios
    depth_ratio = transpiled_metrics.depth / max(1, logical_metrics.depth)
    two_q_ratio = transpiled_metrics.two_qubit_gate_count / max(
        1, logical_metrics.two_qubit_gate_count
    )

    # Explicit SWAPs vs decomposed SWAPs (each SWAP decomposes into 3 CX gates)
    explicit_swaps = transpiled_metrics.swap_count
    decomposed_swaps = max(
        0,
        (transpiled_metrics.two_qubit_gate_count - logical_metrics.two_qubit_gate_count) // 3,
    )
    swaps_introduced = explicit_swaps if explicit_swaps > 0 else decomposed_swaps

    return TranspilationResult(
        optimization_level=optimization_level,
        backend_name=backend_name,
        logical_metrics=logical_metrics,
        transpiled_metrics=transpiled_metrics,
        depth_overhead_ratio=round(depth_ratio, 3),
        two_qubit_overhead_ratio=round(two_q_ratio, 3),
        swaps_introduced=swaps_introduced,
        transpiled_circuit=transpiled_qc,
    )


def compare_optimization_levels(
    circuit: QuantumCircuit,
    backend: Any,
    levels: list[int] | None = None,
    seed: int = 42,
) -> list[TranspilationResult]:
    """Execute comparative transpilation across multiple optimization levels."""
    if levels is None:
        levels = [0, 1, 2, 3]

    results: list[TranspilationResult] = []
    for level in levels:
        res = transpile_qaoa_circuit(
            circuit=circuit,
            backend=backend,
            optimization_level=level,
            seed_transpiler=seed,
        )
        results.append(res)

    return results
