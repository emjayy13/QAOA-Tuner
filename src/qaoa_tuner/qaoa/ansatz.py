"""Parameterized QAOA circuit construction for graph coloring."""

from __future__ import annotations

from qiskit.circuit import Parameter, QuantumCircuit

from qaoa_tuner.qaoa.hamiltonian import CostHamiltonian


def build_qaoa_circuit(
    cost_hamiltonian: CostHamiltonian,
    p: int,
    measure: bool = True,
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Build a parameterized QAOA quantum circuit of depth p.

    Ansatz Structure:
        |psi(gamma, beta)> = prod_{layer=1}^p [ e^{-i beta_layer H_M} e^{-i gamma_layer H_C} ] |+>^{otimes n}

    Gate Implementations:
        - Cost unitary e^{-i gamma_layer c Z_j}  -> RZ(2 * gamma_layer * c, j)
        - Cost unitary e^{-i gamma_layer c Z_j Z_k} -> RZZ(2 * gamma_layer * c, j, k)
        - Mixer unitary e^{-i beta_layer X_j} -> RX(2 * beta_layer, j)

    Args:
        cost_hamiltonian: Target CostHamiltonian instance.
        p: Number of QAOA alternating layers (depth >= 1).
        measure: If True, adds measure_all() at the end.

    Returns:
        tuple: (parameterized_circuit, gammas, betas)
    """
    if p < 1:
        raise ValueError(f"QAOA depth p must be >= 1, got {p}.")

    n = cost_hamiltonian.num_qubits
    qc = QuantumCircuit(n, name=f"QAOA_p{p}")

    gammas = [Parameter(f"gamma_{layer_idx}") for layer_idx in range(p)]
    betas = [Parameter(f"beta_{layer_idx}") for layer_idx in range(p)]

    # 1. Initial State: Equal superposition |+>^{otimes n}
    qc.h(range(n))

    # Pre-parse Pauli terms into active qubit indices for fast circuit generation
    parsed_terms: list[tuple[list[int], float]] = []
    for pauli_str, coeff in cost_hamiltonian.pauli_terms:
        # In Qiskit string, char 0 is qubit n-1, char n-1 is qubit 0
        z_qubits = [
            n - 1 - char_idx
            for char_idx, char in enumerate(pauli_str)
            if char == "Z"
        ]
        parsed_terms.append((z_qubits, coeff))

    # 2. Alternating QAOA Layers
    for layer_idx in range(p):
        # --- Problem Unitary: e^{-i gamma_layer H_C} ---
        for z_qubits, coeff in parsed_terms:
            if len(z_qubits) == 1:
                q = z_qubits[0]
                qc.rz(2 * gammas[layer_idx] * coeff, q)
            elif len(z_qubits) == 2:
                q1, q2 = z_qubits[0], z_qubits[1]
                qc.rzz(2 * gammas[layer_idx] * coeff, q1, q2)
            elif len(z_qubits) == 0:
                pass
            else:
                raise NotImplementedError(
                    f"Higher-order terms (>2 qubits) not supported in standard ansatz: {z_qubits}"
                )

        # --- Mixer Unitary: e^{-i beta_layer H_M} ---
        for q in range(n):
            qc.rx(2 * betas[layer_idx], q)

    # 3. Final Measurement
    if measure:
        qc.measure_all()

    return qc, gammas, betas
