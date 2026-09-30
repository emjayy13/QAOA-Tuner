"""Unit and integration tests for hardware-aware transpilation and resource profiling."""

from qiskit.circuit import QuantumCircuit

from qaoa_tuner.compilation.backends import BackendProvider
from qaoa_tuner.compilation.profiler import profile_circuit
from qaoa_tuner.compilation.transpiler import (
    compare_optimization_levels,
    transpile_qaoa_circuit,
)
from qaoa_tuner.qaoa.ansatz import build_qaoa_circuit
from qaoa_tuner.qaoa.hamiltonian import build_cost_hamiltonian


class TestProfiler:
    """Test quantum circuit resource profiling."""

    def test_profile_bell_state(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        metrics = profile_circuit(qc)
        assert metrics.num_qubits == 2
        assert metrics.one_qubit_gate_count == 1
        assert metrics.two_qubit_gate_count == 1
        assert metrics.total_gate_count == 2
        assert metrics.swap_count == 0
        assert metrics.gate_breakdown == {"h": 1, "cx": 1}

    def test_profile_swap_gate(self):
        qc = QuantumCircuit(3)
        qc.swap(0, 2)
        metrics = profile_circuit(qc)
        assert metrics.swap_count == 1
        assert metrics.two_qubit_gate_count == 1


class TestBackendProvider:
    """Test programmatic backend querying and topology inspection."""

    def test_query_ideal_aer(self):
        backend = BackendProvider.get_backend("ideal")
        info = BackendProvider.query_backend_info(backend)

        assert info.is_simulator is True
        assert "aer" in info.name.lower()

    def test_query_fake_generic_5q(self):
        backend = BackendProvider.get_backend("fake_generic_5q", seed=42)
        info = BackendProvider.query_backend_info(backend)

        assert info.num_qubits == 5
        assert "cx" in info.basis_gates
        assert "rz" in info.basis_gates
        assert info.coupling_map is not None
        assert info.has_noise_model is True

    def test_query_fake_linear_5q(self):
        backend = BackendProvider.get_backend("fake_linear_5q", seed=42)
        info = BackendProvider.query_backend_info(backend)

        assert info.num_qubits == 5
        # Line topology (0-1-2-3-4) has 8 directed edges for bidirectional coupling
        assert len(info.coupling_map) == 8


class TestTranspilation:
    """Test transpilation across optimization levels and hardware topologies."""

    def test_transpile_c4_on_fake_generic(self, cycle_4_graph):
        ham = build_cost_hamiltonian(cycle_4_graph, num_colors=2)
        qc, _, _ = build_qaoa_circuit(ham, p=1, measure=True)

        backend = BackendProvider.get_backend("fake_generic_5q", seed=42)
        res = transpile_qaoa_circuit(qc, backend, optimization_level=1, seed_transpiler=42)

        assert res.optimization_level == 1
        assert res.logical_metrics.num_qubits == 4
        assert res.transpiled_metrics.num_qubits == 5  # physical backend has 5 qubits
        assert res.transpiled_metrics.depth > 0
        assert res.transpiled_metrics.two_qubit_gate_count >= res.logical_metrics.two_qubit_gate_count

        # Check basis gate compliance: all transpiled gates must be in backend basis
        info = BackendProvider.query_backend_info(backend)
        for gate_name in res.transpiled_metrics.gate_breakdown:
            if gate_name not in ["barrier", "measure", "delay", "reset"]:
                assert gate_name in info.basis_gates

    def test_compare_optimization_levels(self, cycle_4_graph):
        ham = build_cost_hamiltonian(cycle_4_graph, num_colors=2)
        qc, _, _ = build_qaoa_circuit(ham, p=1, measure=False)

        backend = BackendProvider.get_backend("fake_generic_5q", seed=42)
        results = compare_optimization_levels(qc, backend, levels=[0, 1, 2, 3], seed=42)

        assert len(results) == 4
        for r in results:
            assert r.transpiled_metrics.total_gate_count > 0

    def test_linear_topology_forces_swaps(self):
        """Verify that interacting distant qubits (q0 and q3) on a 1D chain forces routing."""
        qc = QuantumCircuit(4)
        qc.rzz(0.5, 0, 3)  # Qubits 0 and 3 are separated by 1 and 2

        backend = BackendProvider.get_backend("fake_linear_5q", seed=42)
        res = transpile_qaoa_circuit(qc, backend, optimization_level=1, seed_transpiler=42)

        # On a line, direct 2Q interaction between 0 and 3 cannot be executed in 1 step;
        # Routing must insert SWAPs or additional CX gates
        assert res.swaps_introduced > 0 or res.transpiled_metrics.two_qubit_gate_count > 1
