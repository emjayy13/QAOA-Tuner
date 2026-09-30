import numpy as np

from qaoa_tuner.problem.verifier import find_all_valid_colorings, validate_coloring
from qaoa_tuner.qaoa.ansatz import build_qaoa_circuit
from qaoa_tuner.qaoa.decoder import decode_bitstring, decode_counts
from qaoa_tuner.qaoa.hamiltonian import (
    build_cost_hamiltonian,
    build_mixer_hamiltonian,
)
from qaoa_tuner.qaoa.optimizers import minimize_cobyla, minimize_spsa
from qaoa_tuner.qaoa.runner import QaoaRunner


class TestCostHamiltonian:
    """Test mathematical correctness of Ising and QUBO Cost Hamiltonians."""

    def test_binary_2color_c4_energy(self, cycle_4_graph):
        """Verify ground state energy is 0 for valid 2-colorings of C4."""
        ham = build_cost_hamiltonian(cycle_4_graph, num_colors=2)
        assert ham.num_qubits == 4
        assert ham.encoding == "binary_2color"

        # Valid colorings of C4: {0:0, 1:1, 2:0, 3:1} and {0:1, 1:0, 2:1, 3:0}
        # In Qiskit string: bitstring[3]=q0, bitstring[2]=q1, bitstring[1]=q2, bitstring[0]=q3
        # For {0:0, 1:1, 2:0, 3:1}: q0=0, q1=1, q2=0, q3=1 -> bitstring is "1010"
        energy_valid_1 = ham.evaluate_bitstring("1010")
        energy_valid_2 = ham.evaluate_bitstring("0101")
        assert np.isclose(energy_valid_1, 0.0), f"Expected 0.0, got {energy_valid_1}"
        assert np.isclose(energy_valid_2, 0.0), f"Expected 0.0, got {energy_valid_2}"

        # Monochromatic state (all nodes color 0 => all 4 edges conflict)
        # q0=0, q1=0, q2=0, q3=0 -> "0000"
        energy_all_same = ham.evaluate_bitstring("0000")
        assert np.isclose(energy_all_same, 4.0), f"Expected 4.0, got {energy_all_same}"

    def test_one_hot_kcolor_k3_energy(self, triangle_graph):
        """Verify one-hot k-coloring Hamiltonian on Triangle K3 (k=3, 9 qubits)."""
        ham = build_cost_hamiltonian(triangle_graph, num_colors=3)
        assert ham.num_qubits == 9
        assert ham.encoding == "one_hot_kcolor"

        # Construct a valid coloring: node 0 -> color 0, node 1 -> color 1, node 2 -> color 2
        # Qubits: node 0: q0=1, q1=0, q2=0
        #         node 1: q3=0, q4=1, q5=0
        #         node 2: q6=0, q7=0, q8=1
        # Qiskit bitstring (qubit 8 down to qubit 0):
        # q8 q7 q6 | q5 q4 q3 | q2 q1 q0
        # 1  0  0  | 0  1  0  | 0  0  1  => "100010001"
        valid_bs = "100010001"
        energy_valid = ham.evaluate_bitstring(valid_bs)
        assert np.isclose(energy_valid, 0.0, atol=1e-5), f"Expected 0.0, got {energy_valid}"

    def test_mixer_hamiltonian(self):
        terms = build_mixer_hamiltonian(4)
        assert len(terms) == 4
        for term, coeff in terms:
            assert coeff == 1.0
            assert term.count("X") == 1
            assert term.count("I") == 3


class TestAnsatz:
    """Test QAOA parameterized circuit structure."""

    def test_ansatz_dimensions(self, cycle_4_graph):
        ham = build_cost_hamiltonian(cycle_4_graph, num_colors=2)
        qc, gammas, betas = build_qaoa_circuit(ham, p=2, measure=True)

        assert qc.num_qubits == 4
        assert len(gammas) == 2
        assert len(betas) == 2
        assert len(qc.parameters) == 4
        assert qc.num_clbits == 4
        assert qc.depth() > 0


class TestDecoder:
    """Test bitstring decoding into graph coloring."""

    def test_decode_binary_2color(self, cycle_4_graph):
        # "1010" in Qiskit: bit[3]=q0='0', bit[2]=q1='1', bit[1]=q2='0', bit[0]=q3='1'
        coloring = decode_bitstring("1010", cycle_4_graph, num_colors=2, encoding="binary_2color")
        assert coloring == {0: 0, 1: 1, 2: 0, 3: 1}
        val = validate_coloring(cycle_4_graph, coloring, num_colors=2)
        assert val.is_valid is True

    def test_decode_counts(self, cycle_4_graph):
        counts = {
            "1010": 400,  # valid
            "0101": 400,  # valid
            "0000": 200,  # invalid (4 conflicts)
        }
        res = decode_counts(counts, cycle_4_graph, num_colors=2, encoding="binary_2color")
        assert res.total_shots == 1000
        assert np.isclose(res.valid_coloring_probability, 0.8)
        assert res.best_validation.is_valid is True
        assert res.best_validation.num_conflicts == 0


class TestOptimizers:
    """Test classical optimization algorithms."""

    def test_cobyla_sphere(self):
        def objective(x):
            return (x[0] - 1.5) ** 2 + (x[1] - 2.5) ** 2

        res = minimize_cobyla(objective, initial_point=[0.0, 0.0], max_iter=40)
        assert np.isclose(res.optimal_point[0], 1.5, atol=1e-2)
        assert np.isclose(res.optimal_point[1], 2.5, atol=1e-2)
        assert res.optimal_value < 1e-3

    def test_spsa_sphere(self):
        def objective(x):
            return (x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2

        res = minimize_spsa(objective, initial_point=[0.0, 0.0], max_iter=50, seed=42)
        assert res.optimal_value < 0.2


class TestQaoaRunnerIntegration:
    """End-to-end integration test of QAOA execution on ideal Aer simulation."""

    def test_qaoa_c4_cobyla_solves_graph(self, cycle_4_graph):
        """Verify QAOA p=1 with COBYLA finds a valid 2-coloring for C4."""
        runner = QaoaRunner(
            graph=cycle_4_graph,
            num_colors=2,
            p=1,
            shots=512,
            seed=42,
        )

        exec_res = runner.optimize(
            optimizer="COBYLA",
            max_iter=25,
            initial_point=[0.4, 0.4],
            final_shots=1024,
        )

        # Classical reference solutions
        classical_valid = find_all_valid_colorings(cycle_4_graph, num_colors=2)

        # Verify quantum output matches classical valid solution
        assert exec_res.decoded_result.best_validation.is_valid is True
        assert exec_res.decoded_result.best_validation.num_conflicts == 0
        assert exec_res.decoded_result.best_coloring in classical_valid

        # Random guessing has valid probability 2/16 = 0.125
        # QAOA should achieve significantly higher probability mass on ground states
        assert exec_res.decoded_result.valid_coloring_probability > 0.35

    def test_qaoa_fixed_point_evaluation(self, cycle_4_graph):
        """Verify fixed-point evaluation produces reproducible statistics."""
        runner = QaoaRunner(
            graph=cycle_4_graph,
            num_colors=2,
            p=1,
            shots=1000,
            seed=123,
        )
        res1 = runner.evaluate_point([0.39, 0.39], shots=1000)
        res2 = runner.evaluate_point([0.39, 0.39], shots=1000)

        # Both should find valid colorings with high confidence
        assert res1.best_validation.is_valid is True
        assert res2.best_validation.is_valid is True
        assert abs(res1.valid_coloring_probability - res2.valid_coloring_probability) < 0.1
