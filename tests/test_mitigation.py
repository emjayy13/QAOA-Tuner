"""Tests for Phase 6: readout mitigation, ZNE, and the mitigation comparison benchmark.

Kept small and fast: tiny circuits, few shots, fixed seeds.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator

from qaoa_tuner.experiment.config import ExperimentConfig
from qaoa_tuner.mitigation.readout import (
    ReadoutCalibrator,
    TensoredReadoutCalibrator,
    calibrate_readout_mitigation,
    calibrate_tensored_readout_mitigation,
)
from qaoa_tuner.mitigation.zne import extrapolate_zero_noise, fold_circuit_gates
from qaoa_tuner.noise.models import build_noise_model
from qaoa_tuner.problem.generators import create_cycle_graph


# ---------------------------------------------------------------- ZNE: folding ----
def _small_circuit(measure: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(3)
    qc.h(range(3))
    qc.rzz(0.7, 0, 1)
    qc.rzz(-0.3, 1, 2)
    qc.rx(0.4, 0)
    if measure:
        qc.measure_all()
    return qc


@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_folding_preserves_unitary(scale):
    original = _small_circuit()
    folded = fold_circuit_gates(original, scale)
    assert Operator(folded).equiv(Operator(original))


@pytest.mark.parametrize("scale", [1, 3, 5])
def test_folding_scales_two_qubit_gates_only(scale):
    original = _small_circuit(measure=True)
    folded = fold_circuit_gates(original, scale)
    ops_o, ops_f = original.count_ops(), folded.count_ops()
    assert ops_f["rzz"] == scale * ops_o["rzz"]
    assert ops_f["h"] == ops_o["h"]
    assert ops_f["rx"] == ops_o["rx"]
    assert ops_f["measure"] == ops_o["measure"]
    assert folded.num_clbits == original.num_clbits


def test_folding_keeps_measurements_last():
    folded = fold_circuit_gates(_small_circuit(measure=True), 3)
    names = [inst.operation.name for inst in folded.data]
    last_gate = max(i for i, n in enumerate(names) if n not in ("measure", "barrier"))
    first_measure = min(i for i, n in enumerate(names) if n == "measure")
    assert last_gate < first_measure


@pytest.mark.parametrize("bad", [0, 2, 4, -1, 3.0, 1.5])
def test_folding_rejects_invalid_scale(bad):
    with pytest.raises(ValueError):
        fold_circuit_gates(_small_circuit(), bad)


def test_folding_rejects_gate_after_measurement():
    qc = QuantumCircuit(2, 1)
    qc.measure(0, 0)
    qc.cx(0, 1)
    with pytest.raises(ValueError):
        fold_circuit_gates(qc, 3)


# ------------------------------------------------------------ ZNE: extrapolation ----
def test_linear_extrapolation_recovers_intercept():
    res = extrapolate_zero_noise([1, 3, 5], [0.85, 0.75, 0.65], model="linear")
    assert res.extrapolated_zero_noise_value == pytest.approx(0.9)
    assert res.fit_coefficients[1] == pytest.approx(-0.05)
    assert res.circuit_multiplier == 3
    assert res.noise_scale_sum == 9.0
    assert res.extrapolated_std_error is None


def test_polynomial_extrapolation_exact_quadratic():
    lam = np.array([1, 3, 5], dtype=float)
    vals = 1.0 - 0.1 * lam + 0.01 * lam**2
    res = extrapolate_zero_noise(lam, vals, model="polynomial")
    assert res.extrapolated_zero_noise_value == pytest.approx(1.0)


def test_extrapolation_error_propagation_matches_hand_calculation():
    # Intercept weights for x = [1, 3, 5] are [13/12, 1/3, -5/12]; sum of squares = 35/24.
    res = extrapolate_zero_noise([1, 3, 5], [0.8, 0.7, 0.6], std_errs=[0.01] * 3)
    assert res.extrapolated_std_error == pytest.approx(0.01 * np.sqrt(35 / 24))


def test_extrapolation_input_validation():
    with pytest.raises(ValueError):
        extrapolate_zero_noise([1, 3], [0.5])
    with pytest.raises(ValueError):
        extrapolate_zero_noise([1], [0.5])
    with pytest.raises(ValueError):
        extrapolate_zero_noise([1, 3], [0.5, 0.4], model="polynomial")
    with pytest.raises(ValueError):
        extrapolate_zero_noise([1, 3, 5], [0.5, 0.4, 0.3], model="exponential")


# ------------------------------------------------------- readout: correction math ----
def test_tensored_correction_inverts_known_confusion():
    a0 = np.array([[0.90, 0.20], [0.10, 0.80]])  # qubit 0: P(1|0)=0.10, P(0|1)=0.20
    a1 = np.array([[0.95, 0.10], [0.05, 0.90]])  # qubit 1
    cal = TensoredReadoutCalibrator(2, [a0, a1], 1.0, 0)

    true = np.array([0.5, 0.0, 0.0, 0.5])  # |00> and |11>
    measured = np.kron(a1, a0) @ true  # index = 2*bit(q1) + bit(q0) = int(bitstring, 2)
    raw = {format(i, "02b"): float(v) for i, v in enumerate(measured)}

    mitigated = cal.mitigate_probabilities(raw)
    assert mitigated["00"] == pytest.approx(0.5, abs=1e-6)
    assert mitigated["11"] == pytest.approx(0.5, abs=1e-6)
    assert sum(mitigated.values()) == pytest.approx(1.0)


def test_full_matrix_correction_inverts_known_confusion():
    m = np.array([[0.9, 0.1], [0.1, 0.9]])
    cal = ReadoutCalibrator(1, m, ["0", "1"], float(np.linalg.cond(m)), 0)
    measured = m @ np.array([1.0, 0.0])
    mitigated = cal.mitigate_probabilities({"0": float(measured[0]), "1": float(measured[1])})
    assert mitigated["0"] == pytest.approx(1.0, abs=1e-6)


def test_mitigated_counts_preserve_total_shots():
    a = np.array([[0.9, 0.1], [0.1, 0.9]])
    cal = TensoredReadoutCalibrator(2, [a, a], 1.0, 0)
    out = cal.mitigate_counts({"00": 800, "01": 90, "10": 90, "11": 20})
    assert sum(out.values()) == 1000
    assert all(v > 0 for v in out.values())


def test_correction_clips_negative_probabilities():
    a = np.array([[0.9, 0.1], [0.1, 0.9]])
    cal = TensoredReadoutCalibrator(1, [a], 1.0, 0)
    # Physically impossible measured data (more extreme than the channel allows)
    out = cal.mitigate_probabilities({"0": 1.0, "1": 0.0})
    assert all(p >= 0 for p in out.values())
    assert sum(out.values()) == pytest.approx(1.0)


# ------------------------------------------------- readout: calibration on Aer ----
def test_full_calibration_rejects_too_many_qubits():
    with pytest.raises(ValueError):
        calibrate_readout_mitigation(num_qubits=9, shots=10)


def test_tensored_calibration_estimates_readout_error():
    nm = build_noise_model("readout_only")  # 0.025 symmetric
    cal = calibrate_tensored_readout_mitigation(3, shots=4096, noise_model=nm, seed=1)
    assert cal.num_calibration_circuits == 2
    assert cal.calibration_overhead_shots == 2 * 4096
    for mat in cal.qubit_matrices:
        assert mat[1, 0] == pytest.approx(0.025, abs=0.01)  # P(1|0)
        assert mat[0, 1] == pytest.approx(0.025, abs=0.01)  # P(0|1)
        assert mat.sum(axis=0) == pytest.approx([1.0, 1.0])  # columns are distributions


def test_tensored_matches_full_matrix_for_independent_noise():
    nm = build_noise_model("readout_only")
    tens = calibrate_tensored_readout_mitigation(2, shots=8192, noise_model=nm, seed=3)
    full = calibrate_readout_mitigation(num_qubits=2, shots=8192, noise_model=nm, seed=3)
    implied = np.kron(tens.qubit_matrices[1], tens.qubit_matrices[0])
    assert np.max(np.abs(implied - full.matrix)) < 0.02
    assert full.num_calibration_circuits == 4


def test_readout_mitigation_improves_known_state():
    nm = build_noise_model("readout_only")
    qc = QuantumCircuit(3)
    qc.measure_all()  # true state is |000>
    sim = AerSimulator(noise_model=nm, seed_simulator=7)
    counts = sim.run(qc, shots=8192).result().get_counts()

    cal = calibrate_tensored_readout_mitigation(3, shots=8192, noise_model=nm, seed=11)
    mitigated = cal.mitigate_counts(counts)

    raw_p = counts.get("000", 0) / 8192
    mit_p = mitigated.get("000", 0) / 8192
    assert raw_p < 0.96  # readout error is visible in the raw data
    assert mit_p > raw_p
    assert mit_p > 0.97


# ----------------------------------------------------------- config validation ----
def test_config_rejects_unknown_mitigation_method():
    graph = create_cycle_graph(4).to_dict()
    with pytest.raises(ValueError):
        ExperimentConfig(graph_dict=graph, mitigation_method="magic")
    for method in ("none", "readout", "zne", "READOUT"):
        ExperimentConfig(graph_dict=graph, mitigation_method=method)


# ------------------------------------------------- noise benchmark plumbing fix ----
def test_ideal_vs_noisy_passes_config_fields_through(monkeypatch):
    """Plumbing only: the fake engine returns placeholder numbers that are never reported."""
    from qaoa_tuner.noise import benchmark as nb

    seen = []

    class FakeEngine:
        def run(self, cfg):
            seen.append(cfg)
            return SimpleNamespace(
                solution_metrics=SimpleNamespace(valid_coloring_rate=0.5, expected_conflicts=1.0)
            )

    monkeypatch.setattr(nb, "ExperimentEngine", FakeEngine)
    cfg = ExperimentConfig(
        graph_dict=create_cycle_graph(4).to_dict(),
        transpiler_optimization_level=3,
        initial_point=[0.1, 0.2],
        metadata={"tag": "x"},
    )
    nb.compare_ideal_vs_noisy(cfg, "depolarizing_mild")

    assert len(seen) == 2
    assert all(c.transpiler_optimization_level == 3 for c in seen)
    assert all(c.initial_point == [0.1, 0.2] for c in seen)
    assert all(c.metadata == {"tag": "x"} for c in seen)
    assert seen[0].noise_model_name is None
    assert seen[1].noise_model_name == "depolarizing_mild"
    assert seen[0].experiment_id != seen[1].experiment_id


# ------------------------------------------------------- comparison benchmark ----
@pytest.fixture(scope="module")
def tiny_comparison():
    from qaoa_tuner.mitigation.benchmark import compare_mitigation_performance

    cfg = ExperimentConfig(
        graph_dict=create_cycle_graph(4).to_dict(),
        num_colors=2, qaoa_p=1, optimizer="COBYLA", max_iter=4, shots=256, seed=5,
    )
    return compare_mitigation_performance(cfg, "realistic_superconducting", (1, 3))


def test_comparison_structure_and_overhead(tiny_comparison):
    r = tiny_comparison
    for stage in (r.ideal, r.noisy, r.readout, r.zne):
        assert 0.0 <= stage.valid_coloring_rate <= 1.0
        assert stage.expected_conflicts >= 0.0
    assert r.noisy.extra_circuits == 0 and r.noisy.extra_shots == 0
    assert r.readout.extra_circuits == 2  # tensored calibration
    assert r.readout.extra_shots == 2 * 256
    assert r.zne.extra_circuits == 1  # scale 3 on top of the baseline
    assert r.zne.max_two_qubit_gates_per_circuit == 3 * r.noisy.max_two_qubit_gates_per_circuit
    assert r.zne_valid_rate_fit["circuit_multiplier"] == 2


def test_comparison_is_serializable_and_records_versions(tiny_comparison, tmp_path):
    data = json.loads(tiny_comparison.to_json())
    assert set(data["stages"]) == {"ideal", "noisy", "readout", "zne"}
    assert data["config"]["seed"] == 5
    assert {"python", "qiskit", "qiskit_aer"} <= set(data["software"])
    saved = tiny_comparison.save(tmp_path / "out" / "cmp.json")
    assert saved.is_file()


def test_comparison_plot_saves_png(tiny_comparison, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from qaoa_tuner.mitigation.benchmark import plot_mitigation_comparison

    fig = plot_mitigation_comparison(tiny_comparison, tmp_path / "cmp.png")
    plt.close(fig)
    assert (tmp_path / "cmp.png").is_file()


def test_comparison_rejects_ideal_noise_profile():
    from qaoa_tuner.mitigation.benchmark import compare_mitigation_performance

    cfg = ExperimentConfig(graph_dict=create_cycle_graph(4).to_dict())
    with pytest.raises(ValueError):
        compare_mitigation_performance(cfg, "ideal")
    with pytest.raises(ValueError):
        compare_mitigation_performance(cfg, "depolarizing_mild", zne_scale_factors=(3, 5))
