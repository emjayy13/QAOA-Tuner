"""Readout error mitigation: calibration, correction, and overhead bookkeeping.

Readout (measurement) error means a qubit prepared in |0> is sometimes reported as 1,
and vice versa. We model this as a classical stochastic map on bitstring probabilities:

    p_measured = M @ p_true,     M[i, j] = P(measure bitstring i | prepared bitstring j)

Mitigation = (1) CALIBRATE: estimate M by preparing known basis states and measuring,
             (2) MEASURE:   run the real circuit as usual,
             (3) CORRECT:   solve p_true ~= M^{-1} @ p_measured, then project the result back
                            onto valid probabilities (clip negatives, renormalize).

Two calibrators are provided:

* ``ReadoutCalibrator`` (full matrix): 2^n calibration circuits, captures correlated
  readout errors, but is limited to n <= 8 qubits.
* ``TensoredReadoutCalibrator`` (per-qubit): only 2 calibration circuits for any n.
  Assumes readout errors are independent per qubit, i.e. M = A_{n-1} (x) ... (x) A_0.
  This matches the independent readout error in ``qaoa_tuner.noise.models``.

Limitations (important):
* Calibration prepares |1> with an X gate, so gate error on X is folded into the estimated
  readout error (state-preparation-and-measurement, SPAM, error).
* Mitigation amplifies statistical noise: M^{-1} has entries larger than 1.
* It corrects measurement error only. Gate errors are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

_MAX_FULL_QUBITS = 8
_MAX_TENSORED_QUBITS = 16  # correction builds a 2^n probability vector


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def _clean(bitstring: str) -> str:
    """Remove register separators so '01 10' and '0110' are treated the same."""
    return bitstring.replace(" ", "")


def _project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Turn an inverted vector into a valid probability vector.

    Matrix inversion can produce small negative entries (statistical noise amplified by
    M^{-1}). We clip negatives to zero and renormalize. This is simple and transparent,
    not the statistically optimal (least-squares-on-simplex) projection.
    """
    clipped = np.clip(x, 0.0, None)
    total = float(np.sum(clipped))
    if total > 0.0:
        return clipped / total
    return np.ones_like(x, dtype=float) / len(x)  # complete degeneracy fallback


def _probs_to_counts(probs: dict[str, float], total_shots: int) -> dict[str, int]:
    """Convert probabilities back to integer counts that sum exactly to ``total_shots``."""
    counts = {k: int(round(p * total_shots)) for k, p in probs.items()}
    counts = {k: v for k, v in counts.items() if v > 0}
    diff = total_shots - sum(counts.values())
    if diff != 0 and counts:
        top = max(counts, key=counts.get)
        counts[top] += diff
    return counts


def _execute_calibration(
    circuits: list[QuantumCircuit],
    backend: Any,
    noise_model: Any,
    seed: int,
    shots: int,
) -> Any:
    """Transpile (no optimization) and run calibration circuits; return the Aer result.

    If ``backend`` is given, ``noise_model`` is ignored (the backend carries its own noise).
    """
    if backend is None:
        backend = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    compiled = transpile(circuits, backend=backend, optimization_level=0)
    return backend.run(compiled, shots=shots).result()


# ----------------------------------------------------------------------------
# Full-matrix calibrator (captures correlated errors, n <= 8)
# ----------------------------------------------------------------------------
@dataclass
class ReadoutCalibrator:
    """Full 2^n x 2^n assignment-matrix calibrator.

    Attributes:
        num_qubits: Number of calibrated qubits.
        matrix: M of shape (2^n, 2^n), M[i, j] = P(measured i | prepared j).
        bitstrings: Ordered basis labels ['00..0', ..., '11..1'].
        condition_number: cond(M); large values mean the correction amplifies noise.
        calibration_overhead_shots: Total calibration shots executed.
    """

    num_qubits: int
    matrix: np.ndarray
    bitstrings: list[str]
    condition_number: float
    calibration_overhead_shots: int

    @property
    def num_calibration_circuits(self) -> int:
        return 2**self.num_qubits

    def mitigate_probabilities(self, raw_probs: dict[str, float]) -> dict[str, float]:
        """Apply x_est = M^{-1} y, then project onto the probability simplex."""
        dim = len(self.bitstrings)
        y = np.array([raw_probs.get(bs, 0.0) for bs in self.bitstrings], dtype=float)

        try:
            x_raw = np.linalg.solve(self.matrix, y)
        except np.linalg.LinAlgError:
            x_raw = np.linalg.pinv(self.matrix) @ y

        x_proj = _project_to_simplex(x_raw)
        return {bs: float(x_proj[i]) for i, bs in enumerate(self.bitstrings) if x_proj[i] > 1e-7}

    def mitigate_counts(self, raw_counts: dict[str, int]) -> dict[str, int]:
        """Mitigate a counts histogram; output counts sum to the same shot total."""
        total = sum(raw_counts.values())
        if total == 0:
            return {}
        raw_probs = {_clean(k): v / total for k, v in raw_counts.items()}
        return _probs_to_counts(self.mitigate_probabilities(raw_probs), total)


def generate_calibration_circuits(num_qubits: int) -> list[tuple[str, QuantumCircuit]]:
    """Generate the 2^n basis-state preparation circuits (X where the bit is 1, then measure)."""
    circuits: list[tuple[str, QuantumCircuit]] = []
    for idx in range(2**num_qubits):
        bs = format(idx, f"0{num_qubits}b")
        qc = QuantumCircuit(num_qubits, name=f"cal_{bs}")
        # Qiskit convention: char 0 of the string is qubit n-1
        for char_idx, char in enumerate(bs):
            if char == "1":
                qc.x(num_qubits - 1 - char_idx)
        qc.measure_all()
        circuits.append((bs, qc))
    return circuits


def calibrate_readout_mitigation(
    backend: Any = None,
    num_qubits: int = 4,
    shots: int = 1024,
    noise_model: Any = None,
    seed: int = 42,
) -> ReadoutCalibrator:
    """Build the full assignment matrix by running all 2^n basis-state circuits."""
    if num_qubits > _MAX_FULL_QUBITS:
        raise ValueError(
            f"Full-matrix calibration needs 2^n circuits and is limited to n <= "
            f"{_MAX_FULL_QUBITS}, got {num_qubits}. Use calibrate_tensored_readout_mitigation."
        )

    cal = generate_calibration_circuits(num_qubits)
    labels = [bs for bs, _ in cal]
    result = _execute_calibration([qc for _, qc in cal], backend, noise_model, seed, shots)

    dim = 2**num_qubits
    matrix = np.zeros((dim, dim), dtype=float)
    for j in range(dim):
        counts = result.get_counts(j)
        total = sum(counts.values())
        for i, meas_bs in enumerate(labels):
            matrix[i, j] = counts.get(meas_bs, 0) / total

    return ReadoutCalibrator(
        num_qubits=num_qubits,
        matrix=matrix,
        bitstrings=labels,
        condition_number=float(np.linalg.cond(matrix)),
        calibration_overhead_shots=dim * shots,
    )


# ----------------------------------------------------------------------------
# Tensored (per-qubit) calibrator: 2 circuits for any n
# ----------------------------------------------------------------------------
@dataclass
class TensoredReadoutCalibrator:
    """Per-qubit readout calibrator assuming independent readout error on each qubit.

    Attributes:
        num_qubits: Number of calibrated qubits.
        qubit_matrices: ``qubit_matrices[q]`` is the 2x2 matrix A_q with
            A_q[i, j] = P(qubit q measured i | qubit q prepared j). Index = qubit number.
        condition_number: cond(M) = prod_q cond(A_q) for M = tensor product of the A_q.
        calibration_overhead_shots: Total calibration shots executed.
    """

    num_qubits: int
    qubit_matrices: list[np.ndarray]
    condition_number: float
    calibration_overhead_shots: int

    @property
    def num_calibration_circuits(self) -> int:
        return 2

    def mitigate_probabilities(self, raw_probs: dict[str, float]) -> dict[str, float]:
        """Apply A_q^{-1} along each qubit axis of the probability tensor, then project."""
        n = self.num_qubits
        if n > _MAX_TENSORED_QUBITS:
            raise ValueError(f"Tensored correction limited to n <= {_MAX_TENSORED_QUBITS}.")

        vec = np.zeros(2**n, dtype=float)
        for bs, p in raw_probs.items():
            vec[int(_clean(bs), 2)] = p

        # Reshape to (2,)*n. Axis 0 is the leftmost bit = qubit n-1, so qubit q is axis n-1-q.
        tensor = vec.reshape((2,) * n)
        for q, mat in enumerate(self.qubit_matrices):
            axis = n - 1 - q
            inv = np.linalg.inv(mat)
            tensor = np.moveaxis(np.tensordot(inv, tensor, axes=([1], [axis])), 0, axis)

        x_proj = _project_to_simplex(tensor.reshape(-1))
        return {
            format(i, f"0{n}b"): float(x_proj[i]) for i in np.nonzero(x_proj > 1e-7)[0]
        }

    def mitigate_counts(self, raw_counts: dict[str, int]) -> dict[str, int]:
        """Mitigate a counts histogram; output counts sum to the same shot total."""
        total = sum(raw_counts.values())
        if total == 0:
            return {}
        raw_probs = {_clean(k): v / total for k, v in raw_counts.items()}
        return _probs_to_counts(self.mitigate_probabilities(raw_probs), total)


def calibrate_tensored_readout_mitigation(
    num_qubits: int,
    shots: int = 1024,
    noise_model: Any = None,
    seed: int = 42,
    backend: Any = None,
) -> TensoredReadoutCalibrator:
    """Estimate each qubit's 2x2 readout matrix from just two circuits.

    Circuit A prepares |0...0> (no gates) and measures: gives P(1|0) per qubit.
    Circuit B prepares |1...1> (X on every qubit) and measures: gives P(0|1) per qubit.
    """
    if num_qubits < 1:
        raise ValueError(f"num_qubits must be >= 1, got {num_qubits}.")

    all_zero = QuantumCircuit(num_qubits, name="cal_all_0")
    all_zero.measure_all()
    all_one = QuantumCircuit(num_qubits, name="cal_all_1")
    all_one.x(range(num_qubits))
    all_one.measure_all()

    result = _execute_calibration([all_zero, all_one], backend, noise_model, seed, shots)
    counts0 = {_clean(k): v for k, v in result.get_counts(0).items()}
    counts1 = {_clean(k): v for k, v in result.get_counts(1).items()}
    total0, total1 = sum(counts0.values()), sum(counts1.values())

    matrices: list[np.ndarray] = []
    cond = 1.0
    for q in range(num_qubits):
        char_pos = num_qubits - 1 - q  # string position of qubit q
        e01 = sum(c for bs, c in counts0.items() if bs[char_pos] == "1") / total0
        e10 = sum(c for bs, c in counts1.items() if bs[char_pos] == "0") / total1
        mat = np.array([[1.0 - e01, e10], [e01, 1.0 - e10]], dtype=float)
        if abs(np.linalg.det(mat)) < 1e-9:
            raise ValueError(
                f"Readout channel of qubit {q} is not invertible (P(1|0)={e01}, P(0|1)={e10})."
            )
        cond *= float(np.linalg.cond(mat))
        matrices.append(mat)

    return TensoredReadoutCalibrator(
        num_qubits=num_qubits,
        qubit_matrices=matrices,
        condition_number=cond,
        calibration_overhead_shots=2 * shots,
    )


def mitigate_readout_counts(
    counts: dict[str, int],
    calibrator: ReadoutCalibrator | TensoredReadoutCalibrator,
) -> dict[str, int]:
    """Apply a calibrated readout correction to a counts dictionary."""
    return calibrator.mitigate_counts(counts)
