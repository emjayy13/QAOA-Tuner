"""Zero-Noise Extrapolation (ZNE) via unitary gate folding and polynomial extrapolation.

Idea: we cannot turn the hardware noise off, but we CAN make it worse in a controlled way.
1. NOISE SCALING: build circuits that are logically identical but contain more noisy gates,
   so the effective noise strength is lambda times the original (lambda = 1, 3, 5, ...).
2. MEASURE the observable of interest at each lambda.
3. EXTRAPOLATE the measured values back to lambda = 0.

Folding rule for a gate G:   G  ->  G (G^dagger G)^k,   lambda = 1 + 2k.
Since G^dagger G = I, the ideal circuit is unchanged; on noisy hardware every extra gate adds
its own error.

Assumptions and limitations (important):
* Only two-qubit gates are folded. Single-qubit gate error and readout error are NOT scaled,
  so ZNE can only remove the two-qubit-gate part of the noise. Combine with readout
  mitigation for readout error.
* Noise must grow roughly linearly (or polynomially) with the number of gates. Real noise
  (and thermal relaxation, which depends on duration) may not follow the fitted model.
* Extrapolation amplifies shot noise. ``extrapolated_std_error`` reports the statistical
  uncertainty propagated through the fit when per-point errors are supplied; it does NOT
  include model (bias) error.
* Fold AFTER transpilation and run the folded circuit WITHOUT re-optimizing, otherwise the
  transpiler may cancel the G^dagger G pairs and silently undo the noise scaling.
* Only odd integer scale factors are supported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from qiskit.circuit import QuantumCircuit


@dataclass(frozen=True)
class ZNEResult:
    """Outcome of extrapolating one observable to zero noise.

    Attributes:
        scale_factors: Noise scale factors [lambda_1, lambda_2, ...], e.g. [1, 3, 5].
        scaled_values: Observable value measured at each scale factor.
        extrapolated_zero_noise_value: Fitted value at lambda = 0 (the intercept, unclipped).
        fit_model: "linear" or "polynomial" (quadratic).
        fit_coefficients: [c_0, c_1, ...] in ascending order; c_0 is the zero-noise estimate.
        circuit_multiplier: Number of circuit executions relative to one unmitigated run.
        noise_scale_sum: Sum of scale factors = multiplier on two-qubit-gate executions
            (if every scale uses the same number of shots).
        extrapolated_std_error: Propagated shot-noise standard error of the intercept,
            or None if per-point errors were not supplied.
    """

    scale_factors: list[float]
    scaled_values: list[float]
    extrapolated_zero_noise_value: float
    fit_model: str
    fit_coefficients: list[float]
    circuit_multiplier: int
    noise_scale_sum: float
    extrapolated_std_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fold_circuit_gates(circuit: QuantumCircuit, scale_factor: int = 1) -> QuantumCircuit:
    """Return a copy of ``circuit`` whose two-qubit gates are folded to scale noise.

    Each two-qubit gate G becomes G (G^dagger G)^k with k = (scale_factor - 1) / 2.
    Measurements and barriers must be at the end of the circuit (as produced by
    ``measure_all``); they are re-attached after the unitary part.

    Args:
        circuit: Circuit to fold. Register structure and global phase are preserved.
        scale_factor: Odd positive integer (1, 3, 5, ...).

    Raises:
        ValueError: If the scale factor is not an odd positive integer, or if a gate
            appears after a measurement (mid-circuit measurement is not supported).
    """
    if (
        not isinstance(scale_factor, (int, np.integer))
        or scale_factor < 1
        or scale_factor % 2 == 0
    ):
        raise ValueError(
            f"ZNE gate folding requires an odd positive integer scale factor "
            f"(1, 3, 5, ...), got {scale_factor!r}."
        )

    if scale_factor == 1:
        return circuit.copy()

    k = (int(scale_factor) - 1) // 2
    # copy_empty_like keeps the SAME qubits, clbits, registers and global phase,
    # so the original qubit objects can be re-used when appending.
    folded = circuit.copy_empty_like(name=f"{circuit.name}_fold_{scale_factor}")

    deferred = []
    seen_measure = False
    for inst in circuit.data:
        op = inst.operation
        if op.name in ("measure", "barrier"):
            seen_measure = seen_measure or op.name == "measure"
            deferred.append(inst)
            continue
        if seen_measure:
            raise ValueError("Gate found after a measurement; mid-circuit measurement unsupported.")

        folded.append(op, inst.qubits, inst.clbits)
        if len(inst.qubits) == 2:
            inverse = op.inverse()
            for _ in range(k):
                folded.append(inverse, inst.qubits, inst.clbits)
                folded.append(op, inst.qubits, inst.clbits)

    for inst in deferred:
        folded.append(inst.operation, inst.qubits, inst.clbits)

    return folded


def extrapolate_zero_noise(
    scale_factors: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    model: str = "linear",
    std_errs: list[float] | np.ndarray | None = None,
) -> ZNEResult:
    """Fit value(lambda) and return its intercept at lambda = 0.

    Models:
        "linear":     y = c0 + c1 * lambda
        "polynomial": y = c0 + c1 * lambda + c2 * lambda^2   (needs >= 3 points)

    The fit is ordinary least squares: c = pinv(V) @ y with V the Vandermonde matrix.
    Because the intercept is a fixed linear combination w . y of the data (w = first row
    of pinv(V)), independent point errors sigma_i propagate as sqrt(sum (w_i sigma_i)^2).

    Args:
        scale_factors: Noise scale factors, e.g. [1, 3, 5].
        values: Observable at each scale factor.
        model: "linear" or "polynomial".
        std_errs: Optional standard error of each value (e.g. binomial shot noise).
    """
    lambdas = np.asarray(scale_factors, dtype=float)
    y = np.asarray(values, dtype=float)

    if len(lambdas) != len(y):
        raise ValueError(f"Length mismatch: {len(lambdas)} scale factors vs {len(y)} values.")
    if len(lambdas) < 2:
        raise ValueError("ZNE requires at least 2 scale points to extrapolate.")

    clean = model.lower()
    if clean == "linear":
        degree = 1
    elif clean in ("polynomial", "quadratic"):
        if len(lambdas) < 3:
            raise ValueError("Polynomial (quadratic) ZNE needs at least 3 scale points.")
        degree = 2
    else:
        raise ValueError(f"Unsupported ZNE model: '{model}'. Supported: 'linear', 'polynomial'.")

    vander = np.vander(lambdas, degree + 1, increasing=True)  # columns: 1, lambda, lambda^2
    pseudo_inverse = np.linalg.pinv(vander)
    coeffs = pseudo_inverse @ y

    std_error = None
    if std_errs is not None:
        sigma = np.asarray(std_errs, dtype=float)
        if len(sigma) != len(y):
            raise ValueError("std_errs must have the same length as values.")
        std_error = float(np.sqrt(np.sum((pseudo_inverse[0] * sigma) ** 2)))

    return ZNEResult(
        scale_factors=[float(s) for s in lambdas],
        scaled_values=[float(v) for v in y],
        extrapolated_zero_noise_value=float(coeffs[0]),
        fit_model=clean,
        fit_coefficients=[float(c) for c in coeffs],
        circuit_multiplier=len(lambdas),
        noise_scale_sum=float(np.sum(lambdas)),
        extrapolated_std_error=std_error,
    )
