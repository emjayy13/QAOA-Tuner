"""Classical optimization algorithms for QAOA variational parameter training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class OptimizerResult:
    """Summary of classical optimizer execution.

    Attributes:
        optimal_point: Optimized parameter vector [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}].
        optimal_value: Minimum energy / objective value found.
        num_evaluations: Total function evaluations executed.
        history: Sequence of objective function values across iterations.
        optimizer_name: "COBYLA" or "SPSA".
    """

    optimal_point: list[float]
    optimal_value: float
    num_evaluations: int
    history: list[float] = field(default_factory=list)
    optimizer_name: str = "COBYLA"


def minimize_cobyla(
    objective_fn: Callable[[np.ndarray], float],
    initial_point: np.ndarray | list[float],
    max_iter: int = 50,
    tol: float = 1e-4,
) -> OptimizerResult:
    """Optimize parameters using the gradient-free COBYLA algorithm.

    Why COBYLA:
        - Gradient-free: Constructs successive linear approximations to the objective.
        - Reliable for low-dimensional, noise-free or low-shot parameter spaces (p <= 3 => 2p <= 6 params).
        - Standard baseline in Qiskit variational algorithms.

    Args:
        objective_fn: Callable returning scalar cost for parameter vector theta.
        initial_point: Starting parameter values.
        max_iter: Maximum iterations / function evaluations.
        tol: Convergence tolerance.
    """
    x0 = np.asarray(initial_point, dtype=float)
    history: list[float] = []

    def callback_wrapper(x: np.ndarray) -> float:
        val = float(objective_fn(x))
        history.append(val)
        return val

    res = minimize(
        callback_wrapper,
        x0,
        method="COBYLA",
        options={"maxiter": max_iter, "tol": tol},
    )

    return OptimizerResult(
        optimal_point=[float(x) for x in res.x],
        optimal_value=float(res.fun),
        num_evaluations=len(history),
        history=history,
        optimizer_name="COBYLA",
    )


def minimize_spsa(
    objective_fn: Callable[[np.ndarray], float],
    initial_point: np.ndarray | list[float],
    max_iter: int = 50,
    a: float = 0.2,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: int | None = None,
) -> OptimizerResult:
    """Optimize parameters using Simultaneous Perturbation Stochastic Approximation (SPSA).

    Why SPSA:
        - Specifically designed for noisy objective functions (e.g. shot noise, quantum hardware noise).
        - Estimates full d-dimensional gradient using strictly 2 function evaluations per iteration:
          g_k = (f(theta + c_k * Delta) - f(theta - c_k * Delta)) / (2 * c_k) * Delta^{-1}
          where Delta is a random Bernoulli perturbation vector in {-1, +1}^d.
        - Robust against local stochastic traps in NISQ optimization landscapes.

    Args:
        objective_fn: Callable returning scalar cost for parameter vector theta.
        initial_point: Starting parameter values.
        max_iter: Number of SPSA update steps (2 evaluations per step).
        a: Initial step size parameter.
        c: Initial perturbation magnitude.
        alpha: Step size decay exponent (standard: 0.602).
        gamma: Perturbation decay exponent (standard: 0.101).
        seed: Random seed for deterministic reproducibility.
    """
    rng = np.random.default_rng(seed)
    theta = np.asarray(initial_point, dtype=float).copy()
    dim = len(theta)

    history: list[float] = []
    # Initial evaluation
    initial_val = float(objective_fn(theta))
    history.append(initial_val)

    best_theta = theta.copy()
    best_val = initial_val

    A_stability = max_iter * 0.1

    for k in range(max_iter):
        a_k = a / ((k + 1 + A_stability) ** alpha)
        c_k = c / ((k + 1) ** gamma)

        # Simultaneous perturbation vector Delta in {-1, +1}^dim
        delta = rng.choice([-1.0, 1.0], size=dim)

        theta_plus = theta + c_k * delta
        theta_minus = theta - c_k * delta

        y_plus = float(objective_fn(theta_plus))
        y_minus = float(objective_fn(theta_minus))

        # Simultaneous perturbation gradient estimate
        g_k = (y_plus - y_minus) / (2.0 * c_k * delta)

        # Gradient update step
        theta = theta - a_k * g_k

        # Current point evaluation
        current_val = float(objective_fn(theta))
        history.append(current_val)

        if current_val < best_val:
            best_val = current_val
            best_theta = theta.copy()

    return OptimizerResult(
        optimal_point=[float(x) for x in best_theta],
        optimal_value=best_val,
        num_evaluations=len(history) + 2 * max_iter,
        history=history,
        optimizer_name="SPSA",
    )
