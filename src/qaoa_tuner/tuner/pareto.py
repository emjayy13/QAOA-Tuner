"""Transparent Pareto-dominance analysis.

Definition used here (nothing more sophisticated):

    A DOMINATES B  if  A is at least as good as B on EVERY objective
                       and strictly better on AT LEAST ONE objective.

A configuration is Pareto-efficient (on the "frontier") if no other configuration dominates
it. The frontier is the set of defensible trade-offs: moving from one frontier point to
another always gives up something on at least one objective.

Optional per-objective ``tolerance`` (epsilon-dominance): differences smaller than the
tolerance count as ties. Use it for noisy metrics such as a valid-coloring rate measured
with finite shots (standard error ~0.01), so that a 0.003 difference cannot knock a
configuration off the frontier. With tolerances, "dominates" is no longer transitive, but
the frontier is still well defined (points that nobody dominates). Default: strict (0).

Every function is O(n^2) in the number of configurations, which is fine for tens to
hundreds of points.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Objective:
    """A named metric and whether larger values are better."""

    name: str
    maximize: bool = True


# Quality (max), hardware cost (min), robustness = quality lost to noise (min).
CORE_OBJECTIVES: tuple[Objective, ...] = (
    Objective("valid_coloring_rate", maximize=True),
    Objective("two_qubit_gate_budget", maximize=False),
    Objective("quality_degradation", maximize=False),
)

# Default: the core three plus shot overhead. Without it, readout mitigation looks free
# (its calibration circuits contain no two-qubit gates) and would dominate "no mitigation"
# whenever it helps at all. With it, mitigation has to earn its extra shots.
DEFAULT_OBJECTIVES: tuple[Objective, ...] = CORE_OBJECTIVES + (
    Objective("total_shots", maximize=False),
)


def _oriented(point: Mapping[str, float], objective: Objective) -> float:
    """Return the value so that larger is always better."""
    if objective.name not in point:
        raise KeyError(f"Point has no metric '{objective.name}'. Available: {sorted(point)}")
    value = float(point[objective.name])
    if not math.isfinite(value):
        raise ValueError(f"Metric '{objective.name}' is not finite: {value}")
    return value if objective.maximize else -value


def dominates(
    a: Mapping[str, float],
    b: Mapping[str, float],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
    tolerance: Mapping[str, float] | None = None,
) -> bool:
    """True if ``a`` dominates ``b`` (see module docstring)."""
    tol = tolerance or {}
    strictly_better = False
    for obj in objectives:
        t = tol.get(obj.name, 0.0)
        av, bv = _oriented(a, obj), _oriented(b, obj)
        if av < bv - t:  # a is worse by more than the tolerance
            return False
        if av > bv + t:  # a is better by more than the tolerance
            strictly_better = True
    return strictly_better


def dominators(
    points: Sequence[Mapping[str, float]],
    index: int,
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
    tolerance: Mapping[str, float] | None = None,
) -> list[int]:
    """Indices of all points that dominate ``points[index]`` (empty => on the frontier)."""
    return [
        j
        for j, other in enumerate(points)
        if j != index and dominates(other, points[index], objectives, tolerance)
    ]


def pareto_front_indices(
    points: Sequence[Mapping[str, float]],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
    tolerance: Mapping[str, float] | None = None,
) -> list[int]:
    """Indices of Pareto-efficient points, in input order."""
    return [i for i in range(len(points)) if not dominators(points, i, objectives, tolerance)]


def group_equivalent(
    points: Sequence[Mapping[str, float]],
    objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
) -> list[list[int]]:
    """Group point indices whose objective values are EXACTLY equal.

    Equal points never dominate each other, so ties all stay on the frontier. Grouping them
    lets a report show one trade-off once ("reachable by configurations A, B and C").
    Groups appear in order of their first member; indices inside a group stay in input order.
    """
    groups: dict[tuple[float, ...], list[int]] = {}
    for i, point in enumerate(points):
        key = tuple(float(point[obj.name]) for obj in objectives)
        groups.setdefault(key, []).append(i)
    return list(groups.values())
