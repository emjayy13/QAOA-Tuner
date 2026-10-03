"""Deterministic, explainable configuration recommendations (no machine learning).

Input: a ``TunerResult`` (every configuration already measured, Pareto frontier marked).
Output: up to three picks, each with reasons built from the run's own numbers.

Rules, in order (all visible in this file):

1. FRONTIER. Start from the distinct Pareto trade-offs (exact ties already grouped; the lowest
   transpiler level represents a group).
2. QUALITY FLOOR. Drop trade-offs whose valid-coloring rate is below
   ``max(min_quality_fraction * best_rate_in_run, min_quality_absolute)``. Without a floor, a
   cheap configuration that barely solves the problem would be called "low cost". The default
   fraction (0.8) is a starting value chosen for readability, not a derived constant.
3. PICK from the survivors:
   - LOW COST     : lowest two-qubit gate budget; ties -> fewer shots -> higher quality.
   - HIGH QUALITY : highest valid-coloring rate; ties -> lower gate budget -> fewer shots.
   - BALANCED     : closest to the ideal corner (best quality, lowest cost) after rescaling both
                    to [0, 1] over the survivors, using weights ``quality_weight`` and
                    ``cost_weight``: distance = sqrt(wq*(1-q)^2 + wc*c^2).
   Remaining ties always fall back to lower transpiler level, then label order, so the output
   is deterministic.

Descriptive categories, not universal claims: the recommendation only holds for the tested
graph, backend, noise profile, shots and seed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from qaoa_tuner.tuner.results import ParetoGroup, TunerRecord, TunerResult

CATEGORIES = ("low_cost", "balanced", "high_quality")

DISCLAIMER = (
    "These recommendations describe the configurations that were actually run, for this graph, "
    "backend, noise profile, shot count and seed. They are not universally optimal and may "
    "change with other graphs, hardware or random seeds."
)


@dataclass(frozen=True)
class RecommendationSettings:
    """Tunable parameters of the selection rules."""

    min_quality_fraction: float = 0.8
    min_quality_absolute: float | None = None
    quality_weight: float = 0.5
    cost_weight: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_quality_fraction <= 1.0:
            raise ValueError("min_quality_fraction must be between 0 and 1.")
        if self.min_quality_absolute is not None and not 0.0 <= self.min_quality_absolute <= 1.0:
            raise ValueError("min_quality_absolute must be between 0 and 1.")
        if self.quality_weight < 0 or self.cost_weight < 0:
            raise ValueError("Weights must be non-negative.")
        if self.quality_weight + self.cost_weight <= 0:
            raise ValueError("At least one weight must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Recommendation:
    """One recommended configuration and why it was chosen."""

    category: str
    label: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    equivalent_labels: list[str]
    reasons: list[str]
    caveats: list[str]
    same_choice_as: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExcludedCandidate:
    """A frontier trade-off removed by the quality floor."""

    label: str
    valid_coloring_rate: float
    reason: str


@dataclass(frozen=True)
class RecommendationReport:
    """Everything the dashboard or CLI needs to show recommendations and their rationale."""

    status: str  # "ok" | "no_candidates" | "no_results"
    settings: dict[str, Any]
    context: dict[str, Any]
    best_valid_coloring_rate: float | None
    quality_floor: float | None
    num_frontier_trade_offs: int
    num_candidates: int
    excluded: list[ExcludedCandidate]
    recommendations: list[Recommendation]
    messages: list[str]
    disclaimer: str = DISCLAIMER

    def get(self, category: str) -> Recommendation | None:
        return next((r for r in self.recommendations if r.category == category), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


# ----------------------------------------------------------------------------- helpers ----
def _cost(r: TunerRecord) -> float:
    return float(r.two_qubit_gate_budget)


def _tie_tail(r: TunerRecord) -> tuple[int, str]:
    return (r.optimization_level, r.label)


def _metrics(r: TunerRecord) -> dict[str, Any]:
    return {
        "valid_coloring_rate": r.valid_coloring_rate,
        "valid_rate_std_error": r.valid_rate_std_error,
        "expected_conflicts": r.expected_conflicts,
        "ideal_valid_coloring_rate": r.ideal_valid_coloring_rate,
        "quality_degradation": r.quality_degradation,
        "circuit_depth": r.circuit_depth,
        "two_qubit_gates": r.two_qubit_gates,
        "two_qubit_gate_budget": r.two_qubit_gate_budget,
        "total_shots": r.total_shots,
    }


def _config(r: TunerRecord) -> dict[str, Any]:
    return {
        "qaoa_p": r.qaoa_p,
        "optimizer": r.optimizer,
        "optimization_level": r.optimization_level,
        "mitigation": r.mitigation,
    }


def _rate_text(r: TunerRecord) -> str:
    if r.valid_rate_std_error is None:
        return f"{r.valid_coloring_rate:.3f}"
    return f"{r.valid_coloring_rate:.3f} (+/- {r.valid_rate_std_error:.3f})"


def _distinguishable(a: TunerRecord, b: TunerRecord) -> bool | None:
    """Is the valid-rate difference larger than 2 combined standard errors? None = unknown."""
    if a.valid_rate_std_error is None or b.valid_rate_std_error is None:
        return None
    combined = math.sqrt(a.valid_rate_std_error**2 + b.valid_rate_std_error**2)
    return abs(a.valid_coloring_rate - b.valid_coloring_rate) > 2.0 * combined


def _comparison_caveats(rec: TunerRecord, pairs: list[tuple[TunerRecord, str]]) -> list[str]:
    """Statistical caveats for the comparisons made; same-kind notes are merged into one line."""
    untestable: list[str] = []
    within_noise: list[str] = []
    for other, name in pairs:
        if other.label == rec.label:
            continue
        diff = rec.valid_coloring_rate - other.valid_coloring_rate
        verdict = _distinguishable(rec, other)
        if verdict is None:
            untestable.append(f"the {name} pick ({diff:+.3f})")
        elif verdict is False:
            within_noise.append(f"the {name} pick ({diff:+.3f})")
    notes = []
    if untestable:
        notes.append(
            "No standard error is available for one of the rates (readout-mitigated results "
            f"have none), so the valid-rate difference vs {' and '.join(untestable)} cannot be tested."
        )
    if within_noise:
        notes.append(
            f"Valid-rate difference vs {' and '.join(within_noise)} is within 2 combined "
            "standard errors: not statistically distinguishable at this shot count."
        )
    return notes


def _mitigation_reason(r: TunerRecord, by_key: dict[tuple, TunerRecord]) -> str:
    if r.mitigation == "none":
        return "No error mitigation (baseline), so no extra shots or calibration."
    base = by_key.get((r.qaoa_p, r.optimizer, r.optimization_level, "none"))
    if base is None:
        return f"Uses '{r.mitigation}' mitigation (no unmitigated run of this circuit to compare)."
    delta = r.valid_coloring_rate - base.valid_coloring_rate
    multiplier = r.total_shots / base.total_shots if base.total_shots else float("nan")
    return (
        f"'{r.mitigation}' mitigation moves the valid-coloring rate from "
        f"{base.valid_coloring_rate:.3f} (same circuit, unmitigated) to "
        f"{r.valid_coloring_rate:.3f} ({delta:+.3f}), using {multiplier:.1f}x the shots."
    )


def _normalized(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.0] * len(values)
    return [(v - low) / (high - low) for v in values]


def _group_label_list(group: ParetoGroup) -> list[str]:
    return list(group.equivalent_labels)


# ------------------------------------------------------------------------------- engine ----
def recommend(
    result: TunerResult, settings: RecommendationSettings | None = None
) -> RecommendationReport:
    """Build the recommendation report for a tuning result."""
    settings = settings or RecommendationSettings()
    s = result.settings
    context = {
        "graph": s.get("graph_dict", {}).get("name"),
        "num_colors": s.get("num_colors"),
        "noise_profile": result.noise_profile.get("name"),
        "backend": s.get("backend_name"),
        "shots": s.get("shots"),
        "seed": s.get("seed"),
        "num_configurations": len(result.records),
    }

    def empty(status: str, message: str, **extra: Any) -> RecommendationReport:
        return RecommendationReport(
            status=status, settings=settings.to_dict(), context=context,
            best_valid_coloring_rate=extra.get("best"), quality_floor=extra.get("floor"),
            num_frontier_trade_offs=extra.get("n_groups", 0), num_candidates=0,
            excluded=extra.get("excluded", []), recommendations=[], messages=[message],
        )

    if not result.records:
        return empty("no_results", "The tuning result contains no configurations.")

    best = max(r.valid_coloring_rate for r in result.records)
    floor = settings.min_quality_fraction * best
    if settings.min_quality_absolute is not None:
        floor = max(floor, settings.min_quality_absolute)

    groups = result.pareto_groups()
    survivors: list[ParetoGroup] = []
    excluded: list[ExcludedCandidate] = []
    for g in groups:
        rep = g.representative
        if rep.valid_coloring_rate >= floor:
            survivors.append(g)
        else:
            excluded.append(ExcludedCandidate(
                rep.label, rep.valid_coloring_rate,
                f"valid-coloring rate {rep.valid_coloring_rate:.3f} is below the quality floor "
                f"{floor:.3f} (best in run: {best:.3f})",
            ))

    if not survivors:
        return empty(
            "no_candidates",
            f"No Pareto-efficient trade-off reaches the quality floor {floor:.3f}. Lower "
            "min_quality_fraction / min_quality_absolute, or run more or better configurations.",
            best=best, floor=floor, n_groups=len(groups), excluded=excluded,
        )

    reps = [g.representative for g in survivors]
    by_key = {(r.qaoa_p, r.optimizer, r.optimization_level, r.mitigation): r for r in result.records}

    # ---- the three picks -------------------------------------------------------------------
    low = min(reps, key=lambda r: (_cost(r), r.total_shots, -r.valid_coloring_rate, *_tie_tail(r)))
    high = min(reps, key=lambda r: (-r.valid_coloring_rate, _cost(r), r.total_shots, *_tie_tail(r)))

    q_norm = _normalized([r.valid_coloring_rate for r in reps])
    c_norm = _normalized([_cost(r) for r in reps])
    wq, wc = settings.quality_weight, settings.cost_weight
    distances = [
        math.sqrt(wq * (1.0 - q) ** 2 + wc * c**2) for q, c in zip(q_norm, c_norm, strict=True)
    ]
    order = sorted(
        range(len(reps)),
        key=lambda i: (distances[i], _cost(reps[i]), -reps[i].valid_coloring_rate,
                       *_tie_tail(reps[i])),
    )
    bal_idx = order[0]
    balanced = reps[bal_idx]

    picks = {"low_cost": low, "balanced": balanced, "high_quality": high}
    group_of = {g.representative.label: g for g in survivors}
    n = len(reps)
    messages: list[str] = []

    def build(category: str, rec: TunerRecord) -> Recommendation:
        reasons: list[str] = []
        caveats: list[str] = []

        share = rec.valid_coloring_rate / best if best > 0 else 0.0
        reasons.append(
            f"Valid-coloring rate {_rate_text(rec)}: {share:.0%} of the best in this run "
            f"({best:.3f}); quality floor was {floor:.3f}."
        )
        if category == "low_cost":
            reasons.append(
                f"Lowest two-qubit gate budget among the {n} candidates above the floor: "
                f"{rec.two_qubit_gate_budget} ({rec.two_qubit_gates} gates per circuit, depth "
                f"{rec.circuit_depth})."
            )
            ties = [r.label for r in reps if r is not rec and _cost(r) == _cost(rec)]
            if ties:
                reasons.append(
                    f"Ties on gate budget with {', '.join(ties)}; chosen for fewer total shots "
                    f"({rec.total_shots}) or higher quality."
                )
            other, name = high, "high-quality"
            if other.label != rec.label:
                reasons.append(
                    f"Versus the {name} pick: {rec.valid_coloring_rate - other.valid_coloring_rate:+.3f} "
                    f"valid-coloring rate, {rec.two_qubit_gate_budget - other.two_qubit_gate_budget:+d} "
                    f"two-qubit gate budget, {rec.total_shots - other.total_shots:+d} shots."
                )
        elif category == "high_quality":
            reasons.append(
                f"Highest valid-coloring rate among the {n} candidates above the floor; "
                f"degradation from the noise-free value is {rec.quality_degradation:.3f}."
            )
            other, name = low, "low-cost"
            if other.label != rec.label:
                reasons.append(
                    f"Versus the {name} pick: {rec.valid_coloring_rate - other.valid_coloring_rate:+.3f} "
                    f"valid-coloring rate for {rec.two_qubit_gate_budget - other.two_qubit_gate_budget:+d} "
                    f"two-qubit gate budget and {rec.total_shots - other.total_shots:+d} shots."
                )
        else:  # balanced
            i = reps.index(rec)
            reasons.append(
                f"Closest to the ideal corner (best quality, lowest cost) among {n} candidates "
                f"(distance {distances[i]:.3f}, weights quality={wq:g}, cost={wc:g}): reaches "
                f"{q_norm[i]:.0%} of the candidates' quality range at {c_norm[i]:.0%} of their "
                "cost range."
            )
            for other, name in ((low, "low-cost"), (high, "high-quality")):
                if other.label != rec.label:
                    reasons.append(
                        f"Versus the {name} pick: {rec.valid_coloring_rate - other.valid_coloring_rate:+.3f} "
                        f"valid-coloring rate, {rec.two_qubit_gate_budget - other.two_qubit_gate_budget:+d} "
                        f"two-qubit gate budget."
                    )

        reasons.append(_mitigation_reason(rec, by_key))
        group = group_of[rec.label]
        if group.equivalent_labels:
            reasons.append(
                f"Identical on every objective to {', '.join(group.equivalent_labels)} "
                f"(same circuit at a higher transpiler level); the lowest level is shown."
            )

        if category == "low_cost":
            pairs = [(high, "high-quality")]
        elif category == "high_quality":
            pairs = [(low, "low-cost")]
        else:
            pairs = [(low, "low-cost"), (high, "high-quality")]
        caveats.extend(_comparison_caveats(rec, pairs))

        same = [c for c, r in picks.items() if c != category and r.label == rec.label]
        return Recommendation(
            category=category, label=rec.label, config=_config(rec), metrics=_metrics(rec),
            equivalent_labels=_group_label_list(group), reasons=reasons, caveats=caveats,
            same_choice_as=same,
        )

    recommendations = [build(c, picks[c]) for c in CATEGORIES]

    if len({r.label for r in recommendations}) < len(recommendations):
        messages.append(
            "Some categories select the same configuration: among the candidates above the "
            "quality floor there is no further trade-off to separate them."
        )
    if excluded:
        messages.append(
            f"{len(excluded)} Pareto trade-off(s) were excluded by the quality floor "
            f"({floor:.3f}); see 'excluded'."
        )

    return RecommendationReport(
        status="ok", settings=settings.to_dict(), context=context,
        best_valid_coloring_rate=best, quality_floor=floor,
        num_frontier_trade_offs=len(groups), num_candidates=n, excluded=excluded,
        recommendations=recommendations, messages=messages,
    )


# -------------------------------------------------------------------------- text output ----
_TITLES = {"low_cost": "LOW COST", "balanced": "BALANCED", "high_quality": "HIGH QUALITY"}


def format_report(report: RecommendationReport) -> str:
    """Plain-text rendering for the CLI (the dashboard will render the same report object)."""
    c = report.context
    lines = [
        "=" * 100,
        f" Recommendations: {c.get('graph')}, k={c.get('num_colors')}, noise={c.get('noise_profile')}, "
        f"backend={c.get('backend')}, shots={c.get('shots')}, seed={c.get('seed')}",
        "=" * 100,
    ]
    if report.status != "ok":
        lines += [f" {m}" for m in report.messages]
    else:
        lines.append(
            f" {report.num_frontier_trade_offs} Pareto trade-offs; {report.num_candidates} above "
            f"the quality floor {report.quality_floor:.3f} (best in run {report.best_valid_coloring_rate:.3f})"
        )
        for rec in report.recommendations:
            m = rec.metrics
            lines += [
                "",
                f"[{_TITLES[rec.category]}] {rec.label}",
                f"  p={rec.config['qaoa_p']}, optimizer={rec.config['optimizer']}, "
                f"transpiler level={rec.config['optimization_level']}, mitigation={rec.config['mitigation']}",
                f"  valid rate {m['valid_coloring_rate']:.3f}, 2q gates {m['two_qubit_gates']}, "
                f"2q budget {m['two_qubit_gate_budget']}, depth {m['circuit_depth']}, shots {m['total_shots']}",
                "  Why:",
            ]
            lines += [f"   - {r}" for r in rec.reasons]
            if rec.same_choice_as:
                lines.append(f"   - Also the {', '.join(rec.same_choice_as)} pick.")
            if rec.caveats:
                lines.append("  Caveats:")
                lines += [f"   ! {x}" for x in rec.caveats]
        if report.excluded:
            lines += ["", " Excluded by the quality floor:"]
            lines += [f"   x {e.label}: {e.reason}" for e in report.excluded]
        if report.messages:
            lines.append("")
            lines += [f" Note: {m}" for m in report.messages]
    lines += ["", f" {report.disclaimer}"]
    return "\n".join(lines)
