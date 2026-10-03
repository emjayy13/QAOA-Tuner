"""Tests for Phase 8: the recommendation engine.

Most tests use a hand-built tuning result so the expected picks can be worked out by hand.
The numbers in ``_SEED42_FRONTIER`` are COPIED FROM A REAL TUNER RUN (cycle_4, k=2, seed 42,
realistic_superconducting noise, fake_linear_5q) and serve only as regression input for the
selection logic; they are not new results. Fields the engine does not read are placeholders.
"""

from __future__ import annotations

import json

import pytest

from qaoa_tuner.recommendation.engine import (
    DISCLAIMER,
    RecommendationSettings,
    format_report,
    recommend,
)
from qaoa_tuner.tuner.pareto import DEFAULT_OBJECTIVES, pareto_front_indices
from qaoa_tuner.tuner.results import TunerRecord, TunerResult

# label, p, optimizer, level, mitigation, valid, std_err, degradation, gates, budget, depth, shots
_SEED42_FRONTIER = [
    ("p2-COBYLA-O1-zne", 2, "COBYLA", 1, "zne", 0.709, 0.018, 0.177, 22, 198, 40, 3072),
    ("p2-COBYLA-O2-zne", 2, "COBYLA", 2, "zne", 0.709, 0.018, 0.177, 22, 198, 40, 3072),
    ("p2-COBYLA-O3-zne", 2, "COBYLA", 3, "zne", 0.709, 0.018, 0.177, 22, 198, 40, 3072),
    ("p2-COBYLA-O1-readout", 2, "COBYLA", 1, "readout", 0.669, None, 0.217, 22, 22, 40, 3072),
    ("p2-COBYLA-O2-readout", 2, "COBYLA", 2, "readout", 0.669, None, 0.217, 22, 22, 40, 3072),
    ("p2-COBYLA-O3-readout", 2, "COBYLA", 3, "readout", 0.669, None, 0.217, 22, 22, 40, 3072),
    ("p2-COBYLA-O1-none", 2, "COBYLA", 1, "none", 0.635, 0.015, 0.251, 22, 22, 40, 1024),
    ("p2-COBYLA-O2-none", 2, "COBYLA", 2, "none", 0.635, 0.015, 0.251, 22, 22, 40, 1024),
    ("p2-COBYLA-O3-none", 2, "COBYLA", 3, "none", 0.635, 0.015, 0.251, 22, 22, 40, 1024),
    ("p2-SPSA-O1-zne", 2, "SPSA", 1, "zne", 0.500, 0.018, 0.102, 22, 198, 40, 3072),
    ("p1-COBYLA-O2-zne", 1, "COBYLA", 2, "zne", 0.490, 0.019, 0.046, 10, 90, 24, 3072),
    ("p1-COBYLA-O2-readout", 1, "COBYLA", 2, "readout", 0.483, None, 0.053, 10, 10, 24, 3072),
    ("p1-COBYLA-O2-none", 1, "COBYLA", 2, "none", 0.459, 0.016, 0.077, 10, 10, 24, 1024),
    ("p1-SPSA-O2-zne", 1, "SPSA", 2, "zne", 0.268, 0.016, 0.014, 10, 90, 26, 3072),
    ("p1-SPSA-O3-zne", 1, "SPSA", 3, "zne", 0.268, 0.016, 0.014, 10, 90, 26, 3072),
    ("p1-SPSA-O2-readout", 1, "SPSA", 2, "readout", 0.267, None, 0.016, 10, 10, 26, 3072),
    ("p1-SPSA-O3-readout", 1, "SPSA", 3, "readout", 0.267, None, 0.016, 10, 10, 26, 3072),
    ("p1-SPSA-O2-none", 1, "SPSA", 2, "none", 0.259, 0.014, 0.023, 10, 10, 26, 1024),
    ("p1-SPSA-O3-none", 1, "SPSA", 3, "none", 0.259, 0.014, 0.023, 10, 10, 26, 1024),
    # A configuration the frontier must NOT contain (dominated by p2-COBYLA-O1-zne).
    ("p3-SPSA-O2-zne", 3, "SPSA", 2, "zne", 0.6165, 0.018, 0.3679, 42, 378, 73, 3072),
]


def _record(spec) -> TunerRecord:
    label, p, opt, level, mit, valid, se, deg, gates, budget, depth, shots = spec
    return TunerRecord(
        label=label, qaoa_p=p, optimizer=opt, optimization_level=level, mitigation=mit,
        valid_coloring_rate=valid, valid_rate_std_error=se, expected_conflicts=0.0,
        ideal_valid_coloring_rate=valid + deg, quality_degradation=deg, circuit_depth=depth,
        two_qubit_gates=gates, one_qubit_gates=0, logical_two_qubit_gates=0,
        two_qubit_overhead_ratio=0.0, two_qubit_gate_budget=budget, total_shots=shots,
    )


def make_result(specs=None) -> TunerResult:
    """Build a TunerResult, marking the Pareto frontier with the real dominance code."""
    records = [_record(s) for s in (specs if specs is not None else _SEED42_FRONTIER)]
    points = [r.to_dict() for r in records]
    front = set(pareto_front_indices(points, DEFAULT_OBJECTIVES))
    from dataclasses import replace

    records = [replace(r, is_pareto_optimal=i in front) for i, r in enumerate(records)]
    return TunerResult(
        settings={"graph_dict": {"name": "cycle_4"}, "num_colors": 2, "backend_name": "fake_linear_5q",
                  "shots": 1024, "seed": 42},
        noise_profile={"name": "realistic_superconducting"}, backend={},
        objectives=[{"name": o.name, "maximize": o.maximize} for o in DEFAULT_OBJECTIVES],
        tolerance={}, training=[], records=records, wall_time_seconds=0.0, software={},
    )


# ------------------------------------------------------------------ settings ----


def test_settings_validation():
    with pytest.raises(ValueError):
        RecommendationSettings(min_quality_fraction=1.5)
    with pytest.raises(ValueError):
        RecommendationSettings(min_quality_absolute=-0.1)
    with pytest.raises(ValueError):
        RecommendationSettings(quality_weight=-1)
    with pytest.raises(ValueError):
        RecommendationSettings(quality_weight=0, cost_weight=0)
    RecommendationSettings()  # defaults are valid


# --------------------------------------------------------- the hand-worked case ----


def test_default_picks_match_hand_calculation():
    report = recommend(make_result())
    assert report.status == "ok"
    assert report.best_valid_coloring_rate == pytest.approx(0.709)
    assert report.quality_floor == pytest.approx(0.8 * 0.709)
    assert report.num_frontier_trade_offs == 10  # 19 frontier records, exact ties grouped
    assert report.num_candidates == 3  # only the three p2-COBYLA trade-offs clear the floor
    assert report.get("low_cost").label == "p2-COBYLA-O1-none"  # tie on budget -> fewer shots
    assert report.get("high_quality").label == "p2-COBYLA-O1-zne"
    assert report.get("balanced").label == "p2-COBYLA-O1-readout"  # distance 0.38 vs 0.71 and 0.71


def test_dominated_configuration_is_never_recommended():
    labels = {r.label for r in recommend(make_result(), RecommendationSettings(
        min_quality_fraction=0.0)).recommendations}
    assert "p3-SPSA-O2-zne" not in labels


def test_quality_floor_excludes_cheap_low_quality_points_with_reasons():
    report = recommend(make_result())
    excluded = {e.label for e in report.excluded}
    assert len(report.excluded) == 7
    assert {"p1-SPSA-O2-none", "p1-COBYLA-O2-none", "p2-SPSA-O1-zne"} <= excluded
    assert all("below the quality floor" in e.reason for e in report.excluded)
    # Without a floor the cheapest pick is a p=1 configuration whose quality is below the
    # default floor: this is exactly what the floor keeps out of the recommendations.
    no_floor = recommend(make_result(), RecommendationSettings(min_quality_fraction=0.0))
    cheapest = no_floor.get("low_cost")
    assert cheapest.label == "p1-COBYLA-O2-none"  # budget 10; ties broken by shots, then quality
    assert cheapest.metrics["valid_coloring_rate"] < 0.8 * 0.709
    assert recommend(make_result()).get("low_cost").label != cheapest.label


def test_absolute_floor_overrides_fraction_when_higher():
    report = recommend(make_result(), RecommendationSettings(min_quality_absolute=0.65))
    assert report.quality_floor == pytest.approx(0.65)
    assert report.num_candidates == 2
    assert report.get("low_cost").label == "p2-COBYLA-O1-readout"
    assert report.get("high_quality").label == "p2-COBYLA-O1-zne"
    # exact distance tie (both sqrt(0.5)) falls back to the lower gate budget
    assert report.get("balanced").label == "p2-COBYLA-O1-readout"


# ---------------------------------------------------------------- explanations ----


def test_reasons_contain_numbers_computed_from_the_records():
    report = recommend(make_result())
    high_text = " ".join(report.get("high_quality").reasons)
    assert "0.709" in high_text
    assert "(+0.074)" in high_text  # 0.709 - 0.635, ZNE vs the same circuit unmitigated
    assert "3.0x the shots" in high_text  # 3072 / 1024
    low_text = " ".join(report.get("low_cost").reasons)
    assert "No error mitigation" in low_text
    assert "Ties on gate budget with p2-COBYLA-O1-readout" in low_text
    balanced_text = " ".join(report.get("balanced").reasons)
    assert "46% of the candidates' quality range at 0% of their cost range" in balanced_text


def test_equivalent_levels_are_reported_with_the_lowest_level_shown():
    low = recommend(make_result()).get("low_cost")
    assert low.equivalent_labels == ["p2-COBYLA-O2-none", "p2-COBYLA-O3-none"]
    assert any("higher transpiler level" in r for r in low.reasons)


def test_missing_standard_error_is_flagged_not_hidden():
    balanced = recommend(make_result()).get("balanced")  # readout-mitigated: no std error
    assert any("standard error" in c for c in balanced.caveats)
    # low (none) vs high (zne) differ by 0.074 against ~0.023 combined error: no caveat needed
    assert recommend(make_result()).get("low_cost").caveats == []


def test_indistinguishable_picks_get_a_statistical_caveat():
    specs = [
        ("A", 1, "COBYLA", 1, "none", 0.500, 0.018, 0.10, 10, 10, 20, 1024),
        ("B", 2, "COBYLA", 1, "none", 0.505, 0.018, 0.10, 22, 22, 40, 1024),
    ]
    report = recommend(make_result(specs))
    assert report.get("low_cost").label == "A" and report.get("high_quality").label == "B"
    for category in ("low_cost", "high_quality"):
        assert any("not statistically distinguishable" in c for c in report.get(category).caveats)


# ------------------------------------------------------------------ edge cases ----


def test_single_candidate_fills_all_categories_and_says_so():
    report = recommend(make_result(), RecommendationSettings(min_quality_absolute=0.7))
    assert report.num_candidates == 1
    labels = {report.get(c).label for c in ("low_cost", "balanced", "high_quality")}
    assert labels == {"p2-COBYLA-O1-zne"}
    assert set(report.get("balanced").same_choice_as) == {"low_cost", "high_quality"}
    assert any("same configuration" in m for m in report.messages)


def test_no_candidates_reports_the_limitation():
    report = recommend(make_result(), RecommendationSettings(min_quality_absolute=0.95))
    assert report.status == "no_candidates"
    assert report.recommendations == []
    assert len(report.excluded) == 10
    assert any("quality floor" in m for m in report.messages)
    assert "No Pareto-efficient trade-off" in format_report(report)


def test_empty_result():
    empty = make_result([])
    report = recommend(empty)
    assert report.status == "no_results"
    assert report.recommendations == []


def test_weights_move_the_balanced_pick():
    quality_only = recommend(make_result(), RecommendationSettings(quality_weight=1, cost_weight=0))
    assert quality_only.get("balanced").label == quality_only.get("high_quality").label
    cost_only = recommend(make_result(), RecommendationSettings(quality_weight=0, cost_weight=1))
    assert (cost_only.get("balanced").metrics["two_qubit_gate_budget"]
            == cost_only.get("low_cost").metrics["two_qubit_gate_budget"])


# -------------------------------------------------------- determinism and output ----


def test_recommendations_are_deterministic():
    assert recommend(make_result()).to_dict() == recommend(make_result()).to_dict()


def test_report_serialization_and_text(tmp_path):
    report = recommend(make_result())
    data = json.loads(report.to_json())
    assert data["status"] == "ok" and len(data["recommendations"]) == 3
    assert data["disclaimer"] == DISCLAIMER
    assert report.save(tmp_path / "r" / "report.json").is_file()
    text = format_report(report)
    for marker in ("[LOW COST]", "[BALANCED]", "[HIGH QUALITY]", "Why:", DISCLAIMER):
        assert marker in text
