"""Tests for Phase 7: configuration space, Pareto dominance, and the tuner.

Pure-Python tests come first (instant). The tuner smoke tests use one tiny grid
(depth 1, COBYLA, 3 iterations, 256 shots) on a 4-qubit problem.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

from qaoa_tuner.tuner.pareto import (
    CORE_OBJECTIVES,
    DEFAULT_OBJECTIVES,
    Objective,
    dominates,
    dominators,
    group_equivalent,
    pareto_front_indices,
)
from qaoa_tuner.tuner.space import (
    ConfigurationSpace,
    TunerConfig,
    TunerSettings,
    generate_configurations,
)

# ------------------------------------------------------------------ search space ----


def test_default_space_is_the_documented_72_point_grid():
    configs = generate_configurations()
    assert len(configs) == 3 * 2 * 4 * 3
    assert len({c.label for c in configs}) == 72


def test_generation_order_is_deterministic():
    first = [c.label for c in generate_configurations()]
    second = [c.label for c in generate_configurations()]
    assert first == second
    assert first[0] == "p1-COBYLA-O0-none"


def test_custom_space_and_normalization():
    space = ConfigurationSpace(depths=(2,), optimizers=("spsa",), optimization_levels=(1, 3),
                               mitigations=("ZNE",))
    configs = space.generate()
    assert [c.label for c in configs] == ["p2-SPSA-O1-zne", "p2-SPSA-O3-zne"]
    assert space.size == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"qaoa_p": 0, "optimizer": "COBYLA", "optimization_level": 1, "mitigation": "none"},
        {"qaoa_p": 1, "optimizer": "ADAM", "optimization_level": 1, "mitigation": "none"},
        {"qaoa_p": 1, "optimizer": "COBYLA", "optimization_level": 4, "mitigation": "none"},
        {"qaoa_p": 1, "optimizer": "COBYLA", "optimization_level": 1, "mitigation": "magic"},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TunerConfig(**kwargs)


def test_space_rejects_empty_axis_and_duplicates():
    with pytest.raises(ValueError):
        ConfigurationSpace(depths=()).generate()
    with pytest.raises(ValueError):
        ConfigurationSpace(depths=(1, 1)).generate()


def test_settings_validation():
    graph = {"name": "g", "num_nodes": 2, "edges": [[0, 1]]}
    with pytest.raises(ValueError):
        TunerSettings(graph_dict=graph, num_colors=1)
    with pytest.raises(ValueError):
        TunerSettings(graph_dict=graph, shots=0)
    with pytest.raises(ValueError):
        TunerSettings(graph_dict=graph, zne_scale_factors=(3, 5))
    assert TunerSettings(graph_dict=graph).to_dict()["zne_scale_factors"] == [1, 3, 5]


# ------------------------------------------------------------------------ Pareto ----

OBJ = (Objective("quality", True), Objective("cost", False))


def test_dominance_basic():
    good = {"quality": 0.8, "cost": 10}
    bad = {"quality": 0.6, "cost": 20}
    assert dominates(good, bad, OBJ)
    assert not dominates(bad, good, OBJ)


def test_tradeoff_points_do_not_dominate_each_other():
    high_quality = {"quality": 0.9, "cost": 40}
    low_cost = {"quality": 0.6, "cost": 10}
    assert not dominates(high_quality, low_cost, OBJ)
    assert not dominates(low_cost, high_quality, OBJ)


def test_equal_points_do_not_dominate_each_other():
    a = {"quality": 0.7, "cost": 12}
    assert not dominates(a, dict(a), OBJ)
    assert pareto_front_indices([a, dict(a)], OBJ) == [0, 1]


def test_better_on_one_equal_on_other_dominates():
    assert dominates({"quality": 0.7, "cost": 10}, {"quality": 0.7, "cost": 11}, OBJ)
    assert dominates({"quality": 0.8, "cost": 10}, {"quality": 0.7, "cost": 10}, OBJ)


def test_front_on_hand_computed_example():
    points = [
        {"quality": 0.9, "cost": 40},  # 0: best quality                      -> frontier
        {"quality": 0.6, "cost": 10},  # 1: cheapest                          -> frontier
        {"quality": 0.8, "cost": 20},  # 2: middle                            -> frontier
        {"quality": 0.7, "cost": 30},  # 3: dominated by 2 (worse on both)
        {"quality": 0.5, "cost": 15},  # 4: dominated by 1 only (point 2 costs more: 20 > 15)
    ]
    assert pareto_front_indices(points, OBJ) == [0, 1, 2]
    assert dominators(points, 3, OBJ) == [2]
    assert dominators(points, 4, OBJ) == [1]
    assert dominators(points, 0, OBJ) == []


def test_core_objectives_directions():
    a = {"valid_coloring_rate": 0.6, "two_qubit_gate_budget": 8, "quality_degradation": 0.05}
    b = {"valid_coloring_rate": 0.5, "two_qubit_gate_budget": 9, "quality_degradation": 0.06}
    assert dominates(a, b, CORE_OBJECTIVES)
    # Lower degradation but higher cost: incomparable.
    c = {"valid_coloring_rate": 0.6, "two_qubit_gate_budget": 20, "quality_degradation": 0.01}
    assert not dominates(a, c, CORE_OBJECTIVES) and not dominates(c, a, CORE_OBJECTIVES)


def test_default_objectives_add_shot_overhead_to_the_core_three():
    assert DEFAULT_OBJECTIVES[: len(CORE_OBJECTIVES)] == CORE_OBJECTIVES
    shots = DEFAULT_OBJECTIVES[-1]
    assert shots.name == "total_shots" and shots.maximize is False


def test_shot_overhead_stops_free_mitigation_from_dominating_the_baseline():
    base = {"valid_coloring_rate": 0.63, "two_qubit_gate_budget": 22, "quality_degradation": 0.25,
            "total_shots": 1024}
    readout = {"valid_coloring_rate": 0.67, "two_qubit_gate_budget": 22, "quality_degradation": 0.22,
               "total_shots": 3072}
    assert dominates(readout, base, CORE_OBJECTIVES)  # looks free without the shot objective
    assert not dominates(readout, base, DEFAULT_OBJECTIVES)  # costs 3x the shots
    assert pareto_front_indices([base, readout], DEFAULT_OBJECTIVES) == [0, 1]


def test_group_equivalent_collapses_exact_ties_only():
    pts = [
        {"quality": 0.7, "cost": 22},  # 0
        {"quality": 0.9, "cost": 40},  # 1
        {"quality": 0.7, "cost": 22},  # 2: identical to 0
        {"quality": 0.7, "cost": 23},  # 3: differs in cost
    ]
    assert group_equivalent(pts, OBJ) == [[0, 2], [1], [3]]
    assert group_equivalent([], OBJ) == []


def test_tolerance_treats_small_differences_as_ties():
    a = {"quality": 0.700, "cost": 10}
    b = {"quality": 0.695, "cost": 12}
    assert dominates(a, b, OBJ)  # strict: a wins on both
    # With a 0.01 quality tolerance the quality gap is a tie, but a is still cheaper.
    assert dominates(a, b, OBJ, tolerance={"quality": 0.01})
    # Now b is cheaper by less than the cost tolerance and slightly lower quality: all ties.
    c = {"quality": 0.700, "cost": 10.5}
    assert not dominates(a, c, OBJ, tolerance={"cost": 1.0})


def test_pareto_input_validation():
    assert pareto_front_indices([], OBJ) == []
    with pytest.raises(KeyError):
        dominates({"quality": 1.0}, {"quality": 0.5, "cost": 1}, OBJ)
    with pytest.raises(ValueError):
        dominates({"quality": float("nan"), "cost": 1}, {"quality": 0.5, "cost": 1}, OBJ)


# ----------------------------------------------------------------- tuner smoke ----


@pytest.fixture(scope="module")
def tiny_run():
    """One small tuning run shared by the smoke tests: returns (result, progress_calls)."""
    from qaoa_tuner.problem.generators import create_cycle_graph
    from qaoa_tuner.tuner.tuner import ConfigurationTuner

    settings = TunerSettings(
        graph_dict=create_cycle_graph(4).to_dict(), num_colors=2,
        noise_profile="realistic_superconducting", backend_name="fake_linear_5q",
        shots=256, seed=3, max_iter=4, zne_scale_factors=(1, 3),
    )
    space = ConfigurationSpace(depths=(1,), optimizers=("COBYLA",), optimization_levels=(0, 1),
                               mitigations=("none", "readout", "zne"))
    calls = []
    result = ConfigurationTuner(settings, space).run(lambda d, t, label: calls.append((d, t)))
    return result, calls


@pytest.fixture(scope="module")
def tiny_result(tiny_run):
    return tiny_run[0]


def test_tuner_runs_every_configuration_and_reports_progress(tiny_run):
    tiny_result, calls = tiny_run
    assert len(tiny_result.records) == 6
    assert calls[-1] == (6, 6)
    assert [d for d, _ in calls] == [1, 2, 3, 4, 5, 6]
    assert [r.label for r in tiny_result.records][0] == "p1-COBYLA-O0-none"


def test_record_metrics_are_consistent(tiny_result):
    for r in tiny_result.records:
        assert 0.0 <= r.valid_coloring_rate <= 1.0
        assert r.expected_conflicts >= 0.0
        assert r.quality_degradation == pytest.approx(r.ideal_valid_coloring_rate - r.valid_coloring_rate)
        assert r.two_qubit_gates > 0 and r.circuit_depth > 0
        assert r.uncovered_gates == [], "noise model must cover every executed gate"


def test_mitigation_cost_accounting(tiny_result):
    by = {(r.optimization_level, r.mitigation): r for r in tiny_result.records}
    for level in (0, 1):
        none, readout, zne = by[(level, "none")], by[(level, "readout")], by[(level, "zne")]
        assert none.two_qubit_gate_budget == none.two_qubit_gates
        assert readout.two_qubit_gate_budget == none.two_qubit_gate_budget  # calibration: no 2q gates
        assert zne.two_qubit_gate_budget == (1 + 3) * none.two_qubit_gate_budget  # scales 1 and 3
        assert none.total_shots == 256
        assert readout.total_shots == 256 + 2 * 256
        assert zne.total_shots == 2 * 256
        # Same circuit => identical hardware metrics regardless of mitigation
        assert none.circuit_depth == readout.circuit_depth == zne.circuit_depth


def test_pareto_flags_match_independent_recomputation(tiny_result):
    points = [r.to_dict() for r in tiny_result.records]
    expected = set(pareto_front_indices(points, DEFAULT_OBJECTIVES))
    flagged = {i for i, r in enumerate(tiny_result.records) if r.is_pareto_optimal}
    assert flagged == expected
    assert flagged, "a non-empty set always has at least one non-dominated point"
    for r in tiny_result.records:
        assert (r.num_dominators == 0) == r.is_pareto_optimal
        assert len(r.dominated_by) <= min(3, r.num_dominators)


def test_baseline_is_never_dominated_by_a_mitigated_configuration(tiny_result):
    """Mitigated runs use strictly more shots (256 vs 512/768), so with the shot objective
    a 'none' record can only be dominated by another 'none' record."""
    labels = {r.label: r for r in tiny_result.records}
    for r in tiny_result.records:
        if r.mitigation != "none" or r.num_dominators == 0:
            continue
        points = [x.to_dict() for x in tiny_result.records]
        i = tiny_result.records.index(r)
        for j in dominators(points, i, DEFAULT_OBJECTIVES):
            assert tiny_result.records[j].mitigation == "none", labels


def test_pareto_groups_partition_the_frontier(tiny_result):
    groups = tiny_result.pareto_groups()
    by_label = {r.label: r for r in tiny_result.records}
    members = [g.representative.label for g in groups]
    members += [lab for g in groups for lab in g.equivalent_labels]
    assert sorted(members) == sorted(r.label for r in tiny_result.pareto_records())

    names = [o["name"] for o in tiny_result.objectives]
    for g in groups:
        rep = g.representative
        vector = [getattr(rep, n) for n in names]
        for lab in g.equivalent_labels:
            other = by_label[lab]
            assert [getattr(other, n) for n in names] == vector  # exact tie
            assert other.optimization_level >= rep.optimization_level  # lowest level represents


def test_result_serialization_and_provenance(tiny_result, tmp_path):
    data = json.loads(tiny_result.to_json())
    assert data["num_configurations"] == 6
    assert data["settings"]["seed"] == 3
    assert data["noise_profile"]["name"] == "realistic_superconducting"
    assert {"python", "qiskit", "qiskit_aer"} <= set(data["software"])
    assert data["training"][0]["trained_on"] == "ideal simulator"
    assert data["objectives"][0]["name"] == "valid_coloring_rate"
    assert data["objectives"][-1]["name"] == "total_shots"
    assert "pareto_groups" in data and data["pareto_groups"]

    csv_path = tiny_result.save_csv(tmp_path / "t.csv")
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6 and rows[0]["label"] == "p1-COBYLA-O0-none"
    assert tiny_result.save_json(tmp_path / "t.json").is_file()


def test_tuner_is_reproducible_for_a_fixed_seed():
    from qaoa_tuner.problem.generators import create_cycle_graph
    from qaoa_tuner.tuner.tuner import ConfigurationTuner

    settings = TunerSettings(
        graph_dict=create_cycle_graph(4).to_dict(), shots=128, seed=9, max_iter=4,
    )
    space = ConfigurationSpace(depths=(1,), optimizers=("COBYLA",), optimization_levels=(1,),
                               mitigations=("none",))
    a = ConfigurationTuner(settings, space).run().records[0]
    b = ConfigurationTuner(settings, space).run().records[0]
    assert replace(a) == replace(b)


# ------------------------------------------------------------- input validation ----


def test_backend_too_small_gives_readable_error():
    from qaoa_tuner.problem.generators import create_complete_graph
    from qaoa_tuner.tuner.evaluator import ConfigurationEvaluator

    settings = TunerSettings(
        graph_dict=create_complete_graph(3).to_dict(), num_colors=3, backend_name="fake_linear_5q"
    )  # one-hot: 9 qubits > 5
    with pytest.raises(ValueError, match="qubits"):
        ConfigurationEvaluator(settings)


def test_ideal_noise_profile_rejected():
    from qaoa_tuner.problem.generators import create_cycle_graph
    from qaoa_tuner.tuner.evaluator import ConfigurationEvaluator

    settings = TunerSettings(graph_dict=create_cycle_graph(4).to_dict(), noise_profile="ideal")
    with pytest.raises(ValueError, match="no noise"):
        ConfigurationEvaluator(settings)


# ------------------------------------------------- phase 8: saved results + recommendations ----


def test_result_json_round_trip(tiny_result, tmp_path):
    from qaoa_tuner.tuner.results import TunerResult

    loaded = TunerResult.load_json(tiny_result.save_json(tmp_path / "r.json"))
    assert loaded.records == tiny_result.records
    assert loaded.settings["seed"] == tiny_result.settings["seed"]
    assert [g.representative.label for g in loaded.pareto_groups()] == [
        g.representative.label for g in tiny_result.pareto_groups()
    ]


def test_recommendations_from_a_real_tuning_run(tiny_result):
    from qaoa_tuner.recommendation import recommend

    report = recommend(tiny_result)
    assert report.status == "ok"
    frontier = {r.label for r in tiny_result.pareto_records()}
    best = max(r.valid_coloring_rate for r in tiny_result.records)
    assert report.best_valid_coloring_rate == best
    for rec in report.recommendations:
        assert rec.label in frontier
        assert rec.metrics["valid_coloring_rate"] >= report.quality_floor
    low, high = report.get("low_cost"), report.get("high_quality")
    assert high.metrics["valid_coloring_rate"] == best
    assert low.metrics["two_qubit_gate_budget"] <= high.metrics["two_qubit_gate_budget"]


def test_recommendation_cli_end_to_end_and_missing_file(tiny_result, tmp_path, monkeypatch, capsys):
    import sys

    from qaoa_tuner.recommendation.__main__ import main

    path = tiny_result.save_json(tmp_path / "r.json")
    report_path = tmp_path / "rep.json"
    monkeypatch.setattr(sys, "argv", ["prog", str(path), "--save", str(report_path)])
    assert main() == 0
    assert "[HIGH QUALITY]" in capsys.readouterr().out
    assert report_path.is_file()

    monkeypatch.setattr(sys, "argv", ["prog", str(tmp_path / "missing.json")])
    assert main() == 2
    assert "File not found" in capsys.readouterr().err
