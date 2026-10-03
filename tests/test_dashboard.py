"""Tests for Phase 9: dashboard service layer, charts, and (if Streamlit is installed) the app."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from qaoa_tuner.dashboard import charts, service
from qaoa_tuner.tuner.pareto import DEFAULT_OBJECTIVES
from qaoa_tuner.tuner.results import TunerRecord, TunerResult

# ------------------------------------------------------------------------- service ----


@pytest.mark.parametrize("kind,n", [("cycle", 5), ("path", 4), ("complete", 4), ("random", 6)])
def test_build_graph_kinds(kind, n):
    graph = service.build_graph(kind, n, density=0.6, seed=3)
    assert graph.num_nodes == n and graph.num_edges >= 1


def test_random_graph_is_reproducible():
    a = service.build_graph("random", 7, 0.5, seed=11)
    b = service.build_graph("random", 7, 0.5, seed=11)
    assert list(a.edges) == list(b.edges)


@pytest.mark.parametrize(
    "args",
    [("cycle", 2), ("path", 1), ("complete", 1), ("random", 1), ("random", 5, 0.0), ("hexagon", 4)],
)
def test_build_graph_rejects_invalid_input(args):
    with pytest.raises(ValueError):
        service.build_graph(*args)


def test_qubits_needed():
    assert service.qubits_needed(5, 2) == 5
    assert service.qubits_needed(3, 3) == 9


def test_validate_problem_messages():
    c4 = service.build_graph("cycle", 4)
    assert service.validate_problem(c4, 2) == []
    assert any("bipartite" in e for e in service.validate_problem(service.build_graph("cycle", 5), 2))
    assert any("clique" in e for e in service.validate_problem(service.build_graph("complete", 4), 3))
    assert service.validate_problem(service.build_graph("complete", 3), 3) == []  # 9 qubits is allowed
    assert any("qubits" in e for e in service.validate_problem(service.build_graph("complete", 5), 5))
    assert service.validate_problem(c4, 1)


def test_make_settings_normalizes_scale_factors():
    graph = service.build_graph("cycle", 4)
    s = service.make_settings(graph, 2, "readout_only", "fake_linear_5q", 256, 1, 5, [3, 1, 3])
    assert s.zne_scale_factors == (1, 3) and s.num_colors == 2


def test_run_ladder_returns_the_three_mitigation_modes():
    graph = service.build_graph("cycle", 4)
    settings = service.make_settings(graph, 2, "realistic_superconducting", "fake_linear_5q", 128, 3, 4, [1, 3])
    ladder = service.run_ladder(settings, 1, "COBYLA", 1)
    records = ladder.records
    assert [r.mitigation for r in records] == ["none", "readout", "zne"]
    assert len({r.ideal_valid_coloring_rate for r in records}) == 1  # same trained parameters
    assert all(0.0 <= r.valid_coloring_rate <= 1.0 for r in records)
    assert sum(ladder.counts["none"].values()) == 128
    assert sum(ladder.counts["readout"].values()) == 128  # mitigation preserves the shot total
    assert ladder.counts["zne"] is None  # ZNE gives a scalar estimate only


# ------------------------------------------------------------- bitstring decoding ----


def _bits_with(width, ones):
    bits = ["0"] * width
    for q in ones:
        bits[width - 1 - q] = "1"  # rightmost character is qubit 0
    return "".join(bits)


def test_decode_two_color_bitstrings():
    c4 = service.build_graph("cycle", 4)
    colors, violations, valid = service.decode_bitstring(c4, 2, "0101")  # v0=1, v1=0, v2=1, v3=0
    assert colors == [1, 0, 1, 0] and violations == 0 and valid
    assert service.decode_bitstring(c4, 2, "0000")[1:] == (4, False)  # every edge monochromatic
    assert service.decode_bitstring(c4, 2, "0011")[1:] == (2, False)


def test_decode_one_hot_bitstrings():
    k3 = service.build_graph("complete", 3)
    proper = _bits_with(9, [0, 4, 8])  # v0 -> color 0, v1 -> color 1, v2 -> color 2
    assert service.decode_bitstring(k3, 3, proper) == ([0, 1, 2], 0, True)
    clash = _bits_with(9, [0, 3, 7])  # v0 -> 0, v1 -> 0, v2 -> 1: one monochromatic edge
    assert service.decode_bitstring(k3, 3, clash) == ([0, 0, 1], 1, False)
    assert service.decode_bitstring(k3, 3, "0" * 9)[1:] == (3, False)  # no vertex has a color


def test_decode_rejects_wrong_length():
    with pytest.raises(ValueError):
        service.decode_bitstring(service.build_graph("cycle", 4), 2, "010")


def test_describe_outcomes_orders_by_probability():
    c4 = service.build_graph("cycle", 4)
    outcomes = service.describe_outcomes(c4, 2, {"0000": 20, "0101": 70, "1010": 10}, top=2)
    assert [o.bitstring for o in outcomes] == ["0101", "0000"]
    assert outcomes[0].probability == pytest.approx(0.7) and outcomes[0].valid
    assert not outcomes[1].valid
    assert service.describe_outcomes(c4, 2, {}) == []


def test_decoding_agrees_with_the_qaoa_decoder_for_two_colors():
    from qaoa_tuner.qaoa.decoder import decode_counts

    c4 = service.build_graph("cycle", 4)
    counts = {"0101": 60, "0000": 30, "0011": 10}
    reference = decode_counts(counts, c4, 2, "binary_2color")
    outcomes = service.describe_outcomes(c4, 2, counts, top=None)
    expected_violations = sum(o.probability * o.violations for o in outcomes)
    assert reference.expected_conflicts == pytest.approx(expected_violations)
    assert reference.valid_coloring_probability == pytest.approx(
        sum(o.probability for o in outcomes if o.valid)
    )


# --------------------------------------------------------------------- charts / export ----


def _record(label, p, opt, level, mit, valid, budget, shots, pareto):
    return TunerRecord(
        label=label, qaoa_p=p, optimizer=opt, optimization_level=level, mitigation=mit,
        valid_coloring_rate=valid, valid_rate_std_error=None if mit == "readout" else 0.015,
        expected_conflicts=1.0, ideal_valid_coloring_rate=min(1.0, valid + 0.2),
        quality_degradation=0.2, circuit_depth=20 * p, two_qubit_gates=10 * p,
        one_qubit_gates=30, logical_two_qubit_gates=4, two_qubit_overhead_ratio=2.5,
        two_qubit_gate_budget=budget, total_shots=shots, is_pareto_optimal=pareto,
        dominated_by=[] if pareto else ["x", "y"],
    )


def _records():  # plumbing data for chart/CSV tests only (not reported as results)
    out = []
    for p in (1, 2):
        for level in (0, 1):
            for mit, mult, shots in (("none", 1, 1024), ("readout", 1, 3072), ("zne", 9, 3072)):
                out.append(_record(f"p{p}-COBYLA-O{level}-{mit}", p, "COBYLA", level, mit,
                                   0.3 + 0.1 * p + 0.02 * level, 10 * p * mult, shots, p == 2 and level == 1))
    return out


def _png_ok(fig) -> bool:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    return buffer.getbuffer().nbytes > 1000


def test_every_chart_renders_with_data_and_with_none():
    records = _records()
    figures = [
        charts.fig_stage_bars(records[:3]),
        charts.fig_quality_vs_depth(records),
        charts.fig_depth_vs_p(records),
        charts.fig_ideal_vs_achieved(records),
        charts.fig_mitigation_effect(records, 1),
        charts.fig_gates_vs_quality(records),
        charts.fig_pareto(records),
        charts.fig_graph(service.build_graph("cycle", 5)),
        charts.fig_graph(service.build_graph("cycle", 4), [0, 1, None, 0]),
        charts.fig_distribution(
            service.describe_outcomes(service.build_graph("cycle", 4), 2, {"0101": 7, "0000": 3})
        ),
    ]
    assert all(_png_ok(f) for f in figures)
    for fn in (charts.fig_quality_vs_depth, charts.fig_depth_vs_p, charts.fig_ideal_vs_achieved,
               charts.fig_gates_vs_quality, charts.fig_pareto, charts.fig_stage_bars):
        assert _png_ok(fn([]))  # empty input draws a placeholder instead of failing
    assert _png_ok(charts.fig_mitigation_effect(records, 99))
    assert _png_ok(charts.fig_distribution([]))


def test_result_to_csv_layout():
    result = TunerResult(
        settings={}, noise_profile={}, backend={},
        objectives=[{"name": o.name, "maximize": o.maximize} for o in DEFAULT_OBJECTIVES],
        tolerance={}, training=[], records=_records(), wall_time_seconds=0.0, software={},
    )
    lines = service.result_to_csv(result).strip().splitlines()
    assert len(lines) == 1 + len(result.records)
    assert lines[0].startswith("label,qaoa_p,optimizer")
    assert "x;y" in service.result_to_csv(result)  # list fields joined with ';'


# --------------------------------------------------------------- app (needs streamlit) ----

APP = Path(__file__).resolve().parents[1] / "src" / "qaoa_tuner" / "dashboard" / "app.py"
PAGES = ["Overview", "Problem Setup", "Configuration", "Run", "Results", "Compare", "Recommendations"]


def _app():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception
    return at


def _go(at, page):
    at.sidebar.radio(key="page").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_every_page_renders_without_data():
    at = _app()
    for page in PAGES:
        _go(at, page)


def test_overview_start_button_navigates():
    at = _app()
    next(b for b in at.button if b.label == "Start with Problem Setup").click().run()
    assert not at.exception and at.session_state["page"] == "Problem Setup"


def test_invalid_problem_shows_a_readable_error():
    at = _app()
    _go(at, "Problem Setup")
    at.number_input(key="graph_n").set_value(5).run()  # C5 with 2 colors: odd cycle
    assert any("bipartite" in e.value for e in at.error)
    _go(at, "Run")
    assert any("Problem Setup" in i.value for i in at.info)


def test_single_run_then_grid_then_recommendations():
    at = _app()
    _go(at, "Configuration")
    at.number_input(key="cfg_shots").set_value(128).run()
    at.number_input(key="cfg_max_iter").set_value(4).run()

    _go(at, "Run")
    next(b for b in at.button if b.label == "Run selected configuration").click().run()
    assert not at.exception and "ladder" in at.session_state
    _go(at, "Results")
    assert len(at.metric) == 6

    _go(at, "Run")
    at.multiselect(key="grid_depths").set_value([1]).run()
    at.multiselect(key="grid_optimizers").set_value(["COBYLA"]).run()
    at.multiselect(key="grid_levels").set_value([1]).run()
    next(b for b in at.button if b.label == "Run full grid").click().run()
    assert not at.exception and "tuner_result" in at.session_state
    assert len(at.session_state["tuner_result"].records) == 3

    _go(at, "Compare")
    _go(at, "Recommendations")
    assert any("Low cost" in s.value for s in at.subheader)
