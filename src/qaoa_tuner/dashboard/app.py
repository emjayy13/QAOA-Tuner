"""QAOA-Tuner dashboard.  Run with:  streamlit run src/qaoa_tuner/dashboard/app.py

This file only builds the interface. All quantum logic is in the qaoa_tuner packages.
"""

from __future__ import annotations

import logging
import time
import traceback

import streamlit as st

from qaoa_tuner.dashboard import charts, service
from qaoa_tuner.recommendation import RecommendationSettings, recommend
from qaoa_tuner.tuner.space import ConfigurationSpace

logger = logging.getLogger("qaoa_tuner.dashboard")

PAGES = ["Overview", "Problem Setup", "Configuration", "Run", "Results", "Compare", "Recommendations"]
MITIGATION_LABELS = {
    "none": "None",
    "readout": "Readout error mitigation",
    "zne": "Zero Noise Extrapolation (ZNE)",
}
NOISE_LABELS = {
    "realistic_superconducting": "Realistic superconducting (gate, readout and relaxation errors)",
    "depolarizing_mild": "Mild depolarizing gate errors only",
    "readout_only": "Readout errors only",
}
KIND_LABELS = {"cycle": "Cycle", "path": "Path", "complete": "Complete graph", "random": "Random graph"}

DEFAULTS = {
    "page": "Overview",
    "graph_kind": "cycle", "graph_n": 4, "graph_density": 0.5, "graph_seed": 7, "num_colors": 2,
    "cfg_p": 1, "cfg_optimizer": "COBYLA", "cfg_level": 1, "cfg_mitigation": "zne",
    "cfg_noise": "realistic_superconducting", "cfg_backend": "fake_linear_5q",
    "cfg_shots": 1024, "cfg_seed": 42, "cfg_max_iter": 25, "cfg_zne_scales": [1, 3, 5],
    "grid_depths": [1, 2, 3], "grid_optimizers": ["COBYLA", "SPSA"],
    "grid_levels": [0, 1, 2, 3], "grid_mitigations": ["none", "zne", "readout"],
    "rec_fraction": 0.8, "rec_quality_weight": 0.5,
}

st.set_page_config(page_title="QAOA-Tuner", layout="wide")
for _key, _value in DEFAULTS.items():
    st.session_state.setdefault(_key, _value)
    st.session_state[_key] = st.session_state[_key]  # keeps widget values when switching pages


# ------------------------------------------------------------------------------ helpers ----
def guarded(action, label: str):
    """Run ``action``; show a readable message on failure and keep details in the log."""
    try:
        return action()
    except ValueError as exc:
        logger.warning("%s failed: %s", label, exc)
        st.error(f"{label} failed: {exc}")
    except Exception:  # noqa: BLE001 - last line of defence at the UI boundary; logged below
        logger.exception("%s failed unexpectedly", label)
        st.error(f"{label} failed unexpectedly. The technical details were logged.")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
    return None


def current_problem():
    """Build the graph from the Problem Setup controls: (graph or None, error messages)."""
    s = st.session_state
    try:
        graph = service.build_graph(s["graph_kind"], int(s["graph_n"]), float(s["graph_density"]), int(s["graph_seed"]))
    except ValueError as exc:
        return None, [str(exc)]
    return graph, service.validate_problem(graph, int(s["num_colors"]))


def current_settings(graph):
    s = st.session_state
    return service.make_settings(
        graph, int(s["num_colors"]), s["cfg_noise"], s["cfg_backend"], int(s["cfg_shots"]),
        int(s["cfg_seed"]), int(s["cfg_max_iter"]), list(s["cfg_zne_scales"]),
    )


def explain(text: str) -> None:
    st.caption(text)


# -------------------------------------------------------------------------------- pages ----
def page_overview() -> None:
    st.title("QAOA-Tuner")
    st.write(
        "QAOA-Tuner explores how algorithm settings, quantum hardware constraints, noise, and "
        "error mitigation affect quantum graph-coloring experiments."
    )
    st.write(
        "It is an engineering and experimental tool: it runs QAOA configurations in simulation, "
        "measures them, and shows which configurations are good trade-offs. It makes no claim of "
        "a new algorithm or of quantum advantage."
    )
    st.subheader("How to use it")
    st.markdown(
        "1. **Problem Setup**: choose or generate a graph and the number of colors.\n"
        "2. **Configuration**: pick QAOA depth, optimizer, transpiler level, mitigation and noise.\n"
        "3. **Run**: run one circuit (ideal, noisy and mitigated) or the full configuration grid.\n"
        "4. **Results / Compare / Recommendations**: inspect measured results and trade-offs."
    )
    st.subheader("Pipeline")
    st.code(
        "Graph -> QAOA circuit -> transpilation -> noisy simulation -> error mitigation\n"
        "      -> evaluation -> Pareto analysis -> recommendations",
        language=None,
    )
    st.subheader("Status")
    st.write(
        f"Problem and configuration set: **{'yes' if st.session_state.get('graph_ok') else 'not yet'}**; "
        f"single-circuit results: **{'available' if 'ladder' in st.session_state else 'none yet'}**; "
        f"grid results: **{'available' if 'tuner_result' in st.session_state else 'none yet'}**."
    )
    # A callback runs BEFORE the rerun, so it may change the navigation widget's value.
    st.button("Start with Problem Setup", type="primary",
              on_click=lambda: st.session_state.update({"page": "Problem Setup"}))


def page_problem() -> None:
    st.header("Problem Setup")
    left, right = st.columns([1, 1])
    with left:
        st.selectbox("Graph type", list(KIND_LABELS), format_func=KIND_LABELS.get, key="graph_kind")
        st.number_input("Vertices", min_value=2, max_value=12, step=1, key="graph_n",
                        help="Number of nodes. Qubits needed = vertices (2 colors) or vertices x colors.")
        if st.session_state["graph_kind"] == "random":
            st.slider("Edge density", 0.1, 1.0, step=0.05, key="graph_density",
                      help="Probability that each possible edge exists.")
            st.number_input("Random seed", min_value=0, step=1, key="graph_seed",
                            help="Same seed gives the same graph.")
        st.number_input("Number of colors", min_value=2, max_value=6, step=1, key="num_colors",
                        help="Colors available. Two colors work only for graphs without odd cycles.")
    graph, errors = current_problem()
    st.session_state["graph_ok"] = graph is not None and not errors
    st.session_state["graph"] = graph
    with right:
        if graph is not None:
            a, b, c, d = st.columns(4)
            a.metric("Vertices", graph.num_nodes)
            b.metric("Edges", graph.num_edges)
            c.metric("Colors", int(st.session_state["num_colors"]))
            d.metric("Qubits", service.qubits_needed(graph.num_nodes, int(st.session_state["num_colors"])))
            st.pyplot(charts.fig_graph(graph))
    for message in errors:
        st.error(message)
    with st.expander("What is graph coloring?"):
        st.write(
            "Assign one of k colors to every vertex so that no edge joins two vertices of the same "
            "color. A coloring that breaks this rule has *conflicts*. QAOA encodes the problem in "
            "qubits and searches for a low-conflict assignment."
        )


def page_configuration() -> None:
    st.header("Configuration")
    c1, c2 = st.columns(2)
    with c1:
        st.selectbox("QAOA depth (p)", [1, 2, 3], key="cfg_p",
                     help="Number of QAOA layers. More layers can fit better but add gates and noise.")
        st.selectbox("Classical optimizer", ["COBYLA", "SPSA"], key="cfg_optimizer",
                     help="COBYLA: deterministic, derivative-free. SPSA: stochastic, 2 evaluations per step, noise-tolerant.")
        st.selectbox("Transpiler optimization level", [0, 1, 2, 3], key="cfg_level",
                     help="Qiskit transpiler effort. Higher levels try to reduce gates and depth.")
        st.selectbox("Error mitigation", list(MITIGATION_LABELS), format_func=MITIGATION_LABELS.get, key="cfg_mitigation",
                     help="Readout: corrects measurement errors. ZNE: runs noise-amplified circuits and extrapolates to zero noise.")
    with c2:
        st.selectbox("Noise model", list(service.NOISE_PROFILES), format_func=NOISE_LABELS.get, key="cfg_noise",
                     help="Documented, configurable noise on an Aer simulator.")
        st.selectbox("Backend (topology and gates)", list(service.BACKENDS), key="cfg_backend",
                     help="Provides qubit connectivity and basis gates for transpilation. Noise comes from the noise model.")
        st.number_input("Shots", min_value=128, max_value=16384, step=128, key="cfg_shots",
                        help="Measurements per circuit execution.")
    with st.expander("Advanced settings"):
        st.number_input("Random seed", min_value=0, step=1, key="cfg_seed",
                        help="Seeds training, transpilation and simulation for reproducibility.")
        st.number_input("Optimizer iterations", min_value=4, max_value=200, step=1, key="cfg_max_iter",
                        help="Iteration budget for training. SPSA often needs more than COBYLA.")
        st.multiselect("ZNE noise scale factors", [1, 3, 5, 7], key="cfg_zne_scales",
                       help="Odd integers; must include 1 (the unscaled circuit).")
    if 1 not in st.session_state["cfg_zne_scales"]:
        st.error("ZNE scale factors must include 1.")
    elif len(st.session_state["cfg_zne_scales"]) < 2:
        st.error("Choose at least two ZNE scale factors.")
    else:
        st.success("Configuration is valid.")


def page_run() -> None:
    st.header("Run")
    graph, errors = current_problem()
    zne_ok = 1 in st.session_state["cfg_zne_scales"] and len(st.session_state["cfg_zne_scales"]) >= 2
    if errors or graph is None:
        for message in errors:
            st.error(message)
        st.info("Fix the problem in Problem Setup first.")
        return
    if not zne_ok:
        st.error("Fix the ZNE scale factors in Configuration first.")
        return

    s = st.session_state
    st.write(
        f"**Selected configuration:** graph `{graph.name}`, {s['num_colors']} colors, p={s['cfg_p']}, "
        f"{s['cfg_optimizer']}, transpiler level {s['cfg_level']}, "
        f"{MITIGATION_LABELS[s['cfg_mitigation']]}, noise `{s['cfg_noise']}`, backend `{s['cfg_backend']}`, "
        f"{s['cfg_shots']} shots, seed {s['cfg_seed']}."
    )
    settings = current_settings(graph)

    st.subheader("1. Run this circuit")
    explain("Trains QAOA once without noise, then evaluates the same parameters ideal, noisy, "
            "with readout mitigation and with ZNE. The run is synchronous; please wait for it to finish.")
    if st.button("Run selected configuration", type="primary"):
        start = time.perf_counter()
        with st.spinner("Training, transpiling and simulating..."):
            ladder = guarded(
                lambda: service.run_ladder(settings, int(s["cfg_p"]), s["cfg_optimizer"], int(s["cfg_level"])),
                "Run",
            )
        if ladder is not None:
            st.session_state["ladder"] = ladder.records
            st.session_state["ladder_counts"] = ladder.counts
            st.session_state["ladder_meta"] = {"graph": graph.name, "mitigation": s["cfg_mitigation"], "settings": settings.to_dict()}
            st.success(f"Finished in {time.perf_counter() - start:.1f} s. See the Results page.")

    st.subheader("2. Run the full configuration grid")
    explain("Evaluates every combination below, then marks the Pareto-efficient ones.")
    g1, g2 = st.columns(2)
    g1.multiselect("QAOA depths", [1, 2, 3], key="grid_depths")
    g1.multiselect("Optimizers", ["COBYLA", "SPSA"], key="grid_optimizers")
    g2.multiselect("Transpiler levels", [0, 1, 2, 3], key="grid_levels")
    g2.multiselect("Mitigation", list(service.MITIGATIONS), format_func=MITIGATION_LABELS.get, key="grid_mitigations")
    axes = [s["grid_depths"], s["grid_optimizers"], s["grid_levels"], s["grid_mitigations"]]
    if any(len(a) == 0 for a in axes):
        st.warning("Choose at least one value in each list.")
    else:
        total = len(axes[0]) * len(axes[1]) * len(axes[2]) * len(axes[3])
        st.write(f"{total} configurations will be evaluated.")
        if st.button("Run full grid"):
            space = ConfigurationSpace(tuple(axes[0]), tuple(axes[1]), tuple(axes[2]), tuple(axes[3]))
            bar = st.progress(0.0, text="Starting...")

            def on_progress(done: int, count: int, label: str) -> None:
                bar.progress(done / count, text=f"{done}/{count} evaluated (last: {label})")

            start = time.perf_counter()
            result = guarded(lambda: service.run_tuner(settings, space, on_progress), "Grid run")
            if result is not None:
                st.session_state["tuner_result"] = result
                st.success(f"Finished in {time.perf_counter() - start:.1f} s. See Compare and Recommendations.")


def show_solution(meta: dict, chosen, counts: dict) -> None:
    """Decoded colorings and measured distribution for the chosen mitigation mode."""
    shown = counts.get(chosen.mitigation)
    if shown is None:
        shown = counts.get("none")
        st.info("ZNE produces extrapolated metrics only, not a corrected distribution. "
                "Showing the unmitigated noisy distribution instead.")
    if not shown:
        st.info("No measurement counts are available for this run.")
        return
    graph = service.graph_from_dict(meta["settings"]["graph_dict"])
    k = int(meta["settings"]["num_colors"])
    outcomes = service.describe_outcomes(graph, k, shown, top=None)
    valid_mass = sum(o.probability for o in outcomes if o.valid)
    best = next((o for o in outcomes if o.valid), None)
    left, right = st.columns([1, 2])
    with left:
        if best is not None:
            st.pyplot(charts.fig_graph(graph, best.colors))
            st.caption(f"Most probable valid coloring (probability {best.probability:.3f}): "
                       + ", ".join(f"v{v}={c}" for v, c in enumerate(best.colors)))
        else:
            st.pyplot(charts.fig_graph(graph))
            st.warning("No valid coloring was measured in this run.")
    with right:
        st.pyplot(charts.fig_distribution(outcomes[:10]))
    st.caption(f"Total probability on valid colorings in this distribution: {valid_mass:.3f}. "
               "Violations = same-color edges plus vertices without exactly one color.")
    st.dataframe([
        {"bitstring": o.bitstring, "probability": o.probability, "valid": o.valid,
         "violations": o.violations,
         "coloring": ", ".join(f"v{v}={'-' if c is None else c}" for v, c in enumerate(o.colors))}
        for o in outcomes[:10]
    ])


def page_results() -> None:
    st.header("Results")
    ladder = st.session_state.get("ladder")
    if not ladder:
        st.info("No single-circuit result yet. Go to Run and run the selected configuration.")
        return
    meta = st.session_state["ladder_meta"]
    chosen = next((r for r in ladder if r.mitigation == meta["mitigation"]), ladder[0])
    st.write(f"Graph `{meta['graph']}`. Showing: **{MITIGATION_LABELS[chosen.mitigation]}** ({chosen.label}).")
    cols = st.columns(6)
    cols[0].metric("Valid-coloring rate", f"{chosen.valid_coloring_rate:.3f}",
                   delta=f"{-chosen.quality_degradation:+.3f} vs ideal", delta_color="off")
    cols[1].metric("Ideal (no noise)", f"{chosen.ideal_valid_coloring_rate:.3f}")
    cols[2].metric("Expected conflicts", f"{chosen.expected_conflicts:.3f}")
    cols[3].metric("Two-qubit gates", chosen.two_qubit_gates)
    cols[4].metric("Circuit depth", chosen.circuit_depth)
    cols[5].metric("Total shots", chosen.total_shots)
    tab0, tab1, tab2 = st.tabs(["Solution", "Ideal vs noisy vs mitigated", "Table"])
    with tab0:
        show_solution(meta, chosen, st.session_state.get("ladder_counts", {}))
    with tab1:
        st.pyplot(charts.fig_stage_bars(ladder))
        explain("Measured values from this run. Mitigation reduces, but does not remove, the effect of noise.")
    with tab2:
        st.dataframe([
            {"stage": MITIGATION_LABELS[r.mitigation], "valid rate": r.valid_coloring_rate,
             "std error": r.valid_rate_std_error, "expected conflicts": r.expected_conflicts,
             "two-qubit gate budget": r.two_qubit_gate_budget, "total shots": r.total_shots}
            for r in ladder
        ])


def page_compare() -> None:
    st.header("Compare configurations")
    result = st.session_state.get("tuner_result")
    if result is None:
        st.info("No grid results yet. Go to Run and run the full grid.")
        return
    records = result.records
    n_front = len(result.pareto_records())
    st.write(f"{len(records)} configurations measured; {n_front} are Pareto-efficient "
             f"({len(result.pareto_groups())} distinct trade-offs).")
    levels = sorted({r.optimization_level for r in records})
    tabs = st.tabs(["Quality vs depth", "Circuit depth", "Ideal vs noisy", "Mitigation", "Gates vs quality", "Pareto frontier", "Data"])
    with tabs[0]:
        st.pyplot(charts.fig_quality_vs_depth(records))
    with tabs[1]:
        st.pyplot(charts.fig_depth_vs_p(records))
    with tabs[2]:
        st.pyplot(charts.fig_ideal_vs_achieved(records))
    with tabs[3]:
        if st.session_state.get("compare_level") not in levels:
            st.session_state.pop("compare_level", None)  # stale after a re-run with other levels
        level = st.selectbox("Transpiler level", levels, index=min(1, len(levels) - 1), key="compare_level")
        st.pyplot(charts.fig_mitigation_effect(records, level))
    with tabs[4]:
        st.pyplot(charts.fig_gates_vs_quality(records))
    with tabs[5]:
        st.pyplot(charts.fig_pareto(records))
        explain("A configuration is on the Pareto frontier if no other configuration is at least as good "
                "on every objective (quality, gate budget, degradation, shots) and better on one. "
                "Frontier points are different trade-offs, not a single winner.")
    with tabs[6]:
        st.dataframe([r.to_dict() for r in records])
        st.download_button("Download results (CSV)", service.result_to_csv(result), "qaoa_tuner_results.csv", "text/csv")
        st.download_button("Download results (JSON)", result.to_json(), "qaoa_tuner_results.json", "application/json")


def page_recommendations() -> None:
    st.header("Recommendations")
    result = st.session_state.get("tuner_result")
    if result is None:
        st.info("No grid results yet. Go to Run and run the full grid.")
        return
    with st.expander("Selection rules", expanded=False):
        st.slider("Quality floor (fraction of best valid rate)", 0.0, 1.0, step=0.05, key="rec_fraction",
                  help="Pareto trade-offs below this are excluded before picking.")
        st.slider("Balanced pick: weight on quality (rest on cost)", 0.0, 1.0, step=0.05, key="rec_quality_weight")
    wq = float(st.session_state["rec_quality_weight"])
    try:
        settings = RecommendationSettings(float(st.session_state["rec_fraction"]), None, wq, 1.0 - wq)
    except ValueError as exc:
        st.error(str(exc))
        return
    report = recommend(result, settings)
    if report.status != "ok":
        for message in report.messages:
            st.warning(message)
        st.caption(report.disclaimer)
        return
    st.write(f"{report.num_frontier_trade_offs} Pareto trade-offs; {report.num_candidates} reach the quality "
             f"floor of {report.quality_floor:.3f} (best in run {report.best_valid_coloring_rate:.3f}).")
    titles = {"low_cost": "Low cost", "balanced": "Balanced", "high_quality": "High quality"}
    for col, rec in zip(st.columns(3), report.recommendations, strict=True):
        with col, st.container(border=True):
            st.subheader(titles[rec.category])
            c = rec.config
            st.write(f"p={c['qaoa_p']}, {c['optimizer']}, transpiler level {c['optimization_level']}, "
                     f"{MITIGATION_LABELS[c['mitigation']]}")
            m = rec.metrics
            st.metric("Valid-coloring rate", f"{m['valid_coloring_rate']:.3f}")
            st.write(f"Two-qubit gates {m['two_qubit_gates']}; gate budget {m['two_qubit_gate_budget']}; "
                     f"depth {m['circuit_depth']}; shots {m['total_shots']}")
            st.markdown("**Why**")
            st.markdown("\n".join(f"- {r}" for r in rec.reasons))
            if rec.same_choice_as:
                st.caption(f"Also the {', '.join(rec.same_choice_as)} pick.")
            for note in rec.caveats:
                st.warning(note)
    for message in report.messages:
        st.info(message)
    if report.excluded:
        with st.expander(f"Excluded by the quality floor ({len(report.excluded)})"):
            for e in report.excluded:
                st.write(f"- {e.label}: {e.reason}")
    st.caption(report.disclaimer)


# ------------------------------------------------------------------------------ routing ----
st.sidebar.title("QAOA-Tuner")
st.sidebar.radio("Navigate", PAGES, key="page")
{
    "Overview": page_overview,
    "Problem Setup": page_problem,
    "Configuration": page_configuration,
    "Run": page_run,
    "Results": page_results,
    "Compare": page_compare,
    "Recommendations": page_recommendations,
}[st.session_state["page"]]()
