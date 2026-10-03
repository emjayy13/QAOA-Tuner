"""Matplotlib figures for the dashboard (warm palette; no Streamlit, no pyplot global state)."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
from matplotlib.figure import Figure

from qaoa_tuner.tuner.results import TunerRecord

CREAM, PAPER = "#FAF6F0", "#FFFFFF"
TERRACOTTA, AMBER, GREEN, BROWN, GRAY, INK = "#C4623F", "#D9A441", "#5B8A5B", "#8A5A3C", "#9A9088", "#2B2B2B"
MITIGATION_COLORS = {"none": TERRACOTTA, "readout": AMBER, "zne": BROWN}
MITIGATION_NAMES = {"none": "No mitigation", "readout": "Readout mitigation", "zne": "ZNE"}


def _axes(title: str, xlabel: str, ylabel: str, size=(6.4, 4.0)):
    fig = Figure(figsize=size, facecolor=CREAM)
    ax = fig.subplots()
    ax.set_facecolor(PAPER)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, linestyle="--", alpha=0.3, color=GRAY)
    ax.tick_params(colors=INK)
    for spine in ax.spines.values():
        spine.set_color("#D4CCC5")
    return fig, ax


def _empty(fig, ax, message="No data for this chart."):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color=GRAY)
    fig.tight_layout()
    return fig


COLOR_PALETTE = [TERRACOTTA, AMBER, GREEN, "#6B8FA3", "#B07AA1", BROWN]


def fig_graph(graph, coloring: list | None = None) -> Figure:
    """Draw the problem graph; with ``coloring`` (color index or None per vertex) fill the nodes."""
    fig = Figure(figsize=(4.2, 3.6), facecolor=CREAM)
    ax = fig.subplots()
    ax.set_facecolor(CREAM)
    g = nx.Graph()
    g.add_nodes_from(range(graph.num_nodes))
    g.add_edges_from(graph.edges)
    pos = nx.circular_layout(g) if graph.num_nodes <= 8 else nx.spring_layout(g, seed=1)
    if coloring is None:
        node_colors = [AMBER] * graph.num_nodes
    else:
        node_colors = [GRAY if c is None else COLOR_PALETTE[c % len(COLOR_PALETTE)] for c in coloring]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=GRAY, width=1.6)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, edgecolors=INK, node_size=520)
    nx.draw_networkx_labels(g, pos, ax=ax, font_color=INK, font_size=10)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def fig_distribution(outcomes: list) -> Figure:
    """Probability of the most likely measured outcomes; green = valid coloring."""
    fig, ax = _axes("Most probable outcomes  (green = valid coloring)", "Measured bitstring", "Probability")
    if not outcomes:
        return _empty(fig, ax)
    ax.bar([o.bitstring for o in outcomes], [o.probability for o in outcomes],
           color=[GREEN if o.valid else TERRACOTTA for o in outcomes], edgecolor=INK, alpha=0.9)
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    fig.tight_layout()
    return fig


def fig_stage_bars(ladder: list[TunerRecord]) -> Figure:
    """Ideal -> noisy -> readout -> ZNE valid-coloring rate for one circuit."""
    fig, ax = _axes("Ideal vs noisy vs mitigated", "", "Valid-coloring probability")
    by = {r.mitigation: r for r in ladder}
    if "none" not in by:
        return _empty(fig, ax)
    labels = ["Ideal (no noise)", "Noisy"]
    values = [by["none"].ideal_valid_coloring_rate, by["none"].valid_coloring_rate]
    errors = [0.0, by["none"].valid_rate_std_error or 0.0]
    colors = [GREEN, TERRACOTTA]
    for key, name in (("readout", "Noisy + readout"), ("zne", "Noisy + ZNE")):
        if key in by:
            labels.append(name)
            values.append(by[key].valid_coloring_rate)
            errors.append(by[key].valid_rate_std_error or 0.0)
            colors.append(MITIGATION_COLORS[key])
    ax.bar(labels, values, yerr=errors, capsize=5, color=colors, edgecolor=INK, alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ideal vs noisy vs mitigated  (error bars: shot noise)", fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    return fig


def fig_quality_vs_depth(records: list[TunerRecord]) -> Figure:
    """Best valid-coloring rate per QAOA depth, per mitigation (best over optimizer and level)."""
    fig, ax = _axes("Solution quality vs QAOA depth", "QAOA depth p", "Valid-coloring probability")
    best: dict[str, dict[int, float]] = defaultdict(dict)
    ideal: dict[int, float] = {}
    for r in records:
        cur = best[r.mitigation].get(r.qaoa_p, -1.0)
        best[r.mitigation][r.qaoa_p] = max(cur, r.valid_coloring_rate)
        ideal[r.qaoa_p] = max(ideal.get(r.qaoa_p, -1.0), r.ideal_valid_coloring_rate)
    if not ideal:
        return _empty(fig, ax)
    ps = sorted(ideal)
    ax.plot(ps, [ideal[p] for p in ps], "--o", color=GREEN, label="Ideal (no noise)")
    for m, series in best.items():
        xs = sorted(series)
        ax.plot(xs, [series[x] for x in xs], "-o", color=MITIGATION_COLORS[m], label=MITIGATION_NAMES[m])
    ax.set_xticks(ps)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "best over optimizer and transpiler level", fontsize=7, color=GRAY)
    fig.tight_layout()
    return fig


def fig_depth_vs_p(records: list[TunerRecord]) -> Figure:
    """Transpiled circuit depth vs QAOA depth, one line per transpiler level."""
    fig, ax = _axes("Circuit depth vs QAOA depth", "QAOA depth p", "Transpiled circuit depth")
    series: dict[int, dict[int, int]] = defaultdict(dict)
    for r in records:
        cur = series[r.optimization_level].get(r.qaoa_p)
        series[r.optimization_level][r.qaoa_p] = r.circuit_depth if cur is None else min(cur, r.circuit_depth)
    if not series:
        return _empty(fig, ax)
    palette = [GRAY, AMBER, TERRACOTTA, BROWN]
    for level in sorted(series):
        xs = sorted(series[level])
        ax.plot(xs, [series[level][x] for x in xs], "-o", color=palette[level % 4], label=f"level {level}")
    ax.set_xticks(sorted({r.qaoa_p for r in records}))
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "lowest depth over optimizers", fontsize=7, color=GRAY)
    fig.tight_layout()
    return fig


def fig_ideal_vs_achieved(records: list[TunerRecord]) -> Figure:
    """Each configuration: ideal rate (x) vs rate achieved under noise (y); diagonal = no loss."""
    fig, ax = _axes("Ideal vs noisy quality", "Ideal valid-coloring probability", "Achieved under noise")
    if not records:
        return _empty(fig, ax)
    ax.plot([0, 1], [0, 1], color=GRAY, linestyle=":", label="no degradation")
    for m in ("none", "readout", "zne"):
        pts = [r for r in records if r.mitigation == m]
        if pts:
            ax.scatter([r.ideal_valid_coloring_rate for r in pts], [r.valid_coloring_rate for r in pts],
                       color=MITIGATION_COLORS[m], label=MITIGATION_NAMES[m], alpha=0.8, edgecolor=INK, s=36)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_mitigation_effect(records: list[TunerRecord], level: int) -> Figure:
    """Grouped bars none / readout / ZNE for each (p, optimizer) at one transpiler level."""
    fig, ax = _axes(f"Mitigated vs unmitigated (transpiler level {level})", "", "Valid-coloring probability")
    groups = sorted({(r.qaoa_p, r.optimizer) for r in records if r.optimization_level == level})
    if not groups:
        return _empty(fig, ax)
    lookup = {(r.qaoa_p, r.optimizer, r.mitigation): r for r in records if r.optimization_level == level}
    width = 0.26
    for i, m in enumerate(("none", "readout", "zne")):
        xs, ys = [], []
        for j, (p, opt) in enumerate(groups):
            rec = lookup.get((p, opt, m))
            if rec is not None:
                xs.append(j + (i - 1) * width)
                ys.append(rec.valid_coloring_rate)
        ax.bar(xs, ys, width=width, color=MITIGATION_COLORS[m], edgecolor=INK, label=MITIGATION_NAMES[m])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"p={p}\n{opt}" for p, opt in groups], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_gates_vs_quality(records: list[TunerRecord]) -> Figure:
    """Two-qubit gates in the executed circuit vs achieved quality."""
    fig, ax = _axes("Two-qubit gates vs quality", "Two-qubit gates in the circuit", "Valid-coloring probability")
    if not records:
        return _empty(fig, ax)
    for m in ("none", "readout", "zne"):
        pts = [r for r in records if r.mitigation == m]
        if pts:
            ax.scatter([r.two_qubit_gates for r in pts], [r.valid_coloring_rate for r in pts],
                       color=MITIGATION_COLORS[m], label=MITIGATION_NAMES[m], alpha=0.8, edgecolor=INK, s=36)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def fig_pareto(records: list[TunerRecord]) -> Figure:
    """Gate budget vs quality; Pareto-efficient configurations highlighted."""
    fig, ax = _axes("Pareto frontier (quality vs two-qubit gate budget)",
                    "Two-qubit gate budget (all circuits executed)", "Valid-coloring probability")
    if not records:
        return _empty(fig, ax)
    others = [r for r in records if not r.is_pareto_optimal]
    front = sorted((r for r in records if r.is_pareto_optimal), key=lambda r: r.two_qubit_gate_budget)
    ax.scatter([r.two_qubit_gate_budget for r in others], [r.valid_coloring_rate for r in others],
               color=GRAY, alpha=0.45, s=28, label="dominated")
    ax.scatter([r.two_qubit_gate_budget for r in front], [r.valid_coloring_rate for r in front],
               color=GREEN, edgecolor=INK, s=60, label="Pareto-efficient", zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "frontier is over 4 objectives; this is a 2-D projection", fontsize=7, color=GRAY)
    fig.tight_layout()
    return fig
