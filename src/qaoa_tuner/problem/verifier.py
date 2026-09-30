"""Deterministic classical verification and exact solving for graph coloring."""

from __future__ import annotations

from qaoa_tuner.core.exceptions import InvalidGraphError
from qaoa_tuner.core.types import (
    ClassicalColoringResult,
    Coloring,
    ColoringValidationResult,
    Edge,
)
from qaoa_tuner.problem.graph import ColoringGraph


def validate_coloring(
    graph: ColoringGraph,
    coloring: Coloring,
    num_colors: int | None = None,
) -> ColoringValidationResult:
    """Rigorously validate a proposed vertex coloring for a given graph.

    A coloring is strictly valid if and only if:
    1. Every vertex v in [0, N-1] is assigned a color.
    2. If num_colors is specified, all assigned colors c satisfy 0 <= c < num_colors.
    3. For every edge (u, v) in E, coloring[u] != coloring[v].

    Args:
        graph: The target ColoringGraph.
        coloring: Dict mapping node indices to color indices.
        num_colors: Optional constraint on the allowed number of colors.

    Returns:
        ColoringValidationResult with conflict details and validation status.
    """
    uncolored_nodes: list[int] = []
    invalid_color_nodes: list[int] = []

    for node in graph.nodes:
        if node not in coloring:
            uncolored_nodes.append(node)
        else:
            color = coloring[node]
            if not isinstance(color, int) or color < 0:
                invalid_color_nodes.append(node)
            elif num_colors is not None and color >= num_colors:
                invalid_color_nodes.append(node)

    conflicting_edges: list[Edge] = []
    # Only evaluate edge conflicts if both endpoints are colored
    for u, v in graph.edges:
        if u in coloring and v in coloring:
            if coloring[u] == coloring[v]:
                conflicting_edges.append((u, v))

    used_colors = {c for node, c in coloring.items() if node in graph.nodes and isinstance(c, int)}

    is_valid = (
        len(uncolored_nodes) == 0
        and len(invalid_color_nodes) == 0
        and len(conflicting_edges) == 0
    )

    return ColoringValidationResult(
        is_valid=is_valid,
        num_conflicts=len(conflicting_edges),
        conflicting_edges=conflicting_edges,
        num_colors_used=len(used_colors),
        uncolored_nodes=uncolored_nodes,
        invalid_color_nodes=invalid_color_nodes,
    )


def solve_classical_exact(
    graph: ColoringGraph,
    num_colors: int,
) -> ClassicalColoringResult:
    """Find a valid k-coloring using deterministic constraint-satisfaction backtracking.

    Uses degree-based variable ordering to prune the search space efficiently.
    This serves as the definitive classical ground truth against which quantum results
    must be compared.

    Args:
        graph: The graph to color.
        num_colors: Number of colors available (k).

    Returns:
        ClassicalColoringResult containing a valid coloring or unsatisfiable status.
    """
    if num_colors < 1:
        return ClassicalColoringResult(
            is_satisfiable=False,
            num_colors=num_colors,
            coloring=None,
            validation=None,
            nodes_explored=0,
        )

    # Edge case: graphs with no edges need only 1 color
    if graph.num_edges == 0:
        trivial_coloring = {node: 0 for node in graph.nodes}
        validation = validate_coloring(graph, trivial_coloring, num_colors=num_colors)
        return ClassicalColoringResult(
            is_satisfiable=True,
            num_colors=num_colors,
            coloring=trivial_coloring,
            validation=validation,
            nodes_explored=1,
        )

    # Order nodes by descending degree (Most Constrained Variable heuristic)
    ordered_nodes = sorted(graph.nodes, key=lambda n: graph.degree(n), reverse=True)
    assignment: dict[int, int] = {}
    nodes_explored = 0

    def is_safe(node: int, color: int) -> bool:
        for neighbor in graph.neighbors(node):
            if neighbor in assignment and assignment[neighbor] == color:
                return False
        return True

    def backtrack(index: int) -> bool:
        nonlocal nodes_explored
        nodes_explored += 1

        if index == len(ordered_nodes):
            return True

        node = ordered_nodes[index]
        for color in range(num_colors):
            if is_safe(node, color):
                assignment[node] = color
                if backtrack(index + 1):
                    return True
                del assignment[node]

        return False

    satisfiable = backtrack(0)

    if satisfiable:
        # Sort keys back to standard 0..N-1 order
        result_coloring = {n: assignment[n] for n in graph.nodes}
        validation = validate_coloring(graph, result_coloring, num_colors=num_colors)
        return ClassicalColoringResult(
            is_satisfiable=True,
            num_colors=num_colors,
            coloring=result_coloring,
            validation=validation,
            nodes_explored=nodes_explored,
        )

    return ClassicalColoringResult(
        is_satisfiable=False,
        num_colors=num_colors,
        coloring=None,
        validation=None,
        nodes_explored=nodes_explored,
    )


def find_chromatic_number(graph: ColoringGraph) -> int:
    """Compute the exact chromatic number chi(G) of the graph.

    chi(G) is the minimum number of colors required to color G such that no two
    adjacent vertices share the same color.
    """
    if graph.num_edges == 0:
        return 1

    # An upper bound is Delta(G) + 1 by Brooks' Theorem
    upper_bound = graph.max_degree + 1
    for k in range(1, upper_bound + 1):
        res = solve_classical_exact(graph, k)
        if res.is_satisfiable:
            return k

    return upper_bound


def find_all_valid_colorings(graph: ColoringGraph, num_colors: int) -> list[Coloring]:
    """Exhaustively find all valid k-colorings for small graphs.

    Essential for QAOA benchmark verification: allows exact calculation of
    quantum state overlap with the degenerate ground-state manifold.
    """
    if num_colors < 1:
        return []

    if graph.num_nodes > 12:
        raise InvalidGraphError(
            f"find_all_valid_colorings is restricted to small graphs (<= 12 nodes), got {graph.num_nodes}."
        )

    all_valid: list[Coloring] = []
    assignment: dict[int, int] = {}
    ordered_nodes = sorted(graph.nodes)

    def is_safe(node: int, color: int) -> bool:
        for neighbor in graph.neighbors(node):
            if neighbor in assignment and assignment[neighbor] == color:
                return False
        return True

    def backtrack(index: int) -> None:
        if index == len(ordered_nodes):
            all_valid.append(assignment.copy())
            return

        node = ordered_nodes[index]
        for color in range(num_colors):
            if is_safe(node, color):
                assignment[node] = color
                backtrack(index + 1)
                del assignment[node]

    backtrack(0)
    return all_valid
