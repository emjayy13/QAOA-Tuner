"""Standard and random graph generators for graph coloring experiments."""

from __future__ import annotations

import networkx as nx

from qaoa_tuner.core.exceptions import InvalidGraphError
from qaoa_tuner.problem.graph import ColoringGraph


def create_cycle_graph(n: int) -> ColoringGraph:
    """Create a cycle graph C_n with n vertices.

    Properties:
        - 2-colorable if n is even.
        - 3-colorable if n is odd (odd cycle).
    """
    if n < 3:
        raise InvalidGraphError(f"Cycle graph requires at least 3 vertices, got {n}.")
    g = nx.cycle_graph(n)
    return ColoringGraph(g, name=f"cycle_{n}")


def create_complete_graph(n: int) -> ColoringGraph:
    """Create a complete graph K_n with n vertices.

    Properties:
        - Chromatic number chi(K_n) = n.
        - Requires exactly n colors; impossible with < n colors.
    """
    if n < 1:
        raise InvalidGraphError(f"Complete graph requires at least 1 vertex, got {n}.")
    g = nx.complete_graph(n)
    return ColoringGraph(g, name=f"complete_{n}")


def create_path_graph(n: int) -> ColoringGraph:
    """Create a path graph P_n with n vertices.

    Properties:
        - Bipartite (2-colorable) for all n >= 2.
    """
    if n < 1:
        raise InvalidGraphError(f"Path graph requires at least 1 vertex, got {n}.")
    g = nx.path_graph(n)
    return ColoringGraph(g, name=f"path_{n}")


def create_star_graph(n: int) -> ColoringGraph:
    """Create a star graph S_n with 1 center vertex and n-1 leaf vertices.

    Properties:
        - Total vertices: n.
        - Bipartite (2-colorable) for all n >= 2.
    """
    if n < 2:
        raise InvalidGraphError(f"Star graph requires at least 2 vertices, got {n}.")
    g = nx.star_graph(n - 1)
    return ColoringGraph(g, name=f"star_{n}")


def create_bipartite_graph(
    n1: int, n2: int, p: float = 0.5, seed: int | None = None
) -> ColoringGraph:
    """Create a random bipartite graph with partitions of size n1 and n2.

    Properties:
        - Always 2-colorable (chromatic number <= 2).
    """
    if n1 < 1 or n2 < 1:
        raise InvalidGraphError(f"Bipartite partitions must be >= 1, got {n1} and {n2}.")
    if not (0.0 <= p <= 1.0):
        raise InvalidGraphError(f"Edge probability p must be in [0, 1], got {p}.")

    g = nx.bipartite.random_graph(n1, n2, p, seed=seed)
    return ColoringGraph(g, name=f"bipartite_{n1}_{n2}_p{int(p*100)}")


def create_random_erdos_renyi(
    num_nodes: int, p: float, seed: int | None = None
) -> ColoringGraph:
    """Create an Erdos-Renyi random graph G(n, p).

    Args:
        num_nodes: Number of vertices.
        p: Probability of edge creation between any two vertices.
        seed: Random seed for deterministic reproducibility.
    """
    if num_nodes < 1:
        raise InvalidGraphError(f"Number of nodes must be >= 1, got {num_nodes}.")
    if not (0.0 <= p <= 1.0):
        raise InvalidGraphError(f"Edge probability p must be in [0, 1], got {p}.")

    g = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
    return ColoringGraph(g, name=f"er_{num_nodes}_p{int(p*100)}")
