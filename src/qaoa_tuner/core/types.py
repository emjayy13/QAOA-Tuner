"""Domain types and structured result containers for QAOA-Tuner."""

from dataclasses import dataclass, field
from typing import TypedDict

Edge = tuple[int, int]
Coloring = dict[int, int]


class GraphDict(TypedDict):
    """Serializable dictionary representation of a graph."""

    name: str
    num_nodes: int
    edges: list[list[int]]


@dataclass(frozen=True)
class ColoringValidationResult:
    """Detailed validation analysis of a graph coloring assignment.

    Attributes:
        is_valid: True if and only if all nodes are colored validly with zero edge conflicts.
        num_conflicts: Total count of monochromatic edges (edges connecting same-color nodes).
        conflicting_edges: Exact list of edges that violate the coloring condition.
        num_colors_used: Total count of unique colors used in this assignment.
        uncolored_nodes: Nodes missing a color assignment.
        invalid_color_nodes: Nodes assigned a color outside [0, num_colors - 1].
    """

    is_valid: bool
    num_conflicts: int
    conflicting_edges: list[Edge] = field(default_factory=list)
    num_colors_used: int = 0
    uncolored_nodes: list[int] = field(default_factory=list)
    invalid_color_nodes: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ClassicalColoringResult:
    """Result of an exact classical graph coloring solver.

    Attributes:
        is_satisfiable: True if a valid coloring was found for the given num_colors.
        num_colors: The number of colors requested.
        coloring: Mapping of node to color (if satisfiable, else None).
        validation: Detailed validation result of the produced coloring.
        nodes_explored: Number of recursive search states explored by the backtracking algorithm.
    """

    is_satisfiable: bool
    num_colors: int
    coloring: Coloring | None
    validation: ColoringValidationResult | None
    nodes_explored: int = 0
