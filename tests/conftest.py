"""PyTest configuration and shared fixtures for QAOA-Tuner."""

import pytest

from qaoa_tuner.problem.generators import (
    create_complete_graph,
    create_cycle_graph,
    create_path_graph,
)
from qaoa_tuner.problem.graph import ColoringGraph


@pytest.fixture
def triangle_graph() -> ColoringGraph:
    """Complete graph K3 (3 nodes, 3 edges, chi=3)."""
    return create_complete_graph(3)


@pytest.fixture
def cycle_4_graph() -> ColoringGraph:
    """Cycle graph C4 (4 nodes, 4 edges, bipartite, chi=2)."""
    return create_cycle_graph(4)


@pytest.fixture
def cycle_5_graph() -> ColoringGraph:
    """Odd cycle graph C5 (5 nodes, 5 edges, chi=3)."""
    return create_cycle_graph(5)


@pytest.fixture
def path_4_graph() -> ColoringGraph:
    """Path graph P4 (4 nodes, 3 edges, bipartite, chi=2)."""
    return create_path_graph(4)
