"""Comprehensive unit tests for ColoringGraph, generators, and classical verification."""

import networkx as nx
import pytest

from qaoa_tuner.core.exceptions import InvalidGraphError
from qaoa_tuner.problem.generators import (
    create_bipartite_graph,
    create_complete_graph,
    create_cycle_graph,
    create_path_graph,
    create_random_erdos_renyi,
    create_star_graph,
)
from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.problem.verifier import (
    find_all_valid_colorings,
    find_chromatic_number,
    solve_classical_exact,
    validate_coloring,
)


class TestColoringGraph:
    """Test suite for ColoringGraph validation and methods."""

    def test_valid_creation_and_properties(self):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        cg = ColoringGraph(g, name="triangle")

        assert cg.name == "triangle"
        assert cg.num_nodes == 3
        assert cg.num_edges == 3
        assert cg.nodes == [0, 1, 2]
        assert cg.edges == [(0, 1), (0, 2), (1, 2)]
        assert cg.neighbors(1) == [0, 2]
        assert cg.degree(1) == 2
        assert cg.max_degree == 2
        assert cg.is_connected is True
        assert cg.density == 1.0

    def test_relabeling_arbitrary_nodes(self):
        g = nx.Graph()
        g.add_edges_from([("a", "b"), ("b", "c")])
        cg = ColoringGraph(g, name="relabelled")
        assert cg.num_nodes == 3
        assert cg.nodes == [0, 1, 2]
        assert len(cg.edges) == 2

    def test_rejection_empty_graph(self):
        g = nx.Graph()
        with pytest.raises(InvalidGraphError, match="at least 1 vertex"):
            ColoringGraph(g)

    def test_rejection_self_loops(self):
        g = nx.Graph()
        g.add_edge(0, 0)
        with pytest.raises(InvalidGraphError, match="Self-loops"):
            ColoringGraph(g)

    def test_rejection_directed_graph(self):
        dg = nx.DiGraph()
        dg.add_edge(0, 1)
        with pytest.raises(InvalidGraphError, match="undirected"):
            ColoringGraph(dg)

    def test_serialization_roundtrip(self):
        cg1 = create_cycle_graph(4)
        data = cg1.to_dict()
        cg2 = ColoringGraph.from_dict(data)

        assert cg1.num_nodes == cg2.num_nodes
        assert cg1.edges == cg2.edges
        assert cg1.name == cg2.name


class TestGenerators:
    """Test suite for graph generators."""

    def test_cycle_graph(self):
        c4 = create_cycle_graph(4)
        assert c4.num_nodes == 4
        assert c4.num_edges == 4

        with pytest.raises(InvalidGraphError):
            create_cycle_graph(2)

    def test_complete_graph(self):
        k4 = create_complete_graph(4)
        assert k4.num_nodes == 4
        assert k4.num_edges == 6  # 4*3/2

    def test_path_graph(self):
        p5 = create_path_graph(5)
        assert p5.num_nodes == 5
        assert p5.num_edges == 4

    def test_star_graph(self):
        s5 = create_star_graph(5)
        assert s5.num_nodes == 5
        assert s5.num_edges == 4
        assert s5.max_degree == 4

    def test_bipartite_graph(self):
        bp = create_bipartite_graph(3, 3, p=0.8, seed=42)
        assert bp.num_nodes == 6
        # Bipartite graphs must be at most 2-colorable
        assert find_chromatic_number(bp) <= 2

    def test_erdos_renyi_reproducibility(self):
        er1 = create_random_erdos_renyi(6, 0.5, seed=123)
        er2 = create_random_erdos_renyi(6, 0.5, seed=123)
        assert er1.edges == er2.edges


class TestVerifier:
    """Test suite for validate_coloring and solve_classical_exact."""

    def test_validate_coloring_valid_2color(self, cycle_4_graph):
        # C4 is bipartite: nodes 0,2 color 0; nodes 1,3 color 1
        coloring = {0: 0, 1: 1, 2: 0, 3: 1}
        res = validate_coloring(cycle_4_graph, coloring, num_colors=2)
        assert res.is_valid is True
        assert res.num_conflicts == 0
        assert len(res.conflicting_edges) == 0
        assert res.num_colors_used == 2

    def test_validate_coloring_with_conflicts(self, cycle_4_graph):
        # Conflict between node 0 and node 1
        coloring = {0: 0, 1: 0, 2: 1, 3: 1}
        res = validate_coloring(cycle_4_graph, coloring, num_colors=2)
        assert res.is_valid is False
        # Edges in C4: (0,1), (1,2), (2,3), (0,3)
        # Here: coloring[0]==coloring[1]==0 (edge 0-1 conflict)
        # coloring[2]==coloring[3]==1 (edge 2-3 conflict)
        assert res.num_conflicts == 2
        assert (0, 1) in res.conflicting_edges
        assert (2, 3) in res.conflicting_edges

    def test_validate_coloring_uncolored_node(self, cycle_4_graph):
        coloring = {0: 0, 1: 1, 2: 0}  # node 3 is missing
        res = validate_coloring(cycle_4_graph, coloring, num_colors=2)
        assert res.is_valid is False
        assert res.uncolored_nodes == [3]

    def test_validate_coloring_invalid_color_bounds(self, cycle_4_graph):
        coloring = {0: 0, 1: 1, 2: 0, 3: 5}  # color 5 >= num_colors=2
        res = validate_coloring(cycle_4_graph, coloring, num_colors=2)
        assert res.is_valid is False
        assert res.invalid_color_nodes == [3]

    def test_classical_solver_triangle(self, triangle_graph):
        # Triangle requires 3 colors
        res_2 = solve_classical_exact(triangle_graph, num_colors=2)
        assert res_2.is_satisfiable is False
        assert res_2.coloring is None

        res_3 = solve_classical_exact(triangle_graph, num_colors=3)
        assert res_3.is_satisfiable is True
        assert res_3.validation is not None
        assert res_3.validation.is_valid is True
        assert res_3.validation.num_conflicts == 0

    def test_classical_solver_odd_cycle(self, cycle_5_graph):
        # C5 is an odd cycle: cannot be colored with 2 colors
        res_2 = solve_classical_exact(cycle_5_graph, num_colors=2)
        assert res_2.is_satisfiable is False

        res_3 = solve_classical_exact(cycle_5_graph, num_colors=3)
        assert res_3.is_satisfiable is True
        assert res_3.validation.is_valid is True

    def test_find_chromatic_number(self, triangle_graph, cycle_4_graph, cycle_5_graph, path_4_graph):
        assert find_chromatic_number(triangle_graph) == 3
        assert find_chromatic_number(cycle_4_graph) == 2
        assert find_chromatic_number(cycle_5_graph) == 3
        assert find_chromatic_number(path_4_graph) == 2

    def test_find_all_valid_colorings(self, triangle_graph, cycle_4_graph):
        # K3 with 3 colors: 3! = 6 permutations
        k3_colorings = find_all_valid_colorings(triangle_graph, num_colors=3)
        assert len(k3_colorings) == 6
        for c in k3_colorings:
            assert validate_coloring(triangle_graph, c, num_colors=3).is_valid

        # C4 with 2 colors: exactly 2 valid colorings (alternating 0,1,0,1 and 1,0,1,0)
        c4_colorings = find_all_valid_colorings(cycle_4_graph, num_colors=2)
        assert len(c4_colorings) == 2
        for c in c4_colorings:
            assert validate_coloring(cycle_4_graph, c, num_colors=2).is_valid
