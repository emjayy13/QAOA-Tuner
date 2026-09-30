"""Canonical graph representation for graph coloring problems."""

from __future__ import annotations

import networkx as nx

from qaoa_tuner.core.exceptions import InvalidGraphError
from qaoa_tuner.core.types import Edge, GraphDict


class ColoringGraph:
    """A validated undirected graph container designed specifically for graph coloring.

    Ensures vertices are consistently 0-indexed integers [0, N-1] with no self-loops,
    providing intuitive accessors for degree, neighbors, and topology metrics.
    """

    def __init__(self, nx_graph: nx.Graph, name: str = "custom_graph") -> None:
        """Initialize and validate a ColoringGraph from a NetworkX Graph.

        Args:
            nx_graph: Source NetworkX undirected graph.
            name: Human-readable identifier for the graph.

        Raises:
            InvalidGraphError: If the graph is empty, contains self-loops, or has invalid nodes.
        """
        if not isinstance(nx_graph, nx.Graph):
            raise InvalidGraphError(f"Expected networkx.Graph, got {type(nx_graph).__name__}")

        if nx_graph.is_directed():
            raise InvalidGraphError("Graph coloring requires an undirected graph.")

        if nx.number_of_selfloops(nx_graph) > 0:
            raise InvalidGraphError(
                "Self-loops are invalid in vertex coloring as a vertex cannot be adjacent to itself."
            )

        num_nodes = nx_graph.number_of_nodes()
        if num_nodes == 0:
            raise InvalidGraphError("Graph must contain at least 1 vertex.")

        # Ensure canonical 0..N-1 node relabeling
        nodes = sorted(list(nx_graph.nodes()))
        expected_nodes = list(range(num_nodes))

        if nodes != expected_nodes:
            mapping = {old_node: idx for idx, old_node in enumerate(nodes)}
            self._graph = nx.relabel_nodes(nx_graph, mapping)
        else:
            self._graph = nx_graph.copy()

        self._name = name

    @property
    def name(self) -> str:
        """The identifier name of the graph."""
        return self._name

    @property
    def num_nodes(self) -> int:
        """Number of vertices in the graph (|V|)."""
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        """Number of undirected edges in the graph (|E|)."""
        return self._graph.number_of_edges()

    @property
    def nodes(self) -> list[int]:
        """Sorted list of vertex indices [0, 1, ..., N-1]."""
        return list(range(self.num_nodes))

    @property
    def edges(self) -> list[Edge]:
        """Canonical list of edges with (u, v) such that u < v."""
        canon_edges: list[Edge] = []
        for u, v in self._graph.edges():
            canon_edges.append((min(u, v), max(u, v)))
        return sorted(canon_edges)

    def neighbors(self, node: int) -> list[int]:
        """Return sorted list of neighbors for the given node."""
        if node not in self._graph:
            raise InvalidGraphError(f"Node {node} does not exist in graph of size {self.num_nodes}.")
        return sorted(list(self._graph.neighbors(node)))

    def degree(self, node: int) -> int:
        """Degree of the specified node."""
        if node not in self._graph:
            raise InvalidGraphError(f"Node {node} does not exist in graph of size {self.num_nodes}.")
        return int(self._graph.degree(node))

    @property
    def max_degree(self) -> int:
        """Maximum vertex degree Delta(G) across the graph."""
        if self.num_nodes == 0:
            return 0
        return max(dict(self._graph.degree()).values())

    @property
    def is_connected(self) -> bool:
        """Check if the graph forms a single connected component."""
        return bool(nx.is_connected(self._graph))

    @property
    def density(self) -> float:
        """Graph edge density: 2|E| / (|V|(|V|-1))."""
        return float(nx.density(self._graph))

    @property
    def networkx_graph(self) -> nx.Graph:
        """Expose a copy of the underlying NetworkX graph for plotting or advanced queries."""
        return self._graph.copy()

    @classmethod
    def from_edge_list(
        cls, num_nodes: int, edges: list[tuple[int, int]], name: str = "custom_graph"
    ) -> ColoringGraph:
        """Construct a ColoringGraph directly from a node count and edge list."""
        if num_nodes < 1:
            raise InvalidGraphError(f"Number of nodes must be >= 1, got {num_nodes}.")

        g = nx.Graph()
        g.add_nodes_from(range(num_nodes))
        for u, v in edges:
            if u == v:
                raise InvalidGraphError(f"Self-loop detected on node {u}.")
            if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                raise InvalidGraphError(
                    f"Edge ({u}, {v}) refers to a node out of range [0, {num_nodes - 1}]."
                )
            g.add_edge(u, v)

        return cls(g, name=name)

    def to_dict(self) -> GraphDict:
        """Export graph to a serializable dictionary."""
        return {
            "name": self._name,
            "num_nodes": self.num_nodes,
            "edges": [[u, v] for u, v in self.edges],
        }

    @classmethod
    def from_dict(cls, data: GraphDict) -> ColoringGraph:
        """Reconstruct a ColoringGraph from a serializable dictionary."""
        edges = [(e[0], e[1]) for e in data["edges"]]
        return cls.from_edge_list(data["num_nodes"], edges, name=data.get("name", "custom_graph"))

    def __repr__(self) -> str:
        return f"ColoringGraph(name='{self.name}', nodes={self.num_nodes}, edges={self.num_edges})"
