"""Domain-specific exceptions for QAOA-Tuner."""


class GraphColoringError(Exception):
    """Base exception for all graph coloring errors in QAOA-Tuner."""

    pass


class InvalidGraphError(GraphColoringError):
    """Raised when a graph structure is invalid or malformed."""

    pass


class UnsatisfiableColoringError(GraphColoringError):
    """Raised when a graph cannot be colored with the requested number of colors."""

    pass
