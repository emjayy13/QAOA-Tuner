"""Hardware-aware compilation: backend selection, transpilation, and resource profiling."""

from qaoa_tuner.compilation.backends import BackendInfo, BackendProvider
from qaoa_tuner.compilation.profiler import CircuitResourceMetrics, profile_circuit
from qaoa_tuner.compilation.transpiler import (
    TranspilationResult,
    compare_optimization_levels,
    transpile_qaoa_circuit,
)

__all__ = [
    "BackendInfo",
    "BackendProvider",
    "CircuitResourceMetrics",
    "TranspilationResult",
    "compare_optimization_levels",
    "profile_circuit",
    "transpile_qaoa_circuit",
]
