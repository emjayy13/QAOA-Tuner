"""Hardware backend abstractions, topology configurations, and programmatic inspection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator


@dataclass(frozen=True)
class BackendInfo:
    """Programmatically queried metadata about a quantum backend.

    Attributes:
        name: Name of the backend.
        num_qubits: Total number of physical qubits.
        basis_gates: Native gate instruction set supported by the hardware.
        coupling_map: Physical qubit connectivity graph as directed edge pairs.
        is_simulator: True if this backend is a classical simulation.
        has_noise_model: True if this backend models physical decoherence/errors.
    """

    name: str
    num_qubits: int
    basis_gates: list[str]
    coupling_map: list[list[int]] | None
    is_simulator: bool
    has_noise_model: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to a serializable dictionary."""
        return asdict(self)


class BackendProvider:
    """Factory and registry for acquiring and inspecting quantum compilation targets."""

    @staticmethod
    def get_backend(
        name: str = "aer_simulator_ideal",
        num_qubits: int = 5,
        seed: int = 42,
    ) -> Any:
        """Acquire a quantum backend by name with programmatic configuration.

        Supported Names:
            - "aer_simulator_ideal" / "ideal": Ideal unconstrained AerSimulator.
            - "fake_generic_5q": 5-qubit GenericBackendV2 with realistic calibrated noise.
            - "fake_generic_7q": 7-qubit GenericBackendV2.
            - "fake_generic_12q": 12-qubit GenericBackendV2.
            - "fake_linear_5q": 5-qubit backend with constrained 1D line connectivity (0-1-2-3-4).

        Args:
            name: Backend identifier string.
            num_qubits: Target qubit count for dynamic generic backends.
            seed: Random seed for reproducible calibration.

        Returns:
            A Qiskit Backend instance (AerSimulator or GenericBackendV2).
        """
        clean_name = name.lower()

        if clean_name in ["aer_simulator_ideal", "ideal", "aer"]:
            return AerSimulator(seed_simulator=seed)

        elif clean_name in ["fake_generic_5q", "generic_5q"]:
            return GenericBackendV2(num_qubits=5, seed=seed)

        elif clean_name in ["fake_generic_7q", "generic_7q"]:
            return GenericBackendV2(num_qubits=7, seed=seed)

        elif clean_name in ["fake_generic_12q", "generic_12q"]:
            return GenericBackendV2(num_qubits=12, seed=seed)

        elif clean_name in ["fake_linear_5q", "linear_5q"]:
            # 1D line connectivity: forces SWAP routing for non-adjacent qubits
            line_coupling = [
                [0, 1], [1, 0],
                [1, 2], [2, 1],
                [2, 3], [3, 2],
                [3, 4], [4, 3],
            ]
            return GenericBackendV2(num_qubits=5, coupling_map=line_coupling, seed=seed)

        else:
            # Flexible dynamic fallback
            try:
                return GenericBackendV2(num_qubits=max(num_qubits, 5), seed=seed)
            except Exception:
                return AerSimulator(seed_simulator=seed)

    @classmethod
    def query_backend_info(cls, backend: Any) -> BackendInfo:
        """Inspect and extract hardware properties from a backend programmatically.

        Does not hard-code any values. Queries Qiskit BackendV2 / Target APIs directly.
        """
        name = getattr(backend, "name", str(type(backend).__name__))
        num_qubits = getattr(backend, "num_qubits", 0)

        # Inspect target for basis gates and coupling map
        basis_gates: list[str] = []
        coupling_map: list[list[int]] | None = None

        if hasattr(backend, "target") and backend.target is not None:
            basis_gates = sorted(list(backend.target.operation_names))
            cmap = backend.target.build_coupling_map()
            if cmap is not None:
                coupling_map = [list(edge) for edge in cmap.get_edges()]
        elif hasattr(backend, "configuration"):
            config = backend.configuration()
            basis_gates = getattr(config, "basis_gates", [])
            coupling_map = getattr(config, "coupling_map", None)

        is_simulator = isinstance(backend, AerSimulator) or "aer" in name.lower()
        has_noise_model = isinstance(backend, GenericBackendV2) or hasattr(backend, "noise_model")

        return BackendInfo(
            name=name,
            num_qubits=num_qubits,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            is_simulator=is_simulator,
            has_noise_model=has_noise_model,
        )
