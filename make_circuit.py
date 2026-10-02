from qaoa_tuner.problem.graph import ColoringGraph
from qaoa_tuner.qaoa.hamiltonian import build_cost_hamiltonian
from qaoa_tuner.qaoa.ansatz import build_qaoa_circuit
import matplotlib.pyplot as plt
from pathlib import Path


# Cycle graph C4
graph = ColoringGraph.from_edge_list(
    num_nodes=4,
    edges=[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
    ],
    name="cycle_4",
)

# 2-coloring Hamiltonian
cost_hamiltonian = build_cost_hamiltonian(
    graph,
    num_colors=2,
)

# QAOA with p=1
circuit, gammas, betas = build_qaoa_circuit(
    cost_hamiltonian,
    p=1,
    measure=True,
)

print(circuit)

# Draw circuit
fig = circuit.draw(
    output="mpl",
    fold=-1,
)

# Save
output_dir = Path("poster_assets")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / "qaoa_circuit_c4_p1.png"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"\nSaved circuit to: {output_file}")