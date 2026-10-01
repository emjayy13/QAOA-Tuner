# QAOA-Tuner

Noise- and hardware-aware configuration analysis for QAOA graph coloring, built on Qiskit.

This is an engineering and experimental framework. It does not claim a new QAOA algorithm,
a new error-mitigation technique, or quantum advantage.

**Status:** work in progress (phases 1 to 6 implemented: graph coloring core, QAOA solver,
experiment framework, hardware-aware transpilation, noise simulation, error mitigation).
A full README is planned for the polish phase.

## Baseline environment

Python 3.13.9, Qiskit 2.2.3, Qiskit Aer 0.17.2, NumPy 2.3.5, SciPy 1.16.3, NetworkX 3.5.

## Install and test

```bash
pip install -e .
pytest -q
```

## Run a mitigation comparison

```bash
python -m qaoa_tuner.mitigation.benchmark --graph c4 --colors 2 --p 1 --plot
```

Results (config, seed, noise profile, versions, metrics, overhead) are saved as JSON under
`data/mitigation/`.
