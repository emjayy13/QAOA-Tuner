# QAOA-Tuner

**Noise- and Hardware-Aware Configuration Optimization for Quantum Graph Coloring**

QAOA-Tuner is a modular Qiskit framework that runs QAOA graph-coloring experiments under
different algorithm, compilation, noise and error-mitigation settings, measures quality and
cost, and reports the Pareto-efficient configurations through an interactive dashboard.

It is an **engineering and experimental framework**. It does not claim a new QAOA algorithm,
a new error-mitigation technique, a universally optimal configuration, or quantum advantage.
Everything is simulated (Qiskit Aer); no real hardware was used.

![Problem setup](docs/images/dashboard_problem.png)

## Motivation and question

> Given a graph-coloring problem and a target backend, can we automatically identify QAOA
> configurations that give a favorable trade-off between solution quality, circuit complexity
> and noise robustness?

QAOA settings (depth, optimizer), compilation (transpiler level, backend topology), noise and
mitigation all interact. A deeper circuit may be better in theory and worse on noisy hardware.
This project makes those trade-offs measurable and visible instead of reporting a single "best".

## Architecture

```
Graph coloring input
        |
Problem formulation (cost Hamiltonian, one-hot / binary encoding)
        |
QAOA circuit  ->  Configuration grid (depth, optimizer, transpiler level, mitigation)
        |
Qiskit transpilation for a backend  ->  Noisy Aer simulation
        |
Error mitigation (readout, ZNE)  ->  Evaluation metrics
        |
Pareto analysis  ->  Recommendations (low cost / balanced / high quality)
        |
Streamlit dashboard
```

The quantum logic is usable without the dashboard; the dashboard only calls the packages below.

## Installation

Tested with Python 3.13.9, Qiskit 2.2.3, Qiskit Aer 0.17.2, NumPy 2.3.5, SciPy 1.16.3,
NetworkX 3.5, Matplotlib 3.10.6. `pyproject.toml` bounds Qiskit to `>=2.2.3,<2.3` and Aer to
`>=0.17.2,<0.18`.

```bash
git clone https://github.com/emjayy13/QAOA-Tuner.git
cd QAOA-Tuner
pip install -e ".[dashboard]"     # drop [dashboard] if you do not need the web app
pytest -q
```

## Quick start

```bash
# 1. Run the full 72-configuration grid on a 4-vertex cycle with 2 colors
python -m qaoa_tuner.tuner --graph c4 --colors 2

# 2. Ask for recommendations from the saved result
python -m qaoa_tuner.recommendation data/tuner/cycle_4_k2_realistic_superconducting_fake_linear_5q_seed42.json

# 3. Ideal vs noisy vs mitigated comparison for one circuit
python -m qaoa_tuner.mitigation.benchmark --graph c4 --colors 2 --p 1 --plot

# 4. Dashboard
streamlit run src/qaoa_tuner/dashboard/app.py
```

Dashboard pages: Overview, Problem Setup, Configuration, Run, Results, Compare, Recommendations.

![Pareto frontier](docs/images/dashboard_pareto.png)
![Recommendations](docs/images/dashboard_recommendations.png)

## Metrics

| Metric | Meaning |
|---|---|
| Valid-coloring rate | Probability mass on bitstrings that decode to a proper coloring |
| Expected conflicts | Expected number of constraint violations over the distribution |
| Ideal / achieved / degradation | Same trained parameters without noise vs under noise; degradation = ideal - achieved |
| Circuit depth, 1q / 2q gates | Of the circuit actually executed, after transpilation |
| Two-qubit gate budget | Two-qubit gates summed over every circuit run for one estimate (ZNE at scales 1, 3, 5 costs 9x) |
| Total shots | All shots run, including readout calibration |

All results are measured by simulation. Each tuner JSON records the settings, seed, noise
profile, backend, software versions and every metric.

## Noise models

Presets live in `src/qaoa_tuner/noise/models.py` with their documented, configurable values:
`depolarizing_mild` (gate errors), `readout_only`, and `realistic_superconducting` (gate,
readout and thermal-relaxation errors). Noise is applied through an Aer `NoiseModel`; the backend
only supplies topology and basis gates. The tuner checks that the noise model covers every gate in
the executed circuit and records any gate that it does not.

## Error mitigation

Existing techniques, implemented here in a transparent way:

- **Readout mitigation**: calibrate (per-qubit 2x2 matrices from 2 circuits, or a full matrix for
  at most 8 qubits), measure, correct with the inverse and project back onto valid probabilities.
- **Zero-noise extrapolation (ZNE)**: fold two-qubit gates (G -> G G^dagger G) after transpilation,
  run at noise scales 1, 3, 5, fit linear (or quadratic) and extrapolate to zero noise. The
  reported error is the propagated shot noise only.

Neither removes errors completely. ZNE scales two-qubit gate noise only; readout and single-qubit
noise are untouched, and both techniques amplify statistical noise.

## Pareto analysis and recommendations

A configuration dominates another if it is at least as good on every objective and strictly better
on one. Objectives: valid-coloring rate (max), two-qubit gate budget (min), quality degradation
(min), total shots (min). Configurations identical on every objective (typically the same circuit at
a higher transpiler level) are grouped.

Recommendations are deterministic (no machine learning): Pareto trade-offs below a quality floor
(default 80% of the best valid-coloring rate in the run) are excluded, then **low cost** is the lowest
gate budget, **high quality** the highest valid rate, **balanced** the closest to the ideal corner
after rescaling. Every recommendation lists reasons computed from the run's numbers and flags
differences that are within shot noise.

## Example results

All numbers below were produced by this repository's code on **one problem**: 4-vertex cycle,
2 colors, `realistic_superconducting` noise, `fake_linear_5q` topology, 1024 shots. They are not
general conclusions.

Seed 42, 72 configurations: 19 are Pareto-efficient, forming 10 distinct trade-offs. Recommendations
(quality floor 0.567 = 80% of the best 0.709):

| Pick | Configuration | Valid rate | 2q gate budget | Shots |
|---|---|---|---|---|
| Low cost | p=2, COBYLA, level 1, no mitigation | 0.635 | 22 | 1024 |
| Balanced | p=2, COBYLA, level 1, readout mitigation | 0.669 | 22 | 3072 |
| High quality | p=2, COBYLA, level 1, ZNE | 0.709 | 198 | 3072 |

Mitigation across seeds (`p=2, COBYLA, level 1`, valid-coloring rate):

| Seed | none | readout | ZNE |
|---|---|---|---|
| 42 | 0.635 | 0.669 | 0.709 |
| 1 | 0.613 | 0.651 | 0.687 |
| 2 | 0.563 | 0.590 | 0.621 |
| 3 | 0.446 | 0.472 | 0.493 |
| 4 | 0.580 | 0.617 | 0.649 |
| 5 | 0.512 | 0.538 | 0.574 |

Observations (same single problem): the order ZNE > readout > none held in all six runs; ZNE gained
roughly +0.05 to +0.07 and readout roughly +0.03; baseline quality varied strongly with the seed.
p=3 reaches about 0.98 ideal quality but needs 42 to 45 two-qubit gates (depth 73) and drops to
about 0.53 to 0.62 under noise, so it never reached the frontier. Transpiler level 0 vs 1 changed
`p=2, COBYLA` from 28 to 22 two-qubit gates and quality from 0.590 to 0.635 (seed 42).

## Limitations

- Simulation only; small graphs (at most 12 qubits in the dashboard). No real-hardware validation.
- Parameters are trained once on an ideal simulator and then evaluated under noise; training under
  noise could choose different parameters.
- SPSA is under-trained with the default 25 iterations at p=1 and p=2, so the COBYLA/SPSA
  comparison is not a fair optimizer ranking.
- ZNE folds only two-qubit gates, uses odd integer scale factors, and its circuits share one simulator
  seed. Readout calibration assumes independent, uniform per-qubit errors.
- One graph and one noise profile were studied so far; single-seed differences smaller than about
  two standard errors are not evidence.
- Valid-rate standard errors are binomial shot noise only (no model/bias error).

## Future work

Multi-seed and multi-graph aggregation, noise-aware training, dynamical decoupling, per-qubit
calibrated noise, uploaded graphs, larger instances.

## Reproducibility

Every CLI takes `--seed`; the same seed gives the same training, transpilation and simulation.
Saved JSON files contain the full configuration and software versions. Example multi-seed run:

```bash
for s in 1 2 3 4 5; do python -m qaoa_tuner.tuner --seed $s --output-dir data/tuner_seed$s; done
```

## Project structure

```
src/qaoa_tuner/
  problem/         graph model, generators, classical verifier
  qaoa/            Hamiltonian, ansatz, optimizers (COBYLA, SPSA), runner, decoder
  experiment/      config, engine, metrics, result (phase 3 experiment framework)
  compilation/     backends, transpilation, circuit profiling
  noise/           noise models, ideal-vs-noisy benchmark
  mitigation/      readout, ZNE, mitigation benchmark
  tuner/           configuration space, evaluator, Pareto analysis, results (JSON/CSV)
  recommendation/  deterministic low-cost / balanced / high-quality picks
  dashboard/       Streamlit app, service layer, charts
tests/             pytest suite
docs/              PROJECT_STATE.md (project status and design decisions)
```

## Testing

```bash
pytest -q
```

Dashboard tests that drive the app headlessly are skipped when Streamlit is not installed.

## License

MIT (see `LICENSE`).
