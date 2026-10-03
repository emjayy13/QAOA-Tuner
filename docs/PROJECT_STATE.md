# QAOA-Tuner: project state and handoff

Purpose of this file: let a new working session (human or AI) continue the project without the
earlier conversation. Keep it updated at the end of every phase.

To resume with an AI assistant, give it: (1) this file, (2) the original project brief
(`QAOA-Tuner` spec, phases 0-12, kept by the project owner), (3) the zip or paste of any file it
needs to read. The assistant cannot see the GitHub repo (automated access is blocked) and its
sandbox has no Qiskit, so the owner verifies every change by running `pytest` locally.

## Project in one paragraph

QAOA-Tuner: "Noise- and Hardware-Aware Configuration Optimization for Quantum Graph Coloring".
A modular Qiskit framework that runs QAOA graph-coloring configurations under noise, applies
error mitigation, evaluates quality and cost, and reports Pareto-efficient configurations.
It is an engineering/experimental framework. It does NOT claim a new algorithm, a new
mitigation technique, or quantum advantage. Owner: 3rd-year B.Tech CS student targeting a
Qiskit Software Developer internship; code must be simple and explainable in an interview.
Repo: github.com/emjayy13/QAOA-Tuner (branch `main`).

## Environment (baseline, do not change without approval)

Windows, PowerShell (NOT bash), VS Code, conda base. Python 3.13.9, Qiskit 2.2.3, Qiskit Aer
0.17.2, Qiskit Algorithms 0.4.0, Qiskit Optimization 0.7.0, NetworkX 3.5, NumPy 2.3.5,
SciPy 1.16.3, PyTest 8.4.2, Matplotlib 3.10.6. `pip install -e .` works. Package lives in
`src/qaoa_tuner`. Changes are delivered as zips mirroring the repo layout and applied with
`Expand-Archive <zip> -DestinationPath . -Force`.

## Phase status

| Phase | Content | Status |
|---|---|---|
| 0-5 | Plan, graph core, QAOA, experiment engine, transpilation, noise | done, committed |
| 6 | Error mitigation (readout, ZNE) and comparison benchmark | done, tested, commit `phase-6` pending/confirm |
| 7 | Tuner: 72-point grid, Pareto dominance, tie grouping, JSON/CSV | done, 115 tests passing; commit message `phase-7: add configuration tuner with Pareto analysis` |
| 8 | Recommendation engine (low cost / balanced / high quality) | code delivered (`phase8_patch.zip`); owner must run `pytest -q` and the recommendation CLI, then commit `phase-8: add recommendation engine` |
| 9 | Streamlit dashboard | not started |
| 10 | Polish (README, logging, docs, screenshots) | not started |
| 11 | Poster experiments (fixed suite, multi-seed, figures) | not started |
| 12 | Portfolio packaging | not started |

Test count at last confirmed run: 115 passed (before phase 8).

## Immediate next steps (update after each action)

1. Owner: apply `phase8_patch.zip`, run `pytest -q` (expect the earlier 115 plus new phase 8 tests),
   run `python -m qaoa_tuner.tuner --graph c4 --colors 2` then
   `python -m qaoa_tuner.recommendation data\tuner\<file>.json`, send back the output.
2. If green: commit phase 8 and (optionally) decide on committing `data/` folders.
3. Next phase: 9 (Streamlit dashboard). Before it: unify the tuner's execution path with the phase 3
   engine (engine ignores `mitigation_method`), and decide how the dashboard loads saved results
   (`TunerResult.load_json`, `recommend`, `format_report` already exist and are UI-independent).
4. Optional before phase 11: per-scale simulator seeds for ZNE; check phase 4 `swaps_introduced`;
   multi-seed aggregation for the poster.

## Module map (`src/qaoa_tuner`)

- `problem/`: graph.py (ColoringGraph), generators.py, verifier.py (classical verifier)
- `qaoa/`: hamiltonian.py (k=2: n qubits; k>=3 one-hot: n*k qubits), ansatz.py, optimizers.py
  (COBYLA, SPSA), runner.py (QaoaRunner), decoder.py (decode_counts)
- `experiment/`: config.py (ExperimentConfig, validates mitigation_method), engine.py, metrics.py,
  result.py. NOTE: the engine still ignores `mitigation_method`.
- `compilation/`: backends.py (BackendProvider), transpiler.py, profiler.py
- `noise/`: models.py (presets, NoiseConfig, build_noise_model), benchmark.py (ideal vs noisy)
- `mitigation/`: readout.py (full-matrix <=8 qubits, tensored any n), zne.py (fold 2q gates, linear/
  polynomial extrapolation with propagated error), benchmark.py (ideal/noisy/readout/ZNE comparison)
- `tuner/`: space.py (ConfigurationSpace, TunerConfig, TunerSettings), pareto.py (dominance,
  CORE_OBJECTIVES, DEFAULT_OBJECTIVES = core + total_shots, group_equivalent), evaluator.py,
  results.py (TunerRecord, TunerResult, ParetoGroup), tuner.py, `__main__.py` (CLI)
- `recommendation/` (phase 8): engine.py (RecommendationSettings, recommend, format_report;
  pure Python, reads a TunerResult), `__main__.py` (CLI reading a saved tuner JSON).
  `TunerResult.from_dict/load_json` (tuner/results.py) load saved results.
- `cli.py`: phase 3 experiment CLI

## How to run

```powershell
pytest -q
python -m qaoa_tuner.mitigation.benchmark --graph c4 --colors 2 --p 1 --plot
python -m qaoa_tuner.tuner --graph c4 --colors 2            # writes data/tuner/*.json and *.csv
python -m qaoa_tuner.recommendation data\tuner\<file>.json  # phase 8
```

## Key design decisions (and why)

1. Tuner executes the TRANSPILED circuit under the documented noise model (not the engine's
   own compiled circuit), so transpiler level and backend topology can affect quality.
   The tuner therefore has its own execution path; unify with the engine before phase 9.
2. Parameters are trained once per (p, optimizer) on an IDEAL simulator, then evaluated under
   noise. Limitation: training under noise could pick different parameters.
3. Hardware cost = two-qubit gate budget summed over all circuits executed for an estimate
   (ZNE scales 1,3,5 = 9x; readout calibration circuits add no 2q gates). Shot overhead is a
   separate objective (`total_shots`) so mitigation is not "free".
4. Pareto = plain dominance (>= everywhere, > somewhere), optional epsilon tolerance, exact ties
   grouped; the lowest transpiler level represents a group.
5. ZNE folds only two-qubit gates, after transpilation, run without re-optimizing; odd integer
   scale factors only; only 2q-gate noise is scaled (readout and 1q noise are not).
6. Readout calibration uses 2 circuits (tensored) assuming independent per-qubit errors, valid
   for the uniform noise model used here.
7. Recommendations (phase 8): quality floor first (default 80% of best valid-coloring rate in the
   run, adjustable), then deterministic picks; reasons are built from the run's real numbers.
   No machine learning.

## Findings so far (all from owner-run experiments; ONE graph: cycle_4, k=2,
## noise `realistic_superconducting`, backend `fake_linear_5q`, 1024 shots; not general)

- Mitigation ordering for `p2-COBYLA-O1`: ZNE > readout > none in 6 of 6 seeds (42, 1-5).
  Gains over unmitigated: ZNE about +0.05 to +0.07, readout about +0.03 (valid-coloring rate).
  Unmitigated quality varies a lot by seed (0.446 to 0.635), so report means and spreads.
- p=3 reaches about 0.98 ideal quality but needs 42-45 two-qubit gates (depth 73) and falls to
  about 0.53-0.62 under noise; p=2 (22 gates) lands at about 0.63-0.71. On this noise profile
  p=3 never reaches the frontier.
- Transpiler level 0 -> 1 reduced `p2-COBYLA` from 28 to 22 two-qubit gates and raised quality
  (0.590 -> 0.635, seed 42); levels 1-3 gave identical circuits for p=2 and p=3.
- SPSA with 25 iterations looks under-trained at p=1 and p=2 (ideal quality about 0.28 and 0.60 vs
  COBYLA 0.54 and 0.89) but matches COBYLA at p=3. Do NOT conclude SPSA is a worse optimizer.
- Noise model covers every executed gate (`uncovered_gates` empty in all 72 records).

## Known issues and open items

- Phase 4 `swaps_introduced` (compilation/transpiler.py) probably overcounts: it compares transpiled
  2q gates with logical `rzz` gates, but each `rzz` becomes 2 `cx`. Unverified (source not re-read).
  The tuner avoids it and reports raw gate counts and the overhead ratio.
- ZNE circuits at different scale factors share one simulator seed, so their sampling noise may be
  correlated; use a different seed per scale before the phase 11 experiments.
- Engine ignores `mitigation_method`; tuner/mitigation benchmark implement mitigation separately.
- Frontier objective `quality_degradation` rewards circuits that never started high (e.g. low-
  quality p=1 SPSA); phase 8's quality floor handles this for recommendations.
- Single-seed results can mislead; phase 11 should aggregate several seeds.
- README is a stub (phase 10). Decide whether to commit `data/` result folders.
- Tuner CLI said "this can take a few minutes" although a 4-qubit run takes seconds (fixed in phase 8).

## Working rules (from the project brief; keep following)

- One phase at a time; after each: summarize, list files, how to run, what to verify, run tests,
  known limitations, then STOP and wait. Explain before implementing significant components.
- Never fabricate or hard-code results; all reported numbers come from executed experiments.
  Test fixtures built from real runs must be labelled as such.
- Verify Qiskit APIs against the installed version; no new dependencies without approval.
- Keep modules independent (UI never contains quantum logic); prefer simple, explainable code.
- Dashboard (phase 9): warm, light, calm palette (cream background, muted terracotta/amber,
  muted green for positive, charcoal text); NO dark/neon "AI" look.
- Commit per phase: `phase-N: <description>`.
