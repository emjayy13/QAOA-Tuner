"""Run every configuration in the space, then mark the Pareto-efficient ones."""

from __future__ import annotations

import logging
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace

import qiskit
import qiskit_aer

from qaoa_tuner.tuner.evaluator import ConfigurationEvaluator
from qaoa_tuner.tuner.pareto import DEFAULT_OBJECTIVES, Objective, dominators
from qaoa_tuner.tuner.results import TunerRecord, TunerResult
from qaoa_tuner.tuner.space import ConfigurationSpace, TunerSettings

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class ConfigurationTuner:
    """Exhaustive grid evaluation + Pareto analysis (no adaptive search, by design)."""

    def __init__(
        self,
        settings: TunerSettings,
        space: ConfigurationSpace | None = None,
        objectives: Sequence[Objective] = DEFAULT_OBJECTIVES,
        tolerance: Mapping[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.space = space or ConfigurationSpace()
        self.objectives = tuple(objectives)
        self.tolerance = dict(tolerance or {})

    def run(self, progress: ProgressCallback | None = None) -> TunerResult:
        """Evaluate all configurations. ``progress(done, total, label)`` is called after each."""
        start = time.perf_counter()
        configs = self.space.generate()
        evaluator = ConfigurationEvaluator(self.settings)  # validates backend/noise early
        logger.info(
            "Tuning %d configurations: graph=%s k=%d noise=%s backend=%s shots=%d seed=%d",
            len(configs), self.settings.graph_dict["name"], self.settings.num_colors,
            self.settings.noise_profile, self.settings.backend_name,
            self.settings.shots, self.settings.seed,
        )

        records: list[TunerRecord] = []
        for i, config in enumerate(configs, start=1):
            records.append(evaluator.evaluate(config))
            if progress is not None:
                progress(i, len(configs), config.label)

        records = self._mark_pareto(records)
        return TunerResult(
            settings=self.settings.to_dict(),
            noise_profile=evaluator.noise_cfg.to_dict(),
            backend=evaluator.backend_info.to_dict(),
            objectives=[asdict(o) for o in self.objectives],
            tolerance=self.tolerance,
            training=evaluator.training_summary(),
            records=records,
            wall_time_seconds=time.perf_counter() - start,
            software={
                "python": platform.python_version(),
                "qiskit": qiskit.__version__,
                "qiskit_aer": qiskit_aer.__version__,
            },
        )

    def _mark_pareto(self, records: list[TunerRecord]) -> list[TunerRecord]:
        points = [r.to_dict() for r in records]
        marked = []
        for i, record in enumerate(records):
            doms = dominators(points, i, self.objectives, self.tolerance)
            marked.append(
                replace(
                    record,
                    is_pareto_optimal=not doms,
                    num_dominators=len(doms),
                    dominated_by=[records[j].label for j in doms[:3]],
                )
            )
        return marked
