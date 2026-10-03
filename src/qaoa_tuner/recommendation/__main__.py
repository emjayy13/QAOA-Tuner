"""Command line: python -m qaoa_tuner.recommendation <tuner_result.json>"""

from __future__ import annotations

import argparse
import sys

from qaoa_tuner.recommendation.engine import RecommendationSettings, format_report, recommend
from qaoa_tuner.tuner.results import TunerResult


def main() -> int:
    parser = argparse.ArgumentParser(description="Recommend configurations from a tuner result.")
    parser.add_argument("input", help="JSON file written by `python -m qaoa_tuner.tuner`.")
    parser.add_argument("--min-quality-fraction", type=float, default=0.8,
                        help="Quality floor as a fraction of the best valid rate in the run.")
    parser.add_argument("--min-quality", type=float, default=None,
                        help="Absolute quality floor (valid-coloring rate in [0, 1]).")
    parser.add_argument("--quality-weight", type=float, default=0.5)
    parser.add_argument("--cost-weight", type=float, default=0.5)
    parser.add_argument("--save", default=None, help="Also write the report as JSON to this path.")
    args = parser.parse_args()

    try:
        result = TunerResult.load_json(args.input)
        settings = RecommendationSettings(
            min_quality_fraction=args.min_quality_fraction,
            min_quality_absolute=args.min_quality,
            quality_weight=args.quality_weight,
            cost_weight=args.cost_weight,
        )
    except FileNotFoundError:
        print(f"File not found: {args.input}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as exc:
        print(f"Could not use this input: {exc}", file=sys.stderr)
        return 2

    report = recommend(result, settings)
    print(format_report(report))
    if args.save:
        print(f"\nSaved: {report.save(args.save)}")
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
