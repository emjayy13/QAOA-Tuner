"""Recommendation engine: low-cost / balanced / high-quality picks with explanations."""

from qaoa_tuner.recommendation.engine import (
    CATEGORIES,
    DISCLAIMER,
    ExcludedCandidate,
    Recommendation,
    RecommendationReport,
    RecommendationSettings,
    format_report,
    recommend,
)

__all__ = [
    "CATEGORIES",
    "DISCLAIMER",
    "ExcludedCandidate",
    "Recommendation",
    "RecommendationReport",
    "RecommendationSettings",
    "format_report",
    "recommend",
]
