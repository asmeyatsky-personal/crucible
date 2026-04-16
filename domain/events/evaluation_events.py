"""Evaluation-related domain events — re-exported from evaluation entity."""

from domain.entities.evaluation import EvaluationCompletedEvent, RegressionDetectedEvent

__all__ = ["EvaluationCompletedEvent", "RegressionDetectedEvent"]
