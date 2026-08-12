"""Data models for deterministic benchmark execution."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    metric: str
    scenario: str
    params: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaseResult:
    id: str
    metric: str
    passed: bool
    actual: dict[str, Any]
    expected: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "metric": self.metric,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    version: str
    passed: bool
    total: int
    passed_cases: int
    metrics: dict[str, float]
    thresholds: dict[str, float]
    threshold_results: dict[str, bool]
    cases: tuple[CaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "passed": self.passed,
            "total": self.total,
            "passed_cases": self.passed_cases,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "threshold_results": self.threshold_results,
            "cases": [case.to_dict() for case in self.cases],
        }

