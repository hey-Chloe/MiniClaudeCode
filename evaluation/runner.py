"""CLI and library entry point for MiniClaudeCode benchmarks."""

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evaluation.models import BenchmarkCase, BenchmarkReport, CaseResult
from evaluation.scenarios import DEFAULT_SCENARIOS, Scenario


class BenchmarkConfigurationError(ValueError):
    """Raised when a benchmark definition is malformed."""


class BenchmarkRunner:
    def __init__(self, scenarios: Mapping[str, Scenario] | None = None):
        self.scenarios = dict(scenarios or DEFAULT_SCENARIOS)

    def load(self, path: str | Path) -> tuple[str, dict[str, float], list[BenchmarkCase]]:
        source = Path(path)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkConfigurationError(f"benchmark could not be loaded: {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
            raise BenchmarkConfigurationError("benchmark must contain a cases array")

        thresholds = self._thresholds(data.get("thresholds", {}))
        cases: list[BenchmarkCase] = []
        seen_ids: set[str] = set()
        for raw in data["cases"]:
            if not isinstance(raw, dict):
                raise BenchmarkConfigurationError("every benchmark case must be an object")
            try:
                case = BenchmarkCase(
                    id=str(raw["id"]),
                    metric=str(raw["metric"]),
                    scenario=str(raw["scenario"]),
                    params=dict(raw.get("params", {})),
                    expected=dict(raw.get("expected", {})),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise BenchmarkConfigurationError(f"invalid benchmark case: {exc}") from exc
            if not case.id or case.id in seen_ids:
                raise BenchmarkConfigurationError(f"duplicate or empty case id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)
        if not cases:
            raise BenchmarkConfigurationError("benchmark must contain at least one case")
        return str(data.get("version", "unknown")), thresholds, cases

    def run_file(self, path: str | Path) -> BenchmarkReport:
        version, thresholds, cases = self.load(path)
        return self.run(version, thresholds, cases)

    def run(
        self,
        version: str,
        thresholds: dict[str, float],
        cases: list[BenchmarkCase],
    ) -> BenchmarkReport:
        results = tuple(self._run_case(case) for case in cases)
        grouped: dict[str, list[bool]] = defaultdict(list)
        for result in results:
            grouped[result.metric].append(result.passed)
        metrics = {
            metric: sum(outcomes) / len(outcomes)
            for metric, outcomes in grouped.items()
        }
        threshold_results = {
            metric: metric in metrics and metrics[metric] >= threshold
            for metric, threshold in thresholds.items()
        }
        passed = all(result.passed for result in results) and all(
            threshold_results.values()
        )
        return BenchmarkReport(
            version=version,
            passed=passed,
            total=len(results),
            passed_cases=sum(result.passed for result in results),
            metrics=metrics,
            thresholds=thresholds,
            threshold_results=threshold_results,
            cases=results,
        )

    def _run_case(self, case: BenchmarkCase) -> CaseResult:
        scenario = self.scenarios.get(case.scenario)
        if scenario is None:
            return CaseResult(
                case.id,
                case.metric,
                False,
                {},
                case.expected,
                error=f"unknown scenario: {case.scenario}",
            )
        try:
            actual = scenario(case.params)
            passed = all(actual.get(key) == value for key, value in case.expected.items())
            return CaseResult(case.id, case.metric, passed, actual, case.expected)
        except Exception as exc:
            return CaseResult(
                case.id,
                case.metric,
                False,
                {},
                case.expected,
                error=f"scenario failed: {exc}",
            )

    @staticmethod
    def _thresholds(raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            raise BenchmarkConfigurationError("thresholds must be an object")
        thresholds: dict[str, float] = {}
        for metric, value in raw.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BenchmarkConfigurationError(f"invalid threshold for {metric}")
            threshold = float(value)
            if not 0 <= threshold <= 1:
                raise BenchmarkConfigurationError(
                    f"threshold for {metric} must be between 0 and 1"
                )
            thresholds[str(metric)] = threshold
        return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MiniClaudeCode benchmarks")
    parser.add_argument(
        "benchmark",
        nargs="?",
        default=str(Path(__file__).with_name("benchmark.json")),
    )
    parser.add_argument("--output", help="Write the JSON report to this path")
    args = parser.parse_args()

    try:
        report = BenchmarkRunner().run_file(args.benchmark)
    except BenchmarkConfigurationError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, ensure_ascii=False))
        return 2

    rendered = json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

