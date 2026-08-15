"""Data models for repo-level coding benchmark tasks and reports."""

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class CodingTask:
    """One coding task: a fixture repo, an instruction, and ground truth."""

    id: str
    category: str
    task: str
    files: Mapping[str, str]
    ground_truth: Mapping[str, Any]
    expected_files: Mapping[str, str] = field(default_factory=dict)
    hidden_tests: Mapping[str, str] = field(default_factory=dict)
    expected_policy: Mapping[str, str] = field(default_factory=dict)
    pre_checks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodingCaseResult:
    """Per-task outcome plus metrics computed from the agent run."""

    id: str
    category: str
    status: str
    passed: bool
    checks: Mapping[str, bool]
    turns: int
    tool_calls: int
    tool_successes: int
    tool_success_rate: float | None
    policy_actions: Mapping[str, int]
    total_reads: int
    repeated_reads: int
    repeated_read_rate: float | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    context_truncated: bool
    latency_seconds: float
    cost_usd: float | None
    skills_loaded: tuple[str, ...]
    first_pass: bool
    security_blocks: int
    policy_checks_matched: int
    policy_checks_total: int
    recoverable_failures: int = 0
    recovered_failures: int = 0
    recovery_rate: float | None = None
    tools_sent: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "status": self.status,
            "passed": self.passed,
            "checks": dict(self.checks),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_successes": self.tool_successes,
            "tool_success_rate": self.tool_success_rate,
            "policy_actions": dict(self.policy_actions),
            "total_reads": self.total_reads,
            "repeated_reads": self.repeated_reads,
            "repeated_read_rate": self.repeated_read_rate,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "context_truncated": self.context_truncated,
            "latency_seconds": round(self.latency_seconds, 3),
            "cost_usd": self.cost_usd,
            "skills_loaded": list(self.skills_loaded),
            "first_pass": self.first_pass,
            "security_blocks": self.security_blocks,
            "policy_checks_matched": self.policy_checks_matched,
            "policy_checks_total": self.policy_checks_total,
            "recoverable_failures": self.recoverable_failures,
            "recovered_failures": self.recovered_failures,
            "recovery_rate": self.recovery_rate,
            "tools_sent": self.tools_sent,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class CodingReport:
    """Aggregated benchmark report with honest, computable metrics."""

    version: str
    run_type: str
    model: str | None
    total: int
    passed: int
    task_success_rate: float
    tool_success_rate: float | None
    first_pass_rate: float
    average_turns: float
    average_tokens: float
    average_latency_seconds: float
    total_cost_usd: float | None
    repeated_read_rate: float | None
    context_truncation_rate: float
    approval_accuracy: float | None
    security_blocks: int
    safety_block_rate: float | None
    recovery_rate: float | None
    average_tools_sent_per_turn: float
    cases: tuple[CodingCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_type": self.run_type,
            "model": self.model,
            "total": self.total,
            "passed": self.passed,
            "task_success_rate": self.task_success_rate,
            "tool_success_rate": self.tool_success_rate,
            "first_pass_rate": self.first_pass_rate,
            "average_turns": round(self.average_turns, 3),
            "average_tokens": round(self.average_tokens, 1),
            "average_latency_seconds": round(self.average_latency_seconds, 3),
            "total_cost_usd": self.total_cost_usd,
            "repeated_read_rate": self.repeated_read_rate,
            "context_truncation_rate": self.context_truncation_rate,
            "approval_accuracy": self.approval_accuracy,
            "security_blocks": self.security_blocks,
            "safety_block_rate": self.safety_block_rate,
            "recovery_rate": self.recovery_rate,
            "average_tools_sent_per_turn": round(
                self.average_tools_sent_per_turn, 3
            ),
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def aggregate(
        cls,
        cases: tuple[CodingCaseResult, ...],
        *,
        run_type: str,
        model: str | None,
    ) -> "CodingReport":
        total = len(cases)
        passed = sum(1 for case in cases if case.passed)
        tool_calls = sum(case.tool_calls for case in cases)
        tool_successes = sum(case.tool_successes for case in cases)
        total_reads = sum(case.total_reads for case in cases)
        repeated_reads = sum(case.repeated_reads for case in cases)
        expected_checks = sum(case.policy_checks_total for case in cases)
        matched_checks = sum(case.policy_checks_matched for case in cases)
        recoverable_failures = sum(
            case.recoverable_failures for case in cases
        )
        recovered_failures = sum(case.recovered_failures for case in cases)
        denied_calls = sum(
            case.policy_actions.get("deny", 0) for case in cases
        )
        total_turns = sum(case.turns for case in cases)
        total_tools_sent = sum(case.tools_sent for case in cases)
        costs = [case.cost_usd for case in cases if case.cost_usd is not None]
        total_cost = sum(costs) if len(costs) == total and costs else None
        return cls(
            version="1.0",
            run_type=run_type,
            model=model,
            total=total,
            passed=passed,
            task_success_rate=(passed / total) if total else 0.0,
            tool_success_rate=(tool_successes / tool_calls) if tool_calls else None,
            first_pass_rate=(
                sum(1 for case in cases if case.first_pass) / total
            )
            if total
            else 0.0,
            average_turns=(
                sum(case.turns for case in cases) / total
            )
            if total
            else 0.0,
            average_tokens=(
                sum(case.total_tokens for case in cases) / total
            )
            if total
            else 0.0,
            average_latency_seconds=(
                sum(case.latency_seconds for case in cases) / total
            )
            if total
            else 0.0,
            total_cost_usd=total_cost,
            repeated_read_rate=(
                repeated_reads / total_reads if total_reads else None
            ),
            context_truncation_rate=(
                sum(1 for case in cases if case.context_truncated) / total
            )
            if total
            else 0.0,
            approval_accuracy=(
                matched_checks / expected_checks if expected_checks else None
            ),
            security_blocks=sum(case.security_blocks for case in cases),
            safety_block_rate=(
                denied_calls / tool_calls if tool_calls else None
            ),
            recovery_rate=(
                recovered_failures / recoverable_failures
                if recoverable_failures
                else None
            ),
            average_tools_sent_per_turn=(
                total_tools_sent / total_turns if total_turns else 0.0
            ),
            cases=cases,
        )
