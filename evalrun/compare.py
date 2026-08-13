from dataclasses import dataclass

from evalrun.results import ResultRecord

DEFAULT_PASS_RATE_TOLERANCE = 0.02  # 2 percentage points


@dataclass
class Regression:
    case_id: str
    current_passed: bool
    current_error: str | None


@dataclass
class CompareResult:
    baseline_count: int
    current_count: int
    baseline_pass_rate: float
    current_pass_rate: float
    regressions: list[Regression]
    new_case_ids: list[str]
    missing_case_ids: list[str]

    @property
    def pass_rate_drop(self) -> float:
        return self.baseline_pass_rate - self.current_pass_rate


def _pass_rate(records: list[ResultRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.passed) / len(records)


def compare(baseline: list[ResultRecord], current: list[ResultRecord]) -> CompareResult:
    baseline_by_id = {r.case_id: r for r in baseline}
    current_by_id = {r.case_id: r for r in current}

    regressions = []
    for case_id, base_record in baseline_by_id.items():
        cur_record = current_by_id.get(case_id)
        if cur_record is None or not base_record.passed or cur_record.passed:
            continue
        regressions.append(
            Regression(case_id=case_id, current_passed=cur_record.passed, current_error=cur_record.error)
        )

    return CompareResult(
        baseline_count=len(baseline),
        current_count=len(current),
        baseline_pass_rate=_pass_rate(baseline),
        current_pass_rate=_pass_rate(current),
        regressions=regressions,
        new_case_ids=sorted(set(current_by_id) - set(baseline_by_id)),
        missing_case_ids=sorted(set(baseline_by_id) - set(current_by_id)),
    )


def is_regression(result: CompareResult, pass_rate_tolerance: float = DEFAULT_PASS_RATE_TOLERANCE) -> bool:
    return bool(result.regressions) or result.pass_rate_drop > pass_rate_tolerance
