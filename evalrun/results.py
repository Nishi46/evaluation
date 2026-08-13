import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from evalrun.dataset import Case
from evalrun.runner import RawResult
from evalrun.scorers import ScoreResult


@dataclass
class ResultRecord:
    run_id: str
    commit_sha: str
    case_id: str
    model: str
    output: str | None
    scorer: str
    passed: bool
    score: float
    latency_ms: float
    timestamp: str
    error: str | None = None


def generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def get_commit_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_record(
    case: Case,
    raw: RawResult,
    scored: ScoreResult | None,
    run_id: str,
    commit_sha: str,
    model: str,
) -> ResultRecord:
    return ResultRecord(
        run_id=run_id,
        commit_sha=commit_sha,
        case_id=case.id,
        model=model,
        output=raw.output_text,
        scorer=case.scorer,
        passed=scored.passed if scored is not None else False,
        score=scored.score if scored is not None else 0.0,
        latency_ms=raw.latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
        error=raw.error,
    )


def write_results(records: list[ResultRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def read_results(path: str | Path) -> list[ResultRecord]:
    path = Path(path)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(ResultRecord(**json.loads(line)))
    return records
