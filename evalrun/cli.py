from dataclasses import replace
from pathlib import Path

import click

from evalrun.compare import DEFAULT_PASS_RATE_TOLERANCE, compare, is_regression
from evalrun.dataset import load_dataset
from evalrun.results import ResultRecord, build_record, generate_run_id, get_commit_sha, read_results, write_results
from evalrun.runner import DEFAULT_MODEL
from evalrun.runner import run as run_dataset
from evalrun.scorers import score


@click.group()
def main():
    """evalrun - lightweight LLM eval CLI."""


@main.command("run")
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True), help="Path to dataset YAML.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to run against.")
@click.option("--out", "out_path", required=True, type=click.Path(), help="Path to write results JSONL.")
@click.option("--concurrency", default=5, show_default=True, help="Number of concurrent requests.")
def run_cmd(dataset_path: str, model: str, out_path: str, concurrency: int) -> None:
    cases = load_dataset(dataset_path)
    if not cases:
        raise click.ClickException(f"no cases found in {dataset_path}")

    click.echo(f"running {len(cases)} cases against {model} (concurrency={concurrency})...")
    raw_results = run_dataset(cases, model=model, concurrency=concurrency)

    run_id = generate_run_id()
    commit_sha = get_commit_sha()
    cases_by_id = {case.id: case for case in cases}

    records = []
    for raw in raw_results:
        case = cases_by_id[raw.case_id]
        scored = None
        if not raw.error:
            try:
                scored = score(case, raw.output_text)
            except Exception as exc:
                raw = replace(raw, error=f"scoring failed: {exc}")
        records.append(build_record(case, raw, scored, run_id, commit_sha, model))

    write_results(records, out_path)

    passed = sum(1 for r in records if r.passed)
    errored = sum(1 for r in records if r.error)
    summary = f"{passed}/{len(records)} passed"
    if errored:
        summary += f", {errored} errored"
    click.echo(summary)
    click.echo(f"wrote results to {out_path}")


@main.command("report")
@click.option("--results", "results_path", required=True, type=click.Path(exists=True), help="Path to results JSONL.")
@click.option("--format", "fmt", default="markdown", show_default=True, type=click.Choice(["markdown"]))
def report_cmd(results_path: str, fmt: str) -> None:
    records = read_results(results_path)
    if not records:
        raise click.ClickException(f"no results found in {results_path}")

    click.echo(_render_markdown(records))


@main.command("compare")
@click.option("--current", "current_path", required=True, type=click.Path(exists=True), help="Path to current results JSONL.")
@click.option("--baseline", "baseline_path", required=True, type=click.Path(), help="Path to baseline results JSONL.")
@click.option(
    "--pass-rate-tolerance",
    default=DEFAULT_PASS_RATE_TOLERANCE,
    show_default=True,
    help="Allowed aggregate pass-rate drop (as a fraction, e.g. 0.02 = 2 points) before failing.",
)
def compare_cmd(current_path: str, baseline_path: str, pass_rate_tolerance: float) -> None:
    if Path(baseline_path).exists():
        baseline_records = read_results(baseline_path)
    else:
        click.echo(f"no baseline found at {baseline_path} — treating as empty (first run).")
        baseline_records = []
    current_records = read_results(current_path)

    result = compare(baseline_records, current_records)

    click.echo(f"Baseline: {sum(1 for r in baseline_records if r.passed)}/{result.baseline_count} passed ({result.baseline_pass_rate * 100:.1f}%)")
    click.echo(f"Current:  {sum(1 for r in current_records if r.passed)}/{result.current_count} passed ({result.current_pass_rate * 100:.1f}%)")
    change_pts = (result.current_pass_rate - result.baseline_pass_rate) * 100 + 0.0
    click.echo(f"Pass rate change: {change_pts:+.1f} pts (tolerance: {pass_rate_tolerance * 100:.1f} pts)")

    if result.regressions:
        click.echo("\nRegressions (passed on baseline, now failing):")
        for reg in result.regressions:
            suffix = f" (error: {reg.current_error})" if reg.current_error else ""
            click.echo(f"  - {reg.case_id}{suffix}")

    if result.new_case_ids:
        click.echo(f"\nNew cases (not in baseline): {', '.join(result.new_case_ids)}")
    if result.missing_case_ids:
        click.echo(f"\nMissing cases (in baseline, not in current): {', '.join(result.missing_case_ids)}")

    if is_regression(result, pass_rate_tolerance):
        click.echo("\nREGRESSION DETECTED — failing.")
        raise SystemExit(1)

    click.echo("\nNo regressions detected.")


def _render_markdown(records: list[ResultRecord]) -> str:
    total = len(records)
    passed = sum(1 for r in records if r.passed)
    pass_rate = (passed / total * 100) if total else 0.0

    lines = [
        "## Eval Report",
        "",
        f"**{passed}/{total}** cases passed ({pass_rate:.1f}%)",
        "",
        "| Case ID | Scorer | Result | Score | Latency (ms) |",
        "|---|---|---|---|---|",
    ]
    for r in records:
        if r.error:
            result_cell = f"⚠️ ERROR: {r.error}"
        else:
            result_cell = "✅ pass" if r.passed else "❌ fail"
        lines.append(f"| {r.case_id} | {r.scorer} | {result_cell} | {r.score:.2f} | {r.latency_ms:.0f} |")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
