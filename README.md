# evaluation

A lightweight LLM eval framework, à la promptfoo/DeepEval, so prompt/model changes can be regression-tested automatically on every PR instead of eyeballed.

## Scope: Phase 1 — Prompt/Agent Regression Testing

A full eval framework has five layers (dataset, runner, scorer, results store, CI). Rather than build all five generically, Phase 1 is scoped tightly around the **prompt/agent regression testing** use case — it needs the simplest scorers (exact-match, regex, LLM-judge) and the simplest dataset shape (input → expected output), while still exercising the full pipeline end to end.

Other use cases (tool-trajectory scoring, RAG retrieval scoring, safety datasets, multi-model comparison) are deliberately left as extension points via the scorer registry and the dataset `tags` field, not built now.

## Directory Structure

```
evaluation/
├── README.md                       # this file
├── pyproject.toml
├── .env.example                    # ANTHROPIC_API_KEY
├── evalrun/
│   ├── __init__.py
│   ├── dataset.py                  # load/validate YAML dataset
│   ├── runner.py                   # calls Anthropic API against dataset
│   ├── scorers.py                  # scorer registry + exact_match/regex/llm_judge
│   ├── results.py                  # JSONL read/write, run-id/commit keying
│   ├── cli.py                      # `evalrun` entrypoint
│   └── compare.py                  # baseline vs. current diff logic
├── datasets/
│   └── core_regression.yaml        # seed dataset (15-20 cases)
├── results/
│   └── baseline.jsonl              # committed; updated on merge to main
└── .github/
    └── workflows/eval.yml
```

## Dataset Schema

`datasets/core_regression.yaml`, one case per entry:

```yaml
- id: math_basic_01
  input: "What is 17 * 24?"
  expected_output: "408"
  scorer: exact_match
  tags: [deterministic, math]

- id: regex_email_extract
  input: "Extract the email from: Contact Jane at jane@acme.com for details."
  scorer:
    type: regex
    pattern: "jane@acme\\.com"
  tags: [deterministic, extraction]

- id: summary_quality_01
  input: "Summarize in 2 sentences: [article text...]"
  scorer:
    type: llm_judge
    rubric: "Score 1-5: does the summary capture the main thesis and key evidence in ~2 sentences without hallucination?"
    pass_threshold: 4
  tags: [open-ended, summarization]
```

Fields: `id`, `input`, `expected_output` (omitted for judge-scored cases, used as reference for exact/regex), `scorer` (string shorthand or object), `tags` (free metadata now; later used to filter into suites like `tool-use`/`rag`/`safety`).

Seed dataset categories (15-20 cases total): arithmetic/exact-match (2), regex extraction — email/date/phone (3), format compliance — JSON validity/list formatting (2), instruction following — word-count limits/tone/refusal (3), summarization quality via LLM-judge (3), reasoning/QA via LLM-judge (3), edge cases — empty input, basic prompt-injection attempt (2).

## Runner (`evalrun/runner.py`)

- `run(dataset, model="claude-sonnet-5", concurrency=5) -> list[RawResult]`.
- One `messages.create()` call per case via `anthropic.Anthropic()`; no tools/thinking (keep deterministic).
- Retries: rely on SDK's built-in `max_retries`, no hand-rolled backoff.
- Local disk cache in `.eval_cache/` keyed by `hash(model + input + params)`, skipped in CI unless explicitly enabled — avoids re-billing unchanged cases during local iteration.
- Concurrency via `ThreadPoolExecutor` (SDK is sync; no asyncio needed at this scale).
- Output: `RawResult(case_id, output_text, latency_ms, usage, error)`. Scoring is a separate pass — runner never scores.

## Scorers (`evalrun/scorers.py`)

```python
def score(case: dict, output: str) -> ScoreResult:
    """Returns ScoreResult(passed: bool, score: float, detail: str)"""

SCORER_REGISTRY: dict[str, Callable] = {
    "exact_match": exact_match_scorer,
    "regex": regex_scorer,
    "llm_judge": llm_judge_scorer,
}
```

- `exact_match_scorer`: normalized (whitespace/case) string compare against `expected_output`.
- `regex_scorer`: `re.search(pattern, output)`.
- `llm_judge_scorer`: judge model is **the same model as the runner (claude-sonnet-5)** — keeps the framework to a single model dependency for Phase 1. Uses structured output (`output_config.format` JSON schema: `{score: int, reasoning: str}`) rather than free-text parsing.
- The registry is the extension point for future scorer types (tool-trajectory, retrieval-precision, safety-classifier) — new scorers register a function, no runner changes needed.

## Results (`evalrun/results.py`)

JSONL, one line per case per run, written to `results/{run_id}.jsonl`:

```json
{"run_id": "2026-08-10T15-30-00Z", "commit_sha": "a1b2c3d", "case_id": "math_basic_01", "model": "claude-sonnet-5", "output": "408", "scorer": "exact_match", "passed": true, "score": 1.0, "latency_ms": 812, "timestamp": "2026-08-10T15:30:04Z"}
```

`results/baseline.jsonl` is **committed to the repo** and represents main's latest run. A CI step on merge-to-main overwrites it with the new run's output.

## CLI (`evalrun`)

```
evalrun run --dataset datasets/core_regression.yaml --model claude-sonnet-5 --out results/run.jsonl
evalrun compare --current results/run.jsonl --baseline results/baseline.jsonl
evalrun report --results results/run.jsonl --format markdown
```

## GitHub Actions (`.github/workflows/eval.yml`)

- **PR workflow**: triggers on `pull_request` for paths `evalrun/**`, `datasets/**`. Steps: checkout → install → `evalrun run` → `evalrun compare` against committed `results/baseline.jsonl` → post PR comment via `evalrun report --format markdown` (through `actions/github-script` or a comment action) → **fail the job** (non-zero exit) if any case that passed on baseline now fails, or aggregate pass-rate drops beyond a small tolerance. This should be wired as a required status check so regressions block merge.
- **Main workflow**: on push to `main`, runs `evalrun run` and overwrites/commits `results/baseline.jsonl`.
- Secret: `ANTHROPIC_API_KEY` as a repo secret.

## Implementation Order

1. `evalrun/dataset.py` (schema + loader/validator) + seed `datasets/core_regression.yaml`
2. `evalrun/scorers.py` — `exact_match`, `regex` first (no API dependency, unblocks testing)
3. `evalrun/runner.py` (Anthropic SDK wrapper)
4. Add `llm_judge_scorer` once the runner works
5. `evalrun/results.py` (JSONL write/read, run-id/commit keying)
6. `evalrun/cli.py` wiring `run` / `report`
7. `evalrun/compare.py` + `evalrun compare`
8. `.github/workflows/eval.yml` — PR job (fail + comment) and main job (baseline update)
9. This README documents the scorer plugin interface and `tags` extension points for future layers (tool-use, RAG, safety, multi-model)

## Verification

- Run `evalrun run --dataset datasets/core_regression.yaml --out results/run.jsonl` locally against the seed dataset and confirm all cases produce scored output with no errors.
- Run `evalrun compare --current results/run.jsonl --baseline results/baseline.jsonl` with an intentionally broken case (e.g. edit expected output) and confirm it reports a regression.
- Open a test PR touching `datasets/core_regression.yaml` and confirm the workflow runs, posts a comment, and fails the check on an injected regression; confirm it passes when reverted.
