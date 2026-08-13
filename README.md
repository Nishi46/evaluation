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
├── .env.example                    # OLLAMA_HOST (optional; no API key needed)
├── evalrun/
│   ├── __init__.py
│   ├── dataset.py                  # load/validate YAML dataset
│   ├── runner.py                   # calls local Ollama daemon against dataset
│   ├── scorers.py                  # scorer registry + exact_match/regex/llm_judge
│   ├── results.py                  # JSONL read/write, run-id/commit keying
│   ├── cli.py                      # `evalrun` entrypoint
│   └── compare.py                  # baseline vs. current diff logic
├── datasets/
│   └── core_regression.yaml        # seed dataset (18 cases)
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

Seed dataset categories (18 cases total): arithmetic/exact-match (2), regex extraction — email/date/phone (3), format compliance — JSON validity/list formatting (2), instruction following — word-count limits/tone/refusal (3), summarization quality via LLM-judge (3), reasoning/QA via LLM-judge (3), edge cases — empty input, basic prompt-injection attempt (2).

## Runner (`evalrun/runner.py`)

- `run(dataset, model="llama3.1", concurrency=5) -> list[RawResult]`.
- One `chat()` call per case via `ollama.Client()`, talking to a local Ollama daemon (`http://localhost:11434` by default); no tools/thinking (keep deterministic). No API key required — this is why the model swapped from a hosted API to a local one.
- No retries: the `ollama` client makes a single request per call with no built-in retry/backoff. A failed call (daemon not running, model not pulled, connection error) is captured as `RawResult.error` rather than retried or raised — a hand-rolled retry loop is a known gap, not yet built.
- Local disk cache in `.eval_cache/` keyed by `hash(model + input + params)`, skipped in CI unless explicitly enabled — with Ollama there's no billing to avoid, but it still saves real time re-running unchanged cases locally.
- Concurrency via `ThreadPoolExecutor` (client is sync; no asyncio needed at this scale). Ollama serves requests against one local model process, so concurrency mostly overlaps I/O rather than getting true parallel inference — it's still a net win since cache lookups and case setup aren't free.
- Output: `RawResult(case_id, output_text, latency_ms, usage, error)`. Scoring is a separate pass — runner never scores.

## Scorers (`evalrun/scorers.py`)

```python
def score(case: Case, output: str) -> ScoreResult:
    """Returns ScoreResult(passed: bool, score: float, detail: str)"""

SCORER_REGISTRY: dict[str, Callable[[Case, str], ScoreResult]] = {
    "exact_match": exact_match_scorer,
    "regex": regex_scorer,
    "llm_judge": llm_judge_scorer,
}
```

- `exact_match_scorer`: normalized (whitespace/case) string compare against `expected_output`.
- `regex_scorer`: `re.search(pattern, output)`.
- `llm_judge_scorer`: see below.
- The registry is the extension point for future scorer types (tool-trajectory, retrieval-precision, safety-classifier) — new scorers register a function, no runner changes needed.

### LLM-as-judge

`llm_judge_scorer` is used for open-ended cases (summarization, instruction-following, reasoning, edge cases) where exact-match/regex can't capture quality.

- **Judge model = runner model.** The judge is **the same model as the runner (`llama3.1` via local Ollama)**, imported directly as `JUDGE_MODEL = runner.DEFAULT_MODEL`. This keeps the framework to a single model dependency for Phase 1, at the cost of self-preference bias risk (not mitigated).
- **Rubric and threshold come from the dataset.** Each case supplies `rubric` (a 1-5 grading instruction) and an optional `pass_threshold` (default `4`) in its `scorer` config.
- **Fixed grading prompt.** The judge is called with a single user message containing the original task (`case.input`), the model's response (`output`), and the `rubric`, asking it to score 1-5.
- **Structured output.** The call passes `format` as a strict JSON schema (`{score: int, reasoning: str}`) to `ollama.Client().chat()`, which uses grammar-constrained decoding to guarantee the response matches — so the judge returns machine-parseable output rather than free text.
- **Parsing fallback.** `_parse_judge_json` tries `json.loads` first; if that fails, it regex-extracts the first `{...}` block from the response and parses that instead. Ollama's structured output is enforced more reliably than a hosted API's `response_format` hint, so this fallback is mostly belt-and-suspenders at this point rather than a frequently-hit path.
- **Scoring.** `passed = score >= pass_threshold`; the normalized `score` returned is `judge_score / 5.0`. The judge's `reasoning` string is preserved in `ScoreResult.detail` alongside the raw score and threshold, e.g. `judge_score=4/5 threshold=4 reasoning='...'`.
- **Known limitations:** single judge call with no retry/majority-voting on parse failure, and the judge call's temperature isn't pinned (unlike the runner's `TEMPERATURE = 0`), so judge scores aren't fully deterministic run-to-run.

## Results (`evalrun/results.py`)

JSONL, one line per case per run. `run_id` (UTC timestamp, e.g. `2026-08-10T15-30-00Z`) and `commit_sha` are embedded in every record, but the *filename* is just whatever `evalrun run --out` points at — the CLI doesn't derive it from `run_id`. In practice that's `results/run.jsonl` for a PR run and `results/baseline.jsonl` for the committed baseline.

```json
{"run_id": "2026-08-10T15-30-00Z", "commit_sha": "a1b2c3d", "case_id": "math_basic_01", "model": "llama3.1", "output": "408", "scorer": "exact_match", "passed": true, "score": 1.0, "latency_ms": 812, "timestamp": "2026-08-10T15:30:04Z", "error": null}
```

`error` is `null` for a normal scored case; it's populated instead (with `output` usually `null`, `passed: false`, `score: 0.0`) when either the runner call itself failed, or scoring failed after a successful runner call (see "Per-case failure handling" below) — either way the case still gets a record instead of the whole run aborting.

`results/baseline.jsonl` is **committed to the repo** and represents main's latest run. A CI step on merge-to-main overwrites it with the new run's output.

## CLI (`evalrun`)

```
evalrun run --dataset datasets/core_regression.yaml --model llama3.1 --out results/run.jsonl [--max-error-rate 0.5]
evalrun compare --current results/run.jsonl --baseline results/baseline.jsonl [--pass-rate-tolerance 0.02]
evalrun report --results results/run.jsonl --format markdown
```

`evalrun compare` exits `0` if there's no regression, `1` if there is — that's the mechanism the CI job below fails on. `--pass-rate-tolerance` (default `0.02`, i.e. 2 points) sets how much the aggregate pass rate is allowed to drop before that alone counts as a regression; any single case flipping from passed→failed always counts, tolerance or not.

`evalrun compare` also treats a missing `--baseline` file as an empty baseline (every case reported as new, nothing can regress) rather than failing — this bootstraps a fresh repo's first run/PR before `results/baseline.jsonl` exists yet.

### Per-case failure handling

`evalrun run` never lets one bad case take down the whole run — every case still gets a `ResultRecord` written, even on failure:

- If the runner call itself fails (Ollama daemon not running, model not pulled, connection error, etc.), that's `RawResult.error`, scoring is skipped, and the record is written with `passed: false`, `score: 0.0`, `output: null`.
- If the runner call succeeds but scoring then throws (e.g. the judge model returns unparseable JSON, or the judge call itself fails), that exception is caught per-case and recorded as `error: "scoring failed: ..."`, with `output` still populated from the runner so you can see what the model actually said.

This means a single flaky case degrades one row of the results file instead of losing every already-completed call in the run.

That leniency has a failure mode of its own: if something systemic breaks, *every* case errors, but the run would otherwise still exit `0` and write a fully-broken results file — which the `update-baseline` job would then happily commit as `results/baseline.jsonl`, silently zeroing out regression detection for everyone. This actually happened twice while wiring up CI, under the (now-replaced) hosted-API backend: once from a missing repo secret (100% of cases errored with an auth failure) and once from that provider's account hitting a billing/credit limit mid-run (61% errored) — `--max-error-rate` (default `0.5`) caught both and refused to let either become the baseline. The runner is local-only via Ollama now, so neither of those two specific failure modes applies anymore, but the guardrail stays as protection against whatever the next systemic failure turns out to be (Ollama daemon down, model not pulled in CI, etc.).

## GitHub Actions (`.github/workflows/eval.yml`)

- **Triggers**: `pull_request` and `push` to `main`, both filtered to paths `evalrun/**`, `datasets/**`, `pyproject.toml`, and `.github/workflows/eval.yml` itself — so a change to the workflow file gets a real run to validate it, not just a silent no-op. (Path filters are a common trap: a commit that only touches the workflow file won't trigger anything unless the workflow file is in its own filter list — this bit us once already.)
- **No secrets required.** Since the runner moved to local Ollama, there's no API key to configure at all — a meaningful simplification over the hosted-API version of this workflow.
- Both jobs install Ollama fresh on the GH-hosted runner, start the daemon, and pull `llama3.1` before running anything:
  ```
  curl -fsSL https://ollama.com/install.sh | sh
  ollama serve &            # + poll localhost:11434 until it's up
  ollama pull llama3.1
  ```
  There's no model-weight caching across runs yet (each run re-downloads ~5GB), and GH-hosted runners are CPU-only with no GPU, so a full 18-case eval — especially the `llm_judge` cases — is meaningfully slower here than running the same dataset locally. This was a known, accepted tradeoff when choosing Ollama over a hosted API for CI: no billing/secrets to manage, at the cost of CI wall-clock time. Caching `~/.ollama` (or wherever the installed daemon actually stores models — worth confirming from a real run's logs) is the obvious next optimization if this becomes painful.
- **PR workflow** (`eval-pr` job): checkout → install deps → install/start Ollama, pull model → `evalrun run` → `evalrun report --format markdown` (captured to a file) → `evalrun compare` against committed `results/baseline.jsonl` (its exit code is captured, not failed on immediately) → post/update a single PR comment (matched by an HTML marker, so repeated pushes update one comment instead of spamming) containing both the report table and the compare output → **fail the job** last, based on the captured compare exit code, so the comment always posts even when the job is about to fail. Note: making this job a **required status check** (so regressions block merge) is a repo Settings → Branches step, not something the workflow YAML itself configures.
- **Main workflow** (`update-baseline` job): same Ollama setup, then runs `evalrun run` straight into `results/baseline.jsonl` and commits it — a no-op (no commit/push) if the output is identical to what's already committed.

## Implementation Order

1. `evalrun/dataset.py` (schema + loader/validator) + seed `datasets/core_regression.yaml`
2. `evalrun/scorers.py` — `exact_match`, `regex` first (no API dependency, unblocks testing)
3. `evalrun/runner.py` (Ollama wrapper)
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
