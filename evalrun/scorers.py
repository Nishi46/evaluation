import json
import re
from dataclasses import dataclass
from typing import Callable

from huggingface_hub import (
    ChatCompletionInputJSONSchema,
    ChatCompletionInputResponseFormatJSONSchema,
    InferenceClient,
)

from evalrun.dataset import Case
from evalrun.runner import DEFAULT_MODEL as JUDGE_MODEL

JUDGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False,
}


@dataclass
class ScoreResult:
    passed: bool
    score: float
    detail: str


def exact_match_scorer(case: Case, output: str) -> ScoreResult:
    normalize = lambda s: " ".join(s.strip().lower().split())
    passed = normalize(output) == normalize(case.expected_output or "")
    return ScoreResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"expected={case.expected_output!r} got={output!r}",
    )


def regex_scorer(case: Case, output: str) -> ScoreResult:
    pattern = case.scorer_config["pattern"]
    match = re.search(pattern, output)
    passed = match is not None
    return ScoreResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=f"pattern={pattern!r} {'matched' if passed else 'no match'} in output={output!r}",
    )


def _parse_judge_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # not every HF Inference provider enforces the JSON schema strictly;
        # fall back to pulling the first JSON object out of a prose response.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"could not find JSON object in judge output: {text!r}")
        return json.loads(match.group(0))


def llm_judge_scorer(case: Case, output: str) -> ScoreResult:
    rubric = case.scorer_config["rubric"]
    pass_threshold = case.scorer_config.get("pass_threshold", 4)

    client = InferenceClient()
    response = client.chat_completion(
        model=JUDGE_MODEL,
        max_tokens=1024,
        response_format=ChatCompletionInputResponseFormatJSONSchema(
            type="json_schema",
            json_schema=ChatCompletionInputJSONSchema(
                name="judge_score",
                schema=JUDGE_JSON_SCHEMA,
                strict=True,
            ),
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    "You are grading a model's response against a rubric.\n\n"
                    f"Task given to the model:\n{case.input}\n\n"
                    f"Model's response:\n{output}\n\n"
                    f"Rubric:\n{rubric}\n\n"
                    "Score the response from 1 to 5 according to the rubric."
                ),
            }
        ],
    )

    parsed = _parse_judge_json(response.choices[0].message.content)
    score = int(parsed["score"])
    reasoning = parsed["reasoning"]
    passed = score >= pass_threshold

    return ScoreResult(
        passed=passed,
        score=score / 5.0,
        detail=f"judge_score={score}/5 threshold={pass_threshold} reasoning={reasoning!r}",
    )


SCORER_REGISTRY: dict[str, Callable[[Case, str], ScoreResult]] = {
    "exact_match": exact_match_scorer,
    "regex": regex_scorer,
    "llm_judge": llm_judge_scorer,
}


def score(case: Case, output: str) -> ScoreResult:
    scorer_fn = SCORER_REGISTRY[case.scorer]
    return scorer_fn(case, output)
