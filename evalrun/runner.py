import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ollama

from evalrun.dataset import Case

DEFAULT_MODEL = "llama3.1"
MAX_TOKENS = 1024
TEMPERATURE = 0
CACHE_DIR = Path(".eval_cache")


@dataclass
class RawResult:
    case_id: str
    output_text: str | None
    latency_ms: float
    usage: dict[str, int] | None
    error: str | None


def _cache_enabled() -> bool:
    if os.environ.get("EVAL_CACHE") == "1":
        return True
    if os.environ.get("CI"):
        return False
    return True


def _cache_key(model: str, input_text: str, params: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "input": input_text, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_cache(key: str) -> RawResult | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    return RawResult(**json.loads(path.read_text()))


def _write_cache(key: str, result: RawResult) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(asdict(result)))


def _call_case(client: ollama.Client, case: Case, model: str) -> RawResult:
    params = {"max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
    key = _cache_key(model, case.input, params)

    if _cache_enabled():
        cached = _read_cache(key)
        if cached is not None:
            return cached

    start = time.monotonic()
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": case.input}],
            options={"temperature": TEMPERATURE, "num_predict": MAX_TOKENS},
        )
        result = RawResult(
            case_id=case.id,
            output_text=response.message.content,
            latency_ms=(time.monotonic() - start) * 1000,
            usage={
                "input_tokens": response.prompt_eval_count,
                "output_tokens": response.eval_count,
            },
            error=None,
        )
    except Exception as exc:
        return RawResult(
            case_id=case.id,
            output_text=None,
            latency_ms=(time.monotonic() - start) * 1000,
            usage=None,
            error=str(exc),
        )

    if _cache_enabled():
        _write_cache(key, result)
    return result


def run(dataset: list[Case], model: str = DEFAULT_MODEL, concurrency: int = 5) -> list[RawResult]:
    client = ollama.Client()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(lambda case: _call_case(client, case, model), dataset))
