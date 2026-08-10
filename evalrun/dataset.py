from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Case:
    id: str
    input: str
    expected_output: str | None
    scorer: str
    scorer_config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def _normalize_scorer(raw: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if isinstance(raw, str):
        return raw, {}
    raw = dict(raw)
    scorer_type = raw.pop("type")
    return scorer_type, raw


def load_dataset(path: str | Path) -> list[Case]:
    path = Path(path)
    raw_cases = yaml.safe_load(path.read_text())
    if not raw_cases:
        return []

    cases: list[Case] = []
    seen_ids: set[str] = set()
    for entry in raw_cases:
        case_id = entry["id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)

        scorer_type, scorer_config = _normalize_scorer(entry["scorer"])
        cases.append(
            Case(
                id=case_id,
                input=entry["input"],
                expected_output=entry.get("expected_output"),
                scorer=scorer_type,
                scorer_config=scorer_config,
                tags=entry.get("tags", []),
            )
        )
    return cases
