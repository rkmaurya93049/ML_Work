from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))


def validate_grid(grid: list[list[int]]) -> None:
    if not grid or not grid[0]:
        raise ValueError("Grid must be non-empty")
    width = len(grid[0])
    if not (1 <= len(grid) <= 30 and 1 <= width <= 30):
        raise ValueError("ARC grid dimensions must be between 1 and 30")
    for row in grid:
        if len(row) != width:
            raise ValueError("Grid must be rectangular")
        if any((not isinstance(v, int)) or v < 0 or v > 9 for v in row):
            raise ValueError("ARC cells must be integers in [0, 9]")


def validate_submission(challenges: dict, submission: dict) -> None:
    if set(challenges) != set(submission):
        missing = set(challenges) - set(submission)
        extra = set(submission) - set(challenges)
        raise ValueError(f"Task ID mismatch. missing={missing}, extra={extra}")
    for task_id, task in challenges.items():
        preds = submission[task_id]
        if len(preds) != len(task["test"]):
            raise ValueError(f"Wrong number of test predictions for {task_id}")
        for pred in preds:
            if set(pred) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"Each prediction needs attempt_1 and attempt_2: {task_id}")
            validate_grid(pred["attempt_1"])
            validate_grid(pred["attempt_2"])
