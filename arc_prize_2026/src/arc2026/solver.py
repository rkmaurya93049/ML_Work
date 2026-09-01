from __future__ import annotations

from .primitives import Grid, identity, crop_non_background
from .synthesizer import synthesize


def _safe(fn, grid: Grid) -> Grid:
    try:
        out = fn(grid)
        if not out or not out[0] or len(out) > 30 or len(out[0]) > 30:
            return identity(grid)
        return out
    except Exception:
        return identity(grid)


def solve_task(task: dict) -> list[dict[str, Grid]]:
    programs = synthesize(task["train"])
    p1 = programs[0].fn if programs else identity
    p2 = programs[1].fn if len(programs) > 1 else crop_non_background

    outputs = []
    for test_pair in task["test"]:
        grid = test_pair["input"]
        a1 = _safe(p1, grid)
        a2 = _safe(p2, grid)
        if a2 == a1:
            a2 = identity(grid) if a1 != identity(grid) else crop_non_background(grid)
        outputs.append({"attempt_1": a1, "attempt_2": a2})
    return outputs


def solve_challenges(challenges: dict[str, dict]) -> dict[str, list[dict[str, Grid]]]:
    return {task_id: solve_task(task) for task_id, task in challenges.items()}
