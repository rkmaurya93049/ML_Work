from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .meta_ranker import rerank_with_priors
from .primitives import Grid, crop_non_background, identity
from .synthesizer import Candidate, synthesize


@dataclass(frozen=True)
class Prediction:
    grid: Grid
    program: str
    train_score: float
    exact_train: bool


def _is_valid(grid: Grid) -> bool:
    if not grid or not grid[0] or len(grid) > 30 or len(grid[0]) > 30:
        return False
    width = len(grid[0])
    return all(len(row) == width for row in grid) and all(
        isinstance(v, int) and 0 <= v <= 9 for row in grid for v in row
    )


def _safe(fn: Callable[[Grid], Grid], grid: Grid) -> Grid:
    try:
        out = fn(grid)
        return out if _is_valid(out) else identity(grid)
    except Exception:
        return identity(grid)


def _fallbacks(grid: Grid) -> list[Prediction]:
    raw = [
        Prediction(identity(grid), "fallback:identity", 0.0, False),
        Prediction(crop_non_background(grid), "fallback:crop", 0.0, False),
    ]
    seen: set[str] = set()
    out: list[Prediction] = []
    for p in raw:
        key = repr(p.grid)
        if key not in seen and _is_valid(p.grid):
            seen.add(key)
            out.append(p)
    return out


def _predict_candidates(programs: list[Candidate], grid: Grid) -> list[Prediction]:
    predictions: list[Prediction] = []
    seen: set[str] = set()
    for c in programs:
        pred = _safe(c.fn, grid)
        key = repr(pred)
        if key in seen:
            continue
        seen.add(key)
        predictions.append(Prediction(pred, c.name, c.train_score, c.exact_train))
    predictions.extend(p for p in _fallbacks(grid) if repr(p.grid) not in seen)
    return predictions


def _choose_two(predictions: list[Prediction]) -> tuple[Prediction, Prediction]:
    if not predictions:
        raise ValueError("prediction pool must not be empty")
    first = predictions[0]
    second = next((p for p in predictions[1:] if p.grid != first.grid), None)
    if second is None:
        second = first
    return first, second


def solve_task_with_trace(
    task: dict,
    *,
    priors: dict[str, float] | None = None,
) -> tuple[list[dict[str, Grid]], list[dict[str, object]]]:
    programs = synthesize(task["train"])
    if priors:
        programs = rerank_with_priors(programs, priors)

    outputs: list[dict[str, Grid]] = []
    trace: list[dict[str, object]] = []

    for test_pair in task["test"]:
        grid = test_pair["input"]
        ranked = _predict_candidates(programs, grid)
        p1, p2 = _choose_two(ranked)
        outputs.append({"attempt_1": p1.grid, "attempt_2": p2.grid})
        trace.append(
            {
                "attempt_1_program": p1.program,
                "attempt_1_train_score": round(p1.train_score, 6),
                "attempt_1_exact_train": p1.exact_train,
                "attempt_2_program": p2.program,
                "attempt_2_train_score": round(p2.train_score, 6),
                "attempt_2_exact_train": p2.exact_train,
                "candidate_count": len(ranked),
            }
        )
    return outputs, trace


def solve_task(task: dict, *, priors: dict[str, float] | None = None) -> list[dict[str, Grid]]:
    outputs, _ = solve_task_with_trace(task, priors=priors)
    return outputs


def solve_challenges(
    challenges: dict[str, dict],
    *,
    priors: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Grid]]]:
    return {task_id: solve_task(task, priors=priors) for task_id, task in challenges.items()}


def solve_challenges_with_trace(
    challenges: dict[str, dict],
    *,
    priors: dict[str, float] | None = None,
):
    submission: dict[str, list[dict[str, Grid]]] = {}
    traces: dict[str, list[dict[str, object]]] = {}
    for task_id, task in challenges.items():
        submission[task_id], traces[task_id] = solve_task_with_trace(task, priors=priors)
    return submission, traces
