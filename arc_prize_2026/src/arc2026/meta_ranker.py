from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

from .synthesizer import Candidate, synthesize


def learn_program_priors(challenges: dict[str, dict]) -> dict[str, float]:
    """Learn empirical program priors from solved training tasks.

    For each task, all exact-fitting symbolic programs receive fractional credit.
    The resulting prior is intentionally lightweight and interpretable; it can be
    learned offline from ARC training tasks without leaking hidden test labels.
    """
    counts: Counter[str] = Counter()
    tasks = 0
    for task in challenges.values():
        exact = [c for c in synthesize(task["train"], top_k=64) if c.exact_train]
        if not exact:
            continue
        tasks += 1
        credit = 1.0 / len(exact)
        for c in exact:
            counts[c.name] += credit
    if tasks == 0:
        return {}
    return {name: value / tasks for name, value in counts.items()}


def save_program_priors(priors: dict[str, float], path: str | Path) -> None:
    Path(path).write_text(json.dumps(priors, indent=2, sort_keys=True), encoding="utf-8")


def load_program_priors(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


def rerank_with_priors(candidates: list[Candidate], priors: dict[str, float], weight: float = 0.05) -> list[Candidate]:
    """Use learned historical frequency only as a tie-breaker behind demo fit."""
    return sorted(
        candidates,
        key=lambda c: (
            -int(c.exact_train),
            -(c.train_score + weight * priors.get(c.name, 0.0)),
            c.complexity,
            c.name,
        ),
    )
