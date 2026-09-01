from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .objects import (
    crop_largest_object,
    crop_smallest_object,
    fill_object_bounding_boxes,
    keep_largest_object,
    keep_smallest_object,
)
from .primitives import Grid, base_programs
from .relations import (
    complete_horizontal_symmetry,
    complete_rotational_symmetry,
    complete_vertical_symmetry,
    outline_non_background,
    sort_objects_by_size_row,
)
from .synthesizer import Candidate


@dataclass(frozen=True)
class SymbolicProposal:
    """A model-proposed symbolic program.

    `operations` is intentionally restricted to a safe DSL. An LLM or other
    learned proposer may emit operation names, but it never executes arbitrary
    Python. The proposal is accepted only if it reproduces every train output.
    """

    operations: tuple[str, ...]
    rationale: str = ""


def operation_registry():
    registry = {name: fn for name, fn in base_programs()}
    registry.update(
        {
            "crop_largest_object": crop_largest_object,
            "crop_smallest_object": crop_smallest_object,
            "keep_largest_object": keep_largest_object,
            "keep_smallest_object": keep_smallest_object,
            "fill_object_bounding_boxes": fill_object_bounding_boxes,
            "complete_horizontal_symmetry": complete_horizontal_symmetry,
            "complete_vertical_symmetry": complete_vertical_symmetry,
            "complete_rotational_symmetry": complete_rotational_symmetry,
            "outline_non_background": outline_non_background,
            "sort_objects_by_size_row": sort_objects_by_size_row,
        }
    )
    return registry


def compile_proposal(proposal: SymbolicProposal) -> Candidate | None:
    registry = operation_registry()
    if not proposal.operations or any(op not in registry for op in proposal.operations):
        return None

    fns = [registry[op] for op in proposal.operations]

    def program(grid: Grid) -> Grid:
        out = grid
        for fn in fns:
            out = fn(out)
        return out

    return Candidate(
        name="model:" + " -> ".join(proposal.operations),
        fn=program,
        complexity=2 * len(proposal.operations) + 1,
    )


def validate_proposals(train: list[dict], proposals: Iterable[SymbolicProposal]) -> list[Candidate]:
    accepted: list[Candidate] = []
    for proposal in proposals:
        candidate = compile_proposal(proposal)
        if candidate is None:
            continue
        try:
            if all(candidate.fn(pair["input"]) == pair["output"] for pair in train):
                accepted.append(
                    Candidate(
                        candidate.name,
                        candidate.fn,
                        candidate.complexity,
                        train_score=1.0,
                        exact_train=True,
                    )
                )
        except Exception:
            continue
    accepted.sort(key=lambda c: (c.complexity, c.name))
    return accepted
