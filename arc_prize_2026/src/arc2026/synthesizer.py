from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .primitives import Grid, base_programs, crop_non_background, infer_recolor_mapping, recolor, tile, upscale


@dataclass(frozen=True)
class Candidate:
    name: str
    fn: Callable[[Grid], Grid]
    complexity: int


def _consistent(fn: Callable[[Grid], Grid], train: list[dict]) -> bool:
    try:
        return all(fn(pair["input"]) == pair["output"] for pair in train)
    except Exception:
        return False


def _derived_candidates(train: list[dict]) -> list[Candidate]:
    candidates: list[Candidate] = []
    first_in, first_out = train[0]["input"], train[0]["output"]

    mapping = infer_recolor_mapping(first_in, first_out)
    if mapping is not None:
        candidates.append(Candidate(f"recolor:{mapping}", lambda g, m=mapping: recolor(g, m), 2))

    cropped = crop_non_background(first_in)
    crop_mapping = infer_recolor_mapping(cropped, first_out)
    if crop_mapping is not None:
        candidates.append(Candidate("crop+recolor", lambda g, m=crop_mapping: recolor(crop_non_background(g), m), 3))

    ih, iw = len(first_in), len(first_in[0])
    oh, ow = len(first_out), len(first_out[0])
    if oh % ih == 0 and ow % iw == 0:
        ry, rx = oh // ih, ow // iw
        if ry >= 1 and rx >= 1 and (ry > 1 or rx > 1):
            candidates.append(Candidate(f"upscale:{ry}x{rx}", lambda g, y=ry, x=rx: upscale(g, y, x), 2))
            candidates.append(Candidate(f"tile:{ry}x{rx}", lambda g, y=ry, x=rx: tile(g, y, x), 2))

    return candidates


def synthesize(train: list[dict]) -> list[Candidate]:
    pool = [Candidate(name, fn, 1) for name, fn in base_programs()]
    pool.extend(_derived_candidates(train))
    good = [c for c in pool if _consistent(c.fn, train)]
    good.sort(key=lambda c: (c.complexity, c.name))
    return good
