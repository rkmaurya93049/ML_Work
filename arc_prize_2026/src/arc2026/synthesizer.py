from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .objects import (
    crop_largest_object,
    crop_smallest_object,
    fill_object_bounding_boxes,
    keep_largest_object,
    keep_smallest_object,
)
from .primitives import (
    Grid,
    base_programs,
    crop_non_background,
    infer_recolor_mapping,
    recolor,
    tile,
    upscale,
)
from .relations import (
    complete_horizontal_symmetry,
    complete_rotational_symmetry,
    complete_vertical_symmetry,
    infer_count_rendering,
    infer_uniform_foreground_color,
    outline_non_background,
    recolor_foreground,
    render_object_count,
    sort_objects_by_size_row,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    fn: Callable[[Grid], Grid]
    complexity: int
    train_score: float = 0.0
    exact_train: bool = False


def _valid_grid(g: Grid) -> bool:
    if not isinstance(g, list) or not g or not isinstance(g[0], list) or not g[0]:
        return False
    w = len(g[0])
    if len(g) > 30 or w > 30 or any(len(row) != w for row in g):
        return False
    return all(isinstance(v, int) and 0 <= v <= 9 for row in g for v in row)


def _pair_score(pred: Grid, truth: Grid) -> float:
    if pred == truth:
        return 1.0
    if not _valid_grid(pred):
        return 0.0
    if len(pred) != len(truth) or len(pred[0]) != len(truth[0]):
        # Small reward for matching one dimension: useful for ranking shape hypotheses.
        return 0.08 * int(len(pred) == len(truth)) + 0.08 * int(len(pred[0]) == len(truth[0]))
    total = len(truth) * len(truth[0])
    equal = sum(int(a == b) for rp, rt in zip(pred, truth) for a, b in zip(rp, rt))
    return 0.2 + 0.8 * (equal / total)


def _score(fn: Callable[[Grid], Grid], train: list[dict]) -> float:
    scores: list[float] = []
    try:
        for pair in train:
            pred = fn(pair["input"])
            scores.append(_pair_score(pred, pair["output"]))
    except Exception:
        return 0.0
    return sum(scores) / len(scores) if scores else 0.0


def _exact(fn: Callable[[Grid], Grid], train: list[dict]) -> bool:
    try:
        return all(fn(pair["input"]) == pair["output"] for pair in train)
    except Exception:
        return False


def _static_candidates() -> list[Candidate]:
    candidates = [Candidate(name, fn, 1) for name, fn in base_programs()]
    candidates.extend(
        [
            Candidate("crop_largest_object", crop_largest_object, 2),
            Candidate("crop_smallest_object", crop_smallest_object, 2),
            Candidate("keep_largest_object", keep_largest_object, 2),
            Candidate("keep_smallest_object", keep_smallest_object, 2),
            Candidate("fill_object_bounding_boxes", fill_object_bounding_boxes, 2),
            Candidate("complete_horizontal_symmetry", complete_horizontal_symmetry, 2),
            Candidate("complete_vertical_symmetry", complete_vertical_symmetry, 2),
            Candidate("complete_rotational_symmetry", complete_rotational_symmetry, 2),
            Candidate("outline_non_background", outline_non_background, 2),
            Candidate("sort_objects_by_size_row", sort_objects_by_size_row, 3),
        ]
    )
    return candidates


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

    uniform_color = infer_uniform_foreground_color(train)
    if uniform_color is not None:
        candidates.append(
            Candidate(
                f"recolor_foreground:{uniform_color}",
                lambda g, c=uniform_color: recolor_foreground(g, c),
                2,
            )
        )

    count_spec = infer_count_rendering(train)
    if count_spec is not None:
        orientation, color = count_spec
        candidates.append(
            Candidate(
                f"render_object_count:{orientation}:{color}",
                lambda g, o=orientation, c=color: render_object_count(g, o, c),
                3,
            )
        )

    return candidates


def _fingerprint_on_train(fn: Callable[[Grid], Grid], train: list[dict]) -> str | None:
    try:
        outs = [fn(pair["input"]) for pair in train]
        if not all(_valid_grid(g) for g in outs):
            return None
        return repr(outs)
    except Exception:
        return None


def _compose(a: Candidate, b: Candidate) -> Candidate:
    return Candidate(
        f"{a.name} -> {b.name}",
        lambda g, f=a.fn, h=b.fn: h(f(g)),
        a.complexity + b.complexity + 1,
    )


def generate_candidates(train: list[dict], *, max_composed: int = 220) -> list[Candidate]:
    """Generate primitive, data-derived, and bounded two-step DSL programs."""
    base = _static_candidates() + _derived_candidates(train)
    pool = list(base)

    # Keep composition search bounded. Simple geometric/object transforms compose
    # well and are much cheaper than unconstrained program enumeration.
    composed: list[Candidate] = []
    for a in base:
        for b in base:
            if a.name == "identity" or b.name == "identity":
                continue
            c = _compose(a, b)
            if _fingerprint_on_train(c.fn, train) is not None:
                composed.append(c)
            if len(composed) >= max_composed:
                break
        if len(composed) >= max_composed:
            break
    pool.extend(composed)

    # Deduplicate programs by their behavior on demonstrations, preferring lower complexity.
    best_by_behavior: dict[str, Candidate] = {}
    for c in pool:
        fp = _fingerprint_on_train(c.fn, train)
        if fp is None:
            continue
        prior = best_by_behavior.get(fp)
        if prior is None or (c.complexity, c.name) < (prior.complexity, prior.name):
            best_by_behavior[fp] = c

    ranked: list[Candidate] = []
    for c in best_by_behavior.values():
        s = _score(c.fn, train)
        ranked.append(Candidate(c.name, c.fn, c.complexity, s, _exact(c.fn, train)))

    ranked.sort(key=lambda c: (-int(c.exact_train), -c.train_score, c.complexity, c.name))
    return ranked


def synthesize(train: list[dict], *, top_k: int = 24) -> list[Candidate]:
    return generate_candidates(train)[:top_k]
