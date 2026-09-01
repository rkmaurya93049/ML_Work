from __future__ import annotations

from collections import Counter
from typing import Callable
import numpy as np

Grid = list[list[int]]
Program = tuple[str, Callable[[Grid], Grid]]


def _arr(g: Grid) -> np.ndarray:
    return np.asarray(g, dtype=int)


def identity(g: Grid) -> Grid:
    return _arr(g).tolist()


def rot90(g: Grid) -> Grid:
    return np.rot90(_arr(g), 1).tolist()


def rot180(g: Grid) -> Grid:
    return np.rot90(_arr(g), 2).tolist()


def rot270(g: Grid) -> Grid:
    return np.rot90(_arr(g), 3).tolist()


def flip_h(g: Grid) -> Grid:
    return np.fliplr(_arr(g)).tolist()


def flip_v(g: Grid) -> Grid:
    return np.flipud(_arr(g)).tolist()


def transpose(g: Grid) -> Grid:
    return _arr(g).T.tolist()


def anti_transpose(g: Grid) -> Grid:
    return np.fliplr(np.flipud(_arr(g))).T.tolist()


def most_common_color(g: Grid) -> int:
    return Counter(v for row in g for v in row).most_common(1)[0][0]


def crop_non_background(g: Grid) -> Grid:
    a = _arr(g)
    bg = most_common_color(g)
    ys, xs = np.where(a != bg)
    if len(ys) == 0:
        return g
    return a[ys.min():ys.max()+1, xs.min():xs.max()+1].tolist()


def recolor(g: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in g]


def upscale(g: Grid, ry: int, rx: int) -> Grid:
    return np.repeat(np.repeat(_arr(g), ry, axis=0), rx, axis=1).tolist()


def tile(g: Grid, ry: int, rx: int) -> Grid:
    return np.tile(_arr(g), (ry, rx)).tolist()


def infer_recolor_mapping(inp: Grid, out: Grid) -> dict[int, int] | None:
    a, b = _arr(inp), _arr(out)
    if a.shape != b.shape:
        return None
    mapping: dict[int, int] = {}
    for x, y in zip(a.ravel(), b.ravel()):
        x, y = int(x), int(y)
        if x in mapping and mapping[x] != y:
            return None
        mapping[x] = y
    return mapping


def base_programs() -> list[Program]:
    return [
        ("identity", identity),
        ("rot90", rot90),
        ("rot180", rot180),
        ("rot270", rot270),
        ("flip_h", flip_h),
        ("flip_v", flip_v),
        ("transpose", transpose),
        ("anti_transpose", anti_transpose),
        ("crop_non_background", crop_non_background),
    ]
