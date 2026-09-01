from __future__ import annotations

from collections import Counter

import numpy as np

from .objects import connected_components, object_count
from .primitives import Grid, most_common_color


def complete_horizontal_symmetry(grid: Grid) -> Grid:
    a = np.asarray(grid, dtype=int).copy()
    bg = most_common_color(grid)
    mirror = np.fliplr(a)
    mask = a == bg
    a[mask & (mirror != bg)] = mirror[mask & (mirror != bg)]
    return a.tolist()


def complete_vertical_symmetry(grid: Grid) -> Grid:
    a = np.asarray(grid, dtype=int).copy()
    bg = most_common_color(grid)
    mirror = np.flipud(a)
    mask = a == bg
    a[mask & (mirror != bg)] = mirror[mask & (mirror != bg)]
    return a.tolist()


def complete_rotational_symmetry(grid: Grid) -> Grid:
    a = np.asarray(grid, dtype=int).copy()
    bg = most_common_color(grid)
    mirror = np.rot90(a, 2)
    mask = a == bg
    a[mask & (mirror != bg)] = mirror[mask & (mirror != bg)]
    return a.tolist()


def outline_non_background(grid: Grid) -> Grid:
    a = np.asarray(grid, dtype=int)
    bg = most_common_color(grid)
    out = a.copy()
    h, w = a.shape
    for y in range(h):
        for x in range(w):
            if a[y, x] == bg:
                continue
            neighbors = []
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                neighbors.append(0 <= ny < h and 0 <= nx < w and a[ny, nx] != bg)
            if all(neighbors):
                out[y, x] = bg
    return out.tolist()


def infer_uniform_foreground_color(train: list[dict]) -> int | None:
    colors: list[int] = []
    for pair in train:
        out = pair["output"]
        bg = most_common_color(out)
        fg = sorted({v for row in out for v in row if v != bg})
        if len(fg) != 1:
            return None
        colors.append(fg[0])
    return colors[0] if colors and len(set(colors)) == 1 else None


def recolor_foreground(grid: Grid, color: int) -> Grid:
    bg = most_common_color(grid)
    return [[v if v == bg else color for v in row] for row in grid]


def infer_count_rendering(train: list[dict]):
    """Infer simple count-to-line/bar outputs.

    Supported patterns:
    - N objects -> 1xN filled row
    - N objects -> Nx1 filled column
    The fill color must be consistent across demonstrations.
    """
    orientations: list[str] = []
    colors: list[int] = []
    for pair in train:
        n = object_count(pair["input"])
        out = pair["output"]
        h, w = len(out), len(out[0])
        if h == 1 and w == n:
            orientations.append("row")
        elif w == 1 and h == n:
            orientations.append("col")
        else:
            return None
        vals = [v for row in out for v in row]
        if len(set(vals)) != 1:
            return None
        colors.append(vals[0])
    if not orientations or len(set(orientations)) != 1 or len(set(colors)) != 1:
        return None
    return orientations[0], colors[0]


def render_object_count(grid: Grid, orientation: str, color: int) -> Grid:
    n = max(1, object_count(grid))
    if orientation == "row":
        return [[color] * n]
    return [[color] for _ in range(n)]


def sort_objects_by_size_row(grid: Grid) -> Grid:
    """Render object colors as a 1-row sequence ordered by component size.

    This is intentionally conservative and only useful when demonstrations prove
    that this abstraction is the target.
    """
    objs = connected_components(grid)
    if not objs:
        return [[most_common_color(grid)]]
    return [[o.color for o in sorted(objs, key=lambda o: (o.size, o.color))]]


def dominant_non_background_color(grid: Grid) -> int:
    bg = most_common_color(grid)
    counts = Counter(v for row in grid for v in row if v != bg)
    return counts.most_common(1)[0][0] if counts else bg
