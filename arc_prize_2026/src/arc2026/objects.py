from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, deque

import numpy as np

from .primitives import Grid, most_common_color


@dataclass(frozen=True)
class ArcObject:
    color: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        y0, _, y1, _ = self.bbox
        return y1 - y0 + 1

    @property
    def width(self) -> int:
        _, x0, _, x1 = self.bbox
        return x1 - x0 + 1


def connected_components(grid: Grid, *, diagonal: bool = False, ignore_background: bool = True) -> list[ArcObject]:
    a = np.asarray(grid, dtype=int)
    h, w = a.shape
    bg = most_common_color(grid)
    seen: set[tuple[int, int]] = set()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    objects: list[ArcObject] = []
    for y in range(h):
        for x in range(w):
            if (y, x) in seen:
                continue
            color = int(a[y, x])
            if ignore_background and color == bg:
                seen.add((y, x))
                continue

            q = deque([(y, x)])
            seen.add((y, x))
            cells: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                cells.append((cy, cx))
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and int(a[ny, nx]) == color:
                        seen.add((ny, nx))
                        q.append((ny, nx))

            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            objects.append(ArcObject(color, tuple(sorted(cells)), (min(ys), min(xs), max(ys), max(xs))))

    return objects


def crop_object(grid: Grid, obj: ArcObject, *, preserve_background: bool = False) -> Grid:
    a = np.asarray(grid, dtype=int)
    y0, x0, y1, x1 = obj.bbox
    if preserve_background:
        return a[y0:y1 + 1, x0:x1 + 1].tolist()

    bg = most_common_color(grid)
    out = np.full((obj.height, obj.width), bg, dtype=int)
    for y, x in obj.cells:
        out[y - y0, x - x0] = obj.color
    return out.tolist()


def crop_largest_object(grid: Grid) -> Grid:
    objs = connected_components(grid)
    if not objs:
        return [row[:] for row in grid]
    obj = max(objs, key=lambda o: (o.size, o.height * o.width))
    return crop_object(grid, obj)


def crop_smallest_object(grid: Grid) -> Grid:
    objs = connected_components(grid)
    if not objs:
        return [row[:] for row in grid]
    obj = min(objs, key=lambda o: (o.size, o.height * o.width))
    return crop_object(grid, obj)


def keep_largest_object(grid: Grid) -> Grid:
    objs = connected_components(grid)
    if not objs:
        return [row[:] for row in grid]
    bg = most_common_color(grid)
    h, w = len(grid), len(grid[0])
    out = [[bg for _ in range(w)] for _ in range(h)]
    obj = max(objs, key=lambda o: o.size)
    for y, x in obj.cells:
        out[y][x] = obj.color
    return out


def keep_smallest_object(grid: Grid) -> Grid:
    objs = connected_components(grid)
    if not objs:
        return [row[:] for row in grid]
    bg = most_common_color(grid)
    h, w = len(grid), len(grid[0])
    out = [[bg for _ in range(w)] for _ in range(h)]
    obj = min(objs, key=lambda o: o.size)
    for y, x in obj.cells:
        out[y][x] = obj.color
    return out


def object_count(grid: Grid) -> int:
    return len(connected_components(grid))


def color_count(grid: Grid) -> int:
    return len(Counter(v for row in grid for v in row))


def fill_object_bounding_boxes(grid: Grid) -> Grid:
    out = np.asarray(grid, dtype=int).copy()
    for obj in connected_components(grid):
        y0, x0, y1, x1 = obj.bbox
        out[y0:y1 + 1, x0:x1 + 1] = obj.color
    return out.tolist()


def object_signature(grid: Grid) -> tuple[tuple[int, int, int], ...]:
    objs = connected_components(grid)
    return tuple(sorted((o.color, o.size, o.height * o.width) for o in objs))
