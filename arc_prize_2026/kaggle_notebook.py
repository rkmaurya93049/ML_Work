"""Kaggle-ready ARC-AGI-2 submission entry point.

Attach the ARC Prize 2026 ARC-AGI-2 competition data and run this file in a
Kaggle Notebook. It writes /kaggle/working/submission.json.
"""

import json
from collections import Counter
from pathlib import Path
import numpy as np

DATA = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
TEST = DATA / "arc-agi_test_challenges.json"
OUT = Path("/kaggle/working/submission.json")


def arr(g): return np.asarray(g, dtype=int)
def identity(g): return arr(g).tolist()
def crop(g):
    a = arr(g); bg = Counter(a.ravel().tolist()).most_common(1)[0][0]
    ys, xs = np.where(a != bg)
    return g if len(ys) == 0 else a[ys.min():ys.max()+1, xs.min():xs.max()+1].tolist()

BASE = [
    ("identity", identity),
    ("rot90", lambda g: np.rot90(arr(g), 1).tolist()),
    ("rot180", lambda g: np.rot90(arr(g), 2).tolist()),
    ("rot270", lambda g: np.rot90(arr(g), 3).tolist()),
    ("flip_h", lambda g: np.fliplr(arr(g)).tolist()),
    ("flip_v", lambda g: np.flipud(arr(g)).tolist()),
    ("transpose", lambda g: arr(g).T.tolist()),
    ("crop", crop),
]


def recolor_mapping(inp, out):
    a, b = arr(inp), arr(out)
    if a.shape != b.shape: return None
    m = {}
    for x, y in zip(a.ravel(), b.ravel()):
        x, y = int(x), int(y)
        if x in m and m[x] != y: return None
        m[x] = y
    return m


def candidates(task):
    train = task["train"]
    pool = list(BASE)
    m = recolor_mapping(train[0]["input"], train[0]["output"])
    if m is not None:
        pool.append(("recolor", lambda g, m=m: [[m.get(v, v) for v in r] for r in g]))
    good = []
    for name, fn in pool:
        try:
            if all(fn(p["input"]) == p["output"] for p in train): good.append((name, fn))
        except Exception:
            pass
    return good


def solve(task):
    good = candidates(task)
    p1 = good[0][1] if good else identity
    p2 = good[1][1] if len(good) > 1 else crop
    result = []
    for t in task["test"]:
        g = t["input"]
        try: a1 = p1(g)
        except Exception: a1 = g
        try: a2 = p2(g)
        except Exception: a2 = g
        result.append({"attempt_1": a1, "attempt_2": a2})
    return result


with TEST.open("r", encoding="utf-8") as f:
    challenges = json.load(f)
submission = {task_id: solve(task) for task_id, task in challenges.items()}
with OUT.open("w", encoding="utf-8") as f:
    json.dump(submission, f, separators=(",", ":"))
print(f"Created {OUT} for {len(submission)} tasks")
