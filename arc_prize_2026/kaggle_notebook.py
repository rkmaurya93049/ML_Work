"""Kaggle-ready ARC-AGI-2 submission entry point.

This entry point intentionally imports the same `arc2026` package used for local
evaluation so the leaderboard submission and Paper Track description stay in
sync. Attach the competition data plus this project (or a Kaggle Dataset made
from it) to the notebook before rerun.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def _find_src() -> Path:
    candidates = [
        Path.cwd() / "src",
        Path.cwd() / "arc_prize_2026" / "src",
        Path("/kaggle/working/arc_prize_2026/src"),
    ]
    # Kaggle datasets are mounted one directory below /kaggle/input/<dataset-name>.
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.glob("*/arc_prize_2026/src"))
        candidates.extend(input_root.glob("*/src"))

    for candidate in candidates:
        if (candidate / "arc2026" / "solver.py").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find arc_prize_2026/src. Attach this repository/project as a Kaggle "
        "Dataset or copy the project into /kaggle/working before running the notebook."
    )


SRC = _find_src()
sys.path.insert(0, str(SRC))

from arc2026.io import load_json, save_json, validate_submission  # noqa: E402
from arc2026.meta_ranker import learn_program_priors  # noqa: E402
from arc2026.solver import solve_challenges_with_trace  # noqa: E402


DATA = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2")
TRAIN = DATA / "arc-agi_training_challenges.json"
TEST = DATA / "arc-agi_test_challenges.json"
OUT = Path("/kaggle/working/submission.json")
TRACE = Path("/kaggle/working/solver_trace.json")

if not TEST.exists():
    # Some Kaggle mounts omit the intermediate `competitions` directory.
    alternatives = list(Path("/kaggle/input").glob("*/arc-agi_test_challenges.json"))
    if not alternatives:
        raise FileNotFoundError("ARC-AGI-2 test challenges were not found under /kaggle/input")
    DATA = alternatives[0].parent
    TRAIN = DATA / "arc-agi_training_challenges.json"
    TEST = DATA / "arc-agi_test_challenges.json"

challenges = load_json(TEST)
priors = learn_program_priors(load_json(TRAIN)) if TRAIN.exists() else None
submission, trace = solve_challenges_with_trace(challenges, priors=priors)
validate_submission(challenges, submission)
save_json(submission, OUT)
save_json(trace, TRACE)

print(json.dumps({
    "tasks": len(submission),
    "submission": str(OUT),
    "trace": str(TRACE),
    "learned_priors": bool(priors),
}, indent=2))
