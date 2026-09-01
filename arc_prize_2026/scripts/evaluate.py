from __future__ import annotations

import argparse
import json

from arc2026.evaluation import score_submission
from arc2026.io import load_json, save_json, validate_submission
from arc2026.meta_ranker import load_program_priors
from arc2026.solver import solve_challenges_with_trace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", required=True)
    p.add_argument("--solutions", required=True)
    p.add_argument("--priors", default=None, help="Optional program_priors.json learned from training tasks")
    p.add_argument("--trace", default=None, help="Optional path for per-task program-selection diagnostics")
    args = p.parse_args()

    challenges = load_json(args.challenges)
    solutions = load_json(args.solutions)
    priors = load_program_priors(args.priors) if args.priors else None

    submission, traces = solve_challenges_with_trace(challenges, priors=priors)
    validate_submission(challenges, submission)
    metrics = score_submission(submission, solutions)

    if args.trace:
        save_json(traces, args.trace)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
