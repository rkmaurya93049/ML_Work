from __future__ import annotations

import argparse
import json
from arc2026.io import load_json, validate_submission
from arc2026.solver import solve_challenges
from arc2026.evaluation import score_submission


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", required=True)
    p.add_argument("--solutions", required=True)
    args = p.parse_args()

    challenges = load_json(args.challenges)
    solutions = load_json(args.solutions)
    submission = solve_challenges(challenges)
    validate_submission(challenges, submission)
    print(json.dumps(score_submission(submission, solutions), indent=2))


if __name__ == "__main__":
    main()
