from __future__ import annotations

import argparse
from arc2026.io import load_json, save_json, validate_submission
from arc2026.solver import solve_challenges


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", required=True)
    p.add_argument("--output", default="submission.json")
    args = p.parse_args()

    challenges = load_json(args.challenges)
    submission = solve_challenges(challenges)
    validate_submission(challenges, submission)
    save_json(submission, args.output)
    print(f"Wrote {args.output} with {len(submission)} tasks")


if __name__ == "__main__":
    main()
