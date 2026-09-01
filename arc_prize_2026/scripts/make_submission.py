from __future__ import annotations

import argparse

from arc2026.io import load_json, save_json, validate_submission
from arc2026.meta_ranker import load_program_priors
from arc2026.solver import solve_challenges_with_trace


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenges", required=True)
    p.add_argument("--output", default="submission.json")
    p.add_argument("--priors", default=None, help="Optional program_priors.json learned from ARC training tasks")
    p.add_argument("--trace", default=None, help="Optional JSON file describing selected symbolic programs")
    args = p.parse_args()

    challenges = load_json(args.challenges)
    priors = load_program_priors(args.priors) if args.priors else None
    submission, traces = solve_challenges_with_trace(challenges, priors=priors)
    validate_submission(challenges, submission)
    save_json(submission, args.output)
    if args.trace:
        save_json(traces, args.trace)
    print(f"Wrote {args.output} with {len(submission)} tasks")


if __name__ == "__main__":
    main()
