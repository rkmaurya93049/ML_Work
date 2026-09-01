from __future__ import annotations

import argparse

from arc2026.io import load_json
from arc2026.meta_ranker import learn_program_priors, save_program_priors


def main() -> None:
    p = argparse.ArgumentParser(description="Learn symbolic program priors from ARC training tasks.")
    p.add_argument("--challenges", required=True)
    p.add_argument("--output", default="program_priors.json")
    args = p.parse_args()

    challenges = load_json(args.challenges)
    priors = learn_program_priors(challenges)
    save_program_priors(priors, args.output)
    print(f"Wrote {args.output} with {len(priors)} learned program priors")


if __name__ == "__main__":
    main()
