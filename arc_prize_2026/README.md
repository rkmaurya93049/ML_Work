# ARC Prize 2026 — ARC-AGI-2 + Paper Track

Competition-ready ARC-AGI-2 solver and Paper Track research package.

The Paper Track has no independent task dataset; a valid paper must document a real ARC-AGI-2 or ARC-AGI-3 submission. This project targets ARC-AGI-2 and keeps the solver, evaluation code, Kaggle entry point, traces, and paper draft in one reproducible repository.

## What is implemented

The solver is now a hybrid, evidence-driven symbolic system rather than a one-transform baseline.

### 1. Grid DSL

Core primitives include:

- identity
- 90/180/270 degree rotation
- horizontal and vertical reflection
- transpose and anti-transpose
- background-aware crop
- nearest-neighbour upscaling
- tiling
- inferred color remapping

### 2. Object-centric reasoning

`src/arc2026/objects.py` adds:

- 4- or 8-connected component extraction
- object color, size and bounding-box metadata
- largest/smallest object selection
- object-specific cropping
- keep-largest / keep-smallest filtering
- object counting
- bounding-box filling
- object signatures for structural reasoning

### 3. Relational and pattern reasoning

`src/arc2026/relations.py` adds:

- horizontal symmetry completion
- vertical symmetry completion
- 180-degree rotational symmetry completion
- foreground outlining
- uniform foreground recoloring
- object-count-to-row / object-count-to-column rendering
- size-ordered object-color rendering

### 4. Compositional program synthesis

`src/arc2026/synthesizer.py` searches primitive and task-derived programs and also performs bounded two-step composition. Candidate programs are executed on every demonstration, scored for exact or partial agreement, deduplicated by demonstrated behavior, and ranked by:

1. exact demonstration consistency;
2. training-pair similarity;
3. program complexity;
4. deterministic name tie-break.

This enables programs such as:

```text
crop_non_background -> flip_h
crop_largest_object -> recolor_foreground
complete_horizontal_symmetry -> crop_non_background
```

without enumerating an unbounded program space.

### 5. Diverse test-time hypotheses

ARC-AGI-2 allows two predictions for each test output. `solver.py` therefore keeps two distinct predicted grids whenever possible instead of duplicating a single guess. It also writes an auditable trace recording the selected programs, their demonstration scores, and whether they fit every demonstration exactly.

### 6. Learned symbolic priors

`meta_ranker.py` learns empirical frequencies of successful symbolic programs from the public ARC training challenges. These priors are used only as a small tie-breaker behind demonstration evidence, so they never override a better task-local fit.

Build priors with:

```bash
PYTHONPATH=src python scripts/build_program_priors.py \
  --challenges /path/to/arc-agi_training_challenges.json \
  --output program_priors.json
```

### 7. Neural-symbolic / LLM proposal bridge

`hybrid.py` defines a safe `SymbolicProposal` interface. A future LLM or learned proposal model may suggest sequences of named DSL operations, but it cannot execute arbitrary Python. Proposed programs are compiled only from the approved operation registry and accepted only when they reproduce all task demonstrations.

This provides a neural-symbolic extension point without making the baseline dependent on an external API or internet access during Kaggle reruns.

## Project layout

```text
arc_prize_2026/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── kaggle_notebook.py
├── src/arc2026/
│   ├── __init__.py
│   ├── io.py
│   ├── primitives.py
│   ├── objects.py
│   ├── relations.py
│   ├── synthesizer.py
│   ├── meta_ranker.py
│   ├── hybrid.py
│   ├── solver.py
│   └── evaluation.py
├── scripts/
│   ├── build_program_priors.py
│   ├── evaluate.py
│   └── make_submission.py
├── tests/
│   └── test_solver.py
└── paper/
    ├── writeup_draft.md
    └── submission_checklist.md
```

## Data contract

ARC grids are rectangular integer matrices using values 0-9. The competition reruns the notebook against hidden test challenges. Every test output must contain both:

```json
{
  "attempt_1": [[0]],
  "attempt_2": [[0]]
}
```

The project validates this structure before writing a submission.

## Install

```bash
cd arc_prize_2026
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

## Run tests

```bash
pytest
```

The tests include connected-component extraction, symmetry completion, counting, and a synthetic two-step composition task.

## Local evaluation

Without learned priors:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --challenges /path/to/arc-agi_evaluation_challenges.json \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --trace evaluation_trace.json
```

With learned priors:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --challenges /path/to/arc-agi_evaluation_challenges.json \
  --solutions /path/to/arc-agi_evaluation_solutions.json \
  --priors program_priors.json \
  --trace evaluation_trace.json
```

Record both scores. Keep the prior only if it improves or preserves evaluation performance.

## Generate a Kaggle submission

```bash
PYTHONPATH=src python scripts/make_submission.py \
  --challenges /path/to/arc-agi_test_challenges.json \
  --priors program_priors.json \
  --trace solver_trace.json \
  --output submission.json
```

The generated JSON is validated before it is saved.

## Kaggle notebook setup

`kaggle_notebook.py` now uses the exact same `arc2026` package as local evaluation; the old duplicated lightweight solver has been removed from the competition path.

For a Kaggle rerun:

1. create a Kaggle Dataset from this `arc_prize_2026` project, or otherwise attach the project source to the notebook;
2. attach the ARC Prize 2026 ARC-AGI-2 competition data;
3. run `kaggle_notebook.py`;
4. verify that `/kaggle/working/submission.json` exists;
5. keep `/kaggle/working/solver_trace.json` as experiment evidence.

The notebook learns program priors from the public training challenges when that file is mounted, then solves the hidden test challenges without using hidden solutions.

## Paper Track workflow

The paper draft is aligned with the six judging dimensions:

- Accuracy
- Universality
- Progress
- Theory
- Completeness
- Novelty

Do not fill leaderboard score, submission ID, local accuracy, or solved-task counts until they are measured from the actual final system.

A defensible experiment sequence is:

```text
v0: primitive single-step baseline
v1: + object reasoning
v2: + relational/pattern transforms
v3: + compositional search
v4: + learned program prior
v5: + optional externally proposed safe DSL programs
```

For the paper, report ablations between these versions rather than presenting only one final number.

## Current limitations

The system is materially stronger than the original baseline, but it is not a claim of prize-winning performance. Important unsolved families still include richer topology, path construction, iterative simulation, long multi-object correspondences, arithmetic-like spatial relations, and DSL concepts not yet represented in the search library.

The correct next optimization loop is empirical: run the public evaluation set, inspect traces for failures, add reusable abstractions for recurring failure classes, re-evaluate, and only keep transformations that improve generalization.

## Reproducibility rule

Never hard-code leaderboard answers or report unmeasured scores. The Paper Track writeup must correspond to a real ARC-AGI-2 or ARC-AGI-3 submission and the published artifacts must describe the code that generated that submission.
