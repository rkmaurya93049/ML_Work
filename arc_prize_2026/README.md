# ARC Prize 2026 — ARC-AGI-2 + Paper Track

Competition-ready project for the Kaggle ARC Prize 2026 Paper Track, backed by a real ARC-AGI-2 solver pipeline.

## Why ARC-AGI-2?
The Paper Track has no dataset of its own. It requires a writeup documenting a submission to ARC-AGI-2 or ARC-AGI-3. This project targets ARC-AGI-2 because it provides train/evaluation/test challenge JSON files and a deterministic `submission.json` interface.

## Project layout

```text
arc_prize_2026/
├── README.md
├── requirements.txt
├── kaggle_notebook.py
├── src/arc2026/
│   ├── __init__.py
│   ├── io.py
│   ├── primitives.py
│   ├── synthesizer.py
│   ├── solver.py
│   └── evaluation.py
├── scripts/
│   ├── evaluate.py
│   └── make_submission.py
└── paper/
    ├── writeup_draft.md
    └── submission_checklist.md
```

## Method
The baseline uses evidence-driven symbolic program synthesis. For each task it searches a compact library of interpretable grid programs and keeps only transformations that agree with all demonstrations. Candidate families include identity, rotations, reflections, transpose, background-aware bounding-box crop, nearest-neighbor scaling, tiling, and task-inferred color remapping. The top two consistent programs become Kaggle `attempt_1` and `attempt_2`.

This is intentionally a reproducible baseline, not a claim of state-of-the-art performance. It is designed to be extended with object reasoning, learned proposal models, test-time search, or LLM-guided program generation.

## Kaggle data
Attach the ARC Prize 2026 ARC-AGI-2 competition dataset to your notebook. Kaggle normally exposes it under:

```text
/kaggle/input/competitions/arc-prize-2026-arc-agi-2/
```

Expected files include `arc-agi_training-challenges.json`, `arc-agi_training-solutions.json`, `arc-agi_evaluation-challenges.json`, `arc-agi_evaluation-solutions.json`, `arc-agi_test-challenges.json`, and `sample_submission.json`.

## Local evaluation

```bash
cd arc_prize_2026
pip install -r requirements.txt
PYTHONPATH=src python scripts/evaluate.py \
  --challenges /path/to/arc-agi_evaluation_challenges.json \
  --solutions /path/to/arc-agi_evaluation_solutions.json
```

## Generate submission

```bash
PYTHONPATH=src python scripts/make_submission.py \
  --challenges /path/to/arc-agi_test_challenges.json \
  --output submission.json
```

The script guarantees every task ID is present and every test item contains both `attempt_1` and `attempt_2`.

## Kaggle notebook
`kaggle_notebook.py` is a single-file, notebook-friendly entry point. Paste it into a Kaggle Notebook or upload it as a script, attach the competition data, run all cells/code, and confirm that `/kaggle/working/submission.json` is created.

## Paper Track
The `paper/` directory contains a writeup draft aligned with the six judging dimensions: Accuracy, Universality, Progress, Theory, Completeness, and Novelty. Replace placeholders with your actual leaderboard score and submission ID before final submission.

## Important
Do not claim leaderboard accuracy that has not been measured. The paper-track writeup must correspond to a real ARC-AGI-2 or ARC-AGI-3 Kaggle submission.