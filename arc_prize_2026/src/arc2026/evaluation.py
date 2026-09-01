from __future__ import annotations


def score_submission(submission: dict, solutions: dict) -> dict[str, float | int]:
    correct = 0
    total = 0
    solved_tasks = 0
    for task_id, truth_list in solutions.items():
        task_correct = 0
        preds = submission[task_id]
        for i, truth in enumerate(truth_list):
            total += 1
            hit = preds[i]["attempt_1"] == truth or preds[i]["attempt_2"] == truth
            if hit:
                correct += 1
                task_correct += 1
        if task_correct == len(truth_list):
            solved_tasks += 1
    return {
        "correct_outputs": correct,
        "total_outputs": total,
        "accuracy": correct / total if total else 0.0,
        "fully_solved_tasks": solved_tasks,
        "total_tasks": len(solutions),
    }
