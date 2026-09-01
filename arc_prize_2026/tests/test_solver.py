from arc2026.objects import connected_components, crop_largest_object
from arc2026.relations import complete_horizontal_symmetry, render_object_count
from arc2026.solver import solve_task


def test_connected_components_and_largest_crop():
    grid = [
        [0, 1, 0, 2, 2],
        [0, 0, 0, 2, 2],
        [3, 0, 0, 0, 0],
    ]
    objects = connected_components(grid)
    assert sorted(o.size for o in objects) == [1, 1, 4]
    assert crop_largest_object(grid) == [[2, 2], [2, 2]]


def test_symmetry_completion():
    grid = [[1, 0, 0, 0], [0, 2, 0, 0]]
    assert complete_horizontal_symmetry(grid) == [[1, 0, 0, 1], [0, 2, 2, 0]]


def test_count_renderer():
    grid = [[1, 0, 2], [0, 0, 0], [3, 0, 3]]
    assert render_object_count(grid, "row", 7) == [[7, 7, 7, 7]]


def test_solver_finds_composition_crop_then_flip():
    task = {
        "train": [
            {
                "input": [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0]],
                "output": [[2, 1], [4, 3]],
            },
            {
                "input": [[0, 5, 6, 0], [0, 7, 8, 0], [0, 0, 0, 0]],
                "output": [[6, 5], [8, 7]],
            },
        ],
        "test": [{"input": [[0, 0, 0], [9, 1, 0], [2, 3, 0]]}],
    }
    pred = solve_task(task)[0]
    assert pred["attempt_1"] == [[1, 9], [3, 2]] or pred["attempt_2"] == [[1, 9], [3, 2]]
