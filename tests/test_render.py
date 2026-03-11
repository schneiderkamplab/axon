from __future__ import annotations

import torch

from brainsurgery.utils.render import _shape_only, render_tree, summarize_tensor


def test_summarize_tensor_supports_shape_full_and_stats_modes() -> None:
    tensor = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    assert summarize_tensor(tensor, verbosity="shape") == {"shape": [2, 2]}
    assert summarize_tensor(tensor, verbosity="full")["values"] == [[1.0, 3.0], [5.0, 7.0]]

    stats = summarize_tensor(tensor, verbosity="stats")
    assert stats == {"shape": [2, 2], "min": 1.0, "max": 7.0, "mean": 4.0}


def test_shape_only_and_render_tree_handle_nested_lists() -> None:
    repeated = {"shape": [2, 2], "min": 0.0, "max": 1.0, "mean": 0.5}
    tree = {
        "layer": [
            repeated,
            repeated,
            None,
        ]
    }

    assert _shape_only(tree) == {"layer": [{"shape": [2, 2]}, {"shape": [2, 2]}, None]}

    rendered = render_tree(tree, compact=True)
    assert "[0-1]" in rendered
    assert "shape=[2, 2] min=0" in rendered
