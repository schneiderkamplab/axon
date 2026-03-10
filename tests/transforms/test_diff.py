from importlib import import_module

import torch

from brainsurgery.providers import InMemoryStateDict

_module = import_module("brainsurgery.transforms.diff")
globals().update({name: getattr(_module, name) for name in dir(_module) if not name.startswith("_")})


class _Provider:
    def __init__(self) -> None:
        self._state_dicts = {
            "base": InMemoryStateDict(),
            "edited": InMemoryStateDict(),
        }

    def get_state_dict(self, model: str) -> InMemoryStateDict:
        return self._state_dicts[model]


def test_diff_compile_refs_mode() -> None:
    spec = DiffTransform().compile(
        {"left": "base::ln_f\\..*", "right": "edited::ln_f\\..*", "eps": 1e-6},
        default_model=None,
    )

    assert spec.mode == "refs"
    assert spec.left_ref.model == "base"
    assert spec.right_ref.model == "edited"
    assert spec.eps == 1e-6


def test_diff_compile_alias_mode() -> None:
    spec = DiffTransform().compile(
        {"mode": "aliases", "left_alias": "base", "right_alias": "edited"},
        default_model=None,
    )

    assert spec.mode == "aliases"
    assert spec.left_ref == TensorRef(model="base", expr=".*")
    assert spec.right_ref == TensorRef(model="edited", expr=".*")


def test_diff_compile_rejects_wrong_keys_for_alias_mode() -> None:
    try:
        DiffTransform().compile(
            {"mode": "aliases", "left": "base::.*", "right_alias": "edited"},
            default_model=None,
        )
    except DiffTransformError as exc:
        assert "wrong keys" not in str(exc)
        assert "unknown keys for this mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected alias-mode key validation error")


def test_diff_apply_reports_missing_and_differing(capsys) -> None:
    provider = _Provider()
    provider.get_state_dict("base")["same"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    provider.get_state_dict("edited")["same"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    provider.get_state_dict("base")["missing_on_right"] = torch.tensor([0.0], dtype=torch.float32)
    provider.get_state_dict("edited")["missing_on_left"] = torch.tensor([0.0], dtype=torch.float32)
    provider.get_state_dict("base")["shape"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    provider.get_state_dict("edited")["shape"] = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    provider.get_state_dict("base")["dtype"] = torch.tensor([1.0], dtype=torch.float32)
    provider.get_state_dict("edited")["dtype"] = torch.tensor([1.0], dtype=torch.float16)
    provider.get_state_dict("base")["value"] = torch.tensor([1.0], dtype=torch.float32)
    provider.get_state_dict("edited")["value"] = torch.tensor([1.5], dtype=torch.float32)

    spec = DiffTransform().compile({"left": "base::.*", "right": "edited::.*"}, default_model=None)
    result = DiffTransform().apply(spec, provider)

    assert result.count == 5
    output = capsys.readouterr().out
    assert "Missing on left:\n  - missing_on_left\n" in output
    assert "Missing on right:\n  - missing_on_right\n" in output
    assert "  - shape: shape (2,) != (1, 2)\n" in output
    assert "  - dtype: dtype torch.float32 != torch.float16\n" in output
    assert "  - value: values differ\n" in output


def test_diff_apply_alias_mode_uses_all_tensors_and_eps(capsys) -> None:
    provider = _Provider()
    provider.get_state_dict("base")["x"] = torch.tensor([1.0, 2.0], dtype=torch.float32)
    provider.get_state_dict("edited")["x"] = torch.tensor([1.0, 2.0005], dtype=torch.float32)

    spec = DiffTransform().compile(
        {"mode": "aliases", "left_alias": "base", "right_alias": "edited", "eps": 1e-3},
        default_model=None,
    )
    result = DiffTransform().apply(spec, provider)

    assert result.count == 0
    output = capsys.readouterr().out
    assert "Missing on left:\n  (none)\n" in output
    assert "Missing on right:\n  (none)\n" in output
    assert "Differing:\n  (none)\n" in output
    assert "No differences found.\n" in output
