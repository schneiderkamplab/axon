from importlib import import_module

_module = import_module("brainsurgery.transforms.concat")
ConcatTransform = _module.ConcatTransform
ConcatTransformError = _module.ConcatTransformError


def test_concat_compile_requires_list() -> None:
    try:
        ConcatTransform().compile({"from": "a::x", "to": "a::y"}, default_model=None)
    except ConcatTransformError as exc:
        assert "list of at least two" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected concat.from validation error")


def test_concat_compile_rejects_sliced_to() -> None:
    try:
        ConcatTransform().compile({"from": ["a::x", "a::y"], "to": "a::z::[:]"}, default_model=None)
    except ConcatTransformError as exc:
        assert "must not be sliced" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected concat.to sliced validation error")
