from hypothesis import given
import hypothesis.strategies as st
import pytest

from zodipy import model_registry

ALL_MODELS = model_registry.models


@given(st.sampled_from(ALL_MODELS))
def test_get_model(model) -> None:
    model_registry.get_model(model)


def test_get_model_raises_error() -> None:
    with pytest.raises(ValueError):
        model_registry.get_model("metins_model")


def test_register_model() -> None:
    DIRBE_model = model_registry.get_model("DIRBE")
    DIRBE_model.T_0 += 200
    model_registry.register_model("metins_model", DIRBE_model)


def test_register_model_raises_error() -> None:
    with pytest.raises(ValueError):
        model_registry.register_model("DIRBE", model_registry.get_model("DIRBE"))
