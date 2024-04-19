import hypothesis.strategies as st
import pytest
from hypothesis import given

from zodipy import model_registry

ALL_MODELS = model_registry.models


@given(st.sampled_from(ALL_MODELS))
def test_get_model(model: str) -> None:
    """Tests that the model can be retrieved."""
    model_registry.get_model(model)


def test_get_model_raises_error() -> None:
    """Tests that an error is raised when the model is not found."""
    with pytest.raises(ValueError):
        model_registry.get_model("metins_model")


def test_register_model() -> None:
    """Tests that a model can be registered."""
    DIRBE_model = model_registry.get_model("DIRBE")
    DIRBE_model.T_0 += 250
    model_registry.register_model("metins_model", DIRBE_model)


def test_register_model_raises_error() -> None:
    """Tests that an error is raised when the model is already registered."""
    with pytest.raises(ValueError):
        model_registry.register_model("DIRBE", model_registry.get_model("DIRBE"))
