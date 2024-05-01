import random

from hypothesis import given

from zodipy.zodipy import Model

from ._strategies import zodipy_models


@given(zodipy_models())
def test_get_parameters(model: Model) -> None:
    """Tests that the parameters are returned as a dictionary."""
    assert isinstance(model.get_parameters(), dict)


@given(zodipy_models())
def test_update_model(model: Model) -> None:
    """Tests that the model can be updated."""
    parameters = model.get_parameters()
    comp = random.choice(list(parameters["comps"].keys()))
    parameter = random.choice(list(parameters["comps"][comp]))
    parameters["comps"][comp][parameter] = random.random()
    model.update_parameters(parameters)
