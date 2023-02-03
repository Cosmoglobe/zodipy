import random

from hypothesis import given

from zodipy.zodipy import Zodipy
from ._strategies import model


@given(model())
def test_get_parameterse(model: Zodipy) -> None:
    assert isinstance(model.get_parameters(), dict)


@given(model())
def test_update(model: Zodipy) -> None:
    parameters = model.get_parameters()
    comp = random.choice(list(parameters["comps"].keys()))
    parameter = random.choice(list(parameters["comps"][comp]))
    parameters["comps"][comp][parameter] = random.random()
    model.update_parameters(parameters)
