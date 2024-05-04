import random

import numpy as np
import pytest
from astropy import units

from zodipy import Model, model_registry


def test_x_input() -> None:
    """Test that the x attribute is set correctly."""
    with pytest.raises(units.UnitConversionError):
        Model(x=20 * units.s)

    with pytest.raises(TypeError):
        Model(x=20)

    x = np.linspace(20, 25, 10) * units.micron
    with pytest.raises(ValueError):
        Model(x)


def test_weights_input() -> None:
    """Test weights."""
    bandpass = np.random.rand(10)
    x = np.linspace(20, 25, 15) * units.micron

    with pytest.raises(ValueError):
        Model(20 * units.micron, weights=bandpass)

    with pytest.raises(ValueError):
        Model(x, weights=bandpass, name="dirbe")


def test_extrapolate_input() -> None:
    """Test extrapolate."""
    with pytest.raises(ValueError):
        Model(0.5 * units.micron, name="dirbe", extrapolate=False)

    with pytest.raises(ValueError):
        Model(400 * units.micron, name="dirbe", extrapolate=False)

    Model(20 * units.micron, extrapolate=True)

    x = np.linspace(40, 125, 10) * units.GHz
    bandpass = np.random.rand(10)
    with pytest.raises(ValueError):
        Model(x, weights=bandpass, name="planck18", extrapolate=False)

    Model(x, weights=bandpass, name="planck18", extrapolate=True)


def test_get_parameters() -> None:
    """Tests that the parameters are returned as a dictionary."""
    model = Model(20 * units.micron, name="dirbe")
    assert isinstance(model.get_parameters(), dict)


def test_update_model() -> None:
    """Tests that the model can be updated."""
    model = Model(20 * units.micron, name="dirbe")

    parameters = model.get_parameters()
    comp = random.choice(list(parameters["comps"].keys()))
    parameter = random.choice(list(parameters["comps"][comp]))
    parameters["comps"][comp][parameter] = random.random()
    model.update_parameters(parameters)


def test_get_model_raises_error() -> None:
    """Tests that an error is raised when the model is not found."""
    with pytest.raises(ValueError):
        model_registry.get_model("invalid")


def test_register_model() -> None:
    """Tests that a model can be registered."""
    DIRBE_model = model_registry.get_model("DIRBE")
    DIRBE_model.T_0 += 250
    model_registry.register_model("updated model", DIRBE_model)


def test_register_model_raises_error() -> None:
    """Tests that an error is raised when the model is already registered."""
    with pytest.raises(ValueError):
        model_registry.register_model("DIRBE", model_registry.get_model("DIRBE"))
