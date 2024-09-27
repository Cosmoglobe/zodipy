import random
import warnings

import numpy as np
import pytest
from astropy import coordinates, time, units

from zodipy import Model, grid_number_density, model_registry


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
    obstime = time.Time("2021-01-01T00:00:00")
    skycoord = coordinates.SkyCoord(20, 30, unit=units.deg, obstime=obstime)
    emission_before = model.evaluate(skycoord)
    parameters = model.get_parameters()
    comp = random.choice(list(parameters["comps"].keys()))
    parameter = random.choice(list(parameters["comps"][comp]))
    parameters["comps"][comp][parameter] = random.random()
    model.update_parameters(parameters)
    emission_after = model.evaluate(skycoord)
    assert not np.allclose(emission_before, emission_after)


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


def test_grid_number_density() -> None:
    """Test the number density gridding function."""
    x = np.linspace(0, 5, 100) * units.AU
    y = np.linspace(0, 5, 100) * units.AU
    z = np.linspace(0, 2, 100) * units.AU
    obstime = time.Time("2022-01-01")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = grid_number_density(x, y, z, obstime=obstime)
    assert grid.sum(axis=0).shape == (100, 100, 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = grid_number_density(x, y, z, obstime=obstime, model="rrm-experimental")

    assert grid.sum(axis=0).shape == (100, 100, 100)

    model = model_registry.get_model("rrm-experimental")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid = grid_number_density(x, y, z, obstime=obstime, model=model)

    assert grid.sum(axis=0).shape == (100, 100, 100)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError):
            grid = grid_number_density(x, y, z, obstime=obstime, model=2)
