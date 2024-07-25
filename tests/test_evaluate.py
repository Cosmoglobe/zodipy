from __future__ import annotations

import numpy as np
import pytest
from astropy import coordinates as coords
from astropy import time, units
from astropy.coordinates import SkyCoord
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import DataObject, data
from numpy.testing import assert_array_equal

from zodipy import Model

from .dirbe_tabulated import DAYS, DIRBE_START_DAY, LAT, LON, TABULATED_DIRBE_EMISSION
from .strategies import (
    models,
    obs_inst,
    obs_tod,
    sky_coord_inst,
    sky_coord_tod,
)

np.random.seed(42)

TEST_TIME = time.Time("2021-01-01T00:00:00", scale="utc")
TEST_SKY_COORD = SkyCoord(20, 30, unit=units.deg, obstime=TEST_TIME)
TEST_SKY_COORD_GALACTIC = SkyCoord(20, 30, unit=units.deg, obstime=TEST_TIME, frame="galactic")
test_model = Model(x=20 * units.micron)


def test_compare_to_dirbe_idl() -> None:
    """Tests that ZodiPy reproduces the DIRBE software.

    Zodipy should be able to reproduce the tabulated emission from the DIRBE Zoidacal Light
    Prediction Software with a maximum difference of 0.1%.
    """
    for frequency, tabulated_emission in TABULATED_DIRBE_EMISSION.items():
        model = Model(x=frequency * units.micron, name="dirbe")
        for idx, (day, lon, lat) in enumerate(zip(DAYS, LON, LAT)):
            obstime = DIRBE_START_DAY + time.TimeDelta(day - 1, format="jd")
            coord = coords.SkyCoord(
                lon,
                lat,
                unit=units.deg,
                frame=coords.BarycentricMeanEcliptic,
                obstime=obstime,
            )
            emission = model.evaluate(coord)
            assert emission.value == pytest.approx(tabulated_emission[idx], rel=0.01)


@settings(deadline=None)
@given(models(), sky_coord_inst(), obs_inst())
def test_evaluate_inst(
    model: Model,
    sky_coord: SkyCoord,
    obspos: units.Quantity,
) -> None:
    """Test that the evaluate function works for valid user input."""
    emission = model.evaluate(sky_coord, obspos=obspos)
    assert emission.size == sky_coord.size
    assert isinstance(emission, units.Quantity)
    assert emission.unit == units.MJy / units.sr


@settings(deadline=None, suppress_health_check=[HealthCheck.data_too_large])
@given(data(), models(), sky_coord_tod())
def test_evaluate_tod(
    data: DataObject,
    model: Model,
    sky_coord: SkyCoord,
) -> None:
    """Test that the evaluate function works for valid user input."""
    obspos = data.draw(obs_tod(sky_coord.size))
    emission = model.evaluate(sky_coord, obspos=obspos)
    assert emission.size == sky_coord.size
    assert isinstance(emission, units.Quantity)
    assert emission.unit == units.MJy / units.sr


def test_invalid_sky_coord() -> None:
    """Test that the evaluate function raises an error for invalid sky_coord."""
    with pytest.raises(TypeError):
        test_model.evaluate(20)

    # test that obstime is set
    with pytest.raises(ValueError):
        skycoord = SkyCoord(20, 30, unit=units.deg)
        test_model.evaluate(skycoord)

    # test that it crashed for multiple obstime
    with pytest.raises(ValueError):
        skycoord = SkyCoord(
            20,
            30,
            unit=units.deg,
            obstime=time.Time(["2021-01-01T00:00:00", "2021-01-01T00:02:00"]),
        )
        test_model.evaluate(skycoord)


def test_evaluate_invalid_obspos() -> None:
    """Test that the evaluate function raises an error for invalid obspos."""
    test_model = Model(x=20 * units.micron)
    with pytest.raises(ValueError):
        test_model.evaluate(TEST_SKY_COORD, obspos="invalid")

    with pytest.raises(TypeError):
        test_model.evaluate(TEST_SKY_COORD, obspos=[30, 40, 50])

    with pytest.raises(units.UnitConversionError):
        test_model.evaluate(TEST_SKY_COORD, obspos=[30, 40, 50] * units.s)

    with pytest.raises(ValueError):
        test_model.evaluate(TEST_SKY_COORD, obspos=[[30, 40, 50], [30, 40, 50]] * units.AU)


def test_output_shape() -> None:
    """Test that the return_comps function works for valid user input."""
    n_comps = test_model._ipd_model.ncomps
    assert test_model.evaluate(TEST_SKY_COORD, return_comps=True).shape == (n_comps, 1)
    assert test_model.evaluate(TEST_SKY_COORD, return_comps=False).shape == (1,)

    skycoord = SkyCoord(
        [20, 22, 24, 26, 28, 30],
        [30, 28, 26, 24, 22, 20],
        unit=units.deg,
        obstime=TEST_TIME,
    )
    assert test_model.evaluate(skycoord, return_comps=True).shape == (n_comps, 6)
    assert test_model.evaluate(skycoord, return_comps=False).shape == (6,)


def test_input_shape() -> None:
    """Test that shape of the input sky_coord is correct and obspos and obstime are correct."""
    obstimes = time.Time([TEST_TIME.mjd + i for i in range(4)], format="mjd")
    obsposes = (
        coords.get_body("earth", obstimes)
        .transform_to(coords.HeliocentricMeanEcliptic)
        .cartesian.xyz.to(units.AU)
    )
    test_model.evaluate(SkyCoord([20, 30], [30, 40], unit=units.deg, obstime=TEST_TIME))
    test_model.evaluate(
        SkyCoord([20, 30, 40, 50], [30, 40, 30, 20], unit=units.deg, obstime=obstimes)
    )
    test_model.evaluate(
        SkyCoord([20, 30, 40, 50], [30, 40, 30, 20], unit=units.deg, obstime=obstimes),
        obspos=obsposes,
    )

    # obstime > sky_coord
    with pytest.raises(ValueError):
        test_model.evaluate(SkyCoord([20, 30], [30, 40], unit=units.deg, obstime=obstimes))

    # obspos > sky_coord
    with pytest.raises(ValueError):
        test_model.evaluate(SkyCoord(20, 30, unit=units.deg, obstime=TEST_TIME), obspos=obsposes)

    # obspos > obstime
    with pytest.raises(ValueError):
        test_model.evaluate(SkyCoord(20, 30, unit=units.deg, obstime=TEST_TIME), obspos=obsposes)

    # obstimes > obspos
    with pytest.raises(ValueError):
        test_model.evaluate(
            SkyCoord([20, 30, 40, 50], [30, 40, 30, 20], unit=units.deg, obstime=obstimes),
            obspos=[1, 2, 3] * units.AU,
        )
    with pytest.raises(ValueError):
        test_model.evaluate(
            SkyCoord([20, 30, 40, 50], [30, 40, 30, 20], unit=units.deg, obstime=obstimes),
            obspos=[[1, -0.4, 0.2]] * units.AU,
        )
    with pytest.raises(ValueError):
        test_model.evaluate(
            SkyCoord([20, 30, 40, 50], [30, 40, 30, 20], unit=units.deg, obstime=obstimes),
            obspos=obsposes[:2],
        )


def test_return_comps() -> None:
    """Test that the return_comps function works for valid user input."""
    emission_comps = test_model.evaluate(TEST_SKY_COORD, return_comps=True)
    emission = test_model.evaluate(TEST_SKY_COORD, return_comps=False)
    assert_array_equal(emission_comps.sum(axis=0), emission)

    skycoord = SkyCoord(
        [20, 22, 24, 26, 28, 30],
        [30, 28, 26, 24, 22, 20],
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission_comps = test_model.evaluate(skycoord, return_comps=True)
    emission = test_model.evaluate(skycoord, return_comps=False)
    assert_array_equal(emission_comps.sum(axis=0), emission)


def test_multiprocessing_nproc_inst() -> None:
    """Test that the multiprocessing works with n_proc > 1."""
    model = Model(x=20 * units.micron)

    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission_multi = model.evaluate(skycoord, nprocesses=4)
    emission = model.evaluate(skycoord, nprocesses=1)
    assert_array_equal(emission_multi, emission)

    model = Model(x=75 * units.micron, name="rrm-experimental")

    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission_multi = model.evaluate(skycoord, nprocesses=4)
    emission = model.evaluate(skycoord, nprocesses=1)

    assert_array_equal(emission_multi, emission)


def test_multiprocessing_nproc_time() -> None:
    """Test that the multiprocessing works with n_proc > 1."""
    model = Model(x=20 * units.micron)

    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)
    obstime = np.linspace(0, 300, 10000) + TEST_TIME.mjd
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=time.Time(obstime, format="mjd"),
    )
    emission_multi = model.evaluate(skycoord, nprocesses=4)
    emission = model.evaluate(skycoord, nprocesses=1)
    assert_array_equal(emission_multi, emission)

    # model = Model(x=75 * units.micron, name="rrm-experimental")
    # emission_multi = model.evaluate(skycoord, nprocesses=4)
    # emission = model.evaluate(skycoord, nprocesses=1)

    # assert_array_equal(emission_multi, emission)
