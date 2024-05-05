from __future__ import annotations

import multiprocessing

import numpy as np
import pytest
from astropy import coordinates as coords
from astropy import time, units
from astropy.coordinates import SkyCoord
from hypothesis import given, settings
from numpy.testing import assert_array_equal

from zodipy import Model

from .dirbe_tabulated import DAYS, DIRBE_START_DAY, LAT, LON, TABULATED_DIRBE_EMISSION
from .strategies import models, obs, sky_coords

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
@given(models(), sky_coords(), obs())
def test_evaluate(
    model: Model,
    sky_coord: SkyCoord,
    obs: units.Quantity | str,
) -> None:
    """Test that the evaluate function works for valid user input."""
    assert model.evaluate(sky_coord, obspos=obs).size == sky_coord.size


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
    n_comps = test_model._interplanetary_dust_model.ncomps
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


def test_contains_duplicates() -> None:
    """Test that evaluations with and without the test duplicates flag are the same."""
    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)

    unique = np.unique(np.vstack([lon, lat]), axis=1)
    assert len(lon) > unique.shape[1]
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission = test_model.evaluate(skycoord, contains_duplicates=False)
    emission_duplicates = test_model.evaluate(skycoord, contains_duplicates=True)
    assert_array_equal(emission, emission_duplicates)


def test_multiprocessing_nproc() -> None:
    """Test that the multiprocessing works with n_proc > 1."""
    model_multi = Model(x=20 * units.micron, n_proc=4)
    model = Model(x=20 * units.micron, n_proc=1)

    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission_multi = model_multi.evaluate(skycoord)
    emission = model.evaluate(skycoord)
    assert_array_equal(emission_multi, emission)

    model_multi = Model(x=75 * units.micron, n_proc=4, name="rrm-experimental")
    model = Model(x=75 * units.micron, n_proc=1, name="rrm-experimental")

    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission_multi = model_multi.evaluate(skycoord)
    emission = model.evaluate(skycoord)

    assert_array_equal(emission_multi, emission)


def test_multiprocessing_pool() -> None:
    """Test that the multiprocessing works with n_proc > 1."""
    model = Model(x=20 * units.micron, n_proc=1)
    lon = np.random.randint(low=0, high=360, size=10000)
    lat = np.random.randint(low=-90, high=90, size=10000)
    skycoord = SkyCoord(
        lon,
        lat,
        unit=units.deg,
        obstime=TEST_TIME,
    )
    emission = model.evaluate(skycoord)

    try:
        from pytest_cov.embed import cleanup_on_sigterm
    except ImportError:
        pass
    else:
        cleanup_on_sigterm()

    with multiprocessing.Pool(processes=4) as pool:
        model_multi = Model(x=20 * units.micron, pool=pool)
        emission_multi = model_multi.evaluate(skycoord)

    assert_array_equal(emission_multi, emission)
