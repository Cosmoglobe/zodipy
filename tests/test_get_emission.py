from __future__ import annotations

import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data, integers

from zodipy.zodipy import Zodipy

from ._strategies import (
    angles,
    any_obs,
    freq,
    model,
    nside,
    obs,
    pixels,
    random_freq,
    time,
)
from ._tabulated_dirbe import DAYS, LAT, LON, TABULATED_DIRBE_EMISSION

DIRBE_START_DAY = Time("1990-01-01")


@given(model(), time(), nside(), data())
@settings(deadline=None, max_examples=100)
def test_get_emission_pix(
    model: Zodipy,
    time: Time,
    nside: int,
    data: DataObject,
) -> None:
    """Property test for get_emission_pix."""

    observer = data.draw(obs(model, time))
    frequency = data.draw(freq(model))
    pix = data.draw(pixels(nside))

    emission = model.get_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert np.size(emission) == np.size(pix)


@given(model(), time(), nside(), data())
@settings(deadline=None)
def test_get_binned_emission_pix(
    model: Zodipy,
    time: Time,
    nside: int,
    data: DataObject,
) -> None:
    """Property test for get_binned_emission_pix."""

    observer = data.draw(obs(model, time))
    frequency = data.draw(freq(model))
    pix = data.draw(pixels(nside))

    emission_binned = model.get_binned_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)


@given(model(), time(), angles(), data())
@settings(deadline=None)
def test_get_emission_ang(
    model: Zodipy,
    time: Time,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
) -> None:
    """Property test for get_emission_ang."""

    theta, phi = angles

    observer = data.draw(obs(model, time))
    frequency = data.draw(freq(model))

    emission = model.get_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        obs_time=time,
        obs=observer,
    )
    assert emission.size == theta.size == phi.size


@given(model(), time(), nside(), angles(), data())
@settings(deadline=None)
def test_get_binned_emission_ang(
    model: Zodipy,
    time: Time,
    nside: int,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
) -> None:
    """Property test for get_binned_emission_pix."""

    theta, phi = angles

    observer = data.draw(obs(model, time))
    frequency = data.draw(freq(model))

    emission_binned = model.get_binned_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)


@given(model(extrapolate=False), time(), angles(lonlat=True), nside(), data())
@settings(deadline=None)
def test_invalid_freq(
    model: Zodipy,
    time: Time,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    nside: int,
    data: DataObject,
) -> None:
    """
    Property test checking for unsupported spectral range in models where extrapolation is
    set to False.
    """

    theta, phi = angles
    observer = data.draw(obs(model, time))
    pix = data.draw(pixels(nside))

    freq = data.draw(random_freq(unit=model.model.spectrum.unit))
    if not (model.model.spectrum[0] <= freq <= model.model.spectrum[-1]):
        with pytest.raises(ValueError):
            model.get_emission_pix(
                freq,
                pixels=pix,
                nside=nside,
                obs_time=time,
                obs=observer,
            )

        with pytest.raises(ValueError):
            model.get_emission_ang(
                freq,
                theta=theta,
                phi=phi,
                obs_time=time,
                obs=observer,
                lonlat=True,
            )

        with pytest.raises(ValueError):
            model.get_binned_emission_pix(
                freq,
                pixels=pix,
                nside=nside,
                obs_time=time,
                obs=observer,
            )

        with pytest.raises(ValueError):
            model.get_binned_emission_ang(
                freq,
                theta=theta,
                phi=phi,
                nside=nside,
                obs_time=time,
                obs=observer,
                lonlat=True,
            )


def test_compare_to_dirbe_idl() -> None:
    """
    Tests that ZodiPy is able to reproduce tabulated emission from the DIRBE Zoidacal Light
    Prediction Software with a maximum difference of 0.1%.
    """

    model = Zodipy("dirbe")
    for freq, tabulated_emission in TABULATED_DIRBE_EMISSION.items():
        freq *= u.micron
        for idx, (day, lon, lat) in enumerate(zip(DAYS, LON, LAT)):
            time = DIRBE_START_DAY + TimeDelta(day - 1, format="jd")
            lon *= u.deg
            lat *= u.deg

            emission = model.get_emission_ang(
                freq,
                theta=lon,
                phi=lat,
                lonlat=True,
                obs="earth",
                obs_time=time,
            )
            assert emission.value == pytest.approx(tabulated_emission[idx], rel=0.01)


@given(model(), time(), nside(), integers(min_value=1, max_value=100), data())
@settings(max_examples=10)
def test_invalid_pixel(
    model: Zodipy,
    time: Time,
    nside: int,
    random_integer: int,
    data: DataObject,
) -> None:
    """
    Tests that an error is raised when an invalid pixel is passed to get_emission_pix.
    """

    frequency = data.draw(freq(model))
    observer = data.draw(obs(model, time))
    npix = hp.nside2npix(nside)

    with pytest.raises(ValueError):
        model.get_emission_pix(
            frequency,
            pixels=npix + random_integer,
            nside=nside,
            obs_time=time,
            obs=observer,
        )

    with pytest.raises(ValueError):
        model.get_emission_pix(
            frequency,
            pixels=-random_integer,
            nside=nside,
            obs_time=time,
            obs=observer,
        )


@given(model(los_dist_cut=0.2 * u.AU), time(), angles(), data())
@settings(max_examples=20)
def test_invalid_los_dist_cut(
    model: Zodipy,
    time: Time,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
) -> None:
    """
    Tests that an error is raised when a model with a `los_dist_cut` > distance to
    observer is used.
    """

    frequency = data.draw(freq(model))
    observer = data.draw(any_obs(model))
    theta, phi = angles
    if observer != "sun":
        with pytest.raises(ValueError):
            model.get_emission_ang(
                frequency,
                theta=theta,
                phi=phi,
                obs_time=time,
                obs=observer,
            )
