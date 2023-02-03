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
    freq,
    model,
    nside,
    obs,
    pixels,
    random_freq,
    time,
    weights,
)
from ._tabulated_dirbe import DAYS, LAT, LON, TABULATED_DIRBE_EMISSION

DIRBE_START_DAY = Time("1990-01-01")


@given(model(), time(), nside(), data())
@settings(deadline=None)
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

    freq = data.draw(random_freq(unit=model._ipd_model.spectrum.unit))
    if not (model._ipd_model.spectrum[0] <= freq <= model._ipd_model.spectrum[-1]):
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
@settings(max_examples=10, deadline=None)
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


def test_multiprocessing() -> None:
    """
    Testing that model with multiprocessing enabled returns the same value as
    without multiprocessing.
    """

    model = Zodipy(parallel=False)
    model_parallel = Zodipy(parallel=True)

    observer = "earth"
    time = Time("2020-01-01")
    frequency = 78 * u.micron
    nside = 32
    pix = np.random.randint(0, hp.nside2npix(nside), size=1000)
    theta = np.random.uniform(0, 180, size=1000) * u.deg
    phi = np.random.uniform(0, 360, size=1000) * u.deg

    emission_pix = model.get_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    emission_pix_parallel = model_parallel.get_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert np.allclose(emission_pix, emission_pix_parallel)

    emission_binned_pix = model.get_binned_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    emission_binned_pix_parallel = model_parallel.get_binned_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert np.allclose(emission_binned_pix, emission_binned_pix_parallel)

    emission_ang = model.get_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        obs_time=time,
        obs=observer,
    )
    emission_ang_parallel = model_parallel.get_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        obs_time=time,
        obs=observer,
    )
    assert np.allclose(emission_ang, emission_ang_parallel)

    emission_binned_ang = model.get_binned_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=observer,
    )

    emission_binned_ang_parallel = model_parallel.get_binned_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=observer,
    )

    assert np.allclose(emission_binned_ang.value, emission_binned_ang_parallel.value)


@given(model(), time(), nside(), angles(), random_freq(bandpass=True), data())
@settings(deadline=None)
def test_bandpass_integration(
    model: Zodipy,
    time: Time,
    nside: int,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    freqs: u.Quantity[u.Hz] | u.Quantity[u.micron],
    data: DataObject,
) -> None:
    """Property test for bandpass integrations."""
    theta, phi = angles
    model.extrapolate = True
    observer = data.draw(obs(model, time))
    bp_weights = data.draw(weights(freqs))
    emission_binned = model.get_binned_emission_ang(
        freqs,
        weights=bp_weights,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)


@given(model(), time(), nside(), angles(), random_freq(bandpass=True), data())
@settings(deadline=None)
def test_weights(
    model: Zodipy,
    time: Time,
    nside: int,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    freqs: u.Quantity[u.Hz] | u.Quantity[u.micron],
    data: DataObject,
) -> None:
    """Property test for bandpass weights."""

    theta, phi = angles
    model.extrapolate = True
    observer = data.draw(obs(model, time))
    bp_weights = data.draw(weights(freqs))

    model.get_binned_emission_ang(
        freqs,
        weights=bp_weights,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=observer,
    )


def test_custom_weights() -> None:
    model = Zodipy()
    time = Time("2020-01-01")
    nside = 32
    pix = np.arange(hp.nside2npix(nside))
    central_freq = 25
    sigma_freq = 3
    freqs = (
        np.linspace(central_freq - sigma_freq, central_freq + sigma_freq, 100)
        * u.micron
    )
    weights = np.random.randn(len(freqs))
    weights /= np.trapz(weights, freqs.value)

    model.get_emission_pix(
        freq=freqs,
        weights=weights,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs="earth",
    )


def test_custom_obs_pos() -> None:
    model = Zodipy()
    time = Time("2020-01-01")
    nside = 64
    pix = np.arange(hp.nside2npix(nside))

    model.get_emission_pix(
        freq=234 * u.micron,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs_pos=[0.1, 0.2, 1] * u.AU,
    )

    model.get_emission_pix(
        freq=234 * u.micron,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs_pos=[2, 0.1, 4] * u.AU,
    )

    with pytest.raises(TypeError):
        model.get_emission_pix(
            freq=234 * u.micron,
            pixels=pix,
            nside=nside,
            obs_time=time,
            obs_pos=[2, 0.1, 4],
        )

    with pytest.raises(u.UnitsError):
        model.get_emission_pix(
            freq=234 * u.micron,
            pixels=pix,
            nside=nside,
            obs_time=time,
            obs_pos=[2, 0.1, 4] * u.s,
        )
