from __future__ import annotations

from typing import Literal

import astropy.coordinates as coords
import astropy.units as u
import astropy_healpix as hp
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from hypothesis import given, settings
from hypothesis.strategies import DataObject, booleans, data

from zodipy.zodipy import Zodipy

from ._strategies import (
    angles,
    coords_in,
    freqs,
    healpixes,
    obs_positions,
    pixels,
    quantities,
    random_freqs,
    sky_coords,
    times,
    weights,
    zodipy_models,
)
from ._tabulated_dirbe import DAYS, LAT, LON, TABULATED_DIRBE_EMISSION

DIRBE_START_DAY = Time("1990-01-01")


@given(zodipy_models(), sky_coords(), data())
@settings(deadline=None)
def test_get_emission_skycoord(
    model: Zodipy,
    coordinates: coords.SkyCoord,
    data: DataObject,
) -> None:
    """Property test for get_emission_skycoord."""
    observer = data.draw(obs_positions(model, coordinates.obstime))
    frequency = data.draw(freqs(model))
    emission = model.get_emission_skycoord(
        coordinates,
        freq=frequency,
        obs_pos=observer,
    )
    assert emission.size == coordinates.size


@given(zodipy_models(), sky_coords(), healpixes(), data())
@settings(deadline=None)
def test_get_binned_skycoord(
    model: Zodipy,
    coordinates: coords.SkyCoord,
    healpix: hp.HEALPix,
    data: DataObject,
) -> None:
    """Property test for get_binned_emission_pix."""
    observer = data.draw(obs_positions(model, coordinates.obstime))
    frequency = data.draw(freqs(model))
    cut_solar = data.draw(booleans())

    emission_binned = model.get_binned_emission_skycoord(
        coordinates,
        freq=frequency,
        obs_pos=observer,
        nside=healpix.nside,
        solar_cut=data.draw(quantities(20, 50, u.deg)) if cut_solar else None,
    )
    assert emission_binned.shape == (healpix.npix,)


@given(zodipy_models(), times(), healpixes(), data(), coords_in())
@settings(deadline=None)
def test_get_emission_pix(
    model: Zodipy,
    time: Time,
    healpix: hp.HEALPix,
    data: DataObject,
    coord_in: Literal["E", "G", "C"],
) -> None:
    """Property test for get_emission_pix."""
    observer = data.draw(obs_positions(model, time))
    frequency = data.draw(freqs(model))
    pix = data.draw(pixels(healpix.nside))
    emission = model.get_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
        coord_in=coord_in,
    )
    assert np.size(emission) == np.size(pix)


@given(zodipy_models(), times(), healpixes(), data(), coords_in())
@settings(deadline=None)
def test_get_binned_emission_pix(
    model: Zodipy,
    time: Time,
    healpix: hp.HEALPix,
    data: DataObject,
    coord_in: Literal["E", "G", "C"],
) -> None:
    """Property test for get_binned_emission_pix."""
    observer = data.draw(obs_positions(model, time))
    frequency = data.draw(freqs(model))
    pix = data.draw(pixels(healpix.nside))
    cut_solar = data.draw(booleans())
    emission_binned = model.get_binned_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
        solar_cut=data.draw(quantities(20, 50, u.deg)) if cut_solar else None,
        coord_in=coord_in,
    )
    assert emission_binned.shape == (healpix.npix,)


@given(zodipy_models(), times(), angles(), data(), coords_in())
@settings(deadline=None)
def test_get_emission_ang(
    model: Zodipy,
    time: Time,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
    coord_in: Literal["E", "G", "C"],
) -> None:
    """Property test for get_emission_ang."""
    theta, phi = angles

    observer = data.draw(obs_positions(model, time))
    frequency = data.draw(freqs(model))

    emission = model.get_emission_ang(
        theta,
        phi,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
        coord_in=coord_in,
    )
    assert emission.size == theta.size == phi.size


@given(zodipy_models(), times(), healpixes(), angles(), data(), coords_in())
@settings(deadline=None)
def test_get_binned_emission_ang(
    model: Zodipy,
    time: Time,
    healpix: hp.HEALPix,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
    coord_in: Literal["E", "G", "C"],
) -> None:
    """Property test for get_binned_emission_pix."""
    theta, phi = angles

    observer = data.draw(obs_positions(model, time))
    frequency = data.draw(freqs(model))
    cut_solar = data.draw(booleans())

    emission_binned = model.get_binned_emission_ang(
        theta,
        phi,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
        coord_in=coord_in,
        solar_cut=data.draw(quantities(20, 50, u.deg)) if cut_solar else None,
    )
    assert emission_binned.shape == (healpix.npix,)


@given(zodipy_models(extrapolate=False), times(), angles(lonlat=True), healpixes(), data())
@settings(deadline=None)
def test_invalid_freq(
    model: Zodipy,
    time: Time,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    healpix: hp.HEALPix,
    data: DataObject,
) -> None:
    """Property test checking for unsupported spectral range."""
    theta, phi = angles
    observer = data.draw(obs_positions(model, time))
    pix = data.draw(pixels(healpix.nside))

    freq = data.draw(random_freqs(unit=model._ipd_model.spectrum.unit))
    if not (model._ipd_model.spectrum[0] <= freq <= model._ipd_model.spectrum[-1]):
        with pytest.raises(ValueError):
            model.get_emission_pix(
                pix,
                freq=freq,
                nside=healpix.nside,
                obs_time=time,
                obs_pos=observer,
            )

        with pytest.raises(ValueError):
            model.get_emission_ang(
                theta,
                phi,
                freq=freq,
                obs_time=time,
                obs_pos=observer,
                lonlat=True,
            )

        with pytest.raises(ValueError):
            model.get_binned_emission_pix(
                pix,
                nside=healpix.nside,
                freq=freq,
                obs_time=time,
                obs_pos=observer,
            )

        with pytest.raises(ValueError):
            model.get_binned_emission_ang(
                theta,
                phi,
                nside=healpix.nside,
                freq=freq,
                obs_time=time,
                obs_pos=observer,
                lonlat=True,
            )


def test_compare_to_dirbe_idl() -> None:
    """Tests that ZodiPy reproduces the DIRBE software.

    Zodipy should be able to reproduce the tabulated emission from the DIRBE Zoidacal Light
    Prediction Software with a maximum difference of 0.1%.
    """
    model = Zodipy("dirbe")
    for frequency, tabulated_emission in TABULATED_DIRBE_EMISSION.items():
        for idx, (day, lon, lat) in enumerate(zip(DAYS, LON, LAT)):
            time = DIRBE_START_DAY + TimeDelta(day - 1, format="jd")

            emission = model.get_emission_ang(
                lon * u.deg,
                lat * u.deg,
                freq=frequency * u.micron,
                lonlat=True,
                obs_pos="earth",
                obs_time=time,
            )
            assert emission.value == pytest.approx(tabulated_emission[idx], rel=0.01)


def test_multiprocessing() -> None:
    """Tests multiprocessing with for zodipy.

    Tests that model with multiprocessing enabled returns the same value as
    without multiprocessing.
    """
    model = Zodipy()
    model_parallel = Zodipy(n_proc=4)

    observer = "earth"
    time = Time("2020-01-01")
    frequency = 78 * u.micron
    healpix = hp.HEALPix(32)
    pix = np.random.randint(0, healpix.npix, size=1000)
    theta = np.random.uniform(0, 180, size=1000) * u.deg
    phi = np.random.uniform(0, 360, size=1000) * u.deg

    emission_pix = model.get_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    emission_pix_parallel = model_parallel.get_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    assert np.allclose(emission_pix, emission_pix_parallel)

    emission_binned_pix = model.get_binned_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    emission_binned_pix_parallel = model_parallel.get_binned_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    assert np.allclose(emission_binned_pix, emission_binned_pix_parallel)

    emission_ang = model.get_emission_ang(
        theta,
        phi,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    emission_ang_parallel = model_parallel.get_emission_ang(
        theta,
        phi,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    assert np.allclose(emission_ang, emission_ang_parallel)

    emission_binned_ang = model.get_binned_emission_ang(
        theta,
        phi,
        freq=frequency,
        nside=healpix.nside,
        obs_time=time,
        obs_pos=observer,
    )

    emission_binned_ang_parallel = model_parallel.get_binned_emission_ang(
        theta,
        phi,
        freq=frequency,
        nside=healpix.nside,
        obs_time=time,
        obs_pos=observer,
    )

    assert np.allclose(emission_binned_ang.value, emission_binned_ang_parallel.value)


def test_inner_radial_cutoff_multiprocessing() -> None:
    """Testing that model with inner radial cutoffs can be parallelized."""
    model = Zodipy("RRM-experimental")
    model_parallel = Zodipy("RRM-experimental", n_proc=4)

    observer = "earth"
    time = Time("2020-01-01")
    frequency = 78 * u.micron
    healpix = hp.HEALPix(32)
    pix = np.random.randint(0, healpix.npix, size=1000)

    emission_pix = model.get_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    emission_pix_parallel = model_parallel.get_emission_pix(
        pix,
        nside=healpix.nside,
        freq=frequency,
        obs_time=time,
        obs_pos=observer,
    )
    assert np.allclose(emission_pix, emission_pix_parallel)


@given(
    zodipy_models(extrapolate=True),
    times(),
    healpixes(),
    angles(),
    random_freqs(bandpass=True),
    data(),
)
@settings(deadline=None)
def test_bandpass_integration(
    model: Zodipy,
    time: Time,
    healpix: hp.HEALPix,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    freqs: u.Quantity[u.Hz] | u.Quantity[u.micron],
    data: DataObject,
) -> None:
    """Property test for bandpass integrations."""
    theta, phi = angles
    observer = data.draw(obs_positions(model, time))
    bp_weights = data.draw(weights(freqs))
    emission_binned = model.get_binned_emission_ang(
        theta,
        phi,
        freq=freqs,
        weights=bp_weights,
        nside=healpix.nside,
        obs_time=time,
        obs_pos=observer,
    )
    assert emission_binned.shape == (healpix.npix,)


@given(
    zodipy_models(extrapolate=True),
    times(),
    healpixes(),
    angles(),
    random_freqs(bandpass=True),
    data(),
)
@settings(deadline=None)
def test_weights(
    model: Zodipy,
    time: Time,
    healpix: hp.HEALPix,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    freqs: u.Quantity[u.Hz] | u.Quantity[u.micron],
    data: DataObject,
) -> None:
    """Property test for bandpass weights."""
    theta, phi = angles
    observer = data.draw(obs_positions(model, time))
    bp_weights = data.draw(weights(freqs))

    model.get_binned_emission_ang(
        theta,
        phi,
        weights=bp_weights,
        freq=freqs,
        nside=healpix.nside,
        obs_time=time,
        obs_pos=observer,
    )


def test_custom_weights() -> None:
    """Tests bandpass integration with custom weights."""
    model = Zodipy()
    time = Time("2020-01-01")
    healpix = hp.HEALPix(64)
    pix = np.arange(healpix.npix)
    central_freq = 25
    sigma_freq = 3
    freqs = np.linspace(central_freq - sigma_freq, central_freq + sigma_freq, 100) * u.micron
    weights = np.random.randn(len(freqs))
    weights /= np.trapz(weights, freqs.value)

    model.get_emission_pix(
        pix,
        freq=freqs,
        weights=weights,
        nside=healpix.nside,
        obs_time=time,
        obs_pos="earth",
    )


def test_custom_obs_pos() -> None:
    """Tests a user specified observer position."""
    model = Zodipy()
    time = Time("2020-01-01")
    healpix = hp.HEALPix(64)
    pix = np.arange(healpix.npix)

    model.get_emission_pix(
        pix,
        freq=234 * u.micron,
        nside=healpix.nside,
        obs_time=time,
        obs_pos=[0.1, 0.2, 1] * u.AU,
    )

    model.get_emission_pix(
        pix,
        freq=234 * u.micron,
        nside=healpix.nside,
        obs_time=time,
        obs_pos=[2, 0.1, 4] * u.AU,
    )

    with pytest.raises(TypeError):
        model.get_emission_pix(
            pix,
            freq=234 * u.micron,
            nside=healpix.nside,
            obs_time=time,
            obs_pos=[2, 0.1, 4],
        )

    with pytest.raises(u.UnitsError):
        model.get_emission_pix(
            pix,
            freq=234 * u.micron,
            nside=healpix.nside,
            obs_time=time,
            obs_pos=[2, 0.1, 4] * u.s,
        )


def test_pixel_ordering() -> None:
    """Tests that the pixel related functions work for both healpix orderings."""
    model = Zodipy()
    time = Time("2020-01-01")

    healpix_ring = hp.HEALPix(32)
    healpix_nested = hp.HEALPix(32)

    pix_ring = np.arange(healpix_ring.npix)
    lon, lat = healpix_ring.healpix_to_lonlat(pix_ring)
    lon = u.Quantity(lon)
    lat = u.Quantity(lat)
    pix_nested = healpix_nested.lonlat_to_healpix(lon, lat)

    model.get_emission_pix(
        pix_ring,
        freq=234 * u.micron,
        nside=healpix_ring.nside,
        obs_time=time,
        order="ring",
    )
    model.get_emission_pix(
        pix_nested,
        freq=234 * u.micron,
        nside=healpix_nested.nside,
        obs_time=time,
        order="nested",
    )

    model.get_binned_emission_pix(
        pix_ring,
        freq=234 * u.micron,
        nside=healpix_ring.nside,
        obs_time=time,
        order="ring",
    )
    model.get_binned_emission_pix(
        pix_nested,
        freq=234 * u.micron,
        nside=healpix_nested.nside,
        obs_time=time,
        order="nested",
    )

    model.get_binned_emission_ang(
        lon,
        lat,
        lonlat=True,
        freq=234 * u.micron,
        nside=healpix_ring.nside,
        obs_time=time,
        order="ring",
    )
    model.get_binned_emission_ang(
        lon,
        lat,
        lonlat=True,
        freq=234 * u.micron,
        nside=healpix_nested.nside,
        obs_time=time,
        order="nested",
    )

    sky_coord = coords.SkyCoord(
        lon,
        lat,
        frame=coords.BarycentricMeanEcliptic,
        obstime=time,
    )

    model.get_binned_emission_skycoord(
        sky_coord,
        freq=234 * u.micron,
        nside=healpix_ring.nside,
        order="ring",
    )
    model.get_binned_emission_skycoord(
        sky_coord,
        freq=234 * u.micron,
        nside=healpix_nested.nside,
        order="nested",
    )
