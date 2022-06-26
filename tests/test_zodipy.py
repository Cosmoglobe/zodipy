from __future__ import annotations

import astropy.units as u
import healpy as hp
from astropy.time import Time
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data

import zodipy

from ._helpers import get_random_observer, update_cutoff
from ._strategies import angles, freq_strategy, model, nside, pixels_strategy, time


@given(model(), time(), nside(), data())
@settings(max_examples=50, deadline=None)
def test_get_emission_pix(
    model: zodipy.Zodipy,
    time: Time,
    nside: int,
    data: DataObject,
) -> None:

    freq = data.draw(freq_strategy(model))
    pixels = data.draw(pixels_strategy(nside))
    obs = get_random_observer(model)
    update_cutoff(model, obs, time)

    emission = model.get_emission_pix(
        freq,
        pixels=pixels,
        nside=nside,
        obs_time=time,
        obs=obs,
    )
    assert emission.shape == pixels.shape

    emission_binned = model.get_binned_emission_pix(
        freq,
        pixels=pixels,
        nside=nside,
        obs_time=time,
        obs=obs,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)


@given(model(), time(), nside(), angles(), data())
@settings(max_examples=50, deadline=None)
def test_get_emission_ang(
    model: zodipy.Zodipy,
    time: Time,
    nside: int,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
) -> None:

    theta, phi = angles
    freq = data.draw(freq_strategy(model))
    obs = get_random_observer(model)
    update_cutoff(model, obs, time)

    emission = model.get_emission_ang(
        freq,
        theta=theta,
        phi=phi,
        obs_time=time,
        obs=obs,
    )
    assert emission.shape == theta.shape == phi.shape

    emission_binned = model.get_binned_emission_ang(
        freq,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=obs,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)


# test model raises error if quad order < 0
# test model raises error if model_name not in built-in models
# test model raises error if unit of los_cutoff is not AU
# test model raises error if unit of solar_cutoff is not DEG
# test model raises error if obs is not supported

# test get_emission raisese error if obs_dist < los_cutoff
# test get_emission raises error if pix < 0
