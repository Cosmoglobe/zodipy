from __future__ import annotations

import astropy.units as u
import healpy as hp
from astropy.time import Time
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data

import zodipy

from ._strategies import angles, freq, model, nside, obs, pixels, time


@given(model(), time(), nside(), data())
@settings(deadline=None)
def test_get_emission_pix(
    model: zodipy.Zodipy,
    time: Time,
    nside: int,
    data: DataObject,
) -> None:

    observer = data.draw(obs(model))
    frequency = data.draw(freq(model))
    pix = data.draw(pixels(nside))

    emission = model.get_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert len(emission) == len(pix)

    emission_binned = model.get_binned_emission_pix(
        frequency,
        pixels=pix,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)


@given(model(), time(), nside(), angles(), data())
@settings(deadline=None)
def test_get_emission_ang(
    model: zodipy.Zodipy,
    time: Time,
    nside: int,
    angles: tuple[u.Quantity[u.deg], u.Quantity[u.deg]],
    data: DataObject,
) -> None:

    theta, phi = angles

    observer = data.draw(obs(model))
    frequency = data.draw(freq(model))

    emission = model.get_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        obs_time=time,
        obs=observer,
    )
    assert emission.shape == theta.shape == phi.shape

    emission_binned = model.get_binned_emission_ang(
        frequency,
        theta=theta,
        phi=phi,
        nside=nside,
        obs_time=time,
        obs=observer,
    )
    assert emission_binned.shape == (hp.nside2npix(nside),)
