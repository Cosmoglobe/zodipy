import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.time import Time

time = Time.now()


def test_single_pix(DIRBE):
    emission = DIRBE.get_emission_pix(
        34 * u.micron,
        pixels=2342,
        nside=64,
        obs="earth",
        obs_time=time,
    )
    assert np.size(emission) == 1


def test_single_pointing(DIRBE):
    emission = DIRBE.get_emission_ang(
        34 * u.micron,
        theta=1 * u.deg,
        phi=30 * u.deg,
        obs="earth",
        obs_time=time,
    )
    assert np.size(emission) == 1

    emission = DIRBE.get_emission_ang(
        34 * u.micron,
        theta=170 * u.deg,
        phi=-60 * u.deg,
        obs="earth",
        obs_time=time,
        lonlat=True,
    )
    assert np.size(emission) == 1

    # test theta out of range
    with pytest.raises(ValueError):
        DIRBE.get_emission_ang(
            34 * u.micron,
            theta=200 * u.deg,
            phi=150 * u.deg,
            obs="earth",
            obs_time=time,
        )


def test_multi_pointing(DIRBE):
    emission = DIRBE.get_emission_ang(
        34 * u.micron,
        theta=[50, 80, 170] * u.deg,
        phi=[50, 30, 20] * u.deg,
        obs="earth",
        obs_time=time,
    )
    assert np.size(emission) > 1

    emission = DIRBE.get_emission_ang(
        34 * u.micron,
        theta=[50, 80, 180] * u.deg,
        phi=[50, 30, 20] * u.deg,
        obs="earth",
        obs_time=time,
        lonlat=True,
    )
    assert np.size(emission) > 1

    # test theta out of range
    with pytest.raises(ValueError):
        DIRBE.get_emission_ang(
            34 * u.micron,
            theta=[50, 80, 300] * u.deg,
            phi=[50, 30, 20] * u.deg,
            obs="earth",
            obs_time=time,
        )


def test_binned(DIRBE):
    nside = 32
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # test shape == npix
    emission = DIRBE.get_binned_emission_ang(
        34 * u.micron,
        theta=theta * u.deg,
        phi=phi * u.deg,
        nside=nside,
        obs="earth",
        obs_time=time,
    )
    assert np.size(emission) == npix

    emission = DIRBE.get_binned_emission_ang(
        34 * u.micron,
        theta=theta[:100] * u.deg,
        phi=phi[:100] * u.deg,
        nside=nside,
        obs="earth",
        obs_time=time,
    )
    assert np.size(emission) == npix


def test_time(DIRBE):
    # test non supported time
    with pytest.raises(AttributeError):
        DIRBE.get_emission_pix(
            34 * u.micron,
            pixels=2342,
            nside=64,
            obs="earth",
            obs_time="2020-10-10",
        )


def test_obs_pos_vs_obs(DIRBE):
    # test close value for manual position vs obs
    from astropy.coordinates import HeliocentricMeanEcliptic, get_body

    earth_skycoord = get_body("Earth", time=time)
    earth_skycoord = earth_skycoord.transform_to(HeliocentricMeanEcliptic)
    earth_pos = earth_skycoord.represent_as("cartesian").xyz.to(u.AU)

    emission_obs_pos = DIRBE.get_emission_pix(
        34 * u.micron,
        pixels=2342,
        nside=64,
        obs_pos=earth_pos,
        obs_time=time,
    )

    emission_obs = DIRBE.get_emission_pix(
        34 * u.micron,
        pixels=2342,
        nside=64,
        obs="earth",
        obs_time=time,
    )

    assert np.isclose(emission_obs, emission_obs_pos)


def test_return_comps(DIRBE):
    no_comps = DIRBE.get_emission_pix(
        34 * u.micron,
        pixels=2342,
        nside=64,
        obs="earth",
        obs_time=time,
    )

    comps = DIRBE.get_emission_pix(
        34 * u.micron,
        pixels=2342,
        nside=64,
        obs="earth",
        obs_time=time,
        return_comps=True,
    )

    assert no_comps.shape[0] == 1
    assert comps.shape[0] == DIRBE.model.n_comps

    assert np.isclose(comps.sum(axis=0), no_comps)
