import pytest

import astropy.units as u
from astropy.time import Time
import healpy as hp
import numpy as np

time = Time.now()

def test_single_pix(DIRBE):
    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        pixels=2342,
        nside=64,
    )
    assert np.size(emission) == 1

def test_single_pointing(DIRBE):
    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        theta=1*u.deg,
        phi=30*u.deg,
    )
    assert np.size(emission) == 1

    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        theta=170*u.deg,
        phi=-60*u.deg,
        lonlat=True
    )
    assert np.size(emission) == 1

    # test theta out of range
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            theta=200*u.deg,
            phi=150*u.deg, 
        )


def test_pix_nonside(DIRBE):
    # test pix without nside
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            pixels=[3,4,6],
        )

def test_pix_lonlat(DIRBE):
    # test pix + lonlat
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            pixels=[3,4,6],
            nside=32,
            lonlat=True
        )

def test_pix_theta_phi(DIRBE):

    # test pix + phi
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            pixels=[3,4,6],
            phi=[50,30,20]*u.deg,
            nside=32,
        )

    # test pix + theta
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            pixels=[3,4,6],
            theta=[50,80,180]*u.deg,
            nside=32,
        )
    # test pix + theta + phi
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            pixels=[3,4,6],
            theta=[50,80,180]*u.deg,
            phi=[50,30,20]*u.deg,
            nside=32,
        )



def test_multi_pointing(DIRBE):
    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        theta=[50,80,170]*u.deg,
        phi=[50,30,20]*u.deg,
    )
    assert np.size(emission) > 1

    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        theta=[50,80,180]*u.deg,
        phi=[50,30,20]*u.deg,
        lonlat=True
    )
    assert np.size(emission) > 1

    # test theta out of range
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            theta=[50,80,300]*u.deg,
            phi=[50,30,20]*u.deg,
        )
        

def test_binned(DIRBE):
    nside = 32
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # test shape == npix
    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        theta=theta*u.deg,
        phi=phi*u.deg,
        binned=True,
        nside=nside
    )
    assert np.size(emission) == npix

    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        theta=theta[:100]*u.deg,
        phi=phi[:100]*u.deg,
        binned=True,
        nside=nside
    )
    assert np.size(emission) == npix

    # test raise error if no nside is given
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            theta=theta[:100]*u.deg,
            phi=phi[:100]*u.deg,
            binned=True,
        )


def test_nside_phi_theta(DIRBE):
    # test raise error nside is given without binned
    with pytest.raises(ValueError):
        DIRBE.get_emission(
            34*u.micron,
            obs_time=time,
            obs="earth",
            theta=[50,80,300]*u.deg,
            phi=[50,30,20]*u.deg,
            nside=32
        )

def test_observer(DIRBE):
    # test non supported observer
    with pytest.raises(ValueError):
        DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="sdfs",
        pixels=2342,
        nside=64,
    )

def test_time(DIRBE):
    # test non supported observer
    with pytest.raises(TypeError):
        DIRBE.get_emission(
        34*u.micron,
        obs_time="2020-10-10",
        obs="earth",
        pixels=2342,
        nside=64,
    )


def test_obs_pos(DIRBE):
    # test non supported observer
    emission = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs_pos=[0.0, 1.0, 0.0]*u.AU,
        pixels=2342,
        nside=64,
    )

    assert np.size(emission) == 1


def test_obs_pos_vs_obs(DIRBE):
    # test close value for manual position vs obs
    from astropy.coordinates import get_body, HeliocentricMeanEcliptic

    earth_skycoord = get_body("Earth", time=time)
    earth_skycoord = earth_skycoord.transform_to(HeliocentricMeanEcliptic)
    earth_pos = earth_skycoord.represent_as("cartesian").xyz.to(u.AU)

    emission_obs_pos = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs_pos=earth_pos,
        pixels=2342,
        nside=64,
    )

    emission_obs = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        pixels=2342,
        nside=64,
    )

    assert np.isclose(emission_obs, emission_obs_pos)

def test_return_comps(DIRBE):
    no_comps = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        pixels=2342,
        nside=64,
    )

    comps = DIRBE.get_emission(
        34*u.micron,
        obs_time=time,
        obs="earth",
        pixels=2342,
        nside=64,
        return_comps=True,
    )

    assert no_comps.shape[0] == 1
    assert comps.shape[0] == DIRBE.model.ncomps

    assert np.isclose(comps.sum(axis=0), no_comps)

# def test_coord_out(DIRBE):
#    no_comps = DIRBE.get_emission(
#         34*u.micron,
#         obs_time=time,
#         obs="earth",
#         pixels=2342,
#         nside=64,
#     )