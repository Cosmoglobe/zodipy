import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy.time import Time

from zodipy._constants import DISTANCE_FROM_EARTH_TO_L2
from zodipy._types import NumpyArray


def get_obs_and_earth_positions(
    obs_pos: u.Quantity[u.AU] | str, obs_time: Time
) -> tuple[NumpyArray, NumpyArray]:
    """Return the position of the observer and the Earth (3, `n_pointing`, `n_gauss_quad_degree`).

    The lagrange point SEMB-L2 is not included in any of the current available
    ephemerides. We implement an approximation to its position, assuming that
    SEMB-L2 is at all times located at a fixed distance from Earth along the vector
    pointing to Earth from the Sun.
    """
    earth_position = _get_earth_position(obs_time)
    if isinstance(obs_pos, str):
        obs_position = _get_observer_position(obs_pos, obs_time, earth_position)
    else:
        try:
            obs_position = obs_pos.to(u.AU)
        except AttributeError:
            msg = (
                "Observer position must be a string or an astropy Quantity with units of distance."
            )
            raise TypeError(msg) from AttributeError

    return obs_position.reshape(3, 1, 1).value, earth_position.reshape(3, 1, 1).value


def _get_earth_position(obs_time: Time) -> u.Quantity[u.AU]:
    """Return the position of the Earth given an ephemeris and observation time."""
    earth_skycoordinate = coords.get_body("earth", obs_time)
    earth_skycoordinate = earth_skycoordinate.transform_to(coords.HeliocentricMeanEcliptic)
    return earth_skycoordinate.cartesian.xyz.to(u.AU)


def _get_observer_position(
    obs: str, obs_time: Time, earth_pos: u.Quantity[u.AU]
) -> u.Quantity[u.AU]:
    """Return the position of the Earth and the observer."""
    if obs.lower() == "semb-l2":
        return _get_sun_earth_moon_barycenter(earth_pos)

    observer_skycoordinate = coords.get_body(obs, obs_time)
    observer_skycoordinate = observer_skycoordinate.transform_to(coords.HeliocentricMeanEcliptic)

    return observer_skycoordinate.cartesian.xyz.to(u.AU)


def _get_sun_earth_moon_barycenter(
    earth_position: u.Quantity[u.AU],
) -> u.Quantity[u.AU]:
    """Return the approximated position of SEMB-L2 given Earth's position."""
    r_earth = np.linalg.norm(earth_position)
    earth_unit_vector = earth_position / r_earth
    semb_l2_length = r_earth + DISTANCE_FROM_EARTH_TO_L2

    return earth_unit_vector * semb_l2_length
