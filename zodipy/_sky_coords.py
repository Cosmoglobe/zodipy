"""Functions that extract the positions of Solar System bodies using the astropy
`solar_system_ephemeris` API.

The lagrange point SEMB-L2 is not included in any of the current available
ephemerides. We implement an approximation to its position, assuming that 
SEMB-L2 is at all times located at a fixed distance from Earth along the vector 
pointing to Earth from the Sun.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import HeliocentricMeanEcliptic, get_body
from astropy.time import Time

DISTANCE_FROM_EARTH_TO_L2 = u.Quantity(0.009896235034000056, u.AU)
DISTANCE_TO_JUPITER = u.Quantity(5.2, u.AU)


def get_obs_and_earth_positions(
    obs: str, obs_time: Time, obs_pos: u.Quantity[u.AU] | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:

    earth_position = _get_earth_position(obs_time)
    if obs_pos is None:
        obs_pos = _get_observer_position(obs, obs_time, earth_position)

    return obs_pos.reshape(3, 1).value, earth_position.reshape(3, 1).value


def _get_earth_position(obs_time: Time) -> u.Quantity[u.AU]:
    """Returns the position of the Earth given an ephemeris and observation time."""

    earth_skycoordinate = get_body("earth", obs_time)
    earth_skycoordinate = earth_skycoordinate.transform_to(HeliocentricMeanEcliptic)

    return earth_skycoordinate.cartesian.xyz.to(u.AU)


def _get_observer_position(
    obs: str, obs_time: Time, earth_pos: u.Quantity[u.AU]
) -> u.Quantity[u.AU]:
    """Returns the position of the Earth and the observer."""

    if obs.lower() == "semb-l2":
        return _get_sun_earth_moon_barycenter(earth_pos)

    observer_skycoordinate = get_body(obs, obs_time)
    observer_skycoordinate = observer_skycoordinate.transform_to(
        HeliocentricMeanEcliptic
    )

    return observer_skycoordinate.cartesian.xyz.to(u.AU)


def _get_sun_earth_moon_barycenter(
    earth_position: u.Quantity[u.AU],
) -> u.Quantity[u.AU]:
    """Returns the approximated position of SEMB-L2 given Earth's position."""

    r_earth = np.linalg.norm(earth_position)

    earth_unit_vector = earth_position / r_earth
    semb_l2_length = r_earth + DISTANCE_FROM_EARTH_TO_L2

    return earth_unit_vector * semb_l2_length
