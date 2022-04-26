"""

Functions that extract the positions of Solar System bodies using the astropy
`solar_system_ephemeris` API.

The lagrange point SEMB-L2 is not included in any of the current available
ephemerides. We implement an approximation to its position, assuming that 
SEMB-L2 is at all times located at a fixed distance from Earth along the vector 
pointing to Earth from the Sun.

"""

from __future__ import annotations

from astropy.coordinates import HeliocentricMeanEcliptic, get_body
import astropy.units as u
from astropy.time import Time
import numpy as np
from numpy.typing import NDArray


DISTANCE_FROM_EARTH_TO_L2 = 0.009896235034000056 * u.AU


def get_solar_system_positions(
    observer: str,
    time_of_observation: Time,
    observer_position: u.Quantity[u.AU] | None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    earth_position = _get_earth_position(time_of_observation)
    if observer_position is None:
        observer_position = _get_observer_position(
            observer, time_of_observation, earth_position
        )

    return observer_position.reshape(3, 1).value, earth_position.reshape(3, 1).value


def _get_earth_position(time_of_observation: Time) -> u.Quantity[u.AU]:
    """Returns the position of the Earth given an ephemeris and observation time."""

    earth_skycoordinate = get_body("earth", time_of_observation)
    earth_skycoordinate = earth_skycoordinate.transform_to(HeliocentricMeanEcliptic)

    return earth_skycoordinate.cartesian.xyz.to(u.AU)


def _get_observer_position(
    observer: str, time_of_observation: Time, earth_position: u.Quantity[u.AU]
) -> u.Quantity[u.AU]:
    """Returns the position of the Earth and the observer."""

    if observer.lower() == "semb-l2":
        return _get_sun_earth_moon_barycenter(earth_position)

    observer_skycoordinate = get_body(observer, time_of_observation)
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
