from __future__ import annotations

from astropy.units import Quantity
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import HeliocentricMeanEcliptic, get_body
import numpy as np


# L2 not included in the Astropy api for now so we manually implement an 
# support for it by assume that SEMB-L2 is at all times at a fixed distance 
# from Earth along the vector pointing to Earth from the Sun.
DISTANCE_FROM_EARTH_TO_L2 = Quantity(0.009896235034000056, u.AU)


def get_earth_position(time_of_observation: Time) -> Quantity[u.AU]:
    """Returns the position of the Earth given an ephemeris and observation time."""

    earth_skycoordinate = get_body("Earth", time=time_of_observation)
    earth_skycoordinate = earth_skycoordinate.transform_to(HeliocentricMeanEcliptic)
    earth_position = earth_skycoordinate.cartesian.xyz.to(u.AU)

    return earth_position.reshape(3, 1)


def get_observer_position(
    observer: str, time_of_observation: Time, earth_position: Quantity[u.AU]
) -> Quantity[u.AU]:
    """Returns the position of the Earth and the observer."""

    if observer.lower() in ("semb-l2", "l2"):
        return _get_sun_earth_moon_barycenter(earth_position)

    observer_skycoordinate = get_body("Earth", time=time_of_observation)
    observer_skycoordinate = observer_skycoordinate.transform_to(
        HeliocentricMeanEcliptic
    )
    observer_position = observer_skycoordinate.cartesian.xyz.to(u.AU)

    return observer_position.reshape(3, 1)


def _get_sun_earth_moon_barycenter(
    earth_position: Quantity[u.AU],
) -> Quantity[u.AU]:
    """Returns the approximated position of SEMB-L2 given Earth's position."""

    r_earth = np.linalg.norm(earth_position)
    earth_unit_vector = earth_position / r_earth
    semb_l2_length = r_earth + DISTANCE_FROM_EARTH_TO_L2

    return earth_unit_vector * semb_l2_length
