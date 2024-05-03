from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.coordinates as coords
import numpy as np
from astropy import time, units

from zodipy._constants import DISTANCE_FROM_EARTH_TO_SEMB_L2

if TYPE_CHECKING:
    import numpy.typing as npt


def get_sun_earth_moon_barycenter(
    earthpos: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return a SkyCoord of the heliocentric position of the SEMB-L2 point.

    Note that this is an approximate position, as the SEMB-L2 point is not included in
    any of the current available ephemerides. We assume that SEMB-L2 is at all times
    located at a fixed distance from Earth along the vector pointing to Earth from the Sun.
    """
    earth_distance = np.linalg.norm(earthpos)
    SEMB_L2_distance = earth_distance + DISTANCE_FROM_EARTH_TO_SEMB_L2
    earth_unit_vector = earthpos / earth_distance

    return earth_unit_vector * SEMB_L2_distance


def get_earthpos(obs_time: time.Time, ephemeris: str) -> npt.NDArray[np.float64]:
    """Return the sky coordinates of the Earth in the heliocentric frame."""
    return (
        coords.get_body("earth", obs_time, ephemeris=ephemeris)
        .transform_to(coords.HeliocentricMeanEcliptic)
        .cartesian.xyz.to_value(units.AU)
    )


def get_obspos(
    obs: str,
    obstime: time.Time,
    earthpos: npt.NDArray[np.float64],
    ephemeris: str,
) -> npt.NDArray[np.float64]:
    """Return the sky coordinates of the observer in the heliocentric frame."""
    if obs.lower() == "semb-l2":
        return get_sun_earth_moon_barycenter(earthpos)
    return (
        coords.get_body(obs, obstime, ephemeris=ephemeris)
        .transform_to(coords.HeliocentricMeanEcliptic)
        .cartesian.xyz.to_value(units.AU)
    )
