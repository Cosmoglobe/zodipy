from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.coordinates as coords
import numpy as np
from astropy import time, units
from scipy import interpolate

if TYPE_CHECKING:
    import numpy.typing as npt

MEAN_DIST_TO_L2 = 0.009896235034000056


def get_sun_earth_moon_barycenter(
    earthpos: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return a SkyCoord of the heliocentric position of the SEMB-L2 point.

    Note that this is an approximate position, as the SEMB-L2 point is not included in
    any of the current available ephemerides. We assume that SEMB-L2 is at all times
    located at a fixed distance from Earth along the vector pointing to Earth from the Sun.
    """
    earth_distance = np.linalg.norm(earthpos)
    SEMB_L2_distance = earth_distance + MEAN_DIST_TO_L2
    earth_unit_vector = earthpos / earth_distance
    return earth_unit_vector * SEMB_L2_distance


def get_earthpos_xyz(obstime: time.Time, ephemeris: str) -> npt.NDArray[np.float64]:
    """Return the sky coordinates of the Earth in the heliocentric frame."""
    if obstime.size == 1:
        return (
            coords.get_body("earth", obstime, ephemeris=ephemeris)
            .transform_to(coords.HeliocentricMeanEcliptic)
            .cartesian.xyz.to_value(units.AU)
        )
    return get_interpolated_body_xyz("earth", obstime, ephemeris)


def get_interpolated_body_xyz(
    body: str,
    obstimes: time.Time,
    ephemeris: str,
) -> npt.NDArray[np.float64]:
    """Return interpolated Earth positions in the heliocentric frame."""
    dt = (1 * units.hour).to_value(units.day)
    t0, t1 = obstimes[0].mjd, obstimes[-1].mjd
    times = time.Time(np.arange(t0, max(t0 + 365, t1), dt), format="mjd")

    bodypos = (
        coords.get_body(body, times, ephemeris=ephemeris)
        .transform_to(coords.HeliocentricMeanEcliptic)
        .cartesian.xyz.to_value(units.AU)
    )
    f = interpolate.interp1d(times.mjd, bodypos, axis=-1)
    return f(obstimes.mjd)


def get_obspos_xyz(
    obstime: time.Time,
    obspos: str | units.Quantity,
    earthpos: npt.NDArray[np.float64],
    ephemeris: str,
) -> npt.NDArray[np.float64]:
    """Return the sky coordinates of the observer in the heliocentric frame."""
    if isinstance(obspos, str):
        if obspos.lower() == "semb-l2":
            return get_sun_earth_moon_barycenter(earthpos)
        if obspos.lower() == "earth":
            return earthpos

        try:
            if obstime.size == 1:
                return (
                    coords.get_body(obspos, obstime, ephemeris=ephemeris)
                    .transform_to(coords.HeliocentricMeanEcliptic)
                    .cartesian.xyz.to_value(units.AU)
                )
            return get_interpolated_body_xyz(obspos, obstime, ephemeris)
        except KeyError as error:
            valid_obs = [*coords.solar_system_ephemeris.bodies, "semb-l2"]
            msg = f"Invalid observer string: '{obspos}'. Valid observers are: {valid_obs}"
            raise ValueError(msg) from error

    else:
        try:
            return obspos.to_value(units.AU)
        except AttributeError as error:
            msg = "The observer position must be a string or an astropy Quantity."
            raise TypeError(msg) from error
        except units.UnitConversionError as error:
            msg = "The observer position must be in length units."
            raise units.UnitConversionError(msg) from error
