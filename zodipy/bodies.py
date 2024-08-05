from __future__ import annotations

from typing import TYPE_CHECKING

import astropy.coordinates as coords
import numpy as np
from astropy import time, units
from scipy import interpolate

if TYPE_CHECKING:
    import numpy.typing as npt

MEAN_DIST_TO_L2 = 0.009896235034000056


def arrange_obstimes(t0: float, t1: float) -> time.Time:
    """Return a subset of the obstimes used to interpolate in body positions."""
    dt = (1 * units.hour).to_value(units.day)
    return time.Time(np.arange(t0, t1 + dt, dt), format="mjd")


def get_interp_bodypos(
    body: str,
    obstimes: npt.NDArray[np.float64],
    interp_obstimes: time.Time,
    ephemeris: str,
) -> np.ndarray:
    """Return the interpolated heliocentric positions of body."""
    pos = (
        coords.get_body(body, interp_obstimes, ephemeris=ephemeris)
        .transform_to(coords.HeliocentricMeanEcliptic)
        .cartesian.xyz.to_value(units.AU)
    )
    interpolator = interpolate.CubicSpline(interp_obstimes.mjd, pos, axis=-1)
    return interpolator(obstimes)


def get_semb_l2_pos(
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


def get_earthpos_inst(
    obstime: time.Time,
    ephemeris: str,
) -> npt.NDArray[np.float64]:
    """Return the sky coordinates of the Earth in the heliocentric frame."""
    return (
        coords.get_body("earth", obstime, ephemeris=ephemeris)
        .transform_to(coords.HeliocentricMeanEcliptic)
        .cartesian.xyz.to_value(units.AU)
    ).flatten()


def get_obspos_from_body(
    body: str,
    obstime: time.Time,
    interp_obstimes: time.Time | None,
    earthpos: npt.NDArray[np.float64],
    ephemeris: str,
) -> npt.NDArray[np.float64]:
    """Return the sky coordinates of the observer in the heliocentric frame."""
    if body == "semb-l2":
        return get_semb_l2_pos(earthpos)
    if body == "earth":
        return earthpos

    if obstime.size == 1:
        try:
            return (
                coords.get_body(body, obstime, ephemeris=ephemeris)
                .transform_to(coords.HeliocentricMeanEcliptic)
                .cartesian.xyz.to_value(units.AU)
            ).flatten()
        except KeyError as error:
            valid_obs = [*coords.solar_system_ephemeris.bodies, "semb-l2"]
            msg = f"Invalid observer string: '{body}'. Valid observers are: {valid_obs}"
            raise ValueError(msg) from error

    if interp_obstimes is None:  # pragma: no cover
        msg = "interp_obstimes must be provided when obstime is an array."
        raise ValueError(msg)

    return get_interp_bodypos(
        body=body,
        obstimes=obstime.mjd,
        interp_obstimes=interp_obstimes,
        ephemeris=ephemeris,
    )
