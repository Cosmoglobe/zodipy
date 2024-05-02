from __future__ import annotations

import astropy.coordinates as coords
from astropy import time, units

from zodipy._constants import DISTANCE_FROM_EARTH_TO_SEMB_L2

__all__ = ["get_earth_skycoord", "get_obs_skycoord", "string_to_coordinate_frame"]


def get_sun_earth_moon_barycenter_skycoord(earth_skycoord: coords.SkyCoord) -> coords.SkyCoord:
    """Return a SkyCoord of the heliocentric position of the SEMB-L2 point.

    Note that this is an approximate position, as the SEMB-L2 point is not included in
    any of the current available ephemerides. We assume that SEMB-L2 is at all times
    located at a fixed distance from Earth along the vector pointing to Earth from the Sun.
    """
    earth_distance = earth_skycoord.cartesian.norm()
    SEMB_L2_distance = earth_distance + DISTANCE_FROM_EARTH_TO_SEMB_L2
    earth_unit_vector = earth_skycoord.cartesian.xyz / earth_distance

    return coords.SkyCoord(
        *earth_unit_vector * SEMB_L2_distance,
        frame=coords.HeliocentricMeanEcliptic,
        representation_type="cartesian",
    )


def get_earth_skycoord(obs_time: time.Time, ephemeris: str) -> coords.SkyCoord:
    """Return the sky coordinates of the Earth in the heliocentric frame."""
    return coords.get_body("earth", obs_time, ephemeris=ephemeris).transform_to(
        coords.HeliocentricMeanEcliptic
    )


def get_obs_skycoord(
    obs_pos: units.Quantity | str,
    obs_time: time.Time,
    earth_skycoord: coords.SkyCoord,
    ephemeris: str,
) -> coords.SkyCoord:
    """Return the sky coordinates of the observer in the heliocentric frame."""
    if isinstance(obs_pos, str):
        if obs_pos.lower() == "semb-l2":
            return get_sun_earth_moon_barycenter_skycoord(earth_skycoord)
        return coords.get_body(obs_pos, obs_time, ephemeris=ephemeris).transform_to(
            coords.HeliocentricMeanEcliptic
        )

    try:
        return coords.SkyCoord(
            *obs_pos,
            frame=coords.HeliocentricMeanEcliptic,
            representation_type="cartesian",
        )
    except AttributeError:
        msg = "Observer position (`obs_pos`) must be a string or an astropy Quantity."
        raise TypeError(msg) from AttributeError


def string_to_coordinate_frame(frame_literal: str) -> type[coords.BaseCoordinateFrame]:
    """Return the appropriate astropy coordinate frame class from a string literal."""
    if frame_literal == "E":
        return coords.BarycentricMeanEcliptic
    if frame_literal == "G":
        return coords.Galactic
    if frame_literal == "C":
        return coords.ICRS

    msg = (
        f"Invalid frame literal: {frame_literal}. Must be one of 'E' (Ecliptic),"
        "'G' (Galactic), or 'C' (Celestial)."
    )
    raise ValueError(msg)
