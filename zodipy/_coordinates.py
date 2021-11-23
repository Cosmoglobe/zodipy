from typing import Optional, Sequence, Union, Dict

from astroquery.jplhorizons import Horizons
import healpy as hp
import numpy as np

from zodipy._exceptions import TargetNotSupportedError

EpochsType = Optional[Union[float, Sequence[float], Dict[str, str]]]


TARGET_IDS = {
    "planck": "Planck",
    "wmap": "WMAP",
    "l2": "32",
    "sun": "10",
    "mercury": "199",
    "venus": "299",
    "earth": "399",
    "mars": "499",
    "jupiter": "599",
}


def get_target_coordinates(
    target: str,
    epochs: Optional[EpochsType] = None,
) -> np.ndarray:
    """Returns the heliocentric coordinates of the target given an epoch.

    Parameters
    ----------
    target
        Name of the target body.
    epochs
        Either a list of epochs in JD or MJD format or a dictionary
        defining a range of times and dates; the range dictionary has to
        be of the form {``'start'``:'YYYY-MM-DD [HH:MM:SS]',
        ``'stop'``:'YYYY-MM-DD [HH:MM:SS]', ``'step'``:'n[y|d|h|m|s]'}.
        If no epochs are provided, the current time is used.
    return_dates
        Boolean for wheter or not to returns the Julian dates for each
        target location.

    Returns
    -------
    coordinates
        Heliocentric cartesian coordinates of the target.
    dates, optional
        Julian dates of each target coordinate.
    """

    try:
        target = TARGET_IDS[target.lower()]
    except KeyError:
        raise TargetNotSupportedError(
            f"{target} is not a valid observer. Please select one of the following: "
            f"{', '.join(TARGET_IDS)}."
        )

    if epochs is None:
        epochs = [2459215.50000]  # 01-01-2021

    query = Horizons(id=target, id_type="majorbody", location="c@sun", epochs=epochs)
    ephemerides = query.ephemerides()

    R = ephemerides["r"].value
    lon, lat = ephemerides["EclLon"].value, ephemerides["EclLat"].value
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    coordinates = np.stack((x, y, z), axis=1)

    return coordinates


def to_frame(input_map: np.ndarray, coord_out: str, coord_in: str = "E") -> np.ndarray:
    """Rotates a HEALPIX map to another reference frame.

    Parameters
    ----------
    input_map
        Map to rotate.
    coord_out
        Coordinate system of the output_map.
    coord_in
        Coordinate system of the input_map.

    Returns
    -------
    output_map
        Rotated map.
    """

    if coord_in == coord_out:
        return input_map

    nside = hp.get_nside(input_map)
    npix = hp.nside2npix(nside)

    rotator = hp.Rotator(coord=[coord_out, coord_in])
    new_angles = rotator(hp.pix2ang(nside, np.arange(npix)))
    new_pixels = hp.ang2pix(nside, *new_angles)
    output_map = input_map[..., new_pixels]

    return output_map
