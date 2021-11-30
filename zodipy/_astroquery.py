from typing import Optional, Sequence, Union, Dict

from astroquery.jplhorizons import Horizons
import numpy as np


TARGET_MAPPINGS = {
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


def query_target_positions(
    target: str,
    epochs: Optional[Union[float, Sequence[float], Dict[str, str]]] = None,
) -> np.ndarray:
    """Returns the heliocentric cartesian positions of the target given some epoch.

    Parameters
    ----------
    target
        Name of the target body. Must be a valid target in the JPL Horizons API.
    epochs
        Either a list of epochs in JD or MJD format or a dictionary
        defining a range of times and dates; the range dictionary has to
        be of the form {``'start'``:'YYYY-MM-DD [HH:MM:SS]',
        ``'stop'``:'YYYY-MM-DD [HH:MM:SS]', ``'step'``:'n[y|d|h|m|s]'}.
        If no epochs are provided, the current time is used.

    Returns
    -------
        Heliocentric cartesian positions of the target.
    """

    if (target_lower := target.lower()) in TARGET_MAPPINGS:
        target = TARGET_MAPPINGS[target_lower]

    if epochs is None:
        epochs = 2459215.50000  # 01-01-2021

    query = Horizons(id=target, id_type="majorbody", location="c@sun", epochs=epochs)
    ephemerides = query.ephemerides()

    R = ephemerides["r"].value
    lon, lat = ephemerides["EclLon"].value, ephemerides["EclLat"].value
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)

    return np.stack((x, y, z), axis=1)
