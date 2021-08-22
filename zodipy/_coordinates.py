import warnings
from typing import Union, Iterable, Dict

from astroquery.jplhorizons import Horizons
import healpy as hp
import numpy as np


TARGET_ALIASES = {
    'l1' : 'SEMB-L1',
    'l2' : 'SEMB-L2',
    'planck' : 'Planck',
    'wmap' : 'WMAP',
    'earth' : 'Earth-Moon Barycenter',
}


def get_target_coordinates(
    target: str, epochs: Union[float, Iterable, Dict[str, str]],
) -> np.ndarray:
    """Returns the heliocentric cartesian coordinates of the target.
    
    Parameters
    ----------
    target
        Name of the target body.
    epochs : scalar, list-like, or dictionary, optional
            Either a list of epochs in JD or MJD format or a dictionary
            defining a range of times and dates; the range dictionary has to
            be of the form {``'start'``:'YYYY-MM-DD [HH:MM:SS]',
            ``'stop'``:'YYYY-MM-DD [HH:MM:SS]', ``'step'``:'n[y|d|h|m|s]'}.
            If no epochs are provided, the current time is used.

    Returns
    -------
    coordinates
        Heliocentric cartesian coordinates of the target. The shape is 
        (`n_observations`, 3)
    """

    if target.lower() in TARGET_ALIASES:
        target = TARGET_ALIASES[target.lower()]
    else:
        warnings.warn(
            'The K98 model is only valid when observed from a near earth orbit'
        )

    query = Horizons(
        id=target, id_type='majorbody', location='c@sun', epochs=epochs
    )
    ephemerides = query.ephemerides()

    R = ephemerides['r'].value
    lon, lat = ephemerides['EclLon'].value, ephemerides['EclLat'].value
    lon, lat = np.deg2rad(lon), np.deg2rad(lat)

    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    
    coordinates = np.stack((x,y,z), axis=1)

    return coordinates


def change_coordinate_system(
    input_map: np.ndarray, coord_out: str, coord_in: str = 'E'
) -> np.ndarray:
    """Rotates a map from coordinate system to another using healpy.Rotator
    
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