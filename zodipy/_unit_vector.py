from __future__ import annotations
from typing import Sequence

from astropy.units import Quantity
import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import NDArray


def get_unit_vector_from_pixels(
    coord_in: str, pixels: Sequence[int] | NDArray[np.integer], nside: int
) -> NDArray[np.floating]:
    """Returns ecliptic unit vectors from HEALPix pixels representing some pointing."""

    unit_vectors = np.asarray(hp.pix2vec(nside, pixels))

    return np.asarray(hp.Rotator(coord=[coord_in, "E"])(unit_vectors))


def get_unit_vector_from_angles(
    coord_in: str,
    phi: Quantity[u.rad] | Quantity[u.deg],
    theta: Quantity[u.rad] | Quantity[u.deg],
    lonlat: bool = False,
) -> NDArray[np.floating]:
    """Returns ecliptic unit vectors from sky angles representing some pointing."""

    unit_vectors = np.asarray(hp.ang2vec(theta, phi, lonlat=lonlat)).transpose()

    return np.asarray(hp.Rotator(coord=[coord_in, "E"])(unit_vectors))
