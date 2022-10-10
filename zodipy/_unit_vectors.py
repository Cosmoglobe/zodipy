from __future__ import annotations

from typing import Sequence

import astropy.units as u
import healpy as hp
import numpy as np
import numpy.typing as npt


def get_unit_vectors_from_pixels(
    coord_in: str, pixels: Sequence[int] | npt.NDArray[np.int64], nside: int
) -> npt.NDArray[np.float64]:
    """Returns ecliptic unit vectors from HEALPix pixels representing some pointing."""

    unit_vectors = np.asarray(hp.pix2vec(nside, pixels))

    return np.asarray(hp.Rotator(coord=[coord_in, "E"])(unit_vectors))


def get_unit_vectors_from_ang(
    coord_in: str,
    phi: u.Quantity[u.rad] | u.Quantity[u.deg],
    theta: u.Quantity[u.rad] | u.Quantity[u.deg],
    lonlat: bool = False,
) -> npt.NDArray[np.float64]:
    """Returns ecliptic unit vectors from sky angles representing some pointing."""

    unit_vectors = np.asarray(hp.ang2vec(theta, phi, lonlat=lonlat)).transpose()

    return np.asarray(hp.Rotator(coord=[coord_in, "E"])(unit_vectors))
