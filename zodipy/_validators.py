from typing import Tuple

import astropy.units as u
import healpy as hp
import numpy as np

from ._exceptions import FrequencyOutOfBoundsError
from ._typing import FrequencyOrWavelength, Pixels, SkyAngles


@u.quantity_input(freq=[u.GHz, u.m])
def validate_freq(
    freq: FrequencyOrWavelength,
    extrapolate: bool,
    spectrum: FrequencyOrWavelength,
) -> FrequencyOrWavelength:
    """Validate the user inputed frequency or wavelength and converts to units of GHz."""

    if not extrapolate:
        freq = freq.to(spectrum.unit, equivalencies=u.spectral())
        spectrum_min = spectrum.min()
        spectrum_max = spectrum.max()

        if not (spectrum_min <= freq <= spectrum_max):
            raise FrequencyOutOfBoundsError(
                lower_limit=spectrum_min.value,
                upper_limit=spectrum_max.value,
            )

    return freq.to(u.GHz, equivalencies=u.spectral())


@u.quantity_input(theta=[u.deg, u.rad], phi=[u.deg, u.rad])
def validate_ang(
    theta: SkyAngles, phi: SkyAngles, lonlat: bool
) -> Tuple[SkyAngles, SkyAngles]:
    """Validate the user inputed sky angles and reshapes them if nesecarry."""

    theta = theta.to(u.deg) if lonlat else theta.to(u.rad)
    phi = phi.to(u.deg) if lonlat else phi.to(u.rad)

    if theta.isscalar:
        theta = u.Quantity([theta])
    if phi.isscalar:
        phi = u.Quantity([phi])

    return theta, phi


def validate_pixels(pixels: Pixels, nside: int) -> Pixels:
    """Validate the user inputed pixels and reshape them if nesecarry."""

    if (np.max(pixels) > hp.nside2npix(nside)) or (np.min(pixels) < 0):
        raise ValueError("invalid pixel number given nside")

    if np.ndim(pixels) == 0:
        pixels = np.expand_dims(pixels, axis=0)

    return pixels
