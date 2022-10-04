from typing import Tuple, Union

import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from ._typing import FrequencyOrWavelength, Pixels, SkyAngles


@u.quantity_input(equivalencies=u.spectral())
def validate_frequency(
    freq: FrequencyOrWavelength, spectrum: Union[FrequencyOrWavelength, None]
):
    # If model is not set to extrapolate, we raise an error if the bandpass is out
    # of range with the supported spetrum.
    if spectrum is not None:
        freq = freq.to(spectrum.unit, equivalencies=u.spectral())
        spectrum_min = spectrum.min()
        spectrum_max = spectrum.max()

        if freq.isscalar:
            freq_is_in_range = spectrum_min <= freq <= spectrum_max
        else:
            freq_is_in_range = all(
                spectrum_min.value <= freq <= spectrum_max.value and freq
                for freq in freq.value
            )

        if not freq_is_in_range:
            raise ValueError(
                f"The selected model is only valid in the [{spectrum_min},"
                f" {spectrum_max}] range."
            )

    return freq


@u.quantity_input(weights=[u.MJy / u.sr, None])
def validate_weights(
    freq: FrequencyOrWavelength, weights: Union[u.Quantity[u.MJy / u.sr], None]
) -> Union[NDArray[np.floating], None]:
    if weights is not None:
        return (weights / np.trapz(weights, freq)).value

    if weights is None and not freq.isscalar:
        print("Warning: weights not provided, assuming uniform weights.")
        return np.ones_like(freq).value / len(freq)


@u.quantity_input(theta=[u.deg, u.rad], phi=[u.deg, u.rad])
def validate_ang(
    theta: SkyAngles, phi: SkyAngles, lonlat: bool
) -> Tuple[SkyAngles, SkyAngles]:
    theta = theta.to(u.deg) if lonlat else theta.to(u.rad)
    phi = phi.to(u.deg) if lonlat else phi.to(u.rad)

    if theta.isscalar:
        theta = u.Quantity([theta])
    if phi.isscalar:
        phi = u.Quantity([phi])

    return theta, phi


def validate_pixels(pixels: Pixels, nside: int) -> Pixels:
    if (np.max(pixels) > hp.nside2npix(nside)) or (np.min(pixels) < 0):
        raise ValueError("invalid pixel number given nside")

    if np.ndim(pixels) == 0:
        pixels = np.expand_dims(pixels, axis=0)

    return pixels
