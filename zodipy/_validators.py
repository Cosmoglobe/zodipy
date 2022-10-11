from typing import Tuple, Union

import astropy.units as u
import healpy as hp
import numpy as np
import numpy.typing as npt

from ._ipd_model import InterplanetaryDustModel
from ._types import FrequencyOrWavelength, Pixels, SkyAngles


@u.quantity_input(equivalencies=u.spectral())
def validate_frequency_in_model_range(
    freq: FrequencyOrWavelength, model: InterplanetaryDustModel
) -> None:
    """Validate user inputted frequency."""

    freq_in_spectrum_units = freq.to(model.spectrum.unit, equivalencies=u.spectral())

    lower_freq_range = model.spectrum.min()
    upper_freq_range = model.spectrum.max()

    if freq_in_spectrum_units.isscalar:
        freq_is_in_range = (
            lower_freq_range <= freq_in_spectrum_units <= upper_freq_range
        )
    else:
        freq_is_in_range = all(
            lower_freq_range.value <= nu <= upper_freq_range.value and nu
            for nu in freq_in_spectrum_units.value
        )

    if not freq_is_in_range:
        raise ValueError(
            f"Model: {model.name} is only valid in the [{lower_freq_range},"
            f" {upper_freq_range}] range."
        )


@u.quantity_input
def validate_and_normalize_weights(
    freq: FrequencyOrWavelength, weights: Union[u.Quantity[u.MJy / u.sr], None]
) -> Union[npt.NDArray[np.float64], None]:
    """Validate user inputted weights."""

    if weights is not None:
        weights = weights.to(u.MJy / u.sr)
        return (weights / np.trapz(weights, freq)).value

    if weights is None and not freq.isscalar:
        print("Warning: weights not provided, assuming uniform weights.")
        return np.ones_like(freq).value / len(freq)

    return None


@u.quantity_input(theta=[u.deg, u.rad], phi=[u.deg, u.rad])
def validate_ang(
    theta: SkyAngles, phi: SkyAngles, lonlat: bool
) -> Tuple[SkyAngles, SkyAngles]:
    """Validate user inputted sky angles."""

    theta = theta.to(u.deg) if lonlat else theta.to(u.rad)
    phi = phi.to(u.deg) if lonlat else phi.to(u.rad)

    if theta.isscalar:
        theta = u.Quantity([theta])
    if phi.isscalar:
        phi = u.Quantity([phi])

    return theta, phi


def validate_pixels(pixels: Pixels, nside: int) -> Pixels:
    """Validate user inputted pixels."""

    if (np.max(pixels) > hp.nside2npix(nside)) or (np.min(pixels) < 0):
        raise ValueError("invalid pixel number given nside")

    if np.ndim(pixels) == 0:
        pixels = np.expand_dims(pixels, axis=0)

    return pixels
