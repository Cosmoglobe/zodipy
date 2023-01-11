from typing import Sequence, Tuple, Union

import astropy.units as u
import healpy as hp
import numpy as np
import numpy.typing as npt

from ._ipd_model import InterplanetaryDustModel
from ._types import FrequencyOrWavelength, Pixels, SkyAngles


def get_validated_freq(
    freq: FrequencyOrWavelength, model: InterplanetaryDustModel, extrapolate: bool
) -> FrequencyOrWavelength:
    """Validate user inputted frequency."""
    if not isinstance(freq, u.Quantity):
        raise TypeError("Frequency must be an astropy Quantity.")

    if freq.unit.is_equivalent(u.Hz):
        freq = freq.to(u.Hz)
    elif freq.unit.is_equivalent(u.micron):
        freq = freq.to(u.micron)
    else:
        raise u.UnitsError("Frequency must be in units compatible with Hz or micron.")

    if extrapolate:
        return freq

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
            f"Model is only valid in the [{lower_freq_range},"
            f" {upper_freq_range}] range."
        )

    return freq


def get_validated_and_normalized_weights(
    weights: Union[Sequence[float], npt.NDArray[np.floating], None],
    freq: FrequencyOrWavelength,
) -> npt.NDArray[np.float64]:
    """Validate user inputted weights."""
    if weights is None and freq.size > 1:
        raise ValueError(
            "Bandpass weights must be specified if more than one frequency is given."
        )

    if weights is not None:
        if freq.size != len(weights):
            raise ValueError("Number of frequencies and weights must be the same.")
        elif np.any(np.diff(freq) < 0):
            raise ValueError("Bandpass frequencies must be strictly increasing.")

        normalized_weights = np.asarray(weights, dtype=np.float64)
    else:
        normalized_weights = np.array([1], dtype=np.float64)

    if normalized_weights.size > 1:
        return normalized_weights / np.trapz(normalized_weights, freq.value)

    return normalized_weights


@u.quantity_input(theta=[u.deg, u.rad], phi=[u.deg, u.rad])
def get_validated_ang(
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


def get_validated_pix(pixels: Pixels, nside: int) -> Pixels:
    """Validate user inputted pixels."""
    if (np.max(pixels) > hp.nside2npix(nside)) or (np.min(pixels) < 0):
        raise ValueError("invalid pixel number given nside")

    if np.ndim(pixels) == 0:
        pixels = np.expand_dims(pixels, axis=0)

    return pixels
