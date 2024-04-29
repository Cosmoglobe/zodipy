from __future__ import annotations

import numpy as np
import numpy.typing as npt
from astropy import units

from ._ipd_model import InterplanetaryDustModel


def get_validated_freq(
    freq: units.Quantity, model: InterplanetaryDustModel, extrapolate: bool
) -> units.Quantity:
    """Validate user inputted frequency."""
    if not isinstance(freq, units.Quantity):
        msg = "Frequency must be an astropy Quantity."
        raise TypeError(msg)

    if freq.unit.is_equivalent(units.Hz):
        freq = freq.to(units.Hz)
    elif freq.unit.is_equivalent(units.micron):
        freq = freq.to(units.micron)
    else:
        msg = "Frequency must be in units compatible with Hz or micron."
        raise units.UnitsError(msg)

    if extrapolate:
        return freq

    freq_in_spectrum_units = freq.to(model.spectrum.unit, equivalencies=units.spectral())
    lower_freq_range = model.spectrum.min()
    upper_freq_range = model.spectrum.max()

    if freq_in_spectrum_units.isscalar:
        freq_is_in_range = lower_freq_range <= freq_in_spectrum_units <= upper_freq_range
    else:
        freq_is_in_range = all(
            lower_freq_range.value <= nu <= upper_freq_range.value and nu
            for nu in freq_in_spectrum_units.value
        )

    if not freq_is_in_range:
        msg = f"Model is only valid in the [{lower_freq_range}," f" {upper_freq_range}] range."
        raise ValueError(msg)

    return freq


def get_validated_and_normalized_weights(
    weights: npt.ArrayLike | None,
    freq: units.Quantity,
) -> npt.NDArray[np.float64]:
    """Validate user inputted weights."""
    if weights is None and freq.size > 1:
        msg = "Bandpass weights must be specified if more than one frequency is given."
        raise ValueError(msg)

    if weights is not None:
        normalized_weights = np.asarray(weights, dtype=np.float64)
        if freq.size != len(normalized_weights):
            msg = "Number of frequencies and weights must be the same."
            raise ValueError(msg)
        if np.any(np.diff(freq) < 0):
            msg = "Bandpass frequencies must be strictly increasing."
            raise ValueError(msg)

    else:
        normalized_weights = np.array([1], dtype=np.float64)

    if normalized_weights.size > 1:
        return normalized_weights / np.trapz(normalized_weights, freq.value)

    return normalized_weights


def get_validated_ang(
    theta: units.Quantity, phi: units.Quantity, lonlat: bool
) -> tuple[units.Quantity, units.Quantity]:
    """Validate user inputted sky angles and make sure it adheres to the healpy convention."""
    try:
        theta = theta.to(units.deg) if lonlat else theta.to(units.rad)
        phi = phi.to(units.deg) if lonlat else phi.to(units.rad)
    except AttributeError:
        msg = "Sky angles `theta` and `phi` must be astropy Quantities."
        raise TypeError(msg) from AttributeError

    if theta.isscalar:
        theta = np.expand_dims(theta, axis=0)
    if phi.isscalar:
        phi = np.expand_dims(phi, axis=0)

    if not lonlat:
        theta, phi = phi, (np.pi / 2) * units.rad - theta

    return theta, phi
