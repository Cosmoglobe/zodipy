from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import astropy.units as u
import numpy as np
import numpy.typing as npt

from zodipy._ipd_model import InterplanetaryDustModel
from zodipy._validators import validate_and_normalize_weights, validate_frequencies
from zodipy._source_funcs import get_blackbody_emission
from zodipy._constants import (
    N_INTERPOLATION_POINTS,
    MIN_INTERPOLATION_GRID_TEMPERATURE,
    MAX_INTERPOLATION_GRID_TEMPERATURE,
)


@dataclass
class Bandpass:
    frequencies: u.Quantity[u.Hz] | u.Quantity[u.micron]
    weights: npt.NDArray[np.float64]

    def integrate(self, quantity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Integrate a quantity over the bandpass."""

        return np.trapz(self.weights * quantity, self.frequencies.value, axis=-1)

    def switch_convention(self) -> None:
        """Switched the bandpass from frequency to wavelength or vice versa."""

        self.frequencies = self.frequencies.to(
            u.micron if self.frequencies.unit.is_equivalent(u.Hz) else u.Hz,
            equivalencies=u.spectral(),
        )
        if self.frequencies.size > 1:
            self.frequencies = np.flip(self.frequencies)
            self.weights = np.flip(self.weights)
            self.weights /= np.trapz(self.weights, self.frequencies.value)


def validate_and_get_bandpass(
    freq: u.Quantity[u.Hz] | u.Quantity[u.micron],
    weights: Sequence[float] | npt.NDArray[np.floating] | None,
    model: InterplanetaryDustModel,
    extrapolate: bool,
) -> Bandpass:
    """Validate user inputted bandpass and return a Bandpass object."""

    validate_frequencies(freq, model, extrapolate)
    normalized_weights = validate_and_normalize_weights(weights, freq)

    return Bandpass(freq, normalized_weights)


def get_bandpass_interpolation_table(
    bandpass: Bandpass,
    n_points: int = N_INTERPOLATION_POINTS,
    min_temp: float = MIN_INTERPOLATION_GRID_TEMPERATURE,
    max_temp: float = MAX_INTERPOLATION_GRID_TEMPERATURE,
) -> npt.NDArray[np.float64]:
    """Pre-compute the bandpass integrated blackbody emission for a grid of temperatures."""

    # Prepare bandpass to be integrated in power units and in frequency convention.
    if not bandpass.frequencies.unit.is_equivalent(u.Hz):
        bandpass.switch_convention()

    freqs = (
        np.expand_dims(bandpass.frequencies.value, axis=0)
        if bandpass.frequencies.size == 1
        else bandpass.frequencies.value
    )
    weights = bandpass.weights

    integrals = np.zeros(n_points)
    temp_grid = np.linspace(min_temp, max_temp, n_points)
    for idx, temp in enumerate(temp_grid):
        freq_scaling = get_blackbody_emission(freqs, temp)
        integrals[idx] = (
            np.trapz(freq_scaling * weights, freqs) if freqs.size > 1 else freq_scaling
        )

    return np.asarray([temp_grid, integrals])
