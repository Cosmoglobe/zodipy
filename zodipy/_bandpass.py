from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import astropy.units as u
import numpy as np

from zodipy._constants import (
    MAX_INTERPOLATION_GRID_TEMPERATURE,
    MIN_INTERPOLATION_GRID_TEMPERATURE,
    N_INTERPOLATION_POINTS,
)
from zodipy._source_funcs import get_blackbody_emission
from zodipy._validators import get_validated_and_normalized_weights, get_validated_freq

if TYPE_CHECKING:
    import numpy.typing as npt

    from zodipy._ipd_model import InterplanetaryDustModel
    from zodipy._types import FrequencyOrWavelength


@dataclass
class Bandpass:
    frequencies: FrequencyOrWavelength
    weights: npt.NDArray[np.float64]

    def integrate(self, quantity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Integrate a quantity over the bandpass."""
        return np.trapz(self.weights * quantity, self.frequencies.value, axis=-1)

    def switch_convention(self) -> None:
        """Switch the bandpass from frequency to wavelength or the other way around."""
        self.frequencies = self.frequencies.to(
            u.micron if self.frequencies.unit.is_equivalent(u.Hz) else u.Hz,
            equivalencies=u.spectral(),
        )
        if self.frequencies.size > 1:
            self.frequencies = np.flip(self.frequencies)
            self.weights = np.flip(self.weights)
            self.weights /= np.trapz(self.weights, self.frequencies.value)


def validate_and_get_bandpass(
    freq: FrequencyOrWavelength,
    weights: Sequence[float] | npt.NDArray[np.floating] | None,
    model: InterplanetaryDustModel,
    extrapolate: bool,
) -> Bandpass:
    """Validate bandpass and return a `Bandpass`."""
    freq = get_validated_freq(freq, model, extrapolate)
    normalized_weights = get_validated_and_normalized_weights(weights, freq)

    return Bandpass(freq, normalized_weights)


def get_bandpass_interpolation_table(
    bandpass: Bandpass,
    n_points: int = N_INTERPOLATION_POINTS,
    min_temp: float = MIN_INTERPOLATION_GRID_TEMPERATURE,
    max_temp: float = MAX_INTERPOLATION_GRID_TEMPERATURE,
) -> npt.NDArray[np.float64]:
    """Pre-compute the bandpass integrated blackbody emission for a grid of temperatures."""
    if not bandpass.frequencies.unit.is_equivalent(u.Hz):
        bandpass.switch_convention()

    bandpass_integrals = np.zeros(n_points)
    temperature_grid = np.linspace(min_temp, max_temp, n_points)
    for idx, temp in enumerate(temperature_grid):
        blackbody_emission = get_blackbody_emission(bandpass.frequencies.value, temp)
        bandpass_integrals[idx] = (
            bandpass.integrate(blackbody_emission)
            if bandpass.frequencies.size > 1
            else blackbody_emission
        )

    return np.asarray([temperature_grid, bandpass_integrals])
