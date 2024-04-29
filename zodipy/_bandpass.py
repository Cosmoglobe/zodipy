from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from astropy import units
from astropy.modeling.physical_models import BlackBody

from zodipy._constants import (
    MAX_INTERPOLATION_GRID_TEMPERATURE,
    MIN_INTERPOLATION_GRID_TEMPERATURE,
    N_INTERPOLATION_POINTS,
    SPECIFIC_INTENSITY_UNITS,
)
from zodipy._validators import get_validated_and_normalized_weights, get_validated_freq

if TYPE_CHECKING:
    import numpy.typing as npt

    from zodipy._ipd_model import InterplanetaryDustModel


@dataclass
class Bandpass:
    frequencies: units.Quantity
    weights: npt.NDArray[np.float64]

    def integrate(self, quantity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Integrate a quantity over the bandpass."""
        return np.trapz(self.weights * quantity, self.frequencies.value, axis=-1)

    def switch_convention(self) -> None:
        """Switch the bandpass from frequency to wavelength or the other way around."""
        self.frequencies = self.frequencies.to(
            units.micron if self.frequencies.unit.is_equivalent(units.Hz) else units.Hz,
            equivalencies=units.spectral(),
        )
        if self.frequencies.size > 1:
            self.frequencies = np.flip(self.frequencies)
            self.weights = np.flip(self.weights)
            self.weights /= np.trapz(self.weights, self.frequencies.value)

    @property
    def is_center_frequency(self) -> bool:
        """Return True if the bandpass is centered around a single frequency."""
        return self.frequencies.isscalar


def validate_and_get_bandpass(
    freq: units.Quantity,
    weights: npt.ArrayLike | None,
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
    if not bandpass.frequencies.unit.is_equivalent(units.Hz):
        bandpass.switch_convention()

    temperature_grid = np.linspace(min_temp, max_temp, n_points) * units.K
    blackbody = BlackBody(temperature_grid)

    if bandpass.is_center_frequency:
        blackbody_emission = blackbody(bandpass.frequencies)
        return np.asarray([temperature_grid, blackbody_emission.to_value(SPECIFIC_INTENSITY_UNITS)])

    blackbody_emission = blackbody(bandpass.frequencies[:, np.newaxis])
    integrated_blackbody_emission = np.trapz(
        (bandpass.weights / (1 * units.Hz)) * blackbody_emission.transpose(), bandpass.frequencies
    )
    return np.asarray(
        [temperature_grid, integrated_blackbody_emission.to_value(SPECIFIC_INTENSITY_UNITS)]
    )
