from __future__ import annotations

import numpy as np
import numpy.typing as npt
from astropy import units
from astropy.modeling.physical_models import BlackBody
from scipy import integrate

MIN_TEMP = 40 * units.K
MAX_TEMP = 550 * units.K
N_TEMPS = 100
TEMPERATURES = np.linspace(MIN_TEMP, MAX_TEMP, N_TEMPS)
blackbody = BlackBody(TEMPERATURES)


def get_dust_grain_temperature(
    R: npt.NDArray[np.float64], T_0: float, delta: float
) -> npt.NDArray[np.float64]:
    """Return the dust grain temperature given at a radial distance from the Sun.

    Args:
        R: Radial distance from the sun in ecliptic heliocentric coordinates [AU / 1AU].
        T_0: Temperature of dust grains located 1 AU from the Sun [K].
        delta: Powerlaw index.

    Returns:
        Dust grain temperature [K].

    """
    return T_0 * R**-delta


def tabulate_blackbody_emission(
    wavelengths: units.Quantity, weights: units.Quantity | None
) -> npt.NDArray[np.float64]:
    """Tabulate bandpass integrated blackbody specific intensity for a range of temperatures."""
    if weights is None:
        tabulated_blackbody_emission = blackbody(wavelengths)
    else:
        tabulated_blackbody_emission = integrate.trapezoid(
            weights * blackbody(wavelengths[:, np.newaxis]).transpose(), wavelengths
        )

    return np.asarray(
        [
            TEMPERATURES.to_value(units.K),
            tabulated_blackbody_emission.to_value(units.MJy / units.sr),
        ]
    )
