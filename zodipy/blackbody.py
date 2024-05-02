import numpy as np
import numpy.typing as npt
from astropy import units
from astropy.modeling.physical_models import BlackBody
from scipy import integrate

MIN_TEMP = 40 * units.K
MAX_TEMP = 550 * units.K
N_TEMPS = 100
temperatures = np.linspace(MIN_TEMP, MAX_TEMP, N_TEMPS)
blackbody = BlackBody(temperatures)


def get_dust_grain_temperature(
    R: npt.NDArray[np.float64], T_0: float, delta: float
) -> npt.NDArray[np.float64]:
    """Return the dust grain temperature given a radial distance from the Sun.

    Args:
        R: Radial distance from the sun in ecliptic heliocentric coordinates [AU / 1AU].
        T_0: Temperature of dust grains located 1 AU from the Sun [K].
        delta: Powerlaw index.

    Returns:
        Dust grain temperature [K].

    """
    return T_0 * R**-delta


def tabulate_center_wavelength_bnu(wavelength: units.Quantity) -> npt.NDArray[np.float64]:
    """Tabulate blackbody specific intensity for a center wavelength."""
    return np.asarray(
        [
            temperatures.to_value(units.K),
            blackbody(wavelength).to_value(units.MJy / units.sr),
        ]
    )


def tabulate_bandpass_integrated_bnu(
    wavelengths: units.Quantity, normalized_weights: units.Quantity
) -> npt.NDArray[np.float64]:
    """Tabulate bandpass integrated blackbody specific intensity for a range of temperatures."""
    blackbody_emission = blackbody(wavelengths[:, np.newaxis])
    integrated_blackbody_emission = integrate.trapezoid(
        normalized_weights * blackbody_emission.transpose(), wavelengths
    )
    return np.asarray(
        [
            temperatures.to_value(units.K),
            integrated_blackbody_emission.to_value(units.MJy / units.sr),
        ]
    )
