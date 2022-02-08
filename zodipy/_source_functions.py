from typing import Sequence, Union, List, Tuple

import astropy.constants as const
import astropy.units as u
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel


h = const.h.value
c = const.c.value
k_B = const.k_B.value


def blackbody_emission(
    T: Union[float, NDArray[np.float64]],
    freq: Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
    """Returns the blackbody emission for a temperature T and frequency freq.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    freq
        Frequency [GHz].

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    freq *= 1e9
    term1 = (2 * h * freq ** 3) / c ** 2
    term2 = np.expm1((h * freq) / (k_B * T))

    return term1 / term2


def blackbody_emission_wavelen(
    T: Union[float, NDArray[np.float64]], wavelen: Union[float, NDArray[np.float64]]
) -> Union[float, NDArray[np.float64]]:
    """Returns the blackbody emission for a temperature T and wavelength wavelen.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    wavelen
        Frequency [micron].

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    wavelen *= 1e-6
    term1 = (2 * h * c ** 2) / wavelen ** 5
    term2 = np.expm1((h * c) / (wavelen * k_B * T))

    return term1 / term2


def interplanetary_temperature(
    R: NDArray[np.float64], T_0: float, delta: float
) -> NDArray[np.float64]:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic heliocentric coordinates.
    T_0
        Temperature of the solar system at R = 1 AU.
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature.
    """

    return T_0 * R ** -delta


def phase_function(
    Theta: NDArray[np.float64], C: Sequence[float]
) -> NDArray[np.float64]:
    """Returns the phase function.

    Parameters
    ----------
    Theta
        Scattering angle [rad].
    C
        Phase function parameters.

    Returns
    -------
        The Phase funciton.
    """

    phase = C[0] + C[1] * Theta + np.exp(C[2] * Theta)
    N = 1 / (phase.sum())

    return N * phase


def get_interpolated_source_parameters(
    freq: u.Quantity, model: InterplanetaryDustModel, component: Label
) -> Tuple[float, float, List[float]]:
    """Returns interpolated source parameters given a frequency and a component."""

    emissivities = model.source_component_parameters.get("emissivities")
    if emissivities is not None:
        emissivity_spectrum = emissivities["spectrum"]
        emissivity = np.interp(
            freq.to(emissivity_spectrum.unit, equivalencies=u.spectral()).value,
            emissivity_spectrum.value,
            emissivities[component],
        )

    else:
        emissivity = 1.0

    albedos = model.source_component_parameters.get("albedos")
    if albedos is not None:
        albedo_spectrum = albedos["spectrum"]
        albedo = np.interp(
            freq.to(albedo_spectrum.unit, equivalencies=u.spectral()).value,
            albedo_spectrum.value,
            albedos[component],
        )
    else:
        albedo = 0.0

    phases = model.source_parameters.get("phase")
    if phases is not None:
        phase_spectrum = phases["spectrum"]
        phase_coefficients = [
            np.interp(
                freq.to(phase_spectrum.unit, equivalencies=u.spectral()).value,
                phase_spectrum.value,
                phase_coeff,
            )
            for phase_coeff in phases["coefficients"]
        ]
    else:
        phase_coefficients = [0.0, 0.0, 0.0]

    return emissivity, albedo, phase_coefficients
