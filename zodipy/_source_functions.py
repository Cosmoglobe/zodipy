from typing import Any, Dict, Tuple

import astropy.constants as const
import astropy.units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel


h = const.h
c = const.c
k_B = const.k_B
R_sun = const.R_sun
T_sun = 5778 * u.K


@u.quantity_input
def blackbody_emission_nu(
    T: Quantity[u.K],
    freq: Quantity[u.Hz],
) -> Quantity[u.W / u.m ** 2 / u.Hz / u.sr]:
    """Returns the blackbody emission.

    Parameters
    ----------
    T
        Temperature of the blackbody.
    freq
        Frequency.

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    term1 = (2 * h * freq ** 3) / c ** 2
    term2 = np.expm1(((h * freq) / (k_B * T)))

    return term1 / term2 / u.sr


@u.quantity_input
def blackbody_emission_lambda(
    T: Quantity[u.K],
    freq: Quantity[u.micron],
) -> Quantity[u.W / u.m ** 3 / u.sr]:
    """Returns the blackbody emission.

    Parameters
    ----------
    T
        Temperature of the blackbody.
    freq
        Frequency.

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    term1 = (2 * h * c ** 2) / freq ** 5
    term2 = np.expm1(((h * c) / (freq * k_B * T)))

    return term1 / term2 / u.sr


@u.quantity_input
def solar_flux(
    R: Quantity[u.AU], freq: Quantity[u.Hz], T: Quantity[u.K] = T_sun
) -> Quantity[u.W / u.m ** 2 / u.Hz / u.sr]:
    """Returns the solar flux observed at some distance R from the Sun in AU.

    Parameteers
    -----------
    freq
        Frequency [Hz].
    T
        Temperature [K].

    Returns
    -------
        Solar flux at some distance R from the Sun in AU.
    """

    return np.pi * blackbody_emission_nu(T=T, freq=freq) * (R_sun / R) ** 2


@u.quantity_input
def interplanetary_temperature(
    R: Quantity[u.AU], T_0: Quantity[u.K], delta: float
) -> Quantity[u.K]:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic heliocentric coordinates [AU].
    T_0
        Temperature of the solar system at 1 AU [K].
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature [K].
    """

    return T_0 * (R / u.AU) ** -delta


@u.quantity_input
def phase_function(
    Theta: Quantity[u.rad],
    coeffs: Dict[str, Quantity[1 / u.sr, 1 / (u.rad * u.sr), 1 / u.rad]],
) -> NDArray[np.float64]:
    """Returns the phase function.

    Parameters
    ----------
    Theta
        Scattering angle.
    coeffs
        Phase function parameters.

    Returns
    -------
        The Phase funciton.
    """

    C = coeffs["C_0"].value, coeffs["C_1"].value, coeffs["C_2"].value
    π = np.pi
    Θ = Theta.value

    # Analytic integral to N:
    int_term1 = 2 * π
    int_term2 = 2 * C[0]
    int_term3 = π * C[1]
    int_term4 = (np.exp(C[2] * π) + 1) / (C[2] ** 2 + 1)
    N = 1 / (int_term1 * (int_term2 + int_term3 + int_term4))

    return N * (C[0] + C[1] * Θ + np.exp(C[2] * Θ))


def get_interpolated_source_parameters(
    freq: Quantity[u.Hz, u.micron, u.m],
    model: InterplanetaryDustModel,
    component: Label,
) -> Tuple[float, float, Dict[str, Any]]:
    """Returns interpolated source parameters given a frequency and a component."""

    emissivities = model.source_component_parameters.get("emissivities")
    if emissivities is not None:
        emissivity_spectrum = emissivities["spectrum"]
        emissivity = np.interp(
            freq.to(emissivity_spectrum.unit, equivalencies=u.spectral()),
            emissivity_spectrum,
            emissivities[component],
        )

    else:
        emissivity = 1.0

    albedos = model.source_component_parameters.get("albedos")
    if albedos is not None:
        albedo_spectrum = albedos["spectrum"]
        albedo = np.interp(
            freq.to(albedo_spectrum.unit, equivalencies=u.spectral()),
            albedo_spectrum,
            albedos[component],
        )
    else:
        albedo = 0.0

    phases = model.source_parameters.get("phase")
    if phases is not None:
        phase_spectrum = phases["spectrum"]
        phase_coefficients = {
            coeff: np.interp(
                freq.to(phase_spectrum.unit, equivalencies=u.spectral()),
                phase_spectrum,
                phases[coeff],
            )
            for coeff in ["C_0", "C_1", "C_2"]
        }
    else:
        phase_coefficients = {"C_0": 0.0, "C_1": 0.0, "C_2": 0.0}

    return emissivity, albedo, phase_coefficients
