from typing import Any, Dict, Tuple, Union

import astropy.constants as const
import astropy.units as u
from astropy.units import Quantity
import numpy as np
from numpy.typing import NDArray

from zodipy._labels import Label
from zodipy._model import InterplanetaryDustModel


h = const.h.value
c = const.c.value
k_B = const.k_B.value
R_sun = const.R_sun.value
T_sun = 5778


def blackbody_emission_nu(
    T: Union[float, NDArray[np.float64]], freq: Union[float, NDArray[np.float64]]
) -> NDArray[np.float64]:
    """Returns the blackbody emission.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    freq
        Frequency [GHz] .

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """
    freq *= 1e9
    term1 = (2 * h * freq ** 3) / c ** 2
    term2 = np.expm1(((h * freq) / (k_B * T)))

    return term1 / term2


def blackbody_emission_lambda(
    T: Union[float, NDArray[np.float64]],
    freq:Union[float, NDArray[np.float64]],
) -> Union[float, NDArray[np.float64]]:
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
    freq *= 1e-6
    term1 = (2 * h * c ** 2) / freq ** 5
    term2 = np.expm1(((h * c) / (freq * k_B * T)))

    return term1 / term2


def solar_flux(
    R: NDArray[np.float64], freq: float, T: float = T_sun
) -> NDArray[np.float64]:
    """Returns the solar flux observed at some distance R from the Sun in AU.

    Parameteers
    -----------
    freq
        Frequency [GHz].
    T
        Temperature [K].

    Returns
    -------
        Solar flux at some distance R from the Sun in AU.
    """

    return np.pi * blackbody_emission_nu(T=T, freq=freq) * (R_sun / R) ** 2


def interplanetary_temperature(
    R: NDArray[np.float64], T_0: float, delta: float
) -> NDArray[np.float64]:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic heliocentric coordinates [AU / 1AU].
    T_0
        Temperature of the solar system at 1 AU [K].
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature [K].
    """

    return T_0 * R ** -delta


def phase_function(
    Theta: NDArray[np.float64],
    coeffs: Dict[str, float],
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

    C = coeffs["C_0"], coeffs["C_1"], coeffs["C_2"]
    π = np.pi
    Θ = Theta

    # Analytic integral to N:
    int_term1 = 2 * π
    int_term2 = 2 * C[0]
    int_term3 = π * C[1]
    int_term4 = (np.exp(C[2] * π) + 1) / (C[2] ** 2 + 1)
    N = 1 / (int_term1 * (int_term2 + int_term3 + int_term4))

    return N * (C[0] + C[1] * Θ + np.exp(C[2] * Θ))


def get_source_parameters(
    freq: float,
    model: InterplanetaryDustModel,
    component: Label,
) -> Dict[str, Any]:
    """Returns interpolated source parameters given a frequency and a component."""

    freq_ = freq * u.GHz

    parameters: Dict[str, Any] = {}

    emissivities = model.source_component_parameters.get("emissivities")
    if emissivities is not None:
        emissivity_spectrum = emissivities["spectrum"]
        parameters["emissivity"] = np.interp(
            freq_.to(emissivity_spectrum.unit, equivalencies=u.spectral()),
            emissivity_spectrum,
            emissivities[component],
        )

    else:
        parameters["emissivity"] = 1.0

    albedos = model.source_component_parameters.get("albedos")
    if albedos is not None:
        albedo_spectrum = albedos["spectrum"]
        parameters["albedo"] = np.interp(
            freq_.to(albedo_spectrum.unit, equivalencies=u.spectral()),
            albedo_spectrum,
            albedos[component],
        )
    else:
        parameters["albedo"] = 0.0

    phases = model.source_parameters.get("phase")
    if phases is not None:
        phase_spectrum = phases["spectrum"]
        parameters["phase_coeffs"] = {
            coeff: np.interp(
                freq_.to(phase_spectrum.unit, equivalencies=u.spectral()),
                phase_spectrum,
                phases[coeff],
            ).value
            for coeff in ["C_0", "C_1", "C_2"]
        }
    else:
        parameters["phase_coeffs"] = {"C_0": 0.0, "C_1": 0.0, "C_2": 0.0}

    # Extract model parameters
    parameters["T_0"] = model.source_parameters["T_0"].value
    parameters["delta"] = model.source_parameters["delta"]
    parameters["cloud_offset"] = model.components[component].X_0

    return parameters
