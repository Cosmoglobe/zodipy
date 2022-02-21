from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple, Union
from scipy import interpolate

import astropy.units as u
from astropy.units import Quantity
from numpy.typing import NDArray
import numpy as np

from zodipy._labels import Label
from zodipy._integration_config import EPS, RADIAL_CUTOFF
from zodipy._model import InterplanetaryDustModel
from zodipy._source_functions import blackbody_emission_nu, interplanetary_temperature


DensityInterpolator = Callable[
    [Union[Tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]],
    NDArray[np.float64],
]

INTERP_RANGES = {
    Label.CLOUD: {
        "R": np.linspace(EPS.value, RADIAL_CUTOFF.value, 1000),
        "Z": np.linspace(-2, 2, 800),
    },
    Label.BAND1: {
        "R": np.linspace(0.5, RADIAL_CUTOFF.value, 1000),
        "Z": np.linspace(-2, 2, 800),
    },
    Label.BAND2: {
        "R": np.linspace(0.2, RADIAL_CUTOFF.value, 1000),
        "Z": np.linspace(-0.7, 0.7, 500),
    },
    Label.BAND3: {
        "R": np.linspace(0.5, RADIAL_CUTOFF.value, 1000),
        "Z": np.linspace(-2.5, 2.5, 900),
    },
    Label.RING: {"R": np.linspace(0.5, 1.5, 300), "Z": np.linspace(-0.5, 0.5, 500)},
    Label.FEATURE: {
        "R": np.linspace(0.5, 1.5, 300),
        "Z": np.linspace(-0.5, 0.5, 500),
    },
}


def get_density_interpolators(
    model: InterplanetaryDustModel,
) -> Dict[Label, Optional[DensityInterpolator]]:
    """Returns a dictionary containing the tabulated densities of each component."""

    interpolators: Dict[Label, Optional[DensityInterpolator]] = {}
    for label, component in model.components.items():
        R = INTERP_RANGES[label]["R"]
        Z = INTERP_RANGES[label]["Z"]
        RR, ZZ = np.meshgrid(R, Z, indexing="ij")

        # The Earth-trailing feature is not stationary in and cannot be
        # easily tabulated.
        if label is Label.FEATURE:
            interpolators[label] = None

        else:
            tabulated_density = component.compute_density(R_prime=RR, Z_prime=ZZ)
            interpolators[label] = interpolate.RegularGridInterpolator(
                points=(R, Z),
                values=tabulated_density,
                bounds_error=False,
                fill_value=0.0,
            )

    return interpolators


@lru_cache
def tabulated_blackbody_emission_nu(
    freq: float,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Returns tabulated, cached array of blackbody emission."""

    T_range = np.linspace(50, 10000, 5000)
    tabulated_bnu = blackbody_emission_nu(T=T_range, freq=freq)

    return interpolate.interp1d(T_range, tabulated_bnu)


def interp_blackbody_emission_nu(
    freq: float, T: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Returns the interpolated black body emission for a temperature."""

    f = tabulated_blackbody_emission_nu(freq=freq)
    try:
        return f(T)
    except ValueError:
        return blackbody_emission_nu(T=T, freq=freq)


@lru_cache
def tabulated_interplanetary_temperature(
    T_0: float, delta: float
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Returns tabulated, cached array of interplanetary temperatures."""

    R_range = np.linspace(EPS.value, 15, 5000)
    tabulated_T = interplanetary_temperature(R_range, T_0, delta)

    return interpolate.interp1d(R_range, tabulated_T)


def interp_interplanetary_temperature(
    R: NDArray[np.float64], T_0: float, delta: float
) -> NDArray[np.float64]:
    """Returns the intepolated interplanetary temperature."""

    f = tabulated_interplanetary_temperature(T_0=T_0, delta=delta)

    return f(R)


def interp_source_parameters(
    freq: Quantity[u.GHz],
    model: InterplanetaryDustModel,
    component: Label,
) -> Dict[str, Any]:
    """Returns interpolated source parameters given a frequency and a component."""

    parameters: Dict[str, Any] = {}
    for component in model.component_labels:
        emissivities = model.source_component_parameters.get("emissivities")
        if emissivities is not None:
            emissivity_spectrum = emissivities["spectrum"]
            parameters["emissivity"] = np.interp(
                freq.to(emissivity_spectrum.unit, equivalencies=u.spectral()),
                emissivity_spectrum,
                emissivities[component],
            )

        else:
            parameters["emissivity"] = 1.0

        albedos = model.source_component_parameters.get("albedos")
        if albedos is not None:
            albedo_spectrum = albedos["spectrum"]
            parameters["albedo"] = np.interp(
                freq.to(albedo_spectrum.unit, equivalencies=u.spectral()),
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
                    freq.to(phase_spectrum.unit, equivalencies=u.spectral()),
                    phase_spectrum,
                    phases[coeff],
                ).value
                for coeff in ["C0", "C1", "C2"]
            }
        else:
            parameters["phase_coeffs"] = {"C0": 0.0, "C1": 0.0, "C2": 0.0}

    return parameters
