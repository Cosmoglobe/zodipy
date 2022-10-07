from __future__ import annotations

from dataclasses import asdict
from functools import partial
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ._ipd_comps import Band, Cloud, Component, Feature, Ring


def compute_cloud_density(
    X_helio: NDArray[np.floating],
    X_0: NDArray[np.floating],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    mu: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> NDArray[np.floating]:
    """
    Returns the dust density of a component at points in the Solar System
    given by 'X_helio' and the parameters of the component.
    """

    X_cloud = X_helio - X_0
    R_cloud = np.sqrt(X_cloud[0] ** 2 + X_cloud[1] ** 2 + X_cloud[2] ** 2)

    Z_cloud = (
        X_cloud[0] * sin_Omega_rad * sin_i_rad
        - X_cloud[1] * cos_Omega_rad * sin_i_rad
        + X_cloud[2] * cos_i_rad
    )

    ζ = np.abs(Z_cloud / R_cloud)
    μ = mu
    g = np.zeros_like(ζ)

    condition = ζ < μ
    g[condition] = ζ[condition] ** 2 / (2 * μ)
    g[~condition] = ζ[~condition] - (μ / 2)

    return n_0 * R_cloud**-alpha * np.exp(-beta * g**gamma)


def compute_band_density(
    X_helio: NDArray[np.floating],
    X_0: NDArray[np.floating],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    delta_zeta_rad: float,
    p: float,
    v: float,
    delta_r: float,
) -> NDArray[np.floating]:
    """
    Returns the dust density of a component at points in the Solar System
    given by 'X_helio' and the parameters of the component.
    """

    X_band = X_helio - X_0
    R_band = np.sqrt(X_band[0] ** 2 + X_band[1] ** 2 + X_band[2] ** 2)

    Z_band = (
        X_band[0] * sin_Omega_rad * sin_i_rad
        - X_band[1] * cos_Omega_rad * sin_i_rad
        + X_band[2] * cos_i_rad
    )

    ζ = np.abs(Z_band / R_band)
    ζ_over_δ_ζ = ζ / delta_zeta_rad
    term1 = 3 * n_0 / R_band
    term2 = np.exp(-(ζ_over_δ_ζ**6))

    # Differs from eq 8 in K98 by a factor of 1/self.v. See Planck XIV
    # section 4.1.2.
    term3 = 1 + (ζ_over_δ_ζ**p) / v

    term4 = 1 - np.exp(-((R_band / delta_r) ** 20))

    return term1 * term2 * term3 * term4


def compute_ring_density(
    X_helio: NDArray[np.floating],
    X_0: NDArray[np.floating],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    R: float,
    sigma_r: float,
    sigma_z: float,
) -> NDArray[np.floating]:
    """
    Returns the dust density of a component at points in the Solar System
    given by 'X_helio' and the parameters of the component.
    """

    X_ring = X_helio - X_0
    R_ring = np.sqrt(X_ring[0] ** 2 + X_ring[1] ** 2 + X_ring[2] ** 2)

    Z_ring = (
        X_ring[0] * sin_Omega_rad * sin_i_rad
        - X_ring[1] * cos_Omega_rad * sin_i_rad
        + X_ring[2] * cos_i_rad
    )
    # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
    # term. See Planck 2013 XIV, section 4.1.3.
    term1 = -((R_ring - R) ** 2) / sigma_r**2
    term2 = np.abs(Z_ring) / sigma_z

    return n_0 * np.exp(term1 - term2)


def compute_feature_density(
    X_helio: NDArray[np.floating],
    X_0: NDArray[np.floating],
    X_earth: NDArray[np.floating],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    R: float,
    sigma_r: float,
    sigma_z: float,
    theta_rad: float,
    sigma_theta_rad: float,
) -> NDArray[np.floating]:
    """
    Returns the dust density of a component at points in the Solar System
    given by 'X_helio' and the parameters of the component.
    """

    X_feature = X_helio - X_0
    R_feature = np.sqrt(X_feature[0] ** 2 + X_feature[1] ** 2 + X_feature[2] ** 2)

    Z_feature = (
        X_feature[0] * sin_Omega_rad * sin_i_rad
        - X_feature[1] * cos_Omega_rad * sin_i_rad
        + X_feature[2] * cos_i_rad
    )
    X_earth_comp = X_earth - X_0

    θ_comp = np.arctan2(X_feature[1], X_feature[0]) - np.arctan2(
        X_earth_comp[1], X_earth_comp[0]
    )

    Δθ = θ_comp - theta_rad
    condition1 = Δθ < -np.pi
    condition2 = Δθ > np.pi
    Δθ[condition1] = Δθ[condition1] + 2 * np.pi
    Δθ[condition2] = Δθ[condition2] - 2 * np.pi

    # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
    # term. See Planck 2013 XIV, section 4.1.3.
    exp_term = (R_feature - R) ** 2 / sigma_r**2
    exp_term += np.abs(Z_feature) / sigma_z
    exp_term += Δθ**2 / sigma_theta_rad**2

    return n_0 * np.exp(-exp_term)


DensFunc = Callable[..., NDArray[np.floating]]
PartialDensFunc = Callable[[NDArray[np.floating]], NDArray[np.floating]]

DENSITY_FUNCS: dict[type[Component], DensFunc] = {
    Cloud: compute_cloud_density,
    Band: compute_band_density,
    Ring: compute_ring_density,
    Feature: compute_feature_density,
}

# Find nicer solution to this
KEYS_TO_REMOVE = (
    "x_0",
    "y_0",
    "z_0",
    "i",
    "Omega",
    "delta_zeta",
    "theta",
    "sigma_theta",
)


def construct_density_funcs(
    comps: Sequence[Component], X_earth: NDArray[np.floating]
) -> tuple[PartialDensFunc]:
    """Construct the density functions for the components."""

    partial_density_funcs: list[PartialDensFunc] = []
    for comp in comps:
        comp_params = asdict(comp)
        for key in KEYS_TO_REMOVE:
            comp_params.pop(key, None)

        partial_func = partial(DENSITY_FUNCS[type(comp)], **comp_params)

        if isinstance(comp, Feature):
            partial_func = partial(partial_func, X_earth=X_earth)
        partial_density_funcs.append(partial_func)

    return tuple(partial_density_funcs)
