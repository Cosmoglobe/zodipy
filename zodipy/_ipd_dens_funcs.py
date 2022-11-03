from __future__ import annotations

import inspect
from dataclasses import asdict
from functools import partial
from typing import Any, Callable, Sequence

import astropy.units as u
import numpy as np
import numpy.typing as npt

from zodipy._ipd_comps import (
    Band,
    BroadBand,
    Cloud,
    Component,
    Fan,
    Feature,
    FeatureRRM,
    NarrowBand,
    Ring,
    RingRRM,
)

"""The density functions for the different types of components. Common for all of these 
is that the first argument will be `X_helio` (the line of sight from the observer towards
a point on the sky) and the remaining arguments will be parameters that are set by the
`Component` subclasses. These arguments are unpacked automatically, so for a 
ComputeDensityFunc to work, the mapped `Component` class must have all the parameters as 
instance variables. 
"""
ComputeDensityFunc = Callable[..., npt.NDArray[np.float64]]


def compute_cloud_density(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    mu: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> npt.NDArray[np.float64]:
    """Density of the diffuse cloud (see Eq (6). in K98)."""

    X_cloud = X_helio - X_0
    R_cloud = np.sqrt(X_cloud[0] ** 2 + X_cloud[1] ** 2 + X_cloud[2] ** 2)

    Z_cloud = (
        X_cloud[0] * sin_Omega_rad * sin_i_rad
        - X_cloud[1] * cos_Omega_rad * sin_i_rad
        + X_cloud[2] * cos_i_rad
    )

    zeta = np.abs(Z_cloud / R_cloud)

    g = np.where(zeta < mu, zeta**2 / (2 * mu), zeta - (mu / 2))

    return n_0 * R_cloud**-alpha * np.exp(-beta * g**gamma)


def compute_band_density(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    delta_zeta_rad: float,
    p: float,
    v: float,
    delta_r: float,
) -> npt.NDArray[np.float64]:
    """Density of the dust bands (see Eq. (8) in K98)."""

    X_band = X_helio - X_0
    R_band = np.sqrt(X_band[0] ** 2 + X_band[1] ** 2 + X_band[2] ** 2)

    Z_band = (
        X_band[0] * sin_Omega_rad * sin_i_rad
        - X_band[1] * cos_Omega_rad * sin_i_rad
        + X_band[2] * cos_i_rad
    )

    zeta = np.abs(Z_band / R_band)
    zeta_over_delta_zeta = zeta / delta_zeta_rad
    term1 = 3 * n_0 / R_band
    term2 = np.exp(-(zeta_over_delta_zeta**6))

    # Differs from eq 8 in K98 by a factor of 1/self.v. See Planck XIV
    # section 4.1.2.
    term3 = 1 + (zeta_over_delta_zeta**p) / v

    term4 = 1 - np.exp(-((R_band / delta_r) ** 20))

    return term1 * term2 * term3 * term4


def compute_ring_density(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    R: float,
    sigma_r: float,
    sigma_z: float,
) -> npt.NDArray[np.float64]:
    """Density of the circum-solar ring (see Eq. (9) in K98)."""

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
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    X_earth: npt.NDArray[np.float64],
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
) -> npt.NDArray[np.float64]:
    """Density of the Earth-trailing feature (see Eq. (9) in K98)."""

    X_feature = X_helio - X_0
    R_feature = np.sqrt(X_feature[0] ** 2 + X_feature[1] ** 2 + X_feature[2] ** 2)

    Z_feature = (
        X_feature[0] * sin_Omega_rad * sin_i_rad
        - X_feature[1] * cos_Omega_rad * sin_i_rad
        + X_feature[2] * cos_i_rad
    )
    X_earth_comp = X_earth - X_0

    theta_comp = np.arctan2(X_feature[1], X_feature[0]) - np.arctan2(
        X_earth_comp[1], X_earth_comp[0]
    )

    delta_theta = theta_comp - theta_rad
    delta_theta = np.where(delta_theta < -np.pi, +2 * np.pi, delta_theta)
    delta_theta = np.where(delta_theta > np.pi, -2 * np.pi, delta_theta)

    # Differs from eq 9 in K98 by a factor of 1/2 in the first and last
    # term. See Planck 2013 XIV, section 4.1.3.
    exp_term = (R_feature - R) ** 2 / sigma_r**2
    exp_term += np.abs(Z_feature) / sigma_z
    exp_term += delta_theta**2 / sigma_theta_rad**2

    return n_0 * np.exp(-exp_term)


def compute_fan_density(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    Q: float,
    P: float,
    gamma: float,
    Z_0: float,
) -> npt.NDArray[np.float64]:
    """Density of the fan (see Eq (3). in RRM)."""

    X_cloud = X_helio - X_0
    R_cloud = np.sqrt(X_cloud[0] ** 2 + X_cloud[1] ** 2 + X_cloud[2] ** 2)

    Z_cloud = (
        X_cloud[0] * sin_Omega_rad * sin_i_rad
        - X_cloud[1] * cos_Omega_rad * sin_i_rad
        + X_cloud[2] * cos_i_rad
    )
    sin_beta = Z_cloud / R_cloud
    beta = np.arcsin(sin_beta)

    epsilon = np.where(np.abs(Z_cloud) < Z_0, 2 - np.abs(Z_cloud / Z_0), 1)
    f = np.cos(beta) ** Q * np.exp(-P * np.sin(np.abs(beta) ** epsilon))

    return R_cloud**-gamma * f


def compute_narrow_band_density(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    beta_nb_rad: float,
    G: float,
    gamma: float,
    r: float,
    A: float,
) -> npt.NDArray[np.float64]:
    """Density of the fan (see Eq (4). in RRM)."""

    X_nb = X_helio - X_0
    R_nb = np.sqrt(X_nb[0] ** 2 + X_nb[1] ** 2 + X_nb[2] ** 2)

    Z_nb = (
        X_nb[0] * sin_Omega_rad * sin_i_rad
        - X_nb[1] * cos_Omega_rad * sin_i_rad
        + X_nb[2] * cos_i_rad
    )
    sin_beta = Z_nb / R_nb
    beta = np.arcsin(sin_beta)
    f = np.where(
        np.abs(beta) < beta_nb_rad, np.exp(G * (np.abs(beta) - beta_nb_rad)), 0
    )
    return A * (R_nb / r) ** (-gamma) * f


def compute_broad_band_density(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    beta_bb_rad: float,
    sigma_bb_rad: float,
    gamma: float,
    r: float,
    A: float,
) -> npt.NDArray[np.float64]:
    """Density of the fan (see Eq (5). in RRM)."""

    X_bb = X_helio - X_0
    R_bb = np.sqrt(X_bb[0] ** 2 + X_bb[1] ** 2 + X_bb[2] ** 2)

    Z_bb = (
        X_bb[0] * sin_Omega_rad * sin_i_rad
        - X_bb[1] * cos_Omega_rad * sin_i_rad
        + X_bb[2] * cos_i_rad
    )
    sin_beta = Z_bb / R_bb
    beta = np.arcsin(sin_beta)
    f = np.exp(-((beta - beta_bb_rad) ** 2) / (2 * sigma_bb_rad**2)) + np.exp(
        -((beta + beta_bb_rad) ** 2) / (2 * sigma_bb_rad**2)
    )
    return A * (R_bb / r) ** (-gamma) * f


def compute_ring_density_rmm(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    sin_Omega_rad: float,
    cos_Omega_rad: float,
    sin_i_rad: float,
    cos_i_rad: float,
    n_0: float,
    R: float,
    sigma_r: float,
    sigma_z: float,
    A: float,
) -> npt.NDArray[np.float64]:
    return (
        A
        * compute_ring_density(
            X_helio=X_helio,
            X_0=X_0,
            sin_Omega_rad=sin_Omega_rad,
            cos_Omega_rad=cos_Omega_rad,
            sin_i_rad=sin_i_rad,
            cos_i_rad=cos_i_rad,
            n_0=n_0,
            R=R,
            sigma_r=sigma_r,
            sigma_z=sigma_z,
        )
        * 1e-6
        / (1 * u.cm).to_value(u.AU)
    )


def compute_feature_density_rmm(
    X_helio: npt.NDArray[np.float64],
    X_0: npt.NDArray[np.float64],
    X_earth: npt.NDArray[np.float64],
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
    A: float,
) -> npt.NDArray[np.float64]:
    return (
        A
        * compute_feature_density(
            X_helio=X_helio,
            X_0=X_0,
            sin_Omega_rad=sin_Omega_rad,
            cos_Omega_rad=cos_Omega_rad,
            sin_i_rad=sin_i_rad,
            cos_i_rad=cos_i_rad,
            n_0=n_0,
            R=R,
            sigma_r=sigma_r,
            sigma_z=sigma_z,
            X_earth=X_earth,
            sigma_theta_rad=sigma_theta_rad,
            theta_rad=theta_rad,
        )
        * 1e-6
        / (1 * u.cm).to_value(u.AU)
    )


# Mapping of implemented zodiacal component data classes and their density functions.
DENSITY_FUNCS: dict[type[Component], ComputeDensityFunc] = {
    Cloud: compute_cloud_density,
    Band: compute_band_density,
    Ring: compute_ring_density,
    Feature: compute_feature_density,
    Fan: compute_fan_density,
    NarrowBand: compute_narrow_band_density,
    BroadBand: compute_broad_band_density,
    RingRRM: compute_ring_density_rmm,
    FeatureRRM: compute_feature_density_rmm,
}

PartialComputeDensityFunc = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def construct_density_partials(
    comps: Sequence[Component],
    dynamic_params: dict[str, Any],
) -> tuple[PartialComputeDensityFunc, ...]:
    """
    Return a tuple of the density expressions above which has been prepopulated with model and
    configuration parameters, leaving only the `X_helio` argument to be supplied.

    Raises exception for incorrectly defined components or component density functions.
    """

    partial_density_funcs: list[PartialComputeDensityFunc] = []
    for comp in comps:
        comp_dict = asdict(comp)
        func_params = inspect.signature(DENSITY_FUNCS[type(comp)]).parameters.keys()
        residual_params = [key for key in func_params if key not in comp_dict.keys()]
        try:
            residual_params.remove("X_helio")
        except ValueError:
            raise ValueError(
                "X_helio must be be the first argument to the density function of a component."
            )

        if residual_params:
            if residual_params - dynamic_params.keys():
                raise ValueError(
                    f"Argument(s) {residual_params} required by the density function "
                    f"{DENSITY_FUNCS[type(comp)]} are not provided by instance variables in "
                    f"{type(comp)} or by the `computed_parameters` dict."
                )
            comp_dict.update(dynamic_params)

        # Remove excess intermediate parameters from the component dict.
        comp_params = {
            key: value for key, value in comp_dict.items() if key in func_params
        }

        partial_func = partial(DENSITY_FUNCS[type(comp)], **comp_params)
        partial_density_funcs.append(partial_func)

    return tuple(partial_density_funcs)
