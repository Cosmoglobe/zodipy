from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt


@dataclass
class Component(ABC):
    """Base class for storing common model parameters for zodiacal components.

    Parameters
    ----------
    x_0
        x-offset from the Sun in heliocentric ecliptic coordinates [AU].
    y_0
        y-offset from the Sun in heliocentric ecliptic coordinates [AU].
    z_0
        z-offset from the Sun in heliocentric ecliptic coordinates [AU].
    i
        Inclination with respect to the ecliptic plane [deg].
    Omega
        Ascending node [deg].
    """

    x_0: float = field(repr=False)
    y_0: float = field(repr=False)
    z_0: float = field(repr=False)
    i: float = field(repr=False)
    Omega: float = field(repr=False)

    X_0: npt.NDArray[np.float64] = field(init=False)
    sin_i_rad: float = field(init=False)
    cos_i_rad: float = field(init=False)
    sin_Omega_rad: float = field(init=False)
    cos_Omega_rad: float = field(init=False)

    def __post_init__(self) -> None:
        self.X_0 = np.array([self.x_0, self.y_0, self.z_0]).reshape(3, 1, 1)
        self.sin_i_rad = np.sin(np.radians(self.i))
        self.cos_i_rad = np.cos(np.radians(self.i))
        self.sin_Omega_rad = np.sin(np.radians(self.Omega))
        self.cos_Omega_rad = np.cos(np.radians(self.Omega))


@dataclass
class Cloud(Component):
    """DIRBE diffuse cloud.

    Parameters
    ----------
    n_0
        Density at 1 AU.
    alpha
        Radial power-law exponent.
    beta
       Vertical shape parameter.
    gamma
        Vertical power-law exponent.
    mu
        Widening parameter for the modified fan.
    """

    n_0: float
    alpha: float
    beta: float
    gamma: float
    mu: float


@dataclass
class Band(Component):
    """DIRBE asteroidal dust band.

    Parameters
    ----------
    n_0
        Density at 3 AU.
    delta_zeta
        Shape parameter [deg].
    v
        Shape parameter.
    p
        Shape parameter.
    delta_r
        Inner radial cutoff.
    """

    n_0: float
    delta_zeta: float
    v: float
    p: float
    delta_r: float
    delta_zeta_rad: float = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.delta_zeta_rad = np.radians(self.delta_zeta)


@dataclass
class Ring(Component):
    """DIRBE circum-solar ring (excluding the Earth-trailing Feature).

    Parameters
    ----------
    n_0
        Density at 1 AU.
    R
        Radius of the peak density.
    sigma_r
        Radial dispersion.
    sigma_z
        Vertical dispersion.
    """

    n_0: float
    R: float
    sigma_r: float
    sigma_z: float


@dataclass
class Feature(Component):
    """DIRBE Earth-trailing Feature.

    Parameters
    ----------
    n_0
        Density at 1 AU.
    R
        Radius of the peak density.
    sigma_r
        Radial dispersion.
    sigma_z
        Vertical dispersion.
    theta
        Longitude with respect to Earth [deg].
    sigma_theta
        Longitude dispersion [deg].
    """

    n_0: float
    R: float
    sigma_r: float
    sigma_z: float
    theta: float
    sigma_theta: float
    theta_rad: float = field(init=False)
    sigma_theta_rad: float = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.theta_rad = np.radians(self.theta)
        self.sigma_theta_rad = np.radians(self.sigma_theta)


class ComponentLabel(Enum):
    """Labels representing the components in the DIRBE model."""

    CLOUD = "cloud"
    BAND1 = "band1"
    BAND2 = "band2"
    BAND3 = "band3"
    RING = "ring"
    FEATURE = "feature"
