from dataclasses import dataclass

import numpy as np


@dataclass
class BaseParameters:
    """Base parameters for all Zodiacal components.

    Parameters
    ----------
    x0: float
        x offset from the Sun in ecliptic coordinates.
    y0: float
        y offset from the Sun in ecliptic coordinates.
    z0: float
        z offset from the Sun in ecliptic coordinates.
    inclination: float
        Inclination [deg].
    omega: float
        Ascending node [deg].
    """

    x0 : float
    y0 : float
    z0 : float
    inclination : float
    omega : float

    def __post_init__(self) -> None:
        self.inclination = np.deg2rad(self.inclination)
        self.omega = np.deg2rad(self.omega)


@dataclass
class CloudParameters(BaseParameters):
    """Parameters specific to the cloud component.

    Parameters
    ----------
    n0 : float
        Density at 1 AU.
    alpha : float 
        Radial power-law exponent.
    beta : float
        Vertical shape parameter.
    gamma : float
        Vertical power-law exponent.
    mu : float
        Widening parameter for the modified fan.
    """

    n0 : float
    alpha : float
    beta : float
    gamma : float
    mu : float

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class BandParameters(BaseParameters):
    """Parameters specific to the band components.

    Parameters
    ----------
    n0 : float
        Density at 3 AU.
    delta_zeta : float
        Shape parameter [deg].
    v : float
        Shape parameter.
    p : float
        Shape parameter.
    delta_r : float
        Inner radial cutoff. 
    """

    n0 : float
    delta_zeta : float
    v : float
    p : float
    delta_r : float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.delta_zeta = np.deg2rad(self.delta_zeta)


@dataclass
class RingParameters(BaseParameters):
    """Parameters specific to the circumsolar ring component.

    Parameters
    ----------
    n0 : float
        Density at 1 AU.
    R : float
        Radius of the peak density.
    sigma_r : float
        Radial dispersion.
    sigma_z : float 
        Vertical dispersion.
    """

    n0 : float
    R : float
    sigma_r : float
    sigma_z : float

    def __post_init__(self) -> None:
        super().__post_init__()


@dataclass
class FeatureParameters(RingParameters):
    """Parameters specific to the earth-trailing feature component.

    Parameters
    ----------
    theta : float
        Longitude with respect to Earth.
    sigma_theta : float
        Longitude dispersion.
    """

    theta : float
    sigma_theta : float

    def __post_init__(self) -> None:
        super().__post_init__()
        self.theta = np.deg2rad(self.theta)
        self.sigma_theta = np.deg2rad(self.sigma_theta)