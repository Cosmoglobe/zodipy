import astropy.constants as const
import numpy as np


h = const.h.value
c = const.c.value
k_B = const.k_B.value


def blackbody_emission(T: float, nu: float) -> float:
    """Returns the blackbody emission for a temperature T and frequency nu.

    Parameters
    ----------
    T
        Temperature of the blackbody [K].
    nu
        Frequency [Hz].

    Returns
    -------
        Blackbody emission [W / m^2 Hz sr].
    """

    term1 = (2 * h * nu ** 3) / c ** 2
    term2 = np.expm1((h * nu) / (k_B * T))

    return term1 / term2


def interplanetary_temperature(R: float, T_0: float, delta: float) -> float:
    """Returns the Interplanetary Temperature given a radial distance from the Sun.

    Parameters
    ----------
    R
        Radial distance from the sun in ecliptic coordinates.
    T_0
        Temperature of the solar system at R = 1 AU.
    delta
        Powerlaw index.

    Returns
    -------
        Interplanetary temperature.
    """

    return T_0 * R ** -delta


# def phase_function(𝚯: float, C_0: float, C_1: float, C_2: float) -> float:
#     """Returns the phase function.

#     Parameters
#     ----------
#     𝚯
#         Scattering angle [rad].
#     C_0, C_1, C_2
#         Phase function parameter.

#     Returns
#     -------
#         The Phase funciton.
#     """

#     N = 1

#     return N * (C_0 + C_1 * 𝚯 + np.exp(C_2 * 𝚯))


# def get_source_function(
#     nu: float,
#     E_c_nu: float,
#     A_C_nu: float,
#     𝚯: float,
#     R: float,
#     T_0: float,
#     delta: float,
# ) -> float:
#     """Returns the soure function."""

#     T = interplanetary_temperature(R=R, T_0=T_0, delta=delta)
#     B_nu = blackbody_emission(T=T, nu=nu)
#     F_nu = blackbody_emission(T=5750, nu=nu)
#     𝚽_nu = phase_function(𝚯)
#     return A_C_nu * F_nu * 𝚽_nu + (1 - A_C_nu) * E_c_nu * B_nu
