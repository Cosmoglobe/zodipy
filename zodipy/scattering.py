from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt


def get_scattering_angle(
    R_los: float | npt.NDArray[np.float64],
    R_helio: npt.NDArray[np.float64],
    X_los: npt.NDArray[np.float64],
    X_helio: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return the scattering angle between the Sun and a point along the line of sight.

    Args:
        R_los: Distance from observer to a point along the line of sight.
        R_helio: Distance from the Sun to a point along the line of sight.
        X_los: Vector pointing from the observer to a point along the line of sight.
        X_helio: Heliocentric position of point along the line of sight.

    Returns:
        Scattering angle.

    """
    cos_theta = (X_los * X_helio).sum(axis=0) / (R_los * R_helio)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(-cos_theta)


def get_phase_function(
    Theta: npt.NDArray[np.float64], C1: np.float64, C2: np.float64, C3: np.float64
) -> npt.NDArray[np.float64]:
    """Return the phase function.

    Args:
        Theta: Scattering angle.
        C1: Phase_function coefficient.
        C2: Phase_function coefficient.
        C3: Phase_function coefficient.

    Returns:
        The Phase funciton.

    """
    phase_normalization = _get_phase_normalization(C1, C2, C3)
    return phase_normalization * (C1 + C2 * Theta + np.exp(C3 * Theta))


def _get_phase_normalization(C1: np.float64, C2: np.float64, C3: np.float64) -> np.float64:
    """Return the analyitcal integral for the phase normalization factor N."""
    int_term1 = 2 * np.pi
    int_term2 = 2 * C1
    int_term3 = np.pi * C2
    int_term4 = (np.exp(C3 * np.pi) + 1) / (C3**2 + 1)
    return 1 / (int_term1 * (int_term2 + int_term3 + int_term4))
