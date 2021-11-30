from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import astropy.units as u

from zodipy._component_labels import ComponentLabel


@dataclass
class Emissivity:
    """Emissivity parameters for a Zodiacal component.

    Attributes
    ----------
    spectrum
        Frequencies (or wavelengths) corresponding to the emissivity fits.
    components
        Dictionary mapping components to the emissivity fits.
    """

    spectrum: u.Quantity
    components: Dict[ComponentLabel, Sequence[float]]

    def __call__(self, component: ComponentLabel, ν_or_λ: u.Quantity) -> float:
        """Returns the interpolated emissivity given a frequency.

        Parameters
        ----------
        component
            Component label.
        ν_or_λ
            Frequency or wavelength at which to to interpolate in the emissivities.

        Returns
        -------
        emissivity
            Interpolated emissivity scaling factor.
        """

        ν_or_λ = ν_or_λ.to(self.spectrum.unit, equivalencies=u.spectral())

        if not self.spectrum[0] <= ν_or_λ <= self.spectrum[-1]:
            raise ValueError(f"Frequency is out of range")

        emissivity = np.interp(ν_or_λ, self.spectrum, self.components[component])

        return emissivity


def get_emissivities(
    ν_or_λ: u.Quantity,
    emissivity: Optional[Emissivity],
    components: List[ComponentLabel],
) -> List[float]:
    """Returns a list of interpolated emissivities for each component."""

    if emissivity is not None:
        return [emissivity(component, ν_or_λ) for component in components]

    return [1.0 for _ in components]
