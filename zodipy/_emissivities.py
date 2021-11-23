from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from zodipy._component_labels import ComponentLabel


@dataclass
class Emissivity:
    """Emissivity parameters for a Zodiacal component."""

    frequencies: Sequence[float]
    components: Dict[ComponentLabel, Sequence[float]]

    def __call__(self, component: ComponentLabel, freq: float) -> float:
        """Returns the interpolated emissivity given a frequency.

        Parameters
        ----------
        component
            Component label.
        freq
            Frequency at which to evaluate the Zodiacal emission.

        Returns
        -------
        emissivity
            Interpolated emissivity scaling factor.
        """

        if not self.frequencies[0] <= freq <= self.frequencies[-1]:
            raise ValueError(f"Frequency is out of range")

        emissivity = np.interp(freq, self.frequencies, self.components[component])

        return emissivity


def get_emissivities(
    freq: float,
    emissivity: Optional[Emissivity],
    components: List[ComponentLabel],
) -> List[float]:
    """Returns a list of interpolated emissivities for each component."""

    if emissivity is not None:
        return [emissivity(component, freq) for component in components]

    return [1.0 for _ in components]
