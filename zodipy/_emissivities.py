from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


@dataclass
class Emissivities:
    """Emissivity fits for a Zodiacal component."""

    frequencies : Iterable[float]
    components : Dict[str, Iterable[float]]

    def get_emissivity(self, comp: str, freq: float) -> float:
        """Interpolates in the fitted emissivites.
        
        Parameters
        ----------
        comp
            Component label.
        freq
            Frequency at which to evaluate the Zodiacal emission.

        Returns
        -------
        emissivity
            Emissivity scaling factor.
        """

        if not self.frequencies[0] <= freq <= self.frequencies[-1]:
            raise ValueError(f'Frequency is out of range')

        emissivity = np.interp(freq, self.frequencies, self.components[comp])

        return emissivity