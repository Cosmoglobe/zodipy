from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class Emissivity:
    """Class containing emissivity fits for the Zodiacal components."""

    frequencies : Iterable[float]
    components : Dict[str, Tuple[float]]

    def get_emissivity(self, comp: str, freq: float) -> float:
        """Interpolates in the fitted emissivites.
        
        Parameters
        ----------
        comp : str
            Component label, i.e, 'cloud'.
        freq : float
            Frequency at which to evaluate the Zodiacal emission.

        Returns
        -------
        emissivity : float
            Emissivity scaling factor.
        """

        if not self.frequencies[0] <= freq <= self.frequencies[-1]:
            raise ValueError(f'Frequency is out of range')

        emissivity = np.interp(freq, self.frequencies, self.components[comp])

        return emissivity