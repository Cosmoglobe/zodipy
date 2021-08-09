from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np


@dataclass
class Emissivity:
    """Emissivity fits. 
    
    These are values estimated from template fitting used for spectral 
    rescaling of each Zodiacal component.

    Parameters
    ----------
    frequencies : tuple, list, `numpy.ndarray`
        Iterable containing the sharp frequencies at which the 
        emissivities was estimated.
    values : dict
        Dictionary containing the emissivity fits for each component.
    """

    frequencies : Iterable[float]
    cloud : Iterable[Union[float, None]]
    band1 : Iterable[Union[float, None]]
    band2 : Iterable[Union[float, None]]
    band3 : Iterable[Union[float, None]]
    ring : Iterable[Union[float, None]]
    feature : Iterable[Union[float, None]]

    def get_emissivity(self, comp: str, freq: float) -> float:
        """Interpolates to a specific emissivity.
        
        Parameters
        ----------
        comp : str
            Component label, i.e, 'cloud'.
        freq : float
            Frequency at which to evaluate the Zodiacal emission.

        Returns
        -------
        float
            Emissivity scaling factor.
        """

        emissivities = getattr(self, comp)

        if not self.frequencies[0] <= freq <= self.frequencies[-1]:
            raise ValueError(f'Frequency is out of range')

        return np.interp(freq, self.frequencies, emissivities)