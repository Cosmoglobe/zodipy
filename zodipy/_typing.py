from typing import Sequence, Union

import astropy.units as u
import numpy as np
from numpy.typing import NDArray

Pixels = Union[int, Sequence[int], NDArray[np.integer]]
SkyAngles = Union[u.Quantity[u.deg], u.Quantity[u.rad]]
FrequencyOrWavelength = Union[u.Quantity[u.Hz], u.Quantity[u.m]]
