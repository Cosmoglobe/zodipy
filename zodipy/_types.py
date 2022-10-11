from typing import Sequence, TypeVar, Union

import astropy.units as u
import numpy as np
import numpy.typing as npt

Pixels = Union[int, Sequence[int], npt.NDArray[np.integer]]
SkyAngles = Union[u.Quantity[u.deg], u.Quantity[u.rad]]
FrequencyOrWavelength = Union[u.Quantity[u.Hz], u.Quantity[u.m]]
FloatOrNDArray = TypeVar("FloatOrNDArray", float, npt.NDArray[np.float64])
