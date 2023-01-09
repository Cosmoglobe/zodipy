import astropy.constants as const
import astropy.units as u
import numpy as np

h: float = const.h.value
c: float = const.c.value
k_B: float = const.k_B.value

SPECIFIC_INTENSITY_UNITS = u.W / u.Hz / u.m**2 / u.sr

R_0 = np.finfo(np.float64).eps
R_MARS = 1.53
R_VERITAS = 2.16
R_EOS = 3.02
R_THEMIS = 3.14
R_EARTH = 1
R_ASTEROID_BELT = 3.1
R_JUPITER = 5.2
R_KUIPER_BELT = 30
DISTANCE_FROM_EARTH_TO_L2 = u.Quantity(0.009896235034000056, u.AU)

N_INTERPOLATION_POINTS = 100
MIN_INTERPOLATION_GRID_TEMPERATURE = 40
MAX_INTERPOLATION_GRID_TEMPERATURE = 550
