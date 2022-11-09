import astropy.constants as const
import astropy.units as u

h: float = const.h.value
c: float = const.c.value
k_B: float = const.k_B.value

SPECIFIC_INTENSITY_UNITS = u.W / u.Hz / u.m**2 / u.sr

R_MARS = 1.52
R_EOS = 3.02
R_THEMIS = 3.14
R_EARTH = 1
R_ASTEROID_BELT = 3.1
R_JUPITER = 5.2
R_KUIPER_BELT = 30
DISTANCE_FROM_EARTH_TO_L2 = u.Quantity(0.009896235034000056, u.AU)
