import astropy.constants as const
import astropy.units as u

h: float = const.h.value
c: float = const.c.value
k_B: float = const.k_B.value

SPECIFIC_INTENSITY_UNITS = u.W / u.Hz / u.m**2 / u.sr
