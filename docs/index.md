<!-- # ZodiPy -->
<center>
   <img src="img/zodipy_logo.png" alt="ZodiPy logo" width="50%">

   *ZodiPy* is a Python tool for simulating the Interplanetary Dust Emission that a Solar System observer sees, either in the form of timestreams or binned HEALPix maps.

   [![PyPI version](https://badge.fury.io/py/zodipy.svg)](https://badge.fury.io/py/zodipy)
   ![Tests](https://github.com/MetinSa/zodipy/actions/workflows/tests.yml/badge.svg)
   [![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)![ZodiPy Logo](img/zodipy_map.png)
</center>


## Example
```py
import astropy.units as u
from astropy.time import Time
from zodipy import Zodipy


model = Zodipy(model="dirbe")

emission = model.get_emission_ang(
    25 * u.micron,
    theta=[10, 10.1, 10.2] * u.deg,
    phi=[90, 89, 88] * u.deg,
    obs_time=Time("2022-01-01 12:00:00"),
    obs="earth",
    lonlat=True,
)
print(model)
"""
Model(
   name: 'dirbe',
   components: (
      'cloud',
      'band1',
      'band2',
      'band3',
      'ring',
      'feature',
   ),
   thermal: True,
   scattering: True,
)
"""
print(emission)
#> [15.53095493 15.52883577 15.53121942] MJy / sr
```

What's going on here:

- We start by initializing the [`Zodipy`][zodipy.zodipy.Zodipy] class using the DIRBE interplanetary dust model.
- We use the [`get_emission_ang`][zodipy.zodipy.Zodipy.get_emission_ang] method to compute simulated emission from angular sky coordinates.
- The first argument, `25 * u.micron`, specifies the frequency or wavelength of the simulated observation. Note that we use Astropy units for many of the input arguments.
- `theta` and `phi` represent the pointing of the observation. In this scenario, we observe three sky coordinates.
- `obs_time` represents the time of observation which is used internally to compute the position of all required Solar System bodies and the observer.
- `obs` represents the observer, and must be an solar system observer supported by the [Astropy ephemeris](https://docs.astropy.org/en/stable/coordinates/solarsystem.html) used internally.

For more information on using *ZodiPy*, see [the usage section](usage.md).

## Scientific Paper
- [Cosmoglobe: Simulating Zodiacal Emission with ZodiPy](https://arxiv.org/abs/2205.12962)