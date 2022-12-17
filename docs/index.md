
<img src="img/zodipy_logo.png" alt="ZodiPy logo" width="50%">

[![PyPI version](https://badge.fury.io/py/zodipy.svg)](https://badge.fury.io/py/zodipy)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
![Tests](https://github.com/MetinSa/zodipy/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/Cosmoglobe/zodipy/branch/main/graph/badge.svg?token=VZP9L79EUJ)](https://codecov.io/gh/Cosmoglobe/zodipy)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2205.12962-green)](https://arxiv.org/abs/2205.12962)

ZodiPy simulates zodiacal emission in intensity for arbitrary Solar System observers in the form of timestreams or full-sky maps
![ZodiPy Logo](img/zodipy_map.png)


## A simple example
```python
import astropy.units as u
from astropy.time import Time

from zodipy import Zodipy


model = Zodipy("dirbe")

emission = model.get_emission_ang(
    25 * u.micron,
    theta=[10, 10.1, 10.2] * u.deg,
    phi=[90, 89, 88] * u.deg,
    obs_time=Time("2022-01-01 12:00:00"),
    obs="earth",
    lonlat=True,
)

print(emission)
#> [15.53095493 15.52883577 15.53121942] MJy / sr
```

What's going on here:

- We start by initializing the [`Zodipy`][zodipy.zodipy.Zodipy] class where we specify that we want to use the DIRBE interplanetary dust model.
- We use the [`get_emission_ang`][zodipy.zodipy.Zodipy.get_emission_ang] method which is a method to compute simulated emission from angular sky coordinates. See the [reference](reference.md) for other available methods.
- The first argument to the [`get_emission_ang`][zodipy.zodipy.Zodipy.get_emission_ang] method, `25 * u.micron`, specifies the wavelength (or frequency) of the simulated observation. Note that we use Astropy units for many of the input arguments.
- `theta` and `phi` represent the pointing of the observation (co-latitude and longitude). In this example we observe three sky coordinates.
- `obs_time` represents the time of observation which is used internally to compute the position of the observer and all other required solar system bodies.
- `obs` represents the observer, and must be an solar system observer supported by the [Astropy ephemeris](https://docs.astropy.org/en/stable/coordinates/solarsystem.html) used internally. If we wish to be more specific about the observer position, we can use the `obs_pos` keyword instead of `obs`, which takes in a heliocentric cartesian position in units of AU.
- `lonlat` is a boolean which converts the convention of `theta` and `phi` from co-latitude and longitude to longitude and latitude.

For more information on using ZodiPy, see [the usage section](usage.md).
