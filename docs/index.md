
<img src="img/zodipy_logo.png" alt="ZodiPy logo" width="50%">

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat-square)](http://www.astropy.org/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zodipy?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/zodipy.svg?logo=python&style=flat-square)](https://pypi.org/project/zodipy)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://img.shields.io/badge/repo_status-Active-success?style=flat-square)](https://www.repostatus.org/#active)
[![Actions Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/Zodipy/tests.yml?branch=main&logo=github&style=flat-square)](https://github.com/Cosmoglobe/Zodipy/actions)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/zodipy/mkdocs-deploy.yml?branch=main&style=flat-square&logo=github&label=docs)
[![Codecov](https://img.shields.io/codecov/c/github/Cosmoglobe/zodipy?token=VZP9L79EUJ&style=flat-square&logo=codecov)](https://app.codecov.io/gh/Cosmoglobe/zodipy)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2205.12962-green?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2205.12962)
[![ascl:2306.012](https://img.shields.io/badge/ascl-2306.012-blue.svg?colorB=262255&style=flat-square)](https://ascl.net/2306.012)

ZodiPy is an [Astropy affiliated](https://www.astropy.org/affiliated/) package for simulating zodiacal light in intensity for arbitrary Solar system observers.
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

- We start by initializing the [`Zodipy`][zodipy.zodipy.Zodipy] class, which is our interface, where we specify that we want to use the DIRBE interplanetary dust model.
- We use the [`get_emission_ang`][zodipy.zodipy.Zodipy.get_emission_ang] method which is a method to simulate emission from angular sky coordinates (see the [reference](reference.md) for other available simulation methods).
- The first argument to the [`get_emission_ang`][zodipy.zodipy.Zodipy.get_emission_ang] method, `25 * u.micron`, specifies the wavelength of the simulated observation. Note that we use Astropy units for many of the input arguments.
- `theta` and `phi` represent the pointing of the observation (co-latitude and longitude, following the healpy convention). In this example we observe three sky coordinates.
- `obs_time` represents the time of observation, which we need to compute the position of the observer and all other required solar system bodies.
- `obs` represents the observer, and must be an solar system observer supported by the [Astropy ephemeris](https://docs.astropy.org/en/stable/coordinates/solarsystem.html) used internally. If we wish to be more specific about the observer position, we can use the `obs_pos` keyword instead of `obs`, which takes in a heliocentric cartesian position in units of AU.
- Finally, `lonlat` is a boolean which converts the convention of `theta` and `phi` from co-latitude and longitude to longitude and latitude.

For more information on using ZodiPy, see [the usage section](usage.md).
