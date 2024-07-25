
<img src="img/zodipy_logo.png" alt="ZodiPy logo" width="50%">

[![astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg)](http://www.astropy.org/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zodipy)
[![PyPI](https://img.shields.io/pypi/v/zodipy.svg?logo=python)](https://pypi.org/project/zodipy)
[![Actions Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/Zodipy/tests.yml?branch=main&logo=github)](https://github.com/Cosmoglobe/Zodipy/actions)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/zodipy/mkdocs-deploy.yml?branch=main&style=flat-square&logo=github&label=docs)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://img.shields.io/badge/repo_status-Active-success)](https://www.repostatus.org/#active)
[![Codecov](https://img.shields.io/codecov/c/github/Cosmoglobe/zodipy?token=VZP9L79EUJ&logo=codecov)](https://app.codecov.io/gh/Cosmoglobe/zodipy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/161)
[![ascl:2306.012](https://img.shields.io/badge/ascl-2306.012-blue.svg?colorB=262255)](https://ascl.net/2306.012)
[![DOI](https://zenodo.org/badge/394929213.svg)](https://zenodo.org/doi/10.5281/zenodo.10999611)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06648/status.svg)](https://doi.org/10.21105/joss.06648)

ZodiPy is an [Astropy-affiliated](https://www.astropy.org/affiliated/) package for simulating zodiacal light in intensity for arbitrary Solar system observers.
![ZodiPy Logo](img/zodipy_map.png)

## A simple example
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

# Initialize a zodiacal light model at a wavelength/frequency or over a bandpass
model = zodipy.Model(25*u.micron)

# Use Astropy's `SkyCoord` to specify coordinate
lon = [10, 10.1, 10.2] * u.deg
lat = [90, 89, 88] * u.deg
obstimes = Time(["2022-01-01 12:00:00", "2022-01-01 12:01:00", "2022-01-01 12:02:00"])

skycoord = SkyCoord(lon, lat, obstime=obstimes, frame="galactic")

# Compute the zodiacal light as seen from Earth
emission = model.evaluate(skycoord, obspos="earth")

print(emission)
#> [27.52410841 27.66572294 27.81251906] MJy / sr
```

For more information on using ZodiPy, see [the usage section](usage.md).
