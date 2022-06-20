
<img src="docs/img/zodipy_logo.png" width="350">

[![PyPI version](https://badge.fury.io/py/zodipy.svg)](https://badge.fury.io/py/zodipy)
![Tests](https://github.com/MetinSa/zodipy/actions/workflows/tests.yml/badge.svg)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)


---


ZodiPy is a Python tool for simulating the Interplanetary Dust Emission that a Solar System observer sees, either in the form of timestreams or binned HEALPix maps.

![plot](docs/img/zodipy_map.png)


# Usage
See the [documentation](https://cosmoglobe.github.io/zodipy/) for a broader introduction to using ZodiPy.

**Interplanetary Dust models:** select between built in models.
```python
from zodipy import Zodipy

model = Zodipy(model="Planck18")
```

**Get emission from a point on the sky:** choose a frequency/wavelength, an observer, a time of observation, and angular coordinates (co-latitude, longitude).
```python
import astropy.units as u
from astropy.time import Time

model.get_emission_ang(
    25*u.micron,
    theta=10*u.deg,
    phi=40*u.deg,
    obs="earth",
    obs_time=Time.now(),
)
>> <Quantity [16.65684599] MJy / sr>
```

**Get emission from a sequence of angular coordinates:** `theta` and `phi` can be a sequence of angles that can represent some time-ordered pointing.
```python
theta = [10.1, 10.5, 11.1, 11.5] * u.deg
phi = [40.2, 39.9, 39.8, 41.3] * u.deg

model.get_emission_ang(
    25*u.micron,
    theta=theta,
    phi=phi,
    obs="earth",
    obs_time=Time.now(),
    lonlat=True,
)
>> <Quantity [29.11106315, 29.33735654, 29.41248579, 28.30858417] MJy / sr>
```


**Get emission from pixel indices on a HEALPIX grid:** a sequence of pixel indicies along with an NSIDE parameter can be used.
```python
model.get_emission_pix(
    25*u.micron,
    pixels=[24654, 12937, 26135],
    nside=128,
    obs="earth",
    obs_time=Time.now(),
)
>> <Quantity [17.77385144, 19.7889428 , 22.44797121] MJy / sr>
```

**Get binned emission component-wise:** the emission can be binned to a HEALPix map, and also returned component-wise.
```python
import healpy as hp
import numpy as np

nside = 128

model.get_binned_emission_pix(
    25*u.micron,
    pixels=np.arange(hp.nside2npix(nside)),
    nside=nside,
    obs="earth",
    obs_time=Time.now(),
    return_comps=True
).shape
>> (6, 196608)
```

# Documentation
A detailed introduction along with a tutorial of how to use ZodiPy will shortly be available in the [documentation](https://zodipy.readthedocs.io/en/latest/).
# Installing
ZodiPy is available on PyPI and can be installed with ``pip install zodipy`` (Python >= 3.8 required).

# Scientific paper
See [CITATION](https://github.com/Cosmoglobe/zodipy/blob/dev/CITATION.bib)

- [Cosmoglobe: Simulating Zodiacal Emission with ZodiPy (San et al. 2022)](https://arxiv.org/abs/2205.12962)

# Funding
This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreements No 776282 (COMPET-4; BeyondPlanck), 772253 (ERC; bits2cosmology) and 819478 (ERC; Cosmoglobe).

<table align="center">
    <tr>
        <td><img src="https://user-images.githubusercontent.com/28634670/170697040-d5ec2935-29d0-4847-8999-9bc4eaa59e56.jpeg" height="200"></td>
        <td><img src="https://user-images.githubusercontent.com/28634670/170697140-b010aa69-9f9a-44c0-b702-8a05ec0b6d3e.jpeg" height="200"></td>
    </tr>
</table>