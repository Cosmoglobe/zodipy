
<img src="docs/img/zodipy_logo.png" width="350">

[![PyPI version](https://badge.fury.io/py/zodipy.svg)](https://badge.fury.io/py/zodipy)
![Tests](https://github.com/MetinSa/zodipy/actions/workflows/tests.yml/badge.svg)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2205.12962-green)](https://arxiv.org/abs/2205.12962)

---


ZodiPy is a Python tool for simulating the zodiacal emission that a solar system observer sees, either in the form of timestreams or binned HEALPix maps.

![plot](docs/img/zodipy_map.png)

# Help
See the [documentation](https://cosmoglobe.github.io/zodipy/) for more information and examples on how to use ZodiPy for different applications.

# Installation
ZodiPy is installed using `pip install zodipy`.

# A simple example
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
#> [15.35392831 15.35495051 15.35616009] MJy / sr
```

# Scientific paper and citation
For an overview of the ZodiPy model approach and other information regarding zodiacal emission and interplanetary dust modeling, please see the paper [Cosmoglobe: Simulating Zodiacal Emission with ZodiPy (San et al. 2022)](https://arxiv.org/abs/2205.12962). See [CITATION](https://github.com/Cosmoglobe/zodipy/blob/dev/CITATION.bib) if you have used ZodiPy in your work and want to cite the software.

# Funding
This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreements No 776282 (COMPET-4; BeyondPlanck), 772253 (ERC; bits2cosmology) and 819478 (ERC; Cosmoglobe).

<table align="center">
    <tr>
        <td><img src="https://user-images.githubusercontent.com/28634670/170697040-d5ec2935-29d0-4847-8999-9bc4eaa59e56.jpeg" height="200"></td>
        <td><img src="https://user-images.githubusercontent.com/28634670/170697140-b010aa69-9f9a-44c0-b702-8a05ec0b6d3e.jpeg" height="200"></td>
    </tr>
</table>