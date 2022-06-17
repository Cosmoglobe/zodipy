# ZodiPy

![ZodiPy Logo](img/zodipy_map.png)

ZodiPy is a Python tool for simulating the Interplanetary Dust Emission that a Solar System observer sees, either in the form of timestreams or binned HEALPix maps.

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
    obs="earth",
    obs_time=Time("2022-01-01 12:00:00"),
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

- We initialize an the DIRBE interplanetary dust model through the `Zodipy` class.
- We use the `get_emission_ang` methods to compute simulated emission from angular sky coordinates.
- `25 * u.micron` specifies the frequency or wavelength of the simulation.
- `theta=[10, 10.1, 10.2] * u.deg` and `phi=[90, 89, 88] * u.deg` represent a sequence of three sky coordinates for which we want to simulate the emission.
- The emission is simulated from the observer given by `obs="earth"` in line of sights towards the angles given by `theta` and `phi`.
- `obs_time=Time("2022-01-01 12:00:00")` represents the time of observation which is used to compute the position of all required Solar System bodies and the observer.

## Scientific Paper
- [Cosmoglobe: Simulating Zodiacal Emission with ZodiPy](https://arxiv.org/abs/2205.12962)