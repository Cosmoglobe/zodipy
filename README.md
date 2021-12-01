
<img src="imgs/zodipy_logo.png" width="350">

[![PyPI version](https://badge.fury.io/py/zodipy.svg)](https://badge.fury.io/py/zodipy)
![Tests](https://github.com/MetinSa/zodipy/actions/workflows/tests.yml/badge.svg)
[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat)](http://www.astropy.org/)


---


*Zodipy* is a Python simulation tool for Zodiacal Emission (Interplanetary Dust Emission). It allows you to compute the 
simulated emission in a timestream, or at an instant in time.

![plot](imgs/zodi_default.png)

## Installing
Zodipy is available at PyPI and can be installed with ``pip install zodipy``.

## Features
The full set of features and use-cases will be documentated in the nearby future.

**Initializing an Interplantery Dust Model:** *Zodipy* implements the [Kelsall et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K/abstract) Interplanetary Dust Model. Additionally, it is possible to include the various emissivity fits from the Planck collaboration.
```python
import zodipy

# Other options for models are "K98" (default), "Planck13", "Planck15"
model = zodipy.InterplanetaryDustModel(model="Planck18")
```

**Instantaneous emission:** We can make a map of the simulated instantaneous emission seen by an observer using the `get_instantaneous_emission` function, which queries the observer position given an epoch through the JPL Horizons API:
```python
import healpy as hp
import astropy.units as u

emission = model.get_instantaneous_emission(
    800*u.GHz, 
    nside=256, 
    observer="Planck", 
    epochs=59215,  # 2010-01-01 (iso) in MJD
    coord_out="G"
)

hp.mollview(emission, norm="hist")
```
![plot](imgs/zodi_planck.png)

The `epochs` input must follow the convention used in [astroquery](https://astroquery.readthedocs.io/en/latest/jplhorizons/jplhorizons.html). If multiple dates are passed to the function, the returned emission becomes the average over all instantaneous maps.

Additionally, it is possible to retrieve the emission component-wise by setting `return_comps=True` in the function call. Following is an example of what the simulated emission seen from L2 for each component is at 6th of October 2021.

![plot](imgs/comps.png)


**Time-ordered emission:** We can make a time-stream of simulated emission for a sequence of time-ordered pixels using the `get_time_ordered_emission` function. This requires specifying the heliocentric ecliptic cartesian position of the observer (and optionally the Earth) associated with each chunk of pixels. In the following we use the first day of time-ordered pixels from the DIRBE instrument of the COBE satellite (Photometric Band 6, Detector A, first day of observations) to make a simulated time-stream:
```python
import astropy.units as u
import matplotlib.pyplot as plt
import zodipy

model = zodipy.InterplanetaryDustModel()

dirbe_pixels = ...
dirbe_position = ...  
earth_position = ...  

timestream = model.get_time_ordered_emission(
    25*u.micron
    nside=128,
    pixels=dirbe_pixels,
    observer_coordinates=dirbe_position,
    earth_coordinates=earth_position
)

plt.plot(timestream)
```
![plot](imgs/timestream.png)


**Binned time-ordered emission:** By setting `bin=True` in the function call, the simulated emission is binned into a map. In the following, we compare *Zodipy* simulations with the observed time-ordered data by DIRBE.

```python
import astropy.units as u
import matplotlib.pyplot as plt
import zodipy

model = zodipy.InterplanetaryDustModel()

dirbe_pixel_chunks = [...]
dirbe_positions = [...]
earth_positions = [...]

emission = np.zeros(hp.nside2npix(nside))
hits_map = np.zeros(hp.nside2npix(nside))   
    
for day, (pixels, dirbe_position, earth_position) in enumerate(
    zip(dirbe_pixel_chunks, dirbe_positions, earth_positions),
    start=1
):
    
    unique_pixels, counts = np.unique(pixels, return_counts=True)
    hits_map[unique_pixels] += counts

    emission += model.get_time_ordered_emission(
        25*u.micron,
        nside=128,
        pixels=pixels,
        observer_position=dirbe_position,
        earth_position=earth_positions,
        bin=True
    )

    # We make a plot for each week.
    if day % 7 == 0:
        zodi_emission /= hits_map
        hp.mollview(zodi_emission)

        # Reset emission and hits map for next week
        emission = np.zeros(hp.nside2npix(nside)) 
        hits_map = np.zeros(hp.nside2npix(nside)) 
```
| DIRBE TOD | Zodipy TOD Simulation|
| :---: | :---: |
|![plot](imgs/dirbe.gif) | ![plot](imgs/zodipy.gif)|
