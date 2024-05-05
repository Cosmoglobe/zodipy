As an Astropy-affiliated package, ZodiPy is highly integrated with the astropy ecosystem.
To make zodiacal light simulations, the `astropy.units`, `astropy.coordinates`, and `astropy.time` modules are used to provide user input. The coordinates for which ZodiPy will simulate the zodiacal light is specified in through the `astropy.coordinates.SkyCoord` object. Using ZodiPy is very simple and the user will only interact with *one* object `zodipy.Model`, which has *one* method `evaluate`.

!!! danger "Breaking API changes in `v1.0.0`"
    ZodiPy version >=`v1.0.0` no longer supports simulating zodiacal light from non `SkyCoord` coordinate inputs.
## Initializing Zodipy
### Working with the `Model` class
The interface to ZodiPy is the `zodipy.Model` class
```python
import astropy.units as u
import zodipy

model = zodipy.Model(25 * u.micron)
```
It has *one* required positional argument `x`, which represents a center wavelength/frequency or the points of a empirical bandpass. If `x` represents the points of a bandpass, the `weights` argument must also be provided
```python 
import astropy.units as u
import zodipy

points = [3, 4, 5, 6] * u.micron
weights = [0.2, 0.4, 0.3, 0.1]

model = zodipy.Model(points, weights=weights)
```

ZodiPy supports several zodiacal light models (see the [introduction](introduction.md) page for more information regarding the supported models), which are all valid in wavelength/frequency ranges. By default, the `Model` object will initialize using the DIRBE model. To select a different model, we specify the keyword argument `name`
```python
import astropy.units as u
import zodipy

model = zodipy.Model(25 * u.micron, name="planck18")
```

### Multiprocessing
ZodiPy will distribute the input coordinates to cores if the keyword argument `n_proc` is `>= 1` using Python's `multiprocessing` module.
```python
import multiprocessing

import astropy.units as u
import zodipy

model = zodipy.Model(25 * u.micron, n_proc=multiprocessing.cpu_count())
```
Alternatively, a custom pool (`multiprocessing.pool.Pool`) may be provided.

```python
import multiprocessing

import astropy.units as u
import zodipy

pool = multiprocessing.Pool(multiprocesing.cpu_count())
model = zodipy.Model(25 * u.micron, pool=pool)
```
!!! tip 
    For all available optional keyword arguments in `zodipy.Model` see [the API reference](reference.md).

## Simulating zodiacal light
To make zodiacal light simulations ZodiPy needs to know three things: 1) Sky coordinates for which to simulate zodiacal light; 2) The position of the observer to know where the vertex of the rays is positioned; and 3) the time of observation, used to query the position of Earth. 
### The SkyCoord object
The sky coordinates are provided through astropys powerful `astropy.coordinates.SkyCoord` object, which can represent the observed coordinates in several formats. Users unfamiliar with the `SkyCoord` object should visit the [official Astropy documention](https://docs.astropy.org/en/stable/coordinates/index.html) before using ZodiPy to learn the basics.

When using the `SkyCoord` in ZodiPy, the user **must** set the `obstime` and `frame` attributes. For a single observation in galactic coordinates, the `SkyCoord` object may look something like
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

skycoord = SkyCoord(
    40 * u.deg, 
    60 * u.deg, 
    obstime=Time("2020-01-01"), 
    frame="galactic"
)
```
The `astropy.time.Time` object can represent time in many formats, including julian and modified julian dates (see the [documentation](https://docs.astropy.org/en/stable/time/) for the `Time` object).

!!! note "Notes on coordinate frames"
    The coordinates specified to ZodiPy in the `SkyCoord` object must be in either the Ecliptic, Galactic, or Celestial
    frames. The origin of the frames are therefore not important. Astropy provides a set of supported coordinate frames. In ZodiPy we use the following Astropy frames to represent generic Ecliptic, Galactic, and Celestial frames:

    - Ecliptic = `"barycentricmeanecliptic"` / `BarycentricMeanEcliptic`
    - Galactic = `"galactic"` / `Galactic`
    - Celestial = `"icrs"` / `ICRS`

In the following we show three sets of observations in all three coordinate frames
``` py
import astropy.units as u
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic, Galactic, ICRS
from astropy.time import Time

skycoord_ecliptic = SkyCoord(
    40 * u.deg, 
    60 * u.deg, 
    obstime=Time("2020-01-01"), 
    frame=BarycentricMeanEcliptic
)
skycoord_galactic = SkyCoord(
    203 * u.deg, 
    10 * u.deg, 
    obstime=Time("2020-01-01"), 
    frame=Galactic
)
skycoord_celestial = SkyCoord(
    12 * u.deg, 
    40 * u.deg, 
    obstime=Time("2020-01-01"), 
    frame=ICRS
)
```

### The `evaluate` method
Zodiacal light is evaluated by passing in the `SkyCoord` object to the `zodipy.Model.evaluate` method.
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import zodipy

model = zodipy.Model(25 * u.micron)

skycoord = SkyCoord(
    40 * u.deg, 
    60 * u.deg, 
    obstime=Time("2020-01-01"), 
    frame="galactic"
)

emission = model.evaluate(skycoord)
print(emission)
# <Quantity [25.08189292] MJy / sr>
```
By default the observer is assumed to be on Earth. The position of the observer can be explicitly provided as the keyword argument `obspos` in the `evaluate` method

```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import zodipy

model = zodipy.Model(25 * u.micron)

skycoord = SkyCoord(
    40 * u.deg, 
    60 * u.deg, 
    obstime=Time("2020-01-01"), 
    frame="galactic"
)

emission = model.evaluate(skycoord, obspos="mars")
print(emission)
# <Quantity [8.36985535] MJy / sr>

emission = model.evaluate(skycoord, obspos=[0.87, -0.53, 0.001] * u.AU)
print(emission)
# <Quantity [20.37750965] MJy / sr>
```
This argument accepts both a string represented a body recognized by `astropy.coordinates.solar_system_ephemeris` (see the Astropy [documentation](https://docs.astropy.org/en/stable/api/astropy.coordinates.solar_system_ephemeris.html)), or a heliocentric ecliptic cartesian position.

!!! note "Important to know when using ZodiPy"
    ZodiPy assumes that all coordinates provided in a single call to `evalute` is obtained at an instant in time from the position of the observer. This is a good approximation for observations made within small time intervals, e.g less than a few hours. This is due to the time-varying of the zodiacal light which moves by about 1 degree on the sky each day. 
    
    For the best result, long observations should be split into smaller chunks containing observations within a ~ 1 hour period. 

## Examples

### Emission along an Ecliptic scan
In the following we simulate a scan across the Ecliptic plane

``` py title="ecliptic_scan.py"
{!examples/ecliptic_scan.py!}
```

![Ecliptic scan profile](img/timestream.png)


### HEALPIx maps
We can use [healpy](https://healpy.readthedocs.io/en/latest/) or [Astropy-healpix](https://astropy-healpix.readthedocs.io/en/latest/) package to create a `SkyCoord` object directly from a HEALPIx pixelization

=== "healpy"

    ```py
    --8<-- "docs/examples/healpy_map.py"
    ```

=== "healpy + astropy-healpix"

    ```py
    --8<-- "docs/examples/astropy_healpix_map.py"
    ```

![HEALPix map](img/healpix_map.png)

### Component-wise zodiacal light
We can return the zodiacal light for each component by using setting the keyword argument `return_comps` to `True`
``` py title="component_maps.py"
{!examples/component_maps.py!}
```

![Component maps](img/component_maps.png)


### Visualizing the interplanetary dust distribution
We can evaluate the number density of an interplanetary dust model using the `grid_number_density` function
``` py title="number_density.py"
{!examples/number_density.py!}
```

![DIRBE model interplanetary dust distribution](img/number_density.png)

