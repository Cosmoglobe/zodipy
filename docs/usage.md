!!! warning "Breaking API changes in `v1.0.0`"
    In version `v1.0.0` the `get_emission_*` and `get_binned_emission_*` methods were deprecated 
    and removed. Users wanting to simulate zodiacal light directly from HEALPix pixel indices should
    see the [HEALPix](usage.md#healpix) section under [examples](usage/#examples).

As an Astropy-affiliated package, ZodiPy is highly integrated with the Astropy ecosystem, 
particularly with the [`astropy.units`](https://docs.astropy.org/en/stable/units/), 
[`astropy.coordinates`](https://docs.astropy.org/en/stable/coordinates/index.html), and 
[`astropy.time`](https://docs.astropy.org/en/stable/time/index.html) modules.

## Initializing a zodiacal light model
To make zodiacal light simulations we must first import and initialize a zodiacal light model
```py hl_lines="2 4"
import astropy.units as u
import zodipy

model = zodipy.Model(25 * u.micron)
```
The [`zodipy.Model`][zodipy.Model] object has *one* required positional argument, `x`, which can 
either represent a center wavelength/frequency or the points of an empirical bandpass. If `x` 
represents the points of an instrument bandpass, the `weights` argument must also be provided
```py hl_lines="4 5 7"
import astropy.units as u
import zodipy

points = [3, 4, 5, 6] * u.micron
weights = [0.2, 0.4, 0.3, 0.1]

model = zodipy.Model(points, weights=weights)
```

ZodiPy supports [several zodiacal light models](introduction.md) valid at different 
wavelength/frequency ranges. By default, a [`Model`][zodipy.Model] will initialize on the 
[DIRBE model](https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K/abstract). To select another 
model, for instance the [Planck 2013 model](https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/metrics) ,
we need to specify the `name` keyword in the model constructor
```py hl_lines="4"
import astropy.units as u
import zodipy

model = zodipy.Model(25 * u.micron, name="planck13")
```

Other possible keyword arguments to the [`zodipy.Model`][zodipy.Model] are `gauss_quad_degree`, 
which determines the number of discrete points evaluated along each line-of-sight, `extrapolate`, 
which, if set to `True`, uses linear interpolation to extrapolate the frequency/wavelength dependent 
model parameters allowing the model to be evaluated outside of it's original bounds, and finally, 
`ephemeris`, which one can use to specify the
[solar system ephemeris](https://docs.astropy.org/en/stable/coordinates/solarsystem.html). used to 
compute the position of the Earth and optionally the observer.

## Evaluating a zodiacal light model
To make zodiacal light simulations ZodiPy needs some inputs data from the user:

1. **Sky coordinates**. Provided through Astropy's 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object.
2. **Time of observeration(s)**. Also provided in the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object.
3. **Position(s) of the observer**. Provided in a separate argument when evaluating the model.

### The [`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object
Users unfamiliar with Astropy's 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) should first 
visit the [official Astropy docs](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html)
to learn the basics.

For a single observation in galactic coordinates, the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object may 
look something like the following:
```py hl_lines="2 3 5 6 7 8 9 10"
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

skycoord = SkyCoord(
    40 * u.deg, 
    60 * u.deg, 
    obstime=Time("2020-01-01"), 
)
```
where the coordinates here are specified as longitude and latitude values. Note that the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object is very 
flexible and supports many 
[different formats](https://docs.astropy.org/en/stable/coordinates/skycoord.html#examples) for the
coordinates.

The `obstime` keyword is mandatory and is given by the Astropy 
[`Time`](https://docs.astropy.org/en/stable/time/ref_api.html#module-astropy.time) object, which can
 represent time in many formats, including regular and modified Julian dates (see the 
 [`Time` docs](https://docs.astropy.org/en/stable/time/) for more information).

For several observations, each with their own `obstime` input, the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) may instead
look like the following:
```py hl_lines="6 7 8"
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

skycoord = SkyCoord(
    [40, 41, 42] * u.deg, 
    [60, 59, 58] * u.deg, 
    obstime=Time(["2020-01-01", "2020-01-02", "2020-01-03"]), 
)
```
If a single value is given for `obstime`, all coordinates are assumed to be viewed instantaneously 
at that time from a single position in the solar system. Otherwise, each coordinate must have its 
own `obstime` value.

The sky coordinates should represent observer-centric coordinates. The observer-position is 
therefore required to compute the line-of-sight integrals, but this is provided not in the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object, but
 rather in the [`evaluate`][zodipy.Model.evaluate] method, which we will see soon.

The coordinate frame in the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object 
defaults to the [`ICRS`](https://docs.astropy.org/en/stable/api/astropy.coordinates.ICRS.html) frame, 
but can be changed by providing the `frame` keyword argument:
```py hl_lines="9"
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

skycoord = SkyCoord(
    [40, 41, 42] * u.deg, 
    [60, 59, 58] * u.deg, 
    obstime=Time(["2020-01-01", "2020-01-02", "2020-01-03"]), 
    frame="galactic",
)
```

Common coordinate frames are the Ecliptic, Galactic, and Celestial frames (these are represented as 
"E", "G", "C" in [`healpy`](https://healpy.readthedocs.io/en/latest/)), which can be specified 
either through string representations:

- `"barycentricmeanecliptic"` (Ecliptic)
- `"galactic"` (Galactic)
- `"icrs"` (Celestial)

or through frame objects imported from `astropy.coordinates`:

- [`BarycentricMeanEcliptic`](https://docs.astropy.org/en/stable/api/astropy.coordinates.BarycentricMeanEcliptic.html)
- [`Galactic`](https://docs.astropy.org/en/stable/api/astropy.coordinates.Galactic.html)
- [`ICRS`](https://docs.astropy.org/en/stable/api/astropy.coordinates.ICRS.html)

!!! info "Notes on coordinate frames"
    Note that these built-in Astropy frames do not inherently represent observer-centric coordinate 
    frames. However this is fine, since we only need to know the rotation of the coordinates with 
    respect to the ecliptic plane (internally, the coordinates are manually shifted to heliocentric 
    coordinates using the `obspos` value)-

### The [`evaluate`][zodipy.Model.evaluate] method
The zodiacal light is evaluated by providing the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object to the 
[`evaluate`][zodipy.Model.evaluate] method.

```py hl_lines="15"
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
By default, the observer is assumed to be the center of the Earth. In this case the Earth position is 
computed internally using Astropy's 
[solar system ephemeris](https://docs.astropy.org/en/stable/coordinates/solarsystem.html). 
The position of the observer can be explicitly provided through the keyword argument `obspos` in the
 [`evaluate`][zodipy.Model.evaluate] method

```py hl_lines="15 19"
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
This argument accepts both a string representing a body recognized by the 
[solar system ephemeris](https://docs.astropy.org/en/stable/coordinates/solarsystem.html), or a 
heliocentric ecliptic cartesian position.

Similar to with the `obstime` attribute in the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object, the 
value provided to the `obspos` keyword must have shape `(3,)` or `(3, ncoords)`, where `ncoords` is 
the number of coordinates in the 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object. 


### Multiprocessing
ZodiPy will distribute the input coordinates to available cores using Python's 
[`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html) module if the keyword 
argument `nprocesses` in the [`evaluate`][zodipy.Model.evaluate] method is greater or equal to 1.

```py
import multiprocessing

nprocesses = multiprocessing.cpu_count() # 8 cores

emission = model.evaluate(skycoord, nprocesses=nprocesses)
```


## Examples

### Zodiacal light along an Ecliptic scan
In the following, we simulate a scan across the Ecliptic plane:

``` py title="ecliptic_scan.py"
{!examples/ecliptic_scan.py!}
```

![Ecliptic scan profile](img/ecliptic_scan.png)


### HEALPix
We can use [healpy](https://healpy.readthedocs.io/en/latest/) or 
[Astropy-healpix](https://astropy-healpix.readthedocs.io/en/latest/) to create a 
[`SkyCoord`](https://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html) object 
directly from a HEALPix pixelization. In the following two examples, we produce an instantaneous map 
of the sky in HEALPix representation:

=== "healpy"

    ```py
    --8<-- "docs/examples/healpy_map.py"
    ```

=== "astropy-healpix"

    ```py
    --8<-- "docs/examples/astropy_healpix_map.py"
    ```

![HEALPix map](img/healpix_map.png)

### Component-wise zodiacal light
We can return the zodiacal light for each component by using setting the keyword argument 
`return_comps` to `True` in the [`evaluate`][zodipy.Model.evaluate] method:
``` py title="component_maps.py"
{!examples/component_maps.py!}
```

![Component maps](img/component_maps.png)


### Visualizing the interplanetary dust distribution
We can visualize the number density of a supported zodiacal light model by using the 
[`grid_number_density`][zodipy.grid_number_density] function
``` py title="number_density.py"
{!examples/number_density.py!}
```

![DIRBE model interplanetary dust distribution](img/number_density.png)

