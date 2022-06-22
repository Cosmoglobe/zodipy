## Timestreams
Below we illustrate how ZodiPy can be used to create timestreams of the zodiacal emission.

### Timestream along a meridian

In the following example we simulate what an observer on Earth is expected to see on 14 June, 
2022 when looking along the prime meridian (line of constant longitude) at 30 microns, given the 
DIRBE interplanetary dust model.

```python
{!examples/get_emission_ang.py!}
```

![Zodiacal emission timestream](img/timestream.png)

!!! note
    ZodiPy assumes a constant observer position over an input pointing sequence. For an observer on Earth, the true zodiacal emission
    signal will move along the ecliptic on the sky by roughly one degree each day. To account for this effect, the full pointing sequence of an experiment
    must be chunked into small subsequences with timescales corresponding to at maximum a day.

## HEALPix Maps

Below we illustrate how ZodiPy can be used to create simulated binned HEALPix maps of the zodiacal emission.

### Instantaneous Ecliptic map

In the following example we make an instantaneous map of of the zodiacal emission at 857 GHz
as seen by an observer on earth on 14 June, 2022 given the Planck 2018 interplanetary dust model.

```python
{!examples/get_binned_emission.py!}
```
![Zodiacal emission map](img/binned.png)
*Note that the color bar is logarithmic.*

### Instantaneous Galactic map

We can make the same map in galactic coordinates by specifying that the input pointing is in galactic coordinates.

```python hl_lines="18"
{!examples/get_binned_gal_emission.py!}
```
![Zodiacal emission map galactic](img/binned_gal.png)
*Note that the color bar is logarithmic.*

### Component-wise maps

ZodiPy can also return the zodiacal emission component-wise. In the following example we use
the DIRBE model since the later Planck models excluded the circumsolar-ring and Earth-trailing 
feature components. For more information on the interplanetary dust models, please read [Cosmoglobe: Simulating Zodiacal Emission with ZodiPy](https://arxiv.org/abs/2205.12962).

```python hl_lines="18"
{!examples/get_comp_binned_emission.py!}
```
![Component-wise emission maps](img/binned_comp.png)
*Note that the color bar for the Cloud component is logarithmic, while the others are linear.*


## Gridding the Density Distribution of a Model

In the following example we tabulate the density distribution of the DIRBE interplanetary dust model
and plot the cross section of the diffuse cloud components density in the yz-plane.

```python
{!examples/get_density_contour.py!}
```
![Interplanetary dust distribution](img/density_grid.png)
