## Timestreams
Below we illustrate how ZodiPy can be used to create timestreams of the zodiacal emission.


### Emission along a meridian
In the following example we simulate what an observer on Earth is expected to see on 14 June, 
2022 when looking along a meridian (line of constant longitude) at 30 microns, given the 
DIRBE interplanetary dust model.

```python
{!examples/get_emission_ang.py!}
```

![Zodiacal emission timestream](img/timestream.png)

!!! note
    ZodiPy assumes a constant observer position over an input pointing sequence. For an observer on Earth, the true zodiacal emission
    signal will move along the ecliptic on the sky by roughly one degree each day. To account for this effect, the full pointing sequence of an experiment
    must be chunked into small subsequences with timescales corresponding to at maximum a day.


## HEALPix maps
Below we illustrate how ZodiPy can be used to create simulated binned HEALPix maps of the zodiacal emission.


### Instantaneous map in ecliptic coordinates
In the following example we make an instantaneous map of of the zodiacal emission at 857 GHz
as seen by an observer on earth on 14 June, 2022 given the Planck 2018 interplanetary dust model.

```python
{!examples/get_binned_emission.py!}
```
![Zodiacal emission map](img/binned.png)
*Note that the color bar is logarithmic.*

### Solar cutoff angle
Few experiments look directly in towards the Sun. We can initialize `Zodipy` with the `solar_cut` argument to mask all input pointing that looks in towards the sun with an angular distance smaller than the `solar_cut` value.

```python hl_lines="9"
{!examples/get_binned_emission_solar_cutoff.py!}
```
![Zodiacal emission map](img/binned_solar_cutoff.png)


### Instantaneous map in Galactic coordinates
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
*Note that the color for the Cloud component is logarithmic, while the others are linear.*


## Gridding the interplanetary dust density distribution
In the following example we tabulate the density distribution of the DIRBE interplanetary dust model
and plot the cross section of the diffuse cloud components density in the yz-plane.

```python
{!examples/get_density_contour.py!}
```
![Interplanetary dust distribution](img/density_grid.png)
