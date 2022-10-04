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


### Bandpass integrated emission
Instruments do not typically observe at delta frequencies. Usually, we are more interested in finding out
what the emission looks like over some instrument bandpass. ZodiPy accepts a sequence of frequencies to the `freq`
argument in addition to corresponding weights in the `weights` argument. Note that the bandpass weights must be given in power units, i.e. they must be in units compatible with `Jy/sr`. A top hat
bandpass is assumed if a sequence of frequencies are used without providing weights.
```python hl_lines="32 33"
{!examples/get_bandpass_integrated_emission.py!}
```
![Component-wise emission maps](img/center_freq.png)
![Component-wise emission maps](img/bandpass_integrated.png)
!!! warning "Memory usage"
    Bandpass integration in the current implementation uses significantly more memory in the intermediate computations when compared to delta frequency simulations. Consider smaller chunks sizes if this becomes an issue.


## Gridding the interplanetary dust density distribution
In the following example we tabulate the density distribution of the DIRBE interplanetary dust model
and plot the cross section of the diffuse cloud components density in the yz-plane.

```python
{!examples/get_density_contour.py!}
```
![Interplanetary dust distribution](img/density_grid.png)


## Parallel computations
Simulations with large `nside` or with large pointing sequences can be slow to execute due to the massive amounts of line of sights that needs to be computed. We can however speed up calculations by initializing `Zodipy` with `parallel=True`. ZodiPy will then automatically distribute the pointing sequence over all available CPUs on the machine. Optionally, the number of CPUs can also be manually specified by using the `n_proc` keyword when initializing `ZodiPy`.


```python hl_lines="16"
{!examples/get_parallel_emission.py!}
```
!!! warning "Windows users"
    On windows, the parallel code must be executed in a `if __name__ == "__main__"` guard to avoid spawning infinite processes: 
    ```python
    ...
    if __name__ == "__main__":
        emission = model.get_emission_pix(
            ...
        )
    ```
