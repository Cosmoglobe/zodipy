# Zodipy

## Description
Zodipy is a python tool that simulates the instantaneous Zodiacal emission as seen from an observer.

## Usage
The following will produce a HEALPIX map at `NSIDE=128` of the simulated
emission as seen by the Planck satellite today
```python
import zodipy

zodi = zodipy.Zodi(observer='Planck')
emission = zodi.simulate(nside=128, freq=700)
```

The time of observation is can be changed as following
```python
import datetime

time = datetime(2010, 1, 1)
zodi = zodipy.Zodi(observer='L2', observation_time=time)
emission = zodi.simulate(nside=256, freq=800)
```

These maps can then be visualized using Healpy and matplotlib
```python
import healpy as hp
import matplotlib.pyplot as plt

hp.mollview(
    emission, 
    norm='hist', 
    unit='W/m^2 Hz sr', 
    title='Zodiacal Emission as seen from L2 (2010-01-01)', 
)
plt.savefig('imgs/zodi.png')
plt.show()
```

![plot](imgs/zodi.png)


