Introduction
============

`Zodipy` simulates the Zodiacal / Interplanetary Dust emission that a Solar
System observer is predicted to see at the time of observation. The user must
either select a built in observer along with a time of observation, or input the
position of the observer explicitly. Additionally, the user must specify the
pointing of the observer, either in form of angular coordinates ``theta`` and
``phi``, or though pixel numbers refering to the pixels on a HEALPIX grid.

The software is described in further detail in *San et al. 2022* (in preperation).

Installation 
------------ 
`Zodipy` is available on PyPI, and can be installed with ``pip install zodipy``.
It has the following dependencies:

- Python >= 3.8
- astropy >= 5.0.1
- numpy >= 1.21
- healpy
- scipy
- jplephem

Interplanetary Dust Models
--------------------------
The default Interplanetary Dust model used in `zodipy` is the `DIRBE model <https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K>`_. Other options
are:

   - `Planck 2013 <https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>`_
   - `Planck 2015 <https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P>`_
   - `Planck 2018 <https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P>`_

A Quick Example
---------------
Here is a simple example of using `Zodipy` to simulate the Zodiacal emission
from a point on the sky:

.. code-block:: python

   import astropy.units as u
   from astropy.time import Time
   from zodipy import Zodipy

   model = Zodipy(model="Planck18")

   emission = model.get_emission(
      100*u.micron,
      obs="SEMB-L2",
      obs_time=Time.now(),
      theta=23.4*u.deg,
      phi=-30*u.deg
   )

   >> print(emission)
   <Quantity [0.85517232] MJy / sr>
