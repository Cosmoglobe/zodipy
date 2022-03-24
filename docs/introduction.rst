Introduction
============

Zodipy simulates the Zodiacal emission that a Solar
System observer is predicted to see given an Interplanetary Dust model. The user must
either select a built in observer ajd a time of observation, or input the
position of the observer explicitly. Additionally, the user must specify the
pointing of the observer, either in form of angular coordinates ``theta`` and
``phi``, or though pixel numbers refering to the pixels on a HEALPix grid.

The software is described in further detail in *San et al. 2022* (in preperation).

--------------
User Interface
--------------
The :ref:`Zodipy overview` class represents the Zodipy interface. It is intialized as following:

.. code-block:: python

   from zodipy import Zodipy

   model = Zodipy()

--------------------------
Interplanetary Dust models
--------------------------
The default Interplanetary Dust model used in Zodipy is the `DIRBE model <https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K>`_. 
Other built in models are:

   - Planck13 (`Planck Collaboration et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>`_)
   - Planck15 (`Planck Collaboration et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P>`_)
   - Planck18 (`Planck Collaboration et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P>`_)
   - Odegard (`Odegard et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...877...40O/abstract>`_)

.. code-block:: python

   model = Zodipy(model="Planck15")

----------------------------
Simulating Zodiacal emission
----------------------------
Once Zodipy is initialized, the next step is to simulate some emission. This is
done with the ``get_emission`` function. Below is a list of the key arguments a
user is expected to provide (see :ref:`Zodipy overview` for a full list of
argument):

- **freq** (required): must be an ``astropy.quantity.Quantity`` object with units of either frequncy or length.
- **obs_time** (required): must be an ``astropy.time.Time`` object that represents the time of observation.
- **obs**: a string representing the observer, e.g. "earth". Must be supported by the ephemeris or be "SEMB-L2".
- **pixels** and **nside**: ``Pixels`` must be a ``Sequence[int]`` or an ``NDArray[int]`` representing the pixel indicies on a HEALPix grid, and ``nside`` is an integer representing the resolution.
- **theta** and **phi**: Angular coordinates on the sky (co-latitude, longitude). Must be ``astropy.quantity.Quantity`` objects with units of either degrees or radians.

Below is a minimal example where we use the ``get_emission`` function to compute
the emission as predicted by the "Planck18" model and seen by an observer
located at the Sun-Earth-Moon-Barycenter Lagrange point 2, who observes a point on the sky given by the coordinate
``theta`` and ``phi``:

.. code-block:: python

   import astropy.units as u
   from astropy.time import Time
   from zodipy import Zodipy

   model = Zodipy(model="Planck18")

   emission = model.get_emission(
      100*u.micron,
      obs="SEMB-L2",
      obs_time=Time("2022-06-14"),
      theta=23.4*u.deg,
      phi=-30*u.deg
   )

   >> print(emission)
   <Quantity [0.73169181] MJy / sr>

For other use-cases, see the `GitHub README
<https://github.com/MetinSa/zodipy>`_. A Zodipy tutorial will be made in the
future.