Introduction
============

`Zodipy` simulates the Zodiacal / Interplanetary Dust emission that a Solar
System observer is predicted to see at the time of observation. The user must
either select a built in observer along with a time of observation, or input the
position of the observer explicitly. Additionally, the user must specify the
pointing of the observer, either in form of angular coordinates ``theta`` and
``phi``, or though pixel numbers refering to the pixels on a HEALPIX grid.

The software is described in further detail in *San et al. 2022* (in preperation).

--------------
User Interface
--------------
The `Zodipy` interface is given by the :ref:`Zodipy overview` class. It is intialized as following:

.. code-block:: python

   from zodipy import Zodipy

   model = Zodipy()

--------------------------
Interplanetary Dust Models
--------------------------
The default Interplanetary Dust model used in `Zodipy` is the `DIRBE model <https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K>`_. 
Other built in models are:

   - Planck13 (`Planck Collaboration et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>`_)
   - Planck15 (`Planck Collaboration et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P>`_)
   - Planck18 (`Planck Collaboration et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P>`_)

Selecting one of this is done in the intialization of `Zodipy`:

.. code-block:: python

   model = Zodipy(model="Planck15")

-----------------------------
The ``get_emission`` Function
-----------------------------
Once `Zodipy` is initialized, the next step is to simulate some emission. This
is done through the ``get_emission`` function (see :ref:`Zodipy overview`). This
function takes the following main arguments. 

- **freq** (required): must be an ``astropy.quantity.Quantity`` object with units of frequnecy or length, and represents the frequency / wavelength for which we want to simulate the Zodiacal emission. 
- **obs_time** (required): must be an ``astropy.time.Time`` object that represents the time of observation.
- **obs**: a string representing the observer, e.g. "earth".
- **pixels** and **nside**: ``Pixels`` must be a ``sequence[int]`` or an ``NDArray[int]`` representing the pixel indicies on a HEALPIX grid, and ``nside`` is an integer representing the resolution of the HEALPIX grid.
- **theta** and **phi**: Angular coordinates on the sky (co-latitude, longitude). Must be ``astropy.quantity.Quantity`` objects with units of either degrees or radians.

Below is a minimal example where we use the ``get_emission`` function to compute
the emission as predicted by the "Planck18" model and seen by an observer
located at the Sun-Earth-Moon-Barycenter L2, who observes a point on the sky given by the coordinate
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


For more information on the ``get_emission`` function and all of its optional arguments, please see :ref:`Zodipy overview`.