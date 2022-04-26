Introduction
============

Zodipy simulates the Zodiacal emission that a Solar
System observer is predicted to see given an Interplanetary Dust model. The user must
either select a built in observer ajd a time of observation, or input the
position of the observer explicitly. Additionally, the user must specify the
pointing of the observer, either in form of angular coordinates ``theta`` and
``phi``, or though pixel numbers refering to the pixels on a HEALPix grid.

The software is described in further detail in *San et al. 2022* (in preperation).

--------------------------
Interplanetary Dust models
--------------------------
The default Interplanetary Dust model used in Zodipy is the `DIRBE model <https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K>`_. 
Other built in models are:

   - planck13 (`Planck Collaboration et al. 2014 <https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>`_)
   - planck15 (`Planck Collaboration et al. 2016 <https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P>`_)
   - planck18 (`Planck Collaboration et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P>`_)
   - odegard (`Odegard et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...877...40O/abstract>`_)

--------------
User Interface
--------------
The :ref:`Zodipy overview` class represents the Zodipy interface. It is initialized as following:

.. code-block:: python

   from zodipy import Zodipy

   model = Zodipy("planck18")

The interface also accepts the following optional arguments:

.. code-block:: python

   model = Zodipy(
      model="planck18", 
      ephemeris="builtin",                # Ephemeris used to compute position of 
                                          # Solar System bodies. Default is "de432s".
      solar_irradiance_model="gueymard",  # Solar flux model. Only relevant at 
                                          # scattering dominated wavelengths around
                                          # ~1 micron. Default is "dirbe".
      extrapolate=True,                   # Whether or not to extrapolate in the 
                                          # spectral parameters in the model. Default is 
                                          # False.
      gauss_quad_order=125,               # Number of quadrature points in the line of 
                                          # sight integration. Default is 100.
   )

----------------------------
Simulating Zodiacal emission
----------------------------
The `Zodipy` interface has *four* methods with similar interfaces for computing the emission, given the initialized model.
These are: ``get_emission_ang``, ``get_emission_pix``, ``get_binned_emission_ang``, and ``get_binned_emission_pix``. 
See :ref:`Zodipy overview` for an overview of the API.

Below is a minimal example where we use the ``get_emission_ang`` function to compute
the emission as predicted by the "Planck18" model seen by an observer
located at the Sun-Earth-Moon-Barycenter Lagrange point 2, who observes a point on the sky:

.. code-block:: python

   import astropy.units as u
   from astropy.time import Time
   from zodipy import Zodipy

   model = Zodipy(model="planck18")

   emission = model.get_emission_ang(
      654*u.GHz,
      obs="semb-l2",
      obs_time=Time("2022-06-14"),
      theta=23.4*u.deg,
      phi=-30*u.deg
   )

   >> print(emission)
   [0.03166994] MJy / sr

For other use-cases, see the `GitHub README
<https://github.com/MetinSa/zodipy>`_. A `Zodipy` tutorial will be made in the
future.