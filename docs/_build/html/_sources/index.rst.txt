.. Zodipy documentation master file, created by
   sphinx-quickstart on Mon Mar 21 08:52:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Zodipy Documentation
====================

Zodipy is a Python tool for simulating the Interplanetary Dust Emission that a
Solar System observer sees, either in the form of timestreams or binned HEALPIX
maps.

.. image:: ../imgs/zodipy_map.png
   :align: center
   :alt: An example map from Zodipy

Requirements
------------

Zodipy requires Python 3.8 or higher and depends on the following packages and version:

- astropy >= 5.0.1
- numpy >= 1.21
- healpy
- scipy
- jplephem
  
Installation
------------

Zodipy is available on PyPI, and can be installed with:

.. code-block:: bash

   pip install zodipy


User Guide
----------

.. toctree::
   :maxdepth: 2

   introduction.rst
   reference.rst
