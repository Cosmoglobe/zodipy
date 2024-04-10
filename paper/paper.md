---
title: 'ZodiPy: A Python package for zodiacal light simulations'
tags:
  - Python
  - astronomy
  - cosmology
  - zodiacal light
  - interplanetary dust
authors:
  - name: Metin San
    orcid: 0000-0003-4648-8729
    equal-contrib: true
    affiliation: "1"
affiliations:
 - name: Metin San, PhD. Fellow, University of Oslo, Norway
   index: 1
date: 10 April 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---


# Summary
`ZodiPy` is an Astropy-affiliated Python package for zodiacal light simulations. 
Its purpose is to provide the astrophysics and cosmology communities with an 
accessible and easy-to-use Python interface to existing zodiacal light models, 
assisting in the analysis of infrared astrophysical data and enabling quick and easy 
zodiacal light forecasting for future experiments. `ZodiPy` implements the 
[@Kelsall1998] and the [@Planck2014] interplanetary dust models, which allow for
zodiacal light simulations between $1.25-240\mu$m and $30-857$GHz, respectively, 
with the possibility of extrapolating the models to other frequencies.

# Statement of need
Zodiacal light is the main source of diffuse radiation observed in the infrared 
sky between $1-100\mu$m. The light comes from scattering and re-emission of 
sunlight by dust grains in the interplanetary medium. Zodiacal light is one of the 
most challenging foregrounds to model in cosmological studies of the Extragalactic 
Background Light (EBL) in the infrared sky, primarily due to its seasonal nature. 

Traditionally, observers of the infrared sky have had to build their own zodiacal 
light tools (see the
[LAMBDA foreground models page](https://lambda.gsfc.nasa.gov/product/foreground/fg_models.html) 
for a list of existing tools). However, these programs are either 
only usable for specific experiments or otherwise difficult to access by requiring 
licensed programming languages or web  interfaces. Many modern astronomy and 
cosmology pipelines are built in Python due to the wide range of available 
high-quality tools and open-source projects and communities, such as the Astropy 
project [@astropy]. The lack of a general-purpose zodiacal light tool in this space 
was the primary motivation behind the development of the `ZodiPy` package.

`ZodiPy` can be used to simulate zodiacal light for arbitrary Solar system observers, 
meaning that researchers no longer have to spend time developing their own tools from 
scratch. To use `ZodiPy`, the user is required to provide the following data: 

1) A sequence of pointings, either in ecliptic or galactic coordinates. These can 
be specified  as angles on the sky or as HEALPix [@Gorski2005] pixel indices.
2) A center frequency or an instrument bandpass.
3) The time of observation, which corresponds to the pointing sequence. 
4) The heliocentric ecliptic position of the observer. If the observer is located 
at a major solar system object, such as the Earth or the Sun-Earth-Moon barycenter 
L2, the position is instead queried through the `astropy.coordinates` Solar system 
ephemerides. 

The predicted zodiacal light is then obtained by evaluating a sequence of 
line-of-sight integrals from the position of the observer and through a 
model of the three-dimensional interplanetary dust distribution. For 
implementation details and examples of how to apply `ZodiPy` to a real-world 
dataset, see[@San2022].

`ZodiPy` has been rapidly adopted by the astronomy community, and the package has 
already been used by several research projects[@2023arXiv230617219A; 
@2023arXiv230617226R; @Tsumura2023; @Avitan2023; @Hanzawa2024] with applications 
ranging from assisting the coming NASA Roman Space Telescope in determining its 
observational fields to modeling data obtained aboard the Hayabusa2 JAXA satellite. 

# Acknowledgements
The work to develop `ZodiPy`  has received funding from the European Union’s Horizon 
2020 research and innovation programme under grant agreement numbers 819478 
(ERC; Cosmoglobe) and 772253 (ERC;bits2cosmology).

This project relies on the following Python packages: Astropy [@astropy], NumPy 
[@numpy2011; @numpy2020], healpy [@Zonca2019], SciPy [@scipy2020], and jplephem 
[@2011ascl.soft12014R].

# References