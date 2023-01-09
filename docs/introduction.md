# Introduction

ZodiPy simulates the zodiacal emission that a solar system observer is predicted to see given an interplanetary dust model. The user selects between a set of built in models, for which the emission can be computed either in the form of timestreams or binned HEALPix maps. 

ZodiPy attempts to make zodiacal emission simulations more accessible by providing the community with a simple Python interface to existing models. For other zodiacal emission tools, see [Zodiacal Light Models on LAMBDA](https://lambda.gsfc.nasa.gov/product/foreground/fg_models.html). ZodiPy is an open source project and all contributions are welcome.


## Interplanetary Dust Models
Currently, ZodiPy supports the following interplanetary dust models:

**1.25-240 $\boldsymbol{\mu}$m**

- DIRBE ([Kelsall et al. 1998](https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K/abstract))
- RRM (experimental version in development) ([Rowan-Robinson and May 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.2894R/abstract))

**100-857 GHz**

- Planck 2013 ([Planck Collaboration et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>))
- Planck 2015 ([Planck Collaboration et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P))
- Planck 2018 ([Planck Collaboration et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P))
- Odegard ([Odegard et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...877...40O/abstract))

!!! info
    The Planck and Odegard models extend the DIRBE interplanetary dust model to CMB frequencies by fitting the blackbody emissivity of the dust in the respective DIRBE interplanetary dust components to Planck HFI data.



## Scientific Paper
For an overview of the ZodiPy model approach and other information regarding zodiacal emission and interplanetary dust modeling we refer to the scientific paper on ZodiPy:

- [Cosmoglobe: Simulating zodiacal emission with ZodiPy](https://arxiv.org/abs/2205.12962)