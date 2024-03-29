# Introduction

ZodiPy is an open source Python tool for simulating the zodiacal emission that a solar system observer is predicted to see given an interplanetary dust model. We attempts to make zodiacal emission simulations more accessible by providing the community with a simple interface to existing models. For other zodiacal emission tools, see [Zodiacal Light Models on LAMBDA](https://lambda.gsfc.nasa.gov/product/foreground/fg_models.html). All contributions are most welcome.


## Interplanetary Dust Models
ZodiPy supports the following interplanetary dust models:

**1.25-240 $\boldsymbol{\mu}$m**

- DIRBE ([Kelsall et al. 1998](https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K/abstract))
- RRM (experimental) ([Rowan-Robinson and May 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.2894R/abstract))

**100-857 GHz**

- Planck 2013 ([Planck Collaboration et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>))
- Planck 2015 ([Planck Collaboration et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P))
- Planck 2018 ([Planck Collaboration et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P))
- Odegard ([Odegard et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...877...40O/abstract))

!!! info
    The Planck and Odegard models extend the DIRBE interplanetary dust model to CMB frequencies by fitting the blackbody emissivity of the dust in the respective DIRBE interplanetary dust components to Planck HFI data.
    The distribution of the interplanetary dust is exactly the same as in the DIRBE model.

If you see a missing model, please feel free to contact us by opening an issue on GitHub. 


## Scientific Paper
For an overview of the modeling approach used in ZodiPy and other information regarding zodiacal emission and interplanetary dust modeling we refer to the scientific paper on ZodiPy:

- [Cosmoglobe: Simulating zodiacal emission with ZodiPy](https://arxiv.org/abs/2205.12962)