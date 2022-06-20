# Introduction

ZodiPy simulates the Zodiacal emission that a Solar System observer is predicted to see given an interplanetary dust model. The user selects between a set of built in models, for which the emission can be computed either in the form of timestreams or binned HEALPix maps. 

ZodiPy attempts to make Zodiacal emission simulations more accessible by providing the community with a simple Python interface to existing models. For other Zodiacal emission tools, see [Zodiacal Light Models on LAMBDA](https://lambda.gsfc.nasa.gov/product/foreground/fg_models.html). ZodiPy is an open source project and all contributions are welcome.


## Interplanetary Dust Models
Currently, ZodiPy supports the following interplanetary dust models:

- dirbe ([Kelsall et al. 1998](https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K/abstract))
- planck13 ([Planck Collaboration et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>))
- planck15 ([Planck Collaboration et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P>))
- planck18 ([Planck Collaboration et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P>))
- odegard ([Odegard et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...877...40O/abstract))
