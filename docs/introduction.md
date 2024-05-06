# Introduction

ZodiPy is a Python package for zodiacal light simulations. Its purpose is to provide the astrophysics and cosmology communities with an easy-to-use and accessible interface to existing zodiacal light models in Python, assisting in astrophysical data analysis and zodiacal light forecasting for future experiments.

For other zodiacal light tools, see [Zodiacal Light Models on LAMBDA](https://lambda.gsfc.nasa.gov/product/foreground/fg_models.html).

## Supported zodiacal light models

- DIRBE [`"dirbe"`] ([Kelsall et al. 1998](https://ui.adsabs.harvard.edu/abs/1998ApJ...508...44K/abstract))
- Rowan-Robinson and May (experimental) [`"rrm-experimental"`] ([Rowan-Robinson and May 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.429.2894R/abstract))
- Planck 2013 [`"planck13"`] ([Planck Collaboration et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...571A..14P/abstract>))
- Planck 2015 [`"planck15"`] ([Planck Collaboration et al. 2016](https://ui.adsabs.harvard.edu/abs/2016A&A...594A...8P))
- Planck 2018 [`"planck18"`] ([Planck Collaboration et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A&A...641A...3P))
- Odegard [`"odegard"`] ([Odegard et al. 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...877...40O/abstract))

The names in the brackets are the string representations used in the `zodipy.Model` object to select the model.

If you see a missing model or wish to add a new one, please open an issue on GitHub. Contributors are very welcome!


## Related scientific papers
See [CITATION](https://github.com/Cosmoglobe/zodipy/blob/main/CITATION.bib)

- [Cosmoglobe: Simulating zodiacal emission with ZodiPy (San et al. 2022)](https://arxiv.org/abs/2205.12962). 
- [ZodiPy: A Python package for zodiacal light simulations (San 2024)](https://joss.theoj.org/papers/10.21105/joss.06648#). 
