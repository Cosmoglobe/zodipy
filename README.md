
<img src="docs/img/zodipy_logo.png" width="350">

[![astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange.svg)](http://www.astropy.org/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zodipy)
[![PyPI](https://img.shields.io/pypi/v/zodipy.svg?logo=python)](https://pypi.org/project/zodipy)
[![Actions Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/Zodipy/tests.yml?branch=main&logo=github)](https://github.com/Cosmoglobe/Zodipy/actions)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/zodipy/mkdocs-deploy.yml?branch=main&style=flat-square&logo=github&label=docs)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://img.shields.io/badge/repo_status-Active-success)](https://www.repostatus.org/#active)
[![Codecov](https://img.shields.io/codecov/c/github/Cosmoglobe/zodipy?token=VZP9L79EUJ&logo=codecov)](https://app.codecov.io/gh/Cosmoglobe/zodipy)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pyOpenSci](https://tinyurl.com/y22nb8up)](https://github.com/pyOpenSci/software-review/issues/161)
[![ascl:2306.012](https://img.shields.io/badge/ascl-2306.012-blue.svg?colorB=262255)](https://ascl.net/2306.012)
[![DOI](https://zenodo.org/badge/394929213.svg)](https://zenodo.org/doi/10.5281/zenodo.10999611)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06648/status.svg)](https://doi.org/10.21105/joss.06648)
---



ZodiPy is an [Astropy affiliated](https://www.astropy.org/affiliated/#affiliated-package-list) package for simulating zodiacal light in intensity for arbitrary solar system observers.

![plot](docs/img/zodipy_map.png)


## Documentation
See the [documentation](https://cosmoglobe.github.io/zodipy/) for a list of supported zodiacal light models and examples of how to use ZodiPy.

## A simple example
```python
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

import zodipy

# Initialize a zodiacal light model at a wavelength/frequency or over a bandpass
model = zodipy.Model(25*u.micron)

# Use Astropy's `SkyCoord` object to specify coordinates
lon = [10, 10.1, 10.2] * u.deg
lat = [90, 89, 88] * u.deg
obstimes = Time(["2022-01-01 12:00:00", "2022-01-01 12:01:00", "2022-01-01 12:02:00"])
skycoord = SkyCoord(lon, lat, obstime=obstimes, frame="galactic")

# Evaluate the zodiacal light model
emission = model.evaluate(skycoord)

print(emission)
#> [27.52410841 27.66572294 27.81251906] MJy / sr
```

## Related scientific papers
See [CITATION](https://github.com/Cosmoglobe/zodipy/blob/main/CITATION.bib)
- [Cosmoglobe: Simulating zodiacal emission with ZodiPy (San et al. 2022)](https://arxiv.org/abs/2205.12962). 
- [ZodiPy: A Python package for zodiacal light simulations (San 2024)](https://joss.theoj.org/papers/10.21105/joss.06648#). 


## Install
ZodiPy is installed with pip
```bash
pip install zodipy
```

## Dependencies
ZodiPy supports all Python versions >= 3.9, and has the following dependencies:
- [Astropy](https://www.astropy.org/) (>=5.0.1)
- [NumPy](https://numpy.org/)
- [jplephem](https://pypi.org/project/jplephem/)
- [SciPy](https://scipy.org/)

## For developers
### Poetry
ZodiPy uses [Poetry](https://python-poetry.org/) for development. To build and commit to the repository with the existing pre-commit setup, developers need to have Poetry (>= 1.8.0) installed. See the Poetry [documentation](https://python-poetry.org/docs/) for installation guide. 

After poetry has been installed, developers should create a new virtual environment and run the following in the root of the ZodiPy repositry
```
poetry install
```
This will download all dependencies (including dev)from `pyproject.toml`, and `poetry.lock`.

### Tests, linting and formatting, and building documentation
The following tools should be run from the root of the repository with no errors. (These are ran automatically as part of the CI workflows on GitHub, but should be tested locally first)

#### pytest
Testing is done with [pytest](https://docs.pytest.org/en/8.0.x/). To run the tests, run the following command from the repository root
```bash
pytest
``` 
#### ruff
Formating and linting is done with [ruff](https://github.com/astral-sh/ruff). To format and lint, run the following command from the repository root
```bash
ruff check
ruff format
``` 
#### mypy
ZodiPy is fully typed. We use [mypy](https://mypy-lang.org/) as a static type checker. To type check, run the following command from the repositry root

```bash
mypy zodipy/
```
Remeber to add tests when implementing new features to maintain a high code coverage.

#### MkDocs
We use [MkDocs](https://www.mkdocs.org/) to create our documentation. To serve the docs locally on you machine, run the following from the repositry root
```bash
mkdocs serve
```

## Funding
This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreements No 776282 (COMPET-4; BeyondPlanck), 772253 (ERC; bits2cosmology) and 819478 (ERC; Cosmoglobe).


<div style="display: flex; flex-direction: row; justify-content: space-evenly">
    <img style="width: 49%; height: auto; max-width: 500px; align-self: center" src="https://user-images.githubusercontent.com/28634670/170697040-d5ec2935-29d0-4847-8999-9bc4eaa59e56.jpeg"> 
    &nbsp; 
    <img style="width: 49%; height: auto; max-width: 500px; align-self: center" src="https://user-images.githubusercontent.com/28634670/170697140-b010aa69-9f9a-44c0-b702-8a05ec0b6d3e.jpeg">
</div>
