
<img src="docs/img/zodipy_logo.png" width="350">

[![astropy](http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat-square)](http://www.astropy.org/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zodipy?style=flat-square)
[![PyPI](https://img.shields.io/pypi/v/zodipy.svg?logo=python&style=flat-square)](https://pypi.org/project/zodipy)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://img.shields.io/badge/repo_status-Active-success?style=flat-square)](https://www.repostatus.org/#active)
[![Actions Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/Zodipy/tests.yml?branch=main&logo=github&style=flat-square)](https://github.com/Cosmoglobe/Zodipy/actions)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/Cosmoglobe/zodipy/mkdocs-deploy.yml?branch=main&style=flat-square&logo=github&label=docs)
[![Codecov](https://img.shields.io/codecov/c/github/Cosmoglobe/zodipy?token=VZP9L79EUJ&style=flat-square&logo=codecov)](https://app.codecov.io/gh/Cosmoglobe/zodipy)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2205.12962-green?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2205.12962)
[![ascl:2306.012](https://img.shields.io/badge/ascl-2306.012-blue.svg?colorB=262255&style=flat-square)](https://ascl.net/2306.012)
---


ZodiPy is an [Astropy affiliated](https://www.astropy.org/affiliated/) package for simulating zodiacal light in intensity for arbitrary Solar system observers.

![plot](docs/img/zodipy_map.png)

# Help
See the [documentation](https://cosmoglobe.github.io/zodipy/) for more information and examples on how to use ZodiPy for different applications.

# Installation
ZodiPy is installed using `pip install zodipy`.

# Dependencies
ZodiPy supports all Python versions >= 3.9, and has the following dependencies:
- [Astropy](https://www.astropy.org/) (>=5.0.1)
- [NumPy](https://numpy.org/)
- [healpy](https://healpy.readthedocs.io/en/latest/)
- [jplephem](https://pypi.org/project/jplephem/)
- [SciPy](https://scipy.org/)


# A simple example
```python
import astropy.units as u
from astropy.time import Time

from zodipy import Zodipy


model = Zodipy("dirbe")

emission = model.get_emission_ang(
    25 * u.micron,
    theta=[10, 10.1, 10.2] * u.deg,
    phi=[90, 89, 88] * u.deg,
    obs_time=Time("2022-01-01 12:00:00"),
    obs="earth",
)

print(emission)
#> [15.35392831 15.35495051 15.35616009] MJy / sr
```

# Scientific paper and citation
For an overview of the ZodiPy model approach and other information regarding zodiacal emission and interplanetary dust modeling we refer to the scientific paper on ZodiPy:
- [Cosmoglobe: Simulating zodiacal emission with ZodiPy (San et al. 2022)](https://arxiv.org/abs/2205.12962). 

See [CITATION](https://github.com/Cosmoglobe/zodipy/blob/dev/CITATION.bib) if you have used ZodiPy in your work and want to cite the software.


# For developers
Contributing developers will need to download the following additional dependencies to test, lint, format and build documentation locally:
- pytest
- pytest-cov
- hypothesis
- coverage
- ruff
- mypy
- pre-commit
- mkdocs
- pymdown-extensions
- markdown-include
- mkdocs-material
- mkdocstrings
- mkdocstrings-python
- markdown (<3.4.0)

which are required to test and build ZodiPy.

## Poetry
Developers can install ZodiPy through [Poetry](https://python-poetry.org/) (Poetry >= 1.8.0) by first cloning or forking the repository, and then running 
```
poetry install
```
in a virtual environment from the repository root. This will read the `pyproject.toml` file in the repository and install all dependencies. 

## pip
Developers not using Poetry can install ZodiPy in a virtual environment with all dependencies by first cloning or forking the repository and then running 
```
pip install -r requirements-dev.txt
```
from the repositry root. This will read and download all the dependencies from the `requirements-dev.txt` file in the repository. 

Note that developers using Python 3.12 will need to upgrade their pip versions with `python3 -m pip install --upgrade pip` before being able to install ZodiPy. This is due to known incompatibilities between older pip versions and Python 3.12

## Tests, linting and formatting
The following tools should be run from the root of the repository with no errors. (These are ran automatically as part of the CI workflows on GitHub, but should be tested locally first)

- [pytest](https://docs.pytest.org/en/8.0.x/): Tests are run with pytest by simply running `pytest` in the command line in the root of the repository. 
- [ruff](https://github.com/astral-sh/ruff): Formating and linting is done with `ruff` by simply running `ruff check zodipy/` and `ruff format zodipy/` in the command line in the root of the repository. 
- [mypy](https://mypy-lang.org/): Type checking is done with `mypy` by simply running `mypy zodipy/` in the root of the repository.

Remeber to add tests when implementing new features to maintain a high code coverage.

## Documentation
We use [MkDocs](https://www.mkdocs.org/) to create our documentation. The documentation is built locally with `mkdocs build` from the repository root, and served with `mkdocs serve`.


# Funding
This work has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreements No 776282 (COMPET-4; BeyondPlanck), 772253 (ERC; bits2cosmology) and 819478 (ERC; Cosmoglobe).


<div style="display: flex; flex-direction: row; justify-content: space-evenly">
    <img style="width: 49%; height: auto; max-width: 500px; align-self: center" src="https://user-images.githubusercontent.com/28634670/170697040-d5ec2935-29d0-4847-8999-9bc4eaa59e56.jpeg"> 
    &nbsp; 
    <img style="width: 49%; height: auto; max-width: 500px; align-self: center" src="https://user-images.githubusercontent.com/28634670/170697140-b010aa69-9f9a-44c0-b702-8a05ec0b6d3e.jpeg">
</div>
