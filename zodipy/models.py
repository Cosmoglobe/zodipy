from zodipy import emissivities
from zodipy import parameters
from zodipy._model import InterplanetaryDustModel


PLANCK_2013 = InterplanetaryDustModel(
    components=('cloud', 'band1', 'band2', 'band3', 'ring', 'feature'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)


PLANCK_2015 = InterplanetaryDustModel(
    components=('cloud', 'band1', 'band2', 'band3'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2015
)

PLANCK_2018 = InterplanetaryDustModel(
    components=('cloud', 'band1', 'band2', 'band3'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

# Single component models used for testing
# ----------------------------------------
_CLOUD = InterplanetaryDustModel(
    components=('cloud',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BAND1 = InterplanetaryDustModel(
    components=('band1',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BAND2 = InterplanetaryDustModel(
    components=('band2',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BAND3 = InterplanetaryDustModel(
    components=('band3',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BANDS = InterplanetaryDustModel(
    components=('band3',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_RING = InterplanetaryDustModel(
    components=('ring',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)

_FEATURE = InterplanetaryDustModel(
    components=('feature',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)

_CIRCUMSOLAR = InterplanetaryDustModel(
    components=('ring','feature',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)