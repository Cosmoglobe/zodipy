from zodipy import parameters
from zodipy import emissivities
from zodipy._model import Model


PLANCK_2013 = Model(
    components=('cloud', 'band1', 'band2', 'band3', 'ring', 'feature'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)


PLANCK_2015 = Model(
    components=('cloud', 'band1', 'band2', 'band3'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2015
)


_CLOUD = Model(
    components=('cloud',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BAND1 = Model(
    components=('band1',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BAND2 = Model(
    components=('band2',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_BAND3 = Model(
    components=('band3',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)

_RING = Model(
    components=('ring',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)

_FEATURE = Model(
    components=('feature',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)