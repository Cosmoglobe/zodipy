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


PLANCK_2018 = Model(
    components=('cloud', 'band1', 'band2', 'band3'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)