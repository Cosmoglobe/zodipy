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


MODELS = {
    'planck 2013' : PLANCK_2013,
    'planck 2015' : PLANCK_2015,
    'planck 2018' : PLANCK_2018
}
