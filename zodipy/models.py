from zodipy._model import ModelFactory
from zodipy import emissivities
from zodipy import parameters


models = ModelFactory()

models.register_model(
    name='planck 2013',
    components=('cloud', 'band1', 'band2', 'band3', 'ring', 'feature'),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013
)
models.register_model(
    name='planck 2015',
    components=('cloud', 'band1', 'band2', 'band3',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2015
)
models.register_model(
    name='planck 2018',
    components=('cloud', 'band1', 'band2', 'band3',),
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018
)