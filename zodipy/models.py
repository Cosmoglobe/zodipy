from zodipy._model import ModelRegistry
from zodipy._components import ComponentLabel
from zodipy import emissivities
from zodipy import parameters


model_registry = ModelRegistry()

model_registry.register_model(
    name="planck 2013",
    components=[
        ComponentLabel.CLOUD,
        ComponentLabel.BAND1,
        ComponentLabel.BAND2,
        ComponentLabel.BAND3,
        ComponentLabel.RING,
        ComponentLabel.FEATURE,
    ],
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2013,
)
model_registry.register_model(
    name="planck 2015",
    components=[
        ComponentLabel.CLOUD,
        ComponentLabel.BAND1,
        ComponentLabel.BAND2,
        ComponentLabel.BAND3,
    ],
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2015,
)
model_registry.register_model(
    name="planck 2018",
    components=[
        ComponentLabel.CLOUD,
        ComponentLabel.BAND1,
        ComponentLabel.BAND2,
        ComponentLabel.BAND3,
    ],
    parameters=parameters.K98,
    emissivities=emissivities.PLANCK_2018,
)
model_registry.register_model(
    name="K98",
    components=[
        ComponentLabel.CLOUD,
        ComponentLabel.BAND1,
        ComponentLabel.BAND2,
        ComponentLabel.BAND3,
        ComponentLabel.RING,
        ComponentLabel.FEATURE,
    ],
    parameters=parameters.K98,
    emissivities=None,
)
