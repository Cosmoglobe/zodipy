from zodipy import emissivities
from zodipy import parameters
from zodipy._model import IPDModelRegistry
from zodipy._component_labels import ComponentLabel


model_registry = IPDModelRegistry()

model_registry.register_model(
    name="Planck13",
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
    name="Planck15",
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
    name="Planck18",
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
)
