from zodipy import emissivities
from zodipy import parameters
from zodipy._model import IPDModel, model_registry


model_registry.register_model(
    IPDModel(
        name="K98",
        parameters=parameters.K98,
    )
)

model_registry.register_model(
    IPDModel(
        name="Planck13",
        parameters=parameters.PLANCK,
        emissivities=emissivities.PLANCK_2013,
    )
)

model_registry.register_model(
    IPDModel(
        name="Planck15",
        parameters=parameters.PLANCK,
        emissivities=emissivities.PLANCK_2015,
    )
)

model_registry.register_model(
    IPDModel(
        name="Planck18",
        parameters=parameters.PLANCK,
        emissivities=emissivities.PLANCK_2018,
    )
)
