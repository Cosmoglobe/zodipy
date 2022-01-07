from zodipy import source_parameters
from zodipy import component_parameters
from zodipy._model import IPDModel, model_registry


model_registry.register_model(
    IPDModel(
        name="Planck13",
        component_parameters=component_parameters.PLANCK,
        interplanetary_temperature=source_parameters.T_0_K98,
        delta=source_parameters.delta_K98,
        emissivity=source_parameters.EMISSIVITY_PLANCK_13
    )
)

model_registry.register_model(
    IPDModel(
        name="Planck15",
        component_parameters=component_parameters.PLANCK,
        interplanetary_temperature=source_parameters.T_0_K98,
        delta=source_parameters.delta_K98,
        emissivity=source_parameters.EMISSIVITY_PLANCK_15,
    )
)

model_registry.register_model(
    IPDModel(
        name="Planck18",
        component_parameters=component_parameters.PLANCK,
        interplanetary_temperature=source_parameters.T_0_K98,
        delta=source_parameters.delta_K98,
        emissivity=source_parameters.EMISSIVITY_PLANCK_18,
    )
)


model_registry.register_model(
    IPDModel(
        name="DIRBE",
        component_parameters=component_parameters.K98,
        interplanetary_temperature=source_parameters.T_0_K98,
        delta=source_parameters.delta_K98,
        emissivity=source_parameters.EMISSIVITY_DIRBE,
        albedo=source_parameters.ALBEDO_DIRBE,
    )
)
