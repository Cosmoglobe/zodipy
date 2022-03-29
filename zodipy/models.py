from zodipy._model import model_registry
from zodipy import comp_params
from zodipy import source_params


model_registry.register_model(
    name="DIRBE",
    component_parameters=comp_params.K98,
    spectrum=source_params.SPECTRUM_DIRBE,
    emissivities=source_params.EMISSIVITY_DIRBE,
    albedos=source_params.ALBEDO_DIRBE,
    phase_coefficients=source_params.PHASE_FUNC_DIRBE,
)


model_registry.register_model(
    name="Planck13",
    component_parameters=comp_params.K98,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_13,
)


model_registry.register_model(
    name="Planck15",
    component_parameters=comp_params.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_15,
)


model_registry.register_model(
    name="Planck18",
    component_parameters=comp_params.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_18,
)

model_registry.register_model(
    name="Odegard",
    component_parameters=comp_params.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_ODEGARD,
)

model_registry.register_model(
    name="LiteBIRD",
    component_parameters=comp_params.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_ODEGARD,
)
