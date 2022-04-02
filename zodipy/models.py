from zodipy._model import model_registry
from zodipy import components
from zodipy import source_params


model_registry.register_model(
    name="DIRBE",
    components=components.DIRBE,
    spectrum=source_params.SPECTRUM_DIRBE,
    emissivities=source_params.EMISSIVITY_DIRBE,
    albedos=source_params.ALBEDO_DIRBE,
    phase_coefficients=source_params.PHASE_FUNCTION_DIRBE,
)


model_registry.register_model(
    name="Planck13",
    components=components.DIRBE,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_13,
)


model_registry.register_model(
    name="Planck15",
    components=components.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_15,
)


model_registry.register_model(
    name="Planck18",
    components=components.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_18,
)

model_registry.register_model(
    name="Odegard",
    components=components.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_ODEGARD,
)
