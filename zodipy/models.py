from zodipy._model import model_registry
from zodipy import components
from zodipy import source_params


model_registry.register_model(
    name="dirbe",
    components=components.DIRBE,
    spectrum=source_params.SPECTRUM_DIRBE,
    emissivities=source_params.EMISSIVITY_DIRBE,
    albedos=source_params.ALBEDO_DIRBE,
    phase_coefficients=source_params.PHASE_FUNCTION_DIRBE,
)


model_registry.register_model(
    name="planck13",
    components=components.DIRBE,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_13,
)


model_registry.register_model(
    name="planck15",
    components=components.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_15,
)


model_registry.register_model(
    name="planck18",
    components=components.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_18,
)

model_registry.register_model(
    name="odegard",
    components=components.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_ODEGARD,
)
