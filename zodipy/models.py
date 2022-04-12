from ._model import model_registry
from . import components
from . import source_parameters


model_registry.register_model(
    name="dirbe",
    components=components.DIRBE,
    spectrum=source_parameters.SPECTRUM_DIRBE,
    emissivities=source_parameters.EMISSIVITY_DIRBE,
    albedos=source_parameters.ALBEDO_DIRBE,
    phase_coefficients=source_parameters.PHASE_FUNCTION_DIRBE,
)

model_registry.register_model(
    name="planck13",
    components=components.DIRBE,
    spectrum=source_parameters.SPECTRUM_PLANCK,
    emissivities=source_parameters.EMISSIVITY_PLANCK_13,
)

model_registry.register_model(
    name="planck15",
    components=components.PLANCK,
    spectrum=source_parameters.SPECTRUM_PLANCK,
    emissivities=source_parameters.EMISSIVITY_PLANCK_15,
)

model_registry.register_model(
    name="planck18",
    components=components.PLANCK,
    spectrum=source_parameters.SPECTRUM_PLANCK,
    emissivities=source_parameters.EMISSIVITY_PLANCK_18,
)

model_registry.register_model(
    name="odegard",
    components=components.PLANCK,
    spectrum=source_parameters.SPECTRUM_PLANCK,
    emissivities=source_parameters.EMISSIVITY_ODEGARD,
)
