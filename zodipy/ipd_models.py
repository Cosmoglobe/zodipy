from . import ipd_comps, source_params
from ._ipd_model import model_registry

model_registry.register_model(
    name="dirbe",
    comps=ipd_comps.DIRBE,
    spectrum=source_params.SPECTRUM_DIRBE,
    emissivities=source_params.EMISSIVITY_DIRBE,
    albedos=source_params.ALBEDO_DIRBE,
    solar_irradiance=source_params.SOLAR_IRRADIANCE_DIRBE,
    phase_coefficients=source_params.PHASE_FUNCTION_DIRBE,
)

model_registry.register_model(
    name="planck13",
    comps=ipd_comps.DIRBE,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_13,
)

model_registry.register_model(
    name="planck15",
    comps=ipd_comps.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_15,
)

model_registry.register_model(
    name="planck18",
    comps=ipd_comps.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_18,
)

model_registry.register_model(
    name="odegard",
    comps=ipd_comps.PLANCK,
    spectrum=source_params.SPECTRUM_PLANCK,
    emissivities=source_params.EMISSIVITY_ODEGARD,
)
