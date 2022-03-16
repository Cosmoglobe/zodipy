from zodipy._model import model_registry
from zodipy import comp_params
from zodipy import source_params


model_registry.register_model(
    name="DIRBE",
    comp_params=comp_params.K98,
    emissivities=source_params.EMISSIVITY_DIRBE,
    emissivity_spectrum=source_params.SPECTRUM_DIRBE,
    T_0=source_params.T_0_K98,
    delta=source_params.delta_K98,
    albedos=source_params.ALBEDO_DIRBE,
    albedo_spectrum=source_params.SPECTRUM_DIRBE,
    phase_coeffs=source_params.PHASE_FUNC_DIRBE,
    phase_coeffs_spectrum=source_params.SPECTRUM_DIRBE,
    meta={
        "info": "The Interplanetary Dust Model used in the DIRBE analysis. See Kelsall et al. (1998) for more information."
    },
)


model_registry.register_model(
    name="Planck13",
    comp_params=comp_params.K98,
    emissivities=source_params.EMISSIVITY_PLANCK_13,
    emissivity_spectrum=source_params.SPECTRUM_PLANCK,
    T_0=source_params.T_0_K98,
    delta=source_params.delta_K98,
    meta={"info": "The Interplanetary Dust Model used in the Planck 2013 analysis."},
)


model_registry.register_model(
    name="Planck15",
    comp_params=comp_params.PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_15,
    emissivity_spectrum=source_params.SPECTRUM_PLANCK,
    T_0=source_params.T_0_K98,
    delta=source_params.delta_K98,
    meta={"info": "The Interplanetary Dust Model used in the Planck 2015 analysis."},
)


model_registry.register_model(
    name="Planck18",
    comp_params=comp_params.PLANCK,
    emissivities=source_params.EMISSIVITY_PLANCK_18,
    emissivity_spectrum=source_params.SPECTRUM_PLANCK,
    T_0=source_params.T_0_K98,
    delta=source_params.delta_K98,
    meta={"info": "The Interplanetary Dust Model used in the Planck 2018 analysis."},
)
