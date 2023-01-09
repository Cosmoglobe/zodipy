from zodipy import comps, source_params
from zodipy._ipd_model import RRM, Kelsall, model_registry

model_registry.register_model(
    name="dirbe",
    model=Kelsall(
        comps=comps.DIRBE,
        spectrum=source_params.SPECTRUM_DIRBE,
        emissivities=source_params.EMISSIVITY_DIRBE,
        albedos=source_params.ALBEDO_DIRBE,
        solar_irradiance=source_params.SOLAR_IRRADIANCE_DIRBE,
        phase_coefficients=source_params.PHASE_FUNCTION_DIRBE,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="planck13",
    model=Kelsall(
        comps=comps.DIRBE,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_PLANCK_13,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="planck15",
    model=Kelsall(
        comps=comps.PLANCK,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_PLANCK_15,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="planck18",
    model=Kelsall(
        comps=comps.PLANCK,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_PLANCK_18,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="odegard",
    model=Kelsall(
        comps=comps.PLANCK,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_ODEGARD,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="rrm-experimental",
    model=RRM(
        comps=comps.RRM,
        spectrum=source_params.SPECTRUM_IRAS,
        calibration=source_params.CALIBRATION_RRM,
        T_0=source_params.T_0_RRM,
        delta=source_params.DELTA_RMM,
    ),
)
