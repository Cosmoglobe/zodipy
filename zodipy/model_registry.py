from zodipy import component_params, source_params
from zodipy.zodiacal_light_model import RRM, Kelsall, model_registry

model_registry.register_model(
    name="dirbe",
    model=Kelsall(
        comps=component_params.DIRBE,
        spectrum=source_params.SPECTRUM_DIRBE,
        emissivities=source_params.EMISSIVITY_DIRBE,
        albedos=source_params.ALBEDO_DIRBE,
        solar_irradiance=source_params.SOLAR_IRRADIANCE_DIRBE,
        C1=source_params.C1_DIRBE,
        C2=source_params.C2_DIRBE,
        C3=source_params.C3_DIRBE,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="planck13",
    model=Kelsall(
        comps=component_params.DIRBE,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_PLANCK_13,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="planck15",
    model=Kelsall(
        comps=component_params.PLANCK,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_PLANCK_15,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="planck18",
    model=Kelsall(
        comps=component_params.PLANCK,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_PLANCK_18,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="odegard",
    model=Kelsall(
        comps=component_params.PLANCK,
        spectrum=source_params.SPECTRUM_PLANCK,
        emissivities=source_params.EMISSIVITY_ODEGARD,
        T_0=source_params.T_0_DIRBE,
        delta=source_params.DELTA_DIRBE,
    ),
)

model_registry.register_model(
    name="rrm-experimental",
    model=RRM(
        comps=component_params.RRM,
        spectrum=source_params.SPECTRUM_IRAS,
        calibration=source_params.CALIBRATION_RRM,
        T_0=source_params.T_0_RRM,
        delta=source_params.DELTA_RMM,
    ),
)
