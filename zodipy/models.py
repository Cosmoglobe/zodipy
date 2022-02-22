from zodipy._labels import Label
from zodipy._model import model_registry
from zodipy import component_parameters
from zodipy import spectral_parameters
from zodipy import source_parameters


model_registry.register_model(
    name="DIRBE",
    comp_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
        Label.RING,
        Label.FEATURE,
    ],
    comp_params=component_parameters.K98,
    spectral_params={
        "emissivities": spectral_parameters.EMISSIVITY_DIRBE,
        "albedos": spectral_parameters.ALBEDO_DIRBE,
        "phase": spectral_parameters.PHASE_DIRBE,
    },
    source_params={
        "T_0": source_parameters.K98["T_0"],
        "delta": source_parameters.K98["delta"],
    },
    doc=(
        "The Interplanetary Dust Model used in the DIRBE analysis. See "
        "Kelsall et al. (1998) for more information."
    ),
)

model_registry.register_model(
    name="Planck13",
    comp_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
        Label.RING,
        Label.FEATURE,
    ],
    comp_params=component_parameters.K98,
    spectral_params={
        "emissivities": spectral_parameters.EMISSIVITY_PLANCK_13,
    },
    source_params={
        "T_0": source_parameters.K98["T_0"],
        "delta": source_parameters.K98["delta"],
    },
    doc=("The Interplanetary Dust Model used in the Planck 2013 analysis."),
)


model_registry.register_model(
    name="Planck15",
    comp_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
    ],
    comp_params=component_parameters.K98,
    spectral_params={
        "emissivities": spectral_parameters.EMISSIVITY_PLANCK_15,
    },
    source_params={
        "T_0": source_parameters.K98["T_0"],
        "delta": source_parameters.K98["delta"],
    },
    doc=("The Interplanetary Dust Model used in the Planck 2015 analysis."),
)


model_registry.register_model(
    name="Planck18",
    comp_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
    ],
    comp_params=component_parameters.K98,
    spectral_params={
        "emissivities": spectral_parameters.EMISSIVITY_PLANCK_18,
    },
    source_params={
        "T_0": source_parameters.K98["T_0"],
        "delta": source_parameters.K98["delta"],
    },
    doc=("The Interplanetary Dust Model used in the Planck 2018 analysis."),
)
