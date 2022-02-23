from zodipy._labels import CompLabel
from zodipy._model import model_registry
from zodipy import comp_params
from zodipy import spectral_params
from zodipy import source_params


model_registry.register_model(
    name="DIRBE",
    comp_labels=[
        CompLabel.CLOUD,
        CompLabel.BAND1,
        CompLabel.BAND2,
        CompLabel.BAND3,
        CompLabel.RING,
        CompLabel.FEATURE,
    ],
    comp_params=comp_params.K98,
    spectral_params={
        "emissivities": spectral_params.EMISSIVITY_DIRBE,
        "albedos": spectral_params.ALBEDO_DIRBE,
        "phase": spectral_params.PHASE_DIRBE,
    },
    source_params={
        "T_0": source_params.K98["T_0"],
        "delta": source_params.K98["delta"],
    },
    doc=(
        "The Interplanetary Dust Model used in the DIRBE analysis. See "
        "Kelsall et al. (1998) for more information."
    ),
)

model_registry.register_model(
    name="Planck13",
    comp_labels=[
        CompLabel.CLOUD,
        CompLabel.BAND1,
        CompLabel.BAND2,
        CompLabel.BAND3,
        CompLabel.RING,
        CompLabel.FEATURE,
    ],
    comp_params=comp_params.K98,
    spectral_params={
        "emissivities": spectral_params.EMISSIVITY_PLANCK_13,
    },
    source_params={
        "T_0": source_params.K98["T_0"],
        "delta": source_params.K98["delta"],
    },
    doc=("The Interplanetary Dust Model used in the Planck 2013 analysis."),
)


model_registry.register_model(
    name="Planck15",
    comp_labels=[
        CompLabel.CLOUD,
        CompLabel.BAND1,
        CompLabel.BAND2,
        CompLabel.BAND3,
    ],
    comp_params=comp_params.K98,
    spectral_params={
        "emissivities": spectral_params.EMISSIVITY_PLANCK_15,
    },
    source_params={
        "T_0": source_params.K98["T_0"],
        "delta": source_params.K98["delta"],
    },
    doc=("The Interplanetary Dust Model used in the Planck 2015 analysis."),
)


model_registry.register_model(
    name="Planck18",
    comp_labels=[
        CompLabel.CLOUD,
        CompLabel.BAND1,
        CompLabel.BAND2,
        CompLabel.BAND3,
    ],
    comp_params=comp_params.K98,
    spectral_params={
        "emissivities": spectral_params.EMISSIVITY_PLANCK_18,
    },
    source_params={
        "T_0": source_params.K98["T_0"],
        "delta": source_params.K98["delta"],
    },
    doc=("The Interplanetary Dust Model used in the Planck 2018 analysis."),
)
