from zodipy._model import model_registry
from zodipy._labels import Label
from zodipy import source_parameters
from zodipy import component_parameters


model_registry.register_model(
    name="DIRBE",
    component_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
        Label.RING,
        Label.FEATURE,
    ],
    component_parameters=component_parameters.K98,
    source_component_parameters={
        "emissivities": source_parameters.EMISSIVITY_DIRBE,
        "albedos": source_parameters.ALBEDO_DIRBE,
    },
    source_parameters={
        "T_0": source_parameters.T_0_K98,
        "delta": source_parameters.delta_K98,
        "phase": source_parameters.PHASE_DIRBE,
    },
    doc=(
        "The Interplanetary Dust Model used in the DIRBE analysis. See "
        "Kelsall et al. (1998) for more information."
    ),
)

model_registry.register_model(
    name="Planck13",
    component_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
        Label.RING,
        Label.FEATURE,
    ],
    component_parameters=component_parameters.K98,
    source_component_parameters={
        "emissivities": source_parameters.EMISSIVITY_PLANCK_13,
    },
    source_parameters={
        "T_0": source_parameters.T_0_K98,
        "delta": source_parameters.delta_K98,
    },
    doc=("The Interplanetary Dust Model used in the Planck 2013 analysis."),
)


model_registry.register_model(
    name="Planck15",
    component_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
    ],
    component_parameters=component_parameters.K98,
    source_component_parameters={
        "emissivities": source_parameters.EMISSIVITY_PLANCK_15,
    },
    source_parameters={
        "T_0": source_parameters.T_0_K98,
        "delta": source_parameters.delta_K98,
    },
    doc=("The Interplanetary Dust Model used in the Planck 2015 analysis."),
)


model_registry.register_model(
    name="Planck18",
    component_labels=[
        Label.CLOUD,
        Label.BAND1,
        Label.BAND2,
        Label.BAND3,
    ],
    component_parameters=component_parameters.K98,
    source_component_parameters={
        "emissivities": source_parameters.EMISSIVITY_PLANCK_18,
    },
    source_parameters={
        "T_0": source_parameters.T_0_K98,
        "delta": source_parameters.delta_K98,
    },
    doc=("The Interplanetary Dust Model used in the Planck 2018 analysis."),
)
