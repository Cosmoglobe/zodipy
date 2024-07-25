from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from astropy import units

from zodipy.brightness import (
    BrightnessAtStepCallable,
    kelsall_brightness_at_step,
    rrm_brightness_at_step,
)

if TYPE_CHECKING:
    from zodipy.component import ComponentLabel, ZodiacalComponent


@dataclass(repr=False)
class ZodiacalLightModel(abc.ABC):
    """Base class for interplanetary dust models.

    Args:
        comps: Mapping of component labels to component classes.
        spectrum: Frequency or wavelength over which the model is valid.

    """

    comps: Mapping[ComponentLabel, ZodiacalComponent]
    spectrum: units.Quantity

    @property
    @abc.abstractmethod
    def brightness_at_step_callable(cls) -> BrightnessAtStepCallable:
        """Return the callable that computes the brightness at a step."""

    def to_dict(self) -> dict:
        """Return a dictionary representation of the model."""
        _dict: dict = {}
        for key, value in vars(self).items():
            if key == "comps":
                _dict[key] = {}
                for comp_key, comp_value in value.items():
                    _dict[key][comp_key.value] = {
                        k: v
                        for k, v in vars(comp_value).items()
                        if comp_value.__dataclass_fields__[k].init
                    }
            elif isinstance(value, dict):
                _dict[key] = {k.value: v for k, v in value.items()}
            else:
                _dict[key] = value

        return _dict

    @property
    def ncomps(self) -> int:
        """Return the number of components in the model."""
        return len(self.comps)

    def is_valid_at(self, wavelengths: units.Quantity) -> np.bool_:
        """Check if the model is valid at a given wavelength."""
        try:
            wavelengths = wavelengths.to(self.spectrum.unit, equivalencies=units.spectral())
        except units.UnitConversionError as error:
            msg = "Input 'x' must have units convertible to Hz or m."
            raise units.UnitConversionError(msg) from error
        return np.all((self.spectrum.min() <= wavelengths) & (wavelengths <= self.spectrum.max()))


@dataclass(repr=False)
class Kelsall(ZodiacalLightModel):
    """Kelsall et al. (1998) model."""

    T_0: float
    delta: float
    emissivities: Mapping[ComponentLabel, Sequence[float]]
    albedos: Mapping[ComponentLabel, Sequence[float]] | None = None
    solar_irradiance: Sequence[float] | None = None  # In units of MJy/sr
    C1: Sequence[float] | None = None
    C2: Sequence[float] | None = None
    C3: Sequence[float] | None = None

    @property
    def brightness_at_step_callable(cls) -> BrightnessAtStepCallable:
        """Kellsall brightness at a step fuction."""
        return kelsall_brightness_at_step


@dataclass(repr=False)
class RRM(ZodiacalLightModel):
    """Rowan-Robinson and May (2013) model."""

    T_0: Mapping[ComponentLabel, float]
    delta: Mapping[ComponentLabel, float]
    calibration: Sequence[float]

    @property
    def brightness_at_step_callable(cls) -> BrightnessAtStepCallable:
        """RRM brightness at a step fuction."""
        return rrm_brightness_at_step


@dataclass
class ModelRegistry:
    """Container for registered models."""

    _registry: dict[str, ZodiacalLightModel] = field(init=False, default_factory=dict)

    @property
    def models(self) -> list[str]:
        """Return a list of registered models."""
        return list(self._registry.keys())

    def register_model(
        self,
        name: str,
        model: ZodiacalLightModel,
    ) -> None:
        """Register a model with the registry."""
        if (name := name.lower()) in self._registry:
            msg = f"a model by the name {name!s} is already registered."
            raise ValueError(msg)
        if not isinstance(model, ZodiacalLightModel):  # pragma: no cover
            msg = "model must be an instance of ZodiacalLightModel."
            raise TypeError(msg)
        self._registry[name] = model

    def get_model(self, name: str) -> ZodiacalLightModel:
        """Return a model from the registry."""
        if (name := name.lower()) not in self._registry:
            msg = (
                f"{name!r} is not a registered Interplanetary Dust model. "
                f"Avaliable models are: {', '.join(self._registry)}."
            )
            raise ValueError(msg)

        return self._registry[name]


model_registry = ModelRegistry()
