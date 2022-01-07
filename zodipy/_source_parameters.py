from dataclasses import dataclass
from typing import Dict, Sequence

import astropy.units as u
import numpy as np

from zodipy._labels import Label


@dataclass
class SourceParameter:
    values: Dict[Label, Sequence[float]]
    spectrum: u.Quantity
    doc: str = ""

    def __post_init__(self) -> None:
        self.spectrum = self.spectrum.to("GHz", equivalencies=u.spectral())

    def get_interpolated_values(self, freq: u.Quantity) -> Dict[Label, Sequence[float]]:
        if not self.spectrum[0] <= freq <= self.spectrum[-1]:
            raise ValueError(f"frequency {freq} is out of range with spectrum.")
        return {
            label: np.interp(freq, self.spectrum, value)
            for label, value in self.values.items()
        }
