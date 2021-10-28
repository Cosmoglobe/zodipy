from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Dict, Iterable, List, Optional
import warnings

import healpy as hp
import numpy as np
from scipy.interpolate import RectBivariateSpline

from zodipy.los_configs import LOS_configs
from zodipy.models import models
from zodipy._integration import trapezoidal
from zodipy._model import Model
from zodipy._tabulate import get_tabulated_data, JD_to_yday
from zodipy._exceptions import SimulationStrategyNotFoundError
from zodipy._coordinates import EpochsType, get_target_coordinates


TABLE = "/Users/metinsan/Documents/doktor/zodipy/zodipy/data/zodi_table.h5"


@dataclass
class SimulationStrategy(ABC):
    """Base class representing a simulation strategy.

    Attributes
    ----------
    observer
        Observer in the Solar System, e.g 'L2'.
    model
        Interplanetary dust model.
    line_of_sight_config
        Line of sight configuration per component in the model.
    epochs
        Epochs for which to simulate the Zodiacal Emission.
    hit_counts
        Array containing the number of times each pixel is hit during each
        observation. The shape of this array must be (n_observations, npix).
    """

    observer: str
    model: str
    line_of_sight_config: str
    epochs: EpochsType
    hit_counts: Optional[np.ndarray]

    @abstractmethod
    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """Simulates and returns the Zodiacal emission.

        The emission is computed for a given nside and frequency and
        outputted in units of MJy/sr.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq
            Frequency in GHz for which to evaluate the emission.

        Returns
        -------
        emission
            Simulated Zodiacal emission.
        """


@dataclass
class PixelWeightedMeanStrategy(SimulationStrategy):
    """Currently the only implemented simulation strategy.

    This strategy returns the pixel weighted mean of n observations.
    """

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        model = models.get_model(self.model)
        los_config = LOS_configs.get_config(self.line_of_sight_config)

        components = model.components
        emissivities = model.emissivities
        X_observer = get_target_coordinates(self.observer, self.epochs)
        X_earth = get_target_coordinates("earth", self.epochs)

        npix = hp.nside2npix(nside)
        if self.hit_counts is None:
            hits = np.ones(npix)
            hit_counts = np.asarray([hits for _ in range(len(X_observer))])
        elif hp.get_nside(self.hit_counts) != nside:
            hit_counts = hp.ud_grade(self.hit_counts, nside, power=-2)

        X_unit = np.asarray(hp.pix2vec(nside, np.arange(npix)))
        emission = np.zeros((len(components), npix))

        for obs_pos, earth_pos, hit_count in zip(X_observer, X_earth, hit_counts):
            observed_pixels = np.flatnonzero(hit_count)
            unit_vectors = X_unit[:, observed_pixels]

            for comp_idx, (comp_name, comp_class) in enumerate(components.items()):
                if emissivities is not None:
                    comp_emissivity = emissivities.get_emissivity(comp_name, freq)
                else:
                    comp_emissivity = 1

                line_of_sight = los_config[comp_name]

                integrated_comp_emission = trapezoidal(
                    comp_class.get_emission,
                    freq,
                    obs_pos,
                    earth_pos,
                    unit_vectors,
                    line_of_sight,
                    npix,
                    observed_pixels,
                )

                emission[comp_idx, observed_pixels] += (
                    integrated_comp_emission
                    * comp_emissivity
                    * hit_count[observed_pixels]
                )

        with warnings.catch_warnings():
            # Unobserved pixels will be divided by 0 in the below return
            # statement. This is fine since we want to return unobserved
            # pixels as np.NAN. However, a RuntimeWarning is raised which
            # we silence in this context manager.
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return emission / hit_counts.sum(axis=0) * 1e20


@dataclass
class InterpolateFromTableStrategy(SimulationStrategy):
    """Simulation strategy that interpolates from a tabel."""

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        npix = hp.nside2npix(nside)
        frequencies, days, simulations = get_tabulated_data(nside, freq, self.model)
        n_comps = simulations.shape[2]
        emission = np.zeros((n_comps, npix))
        _, dates = get_target_coordinates(self.observer, self.epochs, return_dates=True)
        dates = [JD_to_yday(date) for date in dates]
        for comp in range(n_comps):
            print(comp)
            for pix in range(npix):
                f = RectBivariateSpline(frequencies, days, simulations[:, :, comp, pix])
                for date in dates:
                    emission[comp][pix] += f(freq, date)

        return emission / len(dates)


IMPLEMENTED_STRATEGIES = {
    "los": PixelWeightedMeanStrategy,
    "interp": InterpolateFromTableStrategy,
}


def get_simulation_strategy(
    observer: str,
    epochs: EpochsType,
    hit_counts: Optional[Iterable[np.ndarray]],
    model: Model,
    line_of_sight_config: Dict[str, np.ndarray],
    strategy: str = "los",
) -> SimulationStrategy:
    """Initializes, validates and returns a simulation strategy."""

    try:
        simulation_strategy = IMPLEMENTED_STRATEGIES[strategy]
    except KeyError:
        raise SimulationStrategyNotFoundError(
            f"simulation stratefy {strategy} is not implemented. Available "
            f"strategies are {list(IMPLEMENTED_STRATEGIES.keys())}"
        )

    if hit_counts is not None:
        hit_counts = np.asarray(hit_counts)
        number_of_hit_counts = 1 if np.ndim(hit_counts) == 1 else len(hit_counts)
        number_of_observations = 1 if np.ndim(hit_counts) == 1 else len(epochs)
        if number_of_hit_counts != number_of_observations:
            raise ValueError(
                f"The number of 'hit_counts' ({number_of_hit_counts}) are "
                "not matching the number of observations "
                f"({number_of_observations})"
            )
        if strategy != "los":
            warnings.warn(
                "simulation strategy is set to line-of-sight integration. "
                "Interpolation is not supporting hit counts."
            )
            simulation_strategy = IMPLEMENTED_STRATEGIES["los"]

    return simulation_strategy(
        observer,
        model,
        line_of_sight_config,
        epochs,
        hit_counts,
    )
