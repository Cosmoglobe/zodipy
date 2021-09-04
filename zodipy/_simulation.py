from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
import warnings

import healpy as hp
import numpy as np

from zodipy._integration import trapezoidal
from zodipy._model import InterplanetaryDustModel


@dataclass
class SimulationStrategy(ABC):
    """Base class representing a simulation strategy.

    Attributes
    ----------
    model
        Interplanetary dust model with initialized componentents and
        corresponding emissivities.
    line_of_sight_config
        Dictionary mapping a line_of_sight_config to each component.
    observer_locations
        The location(s) of the observer.
    earth_location
        The location(s) of the Earth.
    hit_counts
        The number of times each pixel is hit during each observation.
    """

    model: InterplanetaryDustModel
    line_of_sight_config: Dict[str, np.ndarray]
    observer_locations: np.ndarray
    earth_locations: np.ndarray
    hit_counts: np.ndarray

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
class InstantaneousStrategy(SimulationStrategy):
    """Simulation strategy for instantaneous emission.

    This strategy simulates the sky as seen at an instant in time.
    """

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        components = self.model.components
        emissivities = self.model.emissivities
        X_observer = self.observer_locations
        X_earth = self.earth_locations
        hit_counts = self.hit_counts

        npix = hp.nside2npix(nside)
        if hit_counts is None:
            hit_counts = np.ones(npix)
        elif hp.get_nside(hit_counts) != nside:
            hit_counts = hp.ud_grade(hit_counts, nside, power=-2)

        pixels = np.flatnonzero(hit_counts)
        X_unit = np.asarray(hp.pix2vec(nside, np.arange(npix)))[:, pixels]
        emission = np.zeros((len(components), npix)) + np.NAN

        for comp_idx, (comp_name, comp) in enumerate(components.items()):
            comp_emissivity = emissivities.get_emissivity(comp_name, freq)
            line_of_sight = self.line_of_sight_config[comp_name]
            integrated_comp_emission = trapezoidal(
                comp.get_emission,
                freq,
                X_observer,
                X_earth,
                X_unit,
                line_of_sight,
                npix,
                pixels,
            )

            emission[comp_idx, pixels] = comp_emissivity * integrated_comp_emission

        return emission * 1e20


@dataclass
class TimeOrderedStrategy(SimulationStrategy):
    """Simulation strategy for time-ordered emission.

    This strategy simulates the sky at multiple different times and returns
    the pixel weighted average of all observations.
    """

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        components = self.model.components
        emissivities = self.model.emissivities
        X_observer = self.observer_locations
        X_earth = self.earth_locations
        hit_counts = self.hit_counts

        npix = hp.nside2npix(nside)
        if hit_counts is None:
            hits = np.ones(npix)
            hit_counts = np.asarray([hits for _ in range(len(X_observer))])
        elif hp.get_nside(hit_counts) != nside:
            hit_counts = hp.ud_grade(hit_counts, nside, power=-2)

        X_unit = np.asarray(hp.pix2vec(nside, np.arange(npix)))
        emission = np.zeros((len(components), npix))

        for observer_pos, earth_pos, hit_count in zip(X_observer, X_earth, hit_counts):
            pixels = np.flatnonzero(hit_count)
            unit_vectors = X_unit[:, pixels]

            for comp_idx, (comp_name, comp) in enumerate(components.items()):
                comp_emissivity = emissivities.get_emissivity(comp_name, freq)
                line_of_sight = self.line_of_sight_config[comp_name]

                integrated_comp_emission = trapezoidal(
                    comp.get_emission,
                    freq,
                    observer_pos,
                    earth_pos,
                    unit_vectors,
                    line_of_sight,
                    npix,
                    pixels,
                )

                emission[comp_idx, pixels] += (
                    integrated_comp_emission * comp_emissivity * hit_count[pixels]
                )

        with warnings.catch_warnings():
            # Unobserved pixels will be divided by 0 in the below return
            # statement. This is fine since we want to return unobserved
            # pixels as np.NAN. However, a RuntimeWarning is raised which
            # we silence in this context manager.
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return emission / hit_counts.sum(axis=0) * 1e20


def get_simulation_strategy(
    model: InterplanetaryDustModel,
    line_of_sight_config: Dict[str, np.ndarray],
    observer_locations: np.ndarray,
    earth_locations: np.ndarray,
    hit_counts: np.ndarray,
) -> SimulationStrategy:
    """Initializes and returns a simulation strategy given initial conditions."""

    number_of_observations = len(observer_locations)
    if hit_counts is not None:
        hit_counts = np.asarray(hit_counts)
        number_of_hit_counts = 1 if np.ndim(hit_counts) == 1 else len(hit_counts)
        if number_of_hit_counts != number_of_observations:
            raise ValueError(
                f"The number of 'hit_counts' ({number_of_hit_counts}) are "
                "not matching the number of observations "
                f"({number_of_observations})"
            )

    if number_of_observations == 1:
        simulation_strategy = InstantaneousStrategy
        observer_locations = observer_locations.squeeze()
        earth_locations = earth_locations.squeeze()
    else:
        simulation_strategy = TimeOrderedStrategy

    return simulation_strategy(
        model, line_of_sight_config, observer_locations, earth_locations, hit_counts
    )
