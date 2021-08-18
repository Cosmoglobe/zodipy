from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians
from typing import Union, Iterable, List
import warnings

import healpy as hp
import numpy as np

from zodipy._model import InterplanetaryDustModel
from zodipy._integration import IntegrationConfig


@dataclass
class SimulationStrategy(ABC):
    """Base class that represents a simulation strategy.
    
    Attributes
    ----------
    model
        Interplanetary dust model with initialized componentents and 
        corresponding emissivities.
    integration_config
        Configuration object that determines how a component is integrated
        along a line-of-sight.
    observer_locations
        The locations of the observer.
    earth_locations
        The locations of the Earth corresponding to the observer locations.
    """

    model: InterplanetaryDustModel
    integration_config: IntegrationConfig
    observer_locations: Iterable
    earth_locations: Iterable

    @abstractmethod
    def simulate(self, nside: int, freq: float, solar_cut: float) -> np.ndarray:
        """Returns the simulated the Zodiacal emission.
        
        The emission is computed given a nside and frequency and outputted
        in units of MJy/sr.

        Parameters
        ----------
        nside
            HEALPIX map resolution parameter.
        freq
            Frequency [GHz] at which to evaluate the IPD model.
        solar_cut
            Angle [deg] between observer and the Sun for which all pixels 
            are masked (for each observation).
            
        Returns
        -------
        emission
            Simulated Zodiacal emission.
        """

    @staticmethod
    def get_observed_pixels(
        X_observer: np.ndarray, 
        X_unit: np.ndarray, 
        solar_cut: Union[float, None]
    ) -> List[np.ndarray]:
        """Returns a list of observed pixels per observation.
        
        All pixels that have an angular distance of larger than some angle
        solar_cut between the observer and the sun are masked.
        """

        if solar_cut is None:
            return Ellipsis

        angular_distance = (
            hp.rotator.angdist(obs , X_unit) for obs in X_observer
        )

        observed_pixels = [
            ang_dist < radians(solar_cut) for ang_dist in angular_distance
        ]

        return observed_pixels


class InstantaneousStrategy(SimulationStrategy):
    """Simulation strategy for instantaneous emission."""

    def __init__(
        self, model, integration_config, observer_locations, earth_locations
    ) -> None:
        """Initializing the strategy."""

        super().__init__(
            model, integration_config, observer_locations, earth_locations
        )

    def simulate(self, nside: int, freq: float, solar_cut: float) -> np.ndarray:
        """See base class for a description."""

        npix = hp.nside2npix(nside)
        pixels = np.arange(npix)

        X_observer  = self.observer_locations
        X_earth  = self.earth_locations
        X_unit = np.asarray(hp.pix2vec(nside, pixels))

        n_observations = len(X_observer)

        pixels = self.get_observed_pixels(X_observer, X_unit, solar_cut)

        components = self.model.components
        emissivities = self.model.emissivities

        # Unobserved pixels are represented as NANs
        emission = np.zeros((n_observations, len(components), npix)) + np.NAN

        for observation_idx, (observer_pos, earth_pos) in enumerate(
            zip(X_observer, X_earth)
        ):
            if solar_cut is None:
                observed_pixels = pixels
            else:
                observed_pixels = pixels[observation_idx]
            unit_vectors = X_unit[:, observed_pixels]

            for comp_idx, (comp_name, comp) in enumerate(components.items()):
                integration_config = self.integration_config[comp_name]
                R = integration_config.R

                comp_emission = comp.get_emission(
                    freq, observer_pos, earth_pos, unit_vectors, R
                )
                integrated_comp_emission = integration_config.integrator(
                    comp_emission, R, dx=integration_config.dR, axis=0
                )

                comp_emissivity = emissivities.get_emissivity(comp_name, freq)
                integrated_comp_emission *= comp_emissivity

                emission[observation_idx, comp_idx, observed_pixels] = (
                    integrated_comp_emission
                )

        with warnings.catch_warnings():
            # np.nanmean throws a RuntimeWarning if all pixels along an 
            # axis is NANs. This may occur when parts of the sky is left
            # unobserved over all observations. Here we manually disable 
            # the warning thay is thrown in the aforementioned scenario.
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return np.nanmean(emission, axis=0) * 1e20