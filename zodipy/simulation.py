from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import healpy as hp
import numpy as np

from zodipy._model import InterplanetaryDustModel
from zodipy._integration import IntegrationConfig


@dataclass
class SimulationStrategy(ABC):
    """Class that represents the simulation part of Zodipy.
    
    The simulation strategy is responsible for simulating the Zodiacal
    emission given an initial setup of the problem.

    Parameters
    ----------
    model : `zodipy._model.InterplanetaryDustModel`
        IPD model with initialized componentents and corresponding 
        emissivities.
    integration_config: `zodipy._integ.IntegrationConfig`
        Configuration object that determines how a component is integrated
        along a line of sight.
    observer_locations : Iterable
        Iterable containing the various locations of an observer. One 
        instantaneous simulation is produced per location.
    earth_locations : Iterable
        Iterable containing the various locations of the Earth 
        corresponding to the observer locations.
    """

    model: InterplanetaryDustModel
    integration_config: IntegrationConfig
    observer_locations: Iterable
    earth_locations: Iterable

    @abstractmethod
    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """Simulates the Zodiacal emission, given a nside and frequency.
        
        The emission is returned in units of MJy/sr.

        Parameters
        ----------
        nside : int
            HEALPIX map resolution parameter.
        freq : float
            Frequency [GHz] at which to evaluate the IPD model.

        Returns
        -------
        emission : `np.ndarray`
            Simulated Zodiacal emission.
        """


class InstantaneousStrategy(SimulationStrategy):
    """Simulation strategy that computes the instantaneous emission.
    
    By instantaneous emission, we mean the emission that is seen at one
    instant in time. The emission is averaged over all observations.
    """

    def __init__(
        self, model, integration_config, observer_locations, earth_locations
    ) -> None:
        super().__init__(
            model, integration_config, observer_locations, earth_locations
        )

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class."""

        X_observer  = self.observer_locations
        X_earth  = self.earth_locations
        X_unit = np.asarray(
            hp.pix2vec(nside, np.arange(npix := hp.nside2npix(nside)))
        )

        emission = np.zeros(
            shape=(
                n_observations := len(X_observer), 
                len(components := self.model.components), 
                npix
            )
        )
        for observation in range(n_observations):
            for comp_idx, (comp_name, comp) in enumerate(components.items()):
                integration_config = self.integration_config[comp_name]
                comp_emission = comp.get_emission(
                    freq, 
                    X_observer[observation], 
                    X_earth[observation], 
                    X_unit, 
                    integration_config.R
                )
                integrated_comp_emission = integration_config.integrator(
                    comp_emission, 
                    integration_config.R, 
                    dx=integration_config.dR, 
                    axis=0
                )

                comp_emissivity = self.model.emissivities.get_emissivity(
                    comp_name, freq
                )
                integrated_comp_emission *= comp_emissivity

                emission[observation, comp_idx] = integrated_comp_emission
        
        return emission.mean(axis=0) * 1e20