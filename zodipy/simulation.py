from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

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
        The location(s) of the observer.
    earth_location
        The location(s) of the Earth.
    """

    model: InterplanetaryDustModel
    integration_config: IntegrationConfig
    observer_locations: Iterable
    earth_locations: Iterable
    hit_maps: np.ndarray


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


@dataclass
class InstantaneousStrategy(SimulationStrategy):
    """Simulation strategy for instantaneous emission."""

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        components = self.model.components
        emissivities = self.model.emissivities

        X_observer  = self.observer_locations
        X_earth  = self.earth_locations

        if (hit_map := self.hit_maps) is not None:
            pixels = np.flatnonzero(hit_map)
        else:
            pixels = Ellipsis

        npix = hp.nside2npix(nside)
        X_unit = np.asarray(hp.pix2vec(nside, np.arange(npix)))[pixels]

        emission = np.zeros((len(components), npix)) 

        for comp_idx, (comp_name, comp) in enumerate(components.items()):
            integration_config = self.integration_config[comp_name]
            R = integration_config.R

            comp_emission = comp.get_emission(
                freq, X_observer, X_earth, X_unit, R
            )
            integrated_comp_emission = integration_config.integrator(
                comp_emission, R, dx=integration_config.dR, axis=0
            )

            comp_emissivity = emissivities.get_emissivity(comp_name, freq)
            integrated_comp_emission *= comp_emissivity

            emission[comp_idx, pixels] = integrated_comp_emission

        return emission * 1e20


@dataclass
class TimeOrderedStrategy(SimulationStrategy):
    """Simulation strategy for time-ordered emission."""

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        npix = hp.nside2npix(nside)

        hit_maps = self.hit_maps
        if hp.get_nside(hit_maps) != nside:
            hit_maps = hp.ud_grade(self.hit_maps, nside, power=-2)

        X_observer  = self.observer_locations
        X_earth  = self.earth_locations
        X_unit = np.asarray(hp.pix2vec(nside, np.arange(npix)))

        components = self.model.components
        emissivities = self.model.emissivities

        emission = np.zeros((len(components), npix))

        for observer_pos, earth_pos, hit_map in zip(X_observer, X_earth, hit_maps):
            pixels = np.flatnonzero(hit_map)
            unit_vectors = X_unit[:, pixels]

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
                emission[comp_idx, pixels] += (
                    integrated_comp_emission * hit_map[pixels]
                )

        return emission / hit_maps.sum(axis=0) * 1e20