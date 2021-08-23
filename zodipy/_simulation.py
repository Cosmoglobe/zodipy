from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import healpy as hp
import numpy as np

from zodipy._integration_config import IntegrationConfig
from zodipy._model import InterplanetaryDustModel


@dataclass
class SimulationStrategy(ABC):
    """Base class representing a simulation strategy.    
    
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
    hit_counts
        The number of times each pixel is hit during each observation.
    """

    model: InterplanetaryDustModel
    integration_config: IntegrationConfig
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
        X_observer  = self.observer_locations
        X_earth  = self.earth_locations
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
    """Simulation strategy for time-ordered emission.
    
    This strategy simulates the sky at multiple different times and returns
    the pixel weighted average of all observations.
    """

    def simulate(self, nside: int, freq: float) -> np.ndarray:
        """See base class for a description."""

        components = self.model.components
        emissivities = self.model.emissivities
        X_observer  = self.observer_locations
        X_earth  = self.earth_locations
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
                    integrated_comp_emission * hit_count[pixels]
                )
        
        with warnings.catch_warnings():
            # Unobserved pixels will be divided by 0 in the below return 
            # statement. This is fine since we want to return unobserved 
            # pixels as np.NAN. However, a RuntimeWarning is raised which 
            # we silence in this context manager.
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return emission / hit_counts.sum(axis=0) * 1e20