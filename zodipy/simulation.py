from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import radians
from typing import Iterable, List
import warnings

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
    def simulate(self, nside: int, freq: float, mask: float) -> np.ndarray:
        """Simulates the Zodiacal emission, given a nside and frequency.
        
        The emission is returned in units of MJy/sr.

        Parameters
        ----------
        nside : int
            HEALPIX map resolution parameter.
        freq : float
            Frequency [GHz] at which to evaluate the IPD model.
        mask : float, optional
            Angle [deg] between observer and the Sun for which all pixels 
            are masked at each observation. A mask of 90 degrees can be 
            selected to simulate an observer that never looks inwards the Sun.
            
        Returns
        -------
        emission : `np.ndarray`
            Simulated Zodiacal emission.
        """

    @staticmethod
    def get_observed_pixels(
        X_observer: np.ndarray, X_unit: np.ndarray, ang: float
    ) -> List[np.ndarray]:
        """Returns a list of the observed pixels per observation.
        
        All pixels that have an angular distance of larger than some angle
        between the observer and the sun are masked.
        
        Parameters
        ----------
        X_observer: `np.ndarray`
            Array containing coordinates of the observer.
        X_unit: `np.ndarray`
            Array containing heliocentric unit vectors.
        ang: float
            Angle for which all pixels are masked [deg].
        
        Returns
        -------
        list
            List containing arrays of unmasked pixels per observation.
        """

        angular_distance = [
            hp.rotator.angdist(obs , X_unit) for obs in X_observer
        ]

        return [ang_dist < radians(ang) for ang_dist in angular_distance]


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

    def simulate(self, nside: int, freq: float, mask: float) -> np.ndarray:
        """See base class for a description."""

        npix = hp.nside2npix(nside)
        pixels = np.arange(npix)

        X_observer  = self.observer_locations
        X_earth  = self.earth_locations
        X_unit = np.asarray(hp.pix2vec(nside, pixels))

        n_observations = len(X_observer)

        if mask is None:
            pixels = [slice(0, npix) for _ in range(n_observations)]
        else:
            pixels = self.get_observed_pixels(X_observer, X_unit, ang=mask)

        components = self.model.components
        emissivities = self.model.emissivities

        # The emission is initialized as NANs representing unobserved pixels
        emission = np.zeros((n_observations, len(components), npix)) + np.NAN

        for observation_idx, (observer_pos, earth_pos) in enumerate(zip(X_observer, X_earth)):
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

                emission[observation_idx, comp_idx, observed_pixels] = integrated_comp_emission

        with warnings.catch_warnings():
            # np.nanmean throws a RuntimeWarning if all pixels along an 
            # axis is NANs. This may occur when parts of the sky is left
            # unobserver over all observations. Here we manually disable 
            # the warning in the aforementioned scenario.
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            return np.nanmean(emission, axis=0) * 1e20