from typing import Dict, Optional, Sequence, Union

import numpy as np
import astropy.units as u

from zodipy._astroquery import query_target_positions
from zodipy._emissivities import get_emissivities
from zodipy._integration_config import integration_config_registry
from zodipy._simulation import instantaneous_emission, time_ordered_emission
from zodipy.models import model_registry


class InterplanetaryDustModel:
    """The Zodipy simulation interface.

    The Interplanetary Dust Model used by Zodipy is the Kelsall et al. (1998)
    Interplanetary Dust Model, which includes five (six) Zodiacal components:
        - The Diffuse Cloud (cloud)
        - Three Asteroidal Bands (band1, band2, band3)
        - The Circumsolar Ring (ring) + The Earth-trailing Feature (feature)

    Optionally, it is possible to scale the K98 emission with component
    specific emissivities, as done by the Planck Collaboration. This is
    achieved by specifiying one of the following implemented models:
        - Planck13 (all five Kelsall components + emissivity fits for each)
        - Planck15 (cloud + bands + new emissivity fits)
        - Planck18 (cloud + bands + the latest emissivity fits)

    NOTE: No extrapolation is done when using the fitted emissivities, and as
    such, these models can only be evaluated within the frequency range covered
    by the Planck HFI Bands for which the emissivities were fitted.
    """

    def __init__(self, model: str = "K98") -> None:
        """Initializes the interface given an Interplanetary Dust Model.

        Parameters
        ----------
        model
            The name of the model to initialize.
        """

        self.model = model_registry.get_model(model)
        integration_config = integration_config_registry.get_config("default")
        self.line_of_sights = [
            line_of_sight for line_of_sight in integration_config.values()
        ]

    @u.quantity_input(freq_or_wavelength=("Hz", "m", "micron"))
    def get_instantaneous_emission(
        self,
        freq_or_wavelength: u.Quantity,
        nside: int,
        *,
        observer: str = "L2",
        epochs: Optional[Union[float, Sequence[float], Dict[str, str]]] = None,
        return_comps: bool = False,
        coord_out: str = "E",
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the instantaneous Zodiacal Emission [MJy/sr].

        By instantaneous emission we mean the emission observed at an instant
        in time. If multiple epochs are given, the returned emission will be
        the mean of all simulated instantaneous observations.

        The observer location, given by the parameter `observer` (and
        optionally the location of the Earth if either of the Feature or the
        Ring components are included in the selected Interplanetary Dust Model)
        are queried from the Horizons JPL ephemerides, given some epoch defined
        by the `epochs` parameter.

        NOTE: This function returns the fullsky emission from at a single time.
        This means that we in the simulation evaluate line-of-sights that
        sometimes points directly towards the inner Solar System and through
        the Sun, where the dust density increases exponentially. Such
        line-of-sights are unlikely to be observed by an actual observer, and
        as such, the simulated emission will appear very bright in these
        regions.

        Parameters
        ----------
        freq_or_wavelength
            Frequency or wavelength at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter of the returned emission map.
        observer
            The name of the observer for which we quiery its location given
            the `epochs` parameter. Defaults to 'L2'.
        epochs
            The observeration times given as a single epoch, or a list of epochs
            in JD or MJD format, or a dictionary defining a range of times and
            dates; the range dictionary has to be of the form
            {'start':'YYYY-MM-DD [HH:MM:SS]', 'stop':'YYYY-MM-DD [HH:MM:SS]',
            'step':'n[y|d|h|m|s]'}. If no epochs are provided, the current time
            is used in UTC.
        return_comps
            If True, the emission is returned component-wise in a dictionary.
            Defaults to False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocentered
            ecliptic coordinates).

        Returns
        -------
        emission
            Simulated (mean) instantaneous Zodiacal emission [MJy/sr].
        """

        observer_positions = query_target_positions(observer, epochs)
        if self.model.includes_earth_neighboring_components:
            earth_positions = query_target_positions("earth", epochs)
        else:
            earth_positions = observer_positions.copy()

        emissivities = get_emissivities(
            ν_or_λ=freq_or_wavelength,
            emissivity=self.model.emissivities,
            components=list(self.model.components.keys()),
        )

        freq = freq_or_wavelength.to("Hz", equivalencies=u.spectral()).value
        emission = instantaneous_emission(
            nside=nside,
            freq=freq,
            components=list(self.model.components.values()),
            emissivities=emissivities,
            observer_positions=observer_positions,
            earth_positions=earth_positions,
            line_of_sights=self.line_of_sights,
            coord_out=coord_out,
        )

        if return_comps:
            return {
                comp.value: emission[idx]
                for idx, comp in enumerate(self.model.components)
            }

        return emission.sum(axis=0)

    @u.quantity_input(freq_or_wavelength=("Hz", "m", "micron"))
    def get_time_ordered_emission(
        self,
        freq_or_wavelength: u.Quantity,
        nside: int,
        *,
        pixels: np.ndarray,
        observer_position: np.ndarray,
        earth_position: Optional[np.ndarray] = None,
        bin: bool = False,
        return_comps: bool = False,
        coord_out: str = "E",
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the Zodiacal emission [MJy/sr] in a timestream.

        Given a sequence of time-ordered pixels, the Zodiacal emission is
        evaluated from a constant location in space given by the
        `observer_position`. The `earth_position` is required for Interplanetary
        Dust models that include the Earth-trailing Feature and Circum-solar
        Ring components.

        Parameters
        ----------
        freq_or_wavelength
            Frequency or wavelength at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter of the returned emission map.
        pixels
            Sequence of time-ordered pixels.
        observer_position
            The heliocentric ecliptic cartesian position of the observer at 
            the time of observing the tods.
        earth_position
            The heliocentric ecliptic cartesian position of the Earth at the 
            time of observing the tods. If None, the observer is assumed to be 
            the Earth. Defaults to None.
        bin
            If True, the time-ordered sequence of emission per pixel is binned
            into a HEALPIX map. Defaults to False.
        return_comps
            If True, the emission is returned component-wise in a dictionary.
            Defaults to False.
        coord_out
            Coordinate frame of the output map. Defaults to 'E' (heliocentric 
            ecliptic coordinates).

        Returns
        -------
        emission
            Simulated timestream of Zodiacal emission [MJy/sr] (optionally
            binned into a HEALPIX map).
        """

        if earth_position is None:
            earth_position = observer_position

        emissivities = get_emissivities(
            ν_or_λ=freq_or_wavelength,
            emissivity=self.model.emissivities,
            components=list(self.model.components.keys()),
        )

        freq = freq_or_wavelength.to("Hz", equivalencies=u.spectral()).value
        emission = time_ordered_emission(
            nside=nside,
            freq=freq,
            components=list(self.model.components.values()),
            emissivities=emissivities,
            line_of_sights=self.line_of_sights,
            observer_position=observer_position,
            earth_position=earth_position,
            pixel_chunk=pixels,
            bin=bin,
            coord_out=coord_out,
        )

        if return_comps:
            return {
                comp.value: emission[idx]
                for idx, comp in enumerate(self.model.components)
            }

        return emission.sum(axis=0)

    def __str__(self) -> str:
        """String representation of the InterplanetaryDustModel."""

        reprs = []
        for label, component in self.model.components.items():
            component_repr = f"{component.__class__.__name__}" + "\n"
            reprs.append(f"({label.value}): {component_repr}")

        main_repr = "InterplanetaryDustModel("
        main_repr += f"\n  name: {self.model.name}"
        main_repr += "\n  components( "
        main_repr += "\n    " + "    ".join(reprs)
        main_repr += "  )"
        main_repr += "\n)"

        return main_repr