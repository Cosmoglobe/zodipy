from typing import Dict, Optional, Sequence, Union

import numpy as np
import astropy.units as u

from zodipy._astroquery import query_target_positions
from zodipy._integration_config import integration_config_registry
from zodipy._labels import LABEL_TO_CLASS
import zodipy._simulation as simulation
from zodipy.models import model_registry


class Zodipy:
    """The Zodipy simulation interface.

    The geometry of the Zodiacal components used in Zodipy is guven by the 
    Kelsall et al. (1998) Interplanetary Dust Model, which includes five (six) 
    Zodiacal components:
        - The Diffuse Cloud (cloud)
        - Three Asteroidal Bands (band1, band2, band3)
        - The Circumsolar Ring (ring) + The Earth-trailing Feature (feature)

    The spectral parameters used when evaluating the line-of-sight parameters
    can be selected by specifying a model, e.g "Planck13", which uses the 
    source parameters (emissivities) fitted the Planck collaboration in their 
    2013 analysis. 

    NOTE: Currently Zodipy only supports the frequency range covered by the 
    spectral parameters in the specified model.
    """

    def __init__(self, model: str = "DIRBE") -> None:
        """Initializes the interface given a model (fitted source parameters).

        Parameters
        ----------
        model
            The name of the model to initialize. Defaults to `DIRBE` which 
            uses the source parameters fit in K98.
        """

        self.model = model_registry.get_model(model)
        self.line_of_sights = integration_config_registry.get_config("default")

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

        The location of the observer, and optionally, the Earth, are queried 
        from the Horizons JPL ephemerides using `astroquery`, given an epoch 
        defined by the `epochs` parameter.

        NOTE: This function returns the fullsky emission at a single time. This
        means that line-of-sights that sometimes points directly towards the 
        inner Solar System and through the Sun, where the dust density 
        increases exponentially are evaluated. Such line-of-sights are unlikely 
        to be observed by an actual observer, and as such, the simulated 
        emission will appear very bright in these npregions.

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

        freq = freq_or_wavelength.to("GHz", equivalencies=u.spectral())

        observer_positions = query_target_positions(observer, epochs)
        if self.model.includes_ring:
            earth_positions = query_target_positions("earth", epochs)
        else:
            earth_positions = observer_positions.copy()

        emission = simulation.instantaneous_emission(
            nside=nside,
            freq=freq.value,
            model=self.model,
            line_of_sights=self.line_of_sights,
            observer_positions=observer_positions,
            earth_positions=earth_positions,
            coord_out=coord_out,
        )

        if return_comps:
            return {
                component_label.value: emission[idx]
                for idx, component_label in enumerate(self.model.components)
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

        freq = freq_or_wavelength.to("GHz", equivalencies=u.spectral())

        if earth_position is None:
            earth_position = observer_position

        emission = simulation.time_ordered_emission(
            nside=nside,
            freq=freq.value,
            model=self.model,
            line_of_sights=self.line_of_sights,
            observer_position=observer_position,
            earth_position=earth_position,
            pixel_chunk=pixels,
            bin=bin,
            coord_out=coord_out,
        )

        if return_comps:
            return {
                component_label.value: emission[idx]
                for idx, component_label in enumerate(self.model.components)
            }

        return emission.sum(axis=0)

    def __str__(self) -> str:
        """String representation of the Interplanetary dust model used."""

        reprs = []
        for label in self.model.components:
            component_class = LABEL_TO_CLASS[label]
            component_repr = f"{component_class.__name__}" + "\n"
            reprs.append(f"({label.value}): {component_repr}")

        main_repr = "InterplanetaryDustModel("
        main_repr += f"\n  name: {self.model.name}"
        main_repr += "\n  components( "
        main_repr += "\n    " + "    ".join(reprs)
        main_repr += "  )"
        main_repr += "\n)"

        return main_repr