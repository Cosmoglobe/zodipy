from typing import Dict, Optional, Union

import numpy as np
import astropy.units as u

from zodipy._coordinates import to_frame, get_target_coordinates, EpochsType
from zodipy._emissivities import get_emissivities
from zodipy._integration_config import integration_config_registry
from zodipy._simulation import instantaneous_emission, time_ordered_emission
from zodipy.models import model_registry


class InterplanetaryDustModel:
    """The Zodipy simulation interface.

    By default, the Interplanetary Dust Model used by Zodipy is the
    Kelsall et al. (1998) Interplanetary Dust Model, which includes
    five Zodiacal components:
        - The Diffuse Cloud (cloud)
        - Three Asteroidal Bands (band1, band2, band3)
        - The Circumsolar Ring (ring)
        - The Earth-trailing Feature (feature)

    The Kelsall model yields the Zodiacal Emission given purely by the
    parametric model, and does not use the emissivity factors fitted by
    the Planck Collaboration.

    Other available models are:
        - Planck13 (all five Kelsall components + emissivity fits for each)
        - Planck15 (cloud + bands + new emissivity fits)
        - Planck18 (cloud + bands + the latest emissivity fits)
    NOTE: These alternative models can only be evaluated within the frequency
    range covered by the Planck HFI Bands.

    Custom IPD models (models consisting of a set of Zodiacal Components
    with custom parameters and emissivities) needs to be registered before
    initializing this class by using `zodipy.register_custom_model`.
    """

    def __init__(self, model: str = "K98") -> None:
        """Initializes the interface given an interplaneteray dust model.

        Parameters
        ----------
        model
            The name of a implemented Interplanetary Dust Model.
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
        epochs: Optional[EpochsType] = None,
        coord: str = "E",
        return_comps: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the instantaneous Zodiacal Emission [MJy/sr].

        By instantaneous emission we mean the emission observed in an instant
        in time.

        The observer location (and optionally the location of the Earth if
        either of the Feature or the Ring components are included) are queried
        from the Horizons JPL ephemerides, given some epoch.

        NOTE: This function returns the fullsky emission from one coordinate
        in space. This means that we evaluate line-of-sights looking in towards
        the inner Solar System where the dust density increases exponentially.
        These are line-of-sights unlikely to be observed by an actual
        experiment.

        Parameters
        ----------
        freq_or_wavelength
            Frequency or wavelength at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter.
        observer
            The observer. Defaults to 'L2'.
        epochs
            The observeration times given as a single epoch, or a list of epochs
            in JD or MJD format, or a dictionary defining a range of times and
            dates; the range dictionary has to be of the form
            {'start':'YYYY-MM-DD [HH:MM:SS]', 'stop':'YYYY-MM-DD [HH:MM:SS]',
            'step':'n[y|d|h|m|s]'}. If no epochs are provided, the current time
            is used in UTC.
        coord
            Coordinate frame of the output map. Defaults to 'E', which is
            ecliptic coordinates.
        return_comps
            If True, the emission of each component in the model is
            returned individually in a dictionary. Defaults to False.

        Returns
        -------
        emission
            Simulated instantaneous Zodiacal emission in units of MJy/sr.
        """

        observer_coordinates = get_target_coordinates(observer, epochs)
        if self.model.includes_earth_neighboring_components:
            earth_coordinates = get_target_coordinates("earth", epochs)
        else:
            earth_coordinates = np.zeros(3)

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
            observer_coords=observer_coordinates,
            earth_coords=earth_coordinates,
            line_of_sights=self.line_of_sights,
        )
        if coord != "E":
            emission = to_frame(emission, coord)

        if not return_comps:
            return emission.sum(axis=0)

        return {
            comp.value: emission[idx] for idx, comp in enumerate(self.model.components)
        }

    @u.quantity_input(freq_or_wavelength=("Hz", "m", "micron"))
    def get_time_ordered_emission(
        self,
        freq_or_wavelength: u.Quantity,
        nside: int,
        *,
        pixels: np.ndarray,
        observer_coordinates: np.ndarray,
        earth_coordinates: Optional[np.ndarray] = None,
        return_comps: bool = False,
        bin: bool = False,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Simulates and returns the Zodiacal emission timestream [MJy/sr].

        Given a timestream of pixels, this function evaluates the Zodiacal
        Emission given a position in space (`observer_coordinates`), and
        optionally Earth's coordinates (`earth_coordinates`) if the
        interplanetary dust model includes either of the Feature or Ring
        components.

        If the bin parameter is set to True, the timestream is binned into a
        HEALPIX map instead.

        Parameters
        ----------
        freq_or_wavelength
            Frequency or wavelength at which to evaluate the Zodiacal emission.
        nside
            HEALPIX map resolution parameter.
        pixels
            Chunk of time-ordered pixels corresponding to a parts of a scanning
            strategy.
        observer_coordinates
            The heliocentric coordinates with shape (3,) of the observer over
            the tod chunk.
        earth_coordinates
            The heliocentric coordinates with shape (3,) of the Earth over the
            tod chunk. Default is None, in which we set the earth_coordinates
            equal to the observer coordinates.
        return_comps
            If True, the emission of each component in the model is
            returned individually in a dictionary. Defaults to False.
        bin
            If True, return a binned HEALPIX map of the emission. If False, the
            timestream is returned.

        Returns
        -------
        emission
            Zodiacal emission [MJy/sr] over a timestream of pixels, or the
            binned Zodiacal emission map if bin is set to True.
        """

        # If no earth_coordinates are specified, we assume that the earth and
        # the observer coordinates are set to the same.
        if earth_coordinates is None:
            earth_coordinates = observer_coordinates

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
            observer_coordinates=observer_coordinates,
            earth_coordinates=earth_coordinates,
            pixel_chunk=pixels,
            bin=bin,
        )

        if not return_comps:
            return emission.sum(axis=0)

        return {
            comp.value: emission[idx] for idx, comp in enumerate(self.model.components)
        }

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