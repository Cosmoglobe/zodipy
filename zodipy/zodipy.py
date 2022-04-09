from functools import partial
from typing import List, Literal, Optional, Sequence, Union

from astropy.coordinates import solar_system_ephemeris
import astropy.units as u
from astropy.units import Quantity, quantity_input
from astropy.time import Time
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from ._emission import get_emission_step
from ._ephemeris import get_earth_position, get_observer_position
from ._integral import trapezoidal_regular_grid
from ._line_of_sight import get_line_of_sight
from .models import model_registry
from ._unit_vector import (
    get_unit_vectors_from_angles,
    get_unit_vectors_from_pixels,
)


__all__ = ("Zodipy",)


class Zodipy:
    """The Zodipy interface.

    Zodipy simulates the Zodiacal emission that a Solar System observer is
    predicted to see given the DIRBE Interplanetary Dust model or other models,
    such as the Planck Interplanetary Dust models which extend the DIRBE model
    to other frequencies.
    """

    def __init__(
        self,
        model: str = "DIRBE",
        ephemeris: str = "de432s",
        extrapolate: bool = False,
    ) -> None:
        """Initializes Zodipy for a given a model and emphemeris.

        Parameters
        ----------
        model
            The name of the interplanetary dust model. Defaults to DIRBE. See all
            available models with `zodipy.MODELS`.
        ephemeris
            Ephemeris used to compute the positions of the observer and Earth.
            Defaults to 'de432s' which requires downloading (and caching) a 10
            MB file. For more information on available ephemeridis, please visit
            https://docs.astropy.org/en/stable/coordinates/solarsystem.html
        extrapolate
            If True, then the spectral quantities in the model will be linearly
            extrapolated to the requested frequency if this is outside of the
            range covered by the model. If False, an Exception will be raised.
            Default is False.
        """

        self.model = model_registry.get_model(model)
        self.ephemeris = ephemeris
        self.extrapolate = extrapolate

    @property
    def ephemeris(self) -> str:
        """The ephemeris used to compute Solar System positions."""

        return self._ephemeris

    @ephemeris.setter
    def ephemeris(self, value: str):
        try:
            solar_system_ephemeris.set(value)
        except (ValueError, AttributeError):
            raise ValueError(f"{value!r} is not a supported astropy ephemeris.")

        self._ephemeris = value

    @property
    def supported_observers(self) -> List[str]:
        """Returns all observers suported by the ephemeridis."""

        return list(solar_system_ephemeris.bodies) + ["semb-l2"]

    @quantity_input
    def get_emission(
        self,
        freq: Union[Quantity[u.Hz], Quantity[u.m]],
        *,
        obs: str = "earth",
        obs_time: Time = Time.now(),
        obs_pos: Optional[Quantity[u.AU]] = None,
        pixels: Optional[Union[int, Sequence[int], NDArray[np.integer]]] = None,
        theta: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
        phi: Optional[Union[Quantity[u.rad], Quantity[u.deg]]] = None,
        nside: Optional[int] = None,
        lonlat: bool = False,
        binned: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> Quantity[u.MJy / u.sr]:
        """Returns the simulated Zodiacal Emission.

        This function takes as arguments a frequency or wavelength (`freq`),
        time of observation (`obs_time`), and a Solar System observer (`obs`).
        The position of the observer is computed using the ephemeris specified
        in the initialization of `Zodipy`. Optionally, the observer position
        can be explicitly specified with the `obs_pos` argument, which
        overrides `obs`. The pointing, for which to compute the emission, is
        specified either in the form of angles on the sky (`theta`, `phi`), or
        as HEALPix pixels (`pixels`) for a given resolution (`nside`).

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or length.
        obs
            The Solar System observer. A list of all support observers (for a
            given ephemeridis) is specified in `supported_observers` attribute
            of the `Zodipy` instance. Defaults to 'earth'.
        obs_time
            Time of observation (`astropy.time.Time`). Defaults to the current
            time.
        obs_pos
            The heliocentric ecliptic cartesian position of the observer in AU.
            Overrides the `obs` argument. Default is None.
        pixels
            A single, or a sequence of HEALPix pixel indicies representing points
            on the celestial sphere. If pixels is given, the `nside` parameter
            specifying the resolution of these pixels must also be provided.
        theta
            Angular co-latitude coordinate of a point, or a sequence of points,
            on the celestial sphere. Must be in the range [0, π] rad. Units
            must be either radians or degrees.
        phi
            Angular longitude coordinate of a point, or a sequence of points, on
            the celestial sphere. Must be in the range [0, 2π] rad. Units must
            be either radians or degrees.
        nside
            HEALPix resolution parameter of the pixels (and optionally the binned
            output map). Must be given if `pixels` is provided or if `binned` is
            set to True.
        lonlat
            If True, input angles (`theta`, `phi`) are assumed to be longitude
            and latitude, otherwise, they are co-latitude and longitude.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        binned
            If True, the emission is binned into a HEALPix map with resolution
            given by the `nside` argument in the coordinate frame corresponding to
            `coord_in`. Defaults to False.
        coord_in
            Coordinate frame of the input pointing. Assumes 'E' (ecliptic
            coordinates) by default.

        Returns
        -------
        emission
            Sequence of simulated Zodiacal emission in units of 'MJy/sr' for
            each input pointing. The output may be interpreted as a timestream
            of Zodiacal emission observed by the Solar System observer if the
            pointing is provided in a time-ordered manner. If `binned` is set to
            True, the output will be a binned HEALPix Zodiacal emission map.
        """

        if obs.lower() not in self.supported_observers:
            raise ValueError(
                f"observer {obs!r} not supported by ephemeridis "
                f"{solar_system_ephemeris._value!r}"
            )

        if pixels is not None:
            if phi is not None or theta is not None:
                raise ValueError(
                    "get_time_ordered_emission() got an argument for both 'pixels'"
                    "and 'theta' or 'phi' but can only use one of the two"
                )
            if nside is None:
                raise ValueError(
                    "get_time_ordered_emission() got an argument for 'pixels', "
                    "but argument 'nside' is missing"
                )
            if lonlat:
                raise ValueError(
                    "get_time_ordered_emission() has 'lonlat' set to True "
                    "but 'theta' and 'phi' is not given"
                )
            if np.ndim(pixels) == 0:
                pixels = np.expand_dims(pixels, axis=0)

        if binned and nside is None:
            raise ValueError(
                "get_time_ordered_emission() has 'binned' set to True, but "
                "argument 'nside' is missing"
            )

        if (theta is not None and phi is None) or (phi is not None and theta is None):
            raise ValueError(
                "get_time_ordered_emission() is missing an argument for either "
                "'phi' or 'theta'"
            )

        if (theta is not None) and (phi is not None):
            if theta.size != phi.size:
                raise ValueError(
                    "get_time_ordered_emission() got arguments 'theta' and 'phi' "
                    "with different size "
                )
            theta = theta.to(u.deg) if lonlat else theta.to(u.rad)
            phi = phi.to(u.deg) if lonlat else phi.to(u.rad)

            if theta.ndim == 0:
                theta = np.expand_dims(theta, axis=0)
            if phi.ndim == 0:
                phi = np.expand_dims(phi, axis=0)

        if not isinstance(obs_time, Time):
            raise TypeError("argument 'obs_time' must be of type 'astropy.time.Time'")

        # Get position of Solar System bodies
        earth_position = get_earth_position(obs_time)
        if obs_pos is None:
            observer_position = get_observer_position(obs, obs_time, earth_position)
        else:
            observer_position = obs_pos.value.reshape(3, 1)

        if not self.extrapolate:
            self.model.validate_frequency(freq)

        # Convert to frequency convention (internal computations are done in
        # frequency)
        frequency = freq.to("GHz", equivalencies=u.spectral())

        # Compute binned HEALPix map
        if binned:
            # Input pointing is pixel indicies
            if pixels is not None:
                unique_pixels, counts = np.unique(pixels, return_counts=True)
                unit_vectors = get_unit_vectors_from_pixels(
                    coord_in=coord_in,
                    pixels=unique_pixels,
                    nside=nside,
                )
            # Input pointing is angles
            else:
                unique_angles, counts = np.unique(
                    np.asarray([theta, phi]), return_counts=True, axis=1
                )
                unique_pixels = hp.ang2pix(nside, *unique_angles, lonlat=lonlat)
                unit_vectors = get_unit_vectors_from_angles(
                    coord_in=coord_in,
                    theta=unique_angles[0],
                    phi=unique_angles[1],
                    lonlat=lonlat,
                )

            emission = np.zeros((self.model.n_components, hp.nside2npix(nside)))
            # Compute the integrated emission for each component in the model
            for idx, (label, component) in enumerate(self.model.components.items()):
                source_parameters = self.model.get_source_parameters(label, frequency)
                emissivity, albedo, phase_coefficients = source_parameters

                emission_step_function = partial(
                    get_emission_step,
                    X_obs=observer_position,
                    X_earth=earth_position,
                    u_los=unit_vectors,
                    component=component,
                    frequency=frequency.value,
                    T_0=self.model.T_0,
                    delta=self.model.delta,
                    emissivity=emissivity,
                    albedo=albedo,
                    phase_coefficients=phase_coefficients,
                )
                start, stop, n_steps = get_line_of_sight(
                    component_label=label,
                    observer_position=observer_position,
                    unit_vectors=unit_vectors,
                )
                emission[idx, unique_pixels] = trapezoidal_regular_grid(
                    get_emission_step=emission_step_function,
                    start=start,
                    stop=stop,
                    n_steps=n_steps,
                )

            emission[:, unique_pixels] *= counts

        # Compute emission for each pointing
        else:
            # Input pointing is pixel indicies
            if pixels is not None:
                unique_pixels, indicies = np.unique(pixels, return_inverse=True)
                unit_vectors = get_unit_vectors_from_pixels(
                    coord_in=coord_in,
                    pixels=unique_pixels,
                    nside=nside,
                )
                emission = np.zeros((self.model.n_components, len(pixels)))
            # Input pointing is angles
            else:
                unique_angles, indicies = np.unique(
                    np.asarray([theta, phi]), return_inverse=True, axis=1
                )
                unit_vectors = get_unit_vectors_from_angles(
                    coord_in=coord_in,
                    theta=unique_angles[0],
                    phi=unique_angles[1],
                    lonlat=lonlat,
                )

                emission = np.zeros((self.model.n_components, len(theta)))

            # Compute the integrated emission for each component in the model
            for idx, (label, component) in enumerate(self.model.components.items()):
                source_parameters = self.model.get_source_parameters(label, frequency)
                emissivity, albedo, phase_coefficients = source_parameters
                emission_step_function = partial(
                    get_emission_step,
                    X_obs=observer_position,
                    X_earth=earth_position,
                    u_los=unit_vectors,
                    component=component,
                    frequency=frequency.value,
                    T_0=self.model.T_0,
                    delta=self.model.delta,
                    emissivity=emissivity,
                    albedo=albedo,
                    phase_coefficients=phase_coefficients,
                )
                start, stop, n_steps = get_line_of_sight(
                    component_label=label,
                    observer_position=observer_position,
                    unit_vectors=unit_vectors,
                )
                integrated_comp_emission = trapezoidal_regular_grid(
                    get_emission_step=emission_step_function,
                    start=start,
                    stop=stop,
                    n_steps=n_steps,
                )

                emission[idx] = integrated_comp_emission[indicies]

        # Convert from specific intensity units (W Hz^-1 m^-2 sr^-1) to MJy/sr
        emission = (emission << (u.W / u.Hz / u.m**2 / u.sr)).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={self.model.name!r}, "
            f"ephemeris={self.ephemeris!r}, "
            f"extrapolate={self.extrapolate!r})"
        )

    def __str__(self) -> str:
        return repr(self.model)
