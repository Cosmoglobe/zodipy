from functools import partial
from typing import List, Literal, Optional, Sequence, Union

from astropy.coordinates import solar_system_ephemeris
import astropy.units as u
from astropy.units import Quantity, quantity_input
from astropy.time import Time
import healpy as hp
import numpy as np
from numpy.typing import NDArray

from zodipy._emission import get_emission_step
from zodipy._ephemeris import get_earth_position, get_observer_position
from zodipy._integral import trapezoidal_regular_grid
from zodipy._line_of_sight import get_line_of_sight
from zodipy._unit_vector import (
    get_unit_vector_from_angles,
    get_unit_vector_from_pixels,
)
from zodipy.models import model_registry


__all__ = ("Zodipy",)


class Zodipy:
    """The Zodipy interface.

    Zodipy simulates the Zodiacal emission that a Solar System observer is
    predicted to see given the Kelsall et al. (1998) interplanetary dust model.
    """

    def __init__(self, model: str = "DIRBE", ephemeris: str = "de432s") -> None:
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
        """

        self.model = model_registry.get_model(model)
        solar_system_ephemeris.set(ephemeris)

    @property
    def supported_observers(self) -> List[str]:
        """Returns all observers suported by the ephemeridis."""

        return list(solar_system_ephemeris.bodies) + ["l2 or semb-l2"]

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
        """Returns simulated Zodiacal Emission.

        This function takes as arguments a frequency or wavelength (`freq`),
        time of observation (`obs_time`), an a Solar System observer (`obs`).
        The position of the observer is computed using the pehemeris specified
        in the initialization of `Zodipy`. Optionally, the observer position
        can be explicitly specified with the `obs_pos` argument, which
        overrides`obs`. The pointing, for which to compute the emission, is
        specified either in the form of angles on the sky (`theta`, `phi`), or
        as HEALPIX pixels at some resolution (`pixels`, `nside`).

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or m.
        obs
            The solar system observer. A list of all support observers (for a
            given ephemeridis) is specified in `observers` attribute of the
            `zodipy.Zodipy` instance. Defaults to 'earth'.
        obs_time
            Time of observation (`astropy.time.Time` object). Defaults to
            current time.
        obs_pos
            The heliocentric ecliptic cartesian position of the observer in AU.
            Overrides the `obs` argument. Default is None.
        pixels
            A single, or a sequence of HEALPIX pixels representing points on
            the sky. The `nside` parameter, which specifies the resolution of
            these pixels, must also be provided along with this argument.
        theta, phi
            Angular coordinates (co-latitude, longitude ) of a point, or a
            sequence of points, on the sphere. Units must be radians or degrees.
            co-latitude must be in [0, pi] rad, and longitude in [0, 2*pi] rad.
        nside
            HEALPIX map resolution parameter of the pixels (and optionally the
            binned output map). Must be specified if `pixels` is provided or if
            `binned` is set to True.
        lonlat
            If True, input angles (`theta`, `phi`) are assumed to be longitude
            and latitude, otherwise, they are co-latitude and longitude.
            Seeting lonlat to True corresponds to theta=RA, phi=DEC
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        binned
            If True, the emission is binned into a HEALPIX map with resolution
            given by the `nside` argument in the coordinate frame corresponding to
            `coord_in`. Defaults to False.
        coord_in
            Coordinates frame of the pointing. Assumes 'E' (ecliptic coordinates)
            by default.

        Returns
        -------
        emission
            Sequence of simulated Zodiacal emission in units of 'MJy/sr' for
            each pointing. If the pointing is provided in a time-ordered manner,
            then the output of this function can be interpreted as the observered
            Zodiacal Emission timestream.
        """

        if obs.lower() not in self.supported_observers:
            raise ValueError(
                f"observer {obs!r} not supported by ephemeridis "
                f"{solar_system_ephemeris._value!r} or 'Zodipy'"
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
            if np.size(pixels) == 1:
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
            if theta.size == 1:
                phi = np.expand_dims(phi, axis=0)
                theta = np.expand_dims(theta, axis=0)
            if lonlat:
                theta = theta.to(u.deg)
                phi = phi.to(u.deg)
            else:
                theta = theta.to(u.rad)
                phi = phi.to(u.rad)

        if not isinstance(obs_time, Time):
            raise TypeError("argument 'obs_time' must be of type 'astropy.time.Time'")

        earth_position = get_earth_position(obs_time)
        if obs_pos is None:
            observer_position = get_observer_position(obs, obs_time, earth_position)
        else:
            observer_position = obs_pos.reshape(3,1)

        frequency = freq.to("GHz", equivalencies=u.spectral())

        if binned:
            if pixels is not None:
                unique_pixels, counts = np.unique(pixels, return_counts=True)
                unit_vectors = get_unit_vector_from_pixels(
                    coord_in=coord_in,
                    pixels=unique_pixels,
                    nside=nside,
                )
            else:
                unique_angles, counts = np.unique(
                    np.asarray([theta, phi]), return_counts=True, axis=1
                )
                unique_pixels = hp.ang2pix(nside, *unique_angles, lonlat=lonlat)
                unit_vectors = get_unit_vector_from_angles(
                    coord_in=coord_in,
                    theta=unique_angles[0],
                    phi=unique_angles[1],
                    lonlat=lonlat,
                )

            emission = np.zeros((self.model.n_components, hp.nside2npix(nside)))
            for idx, (label, component) in enumerate(self.model.components.items()):
                extrapolated_parameters = (
                    self.model.get_extrapolated_component_parameters(label, frequency)
                )
                emissivity, albedo, phase_coefficients = extrapolated_parameters

                # Create new callable with single argument as the distance along
                # the line of sight (R)
                emission_step_function = partial(
                    get_emission_step,
                    frequency=frequency.value,
                    observer_position=observer_position.value,
                    earth_position=earth_position.value,
                    unit_vectors=unit_vectors,
                    component=component,
                    T_0=self.model.T_0,
                    delta=self.model.delta,
                    emissivity=emissivity,
                    albedo=albedo,
                    phase_coefficients=phase_coefficients,
                )
                start, stop, n_steps = get_line_of_sight(
                    component_label=label,
                    observer_position=observer_position.value,
                    unit_vectors=unit_vectors,
                )
                emission[idx, unique_pixels] = trapezoidal_regular_grid(
                    get_emission_step=emission_step_function,
                    start=start,
                    stop=stop,
                    n_steps=n_steps,
                )

            emission[:, unique_pixels] *= counts

        else:
            if pixels is not None:
                unique_pixels, indicies = np.unique(pixels, return_inverse=True)
                unit_vectors = get_unit_vector_from_pixels(
                    coord_in=coord_in,
                    pixels=unique_pixels,
                    nside=nside,
                )
                emission = np.zeros((self.model.n_components, len(pixels)))

            else:
                unique_angles, indicies = np.unique(
                    np.asarray([theta, phi]), return_inverse=True, axis=1
                )
                unit_vectors = get_unit_vector_from_angles(
                    coord_in=coord_in,
                    theta=unique_angles[0],
                    phi=unique_angles[1],
                    lonlat=lonlat,
                )

                emission = np.zeros((self.model.n_components, len(theta)))

            for idx, (label, component) in enumerate(self.model.components.items()):
                extrapolated_parameters = (
                    self.model.get_extrapolated_component_parameters(label, frequency)
                )
                emissivity, albedo, phase_coefficients = extrapolated_parameters

                # Create new callable with single argument as the distance along
                # the line of sight (R)
                emission_step_function = partial(
                    get_emission_step,
                    frequency=frequency.value,
                    observer_position=observer_position.value,
                    earth_position=earth_position.value,
                    unit_vectors=unit_vectors,
                    component=component,
                    T_0=self.model.T_0,
                    delta=self.model.delta,
                    emissivity=emissivity,
                    albedo=albedo,
                    phase_coefficients=phase_coefficients,
                )
                start, stop, n_steps = get_line_of_sight(
                    component_label=label,
                    observer_position=observer_position.value,
                    unit_vectors=unit_vectors,
                )
                integrated_comp_emission = trapezoidal_regular_grid(
                    get_emission_step=emission_step_function,
                    start=start,
                    stop=stop,
                    n_steps=n_steps,
                )

                emission[idx] = integrated_comp_emission[indicies]

        # The output unit is W/Hz/m^2/sr which we convert to MJy/sr
        emission = (emission << (u.W / u.Hz / u.m ** 2 / u.sr)).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)
