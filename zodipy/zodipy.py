from __future__ import annotations

from functools import partial
from typing import Literal, Sequence

from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import NDArray
import quadpy

from ._emission import get_emission_at_step
from ._decorators import validate_frequency, validate_angles, validate_pixels
from ._ephemeris import get_solar_system_positions
from ._line_of_sight import get_line_of_sights, DISTANCE_TO_JUPITER
from ._source_functions import SPECIFIC_INTENSITY_UNITS
from ._unit_vectors import (
    get_unit_vectors_from_angles,
    get_unit_vectors_from_pixels,
)
from .models import model_registry
from .solar_irradiance_models import solar_irradiance_model_registry


class Zodipy:
    """The Zodipy interface.

    Simulate the Zodiacal emission that a Solar System observer is predicted to
    observer given the DIRBE Interplanetary Dust model or other models which 
    extend the DIRBE model to other frequencies.

    Parameters
    ----------
    model
        The name of the interplanetary dust model. Defaults to DIRBE. See all
        available models with `zodipy.MODELS`.
    ephemeris
        Ephemeris used to compute the positions of the observer and Earth.
        Defaults to 'de432s' which requires downloading (and caching) a ~10
        MB file. For more information on available ephemeridis, please visit
        https://docs.astropy.org/en/stable/coordinates/solarsystem.html
    solar_irradiance_model
        Solar irradiance model to use when computing the scattered emission.
        Only relevant at wavelenghts around 1 micron. Default is the tabulated
        DIRBE Solar flux. Other models requires downloading (and caching)
        small (<1MB) files containing the tabulated model spectra and
        irradiance.
    extrapolate
        If True, then the spectral quantities in the model will be linearly
        extrapolated to the requested frequency if this is outside of the
        range covered by the model. If False, an Exception will be raised.
        Default is False.
    gauss_quad_order
        Order of the Gaussian-Legendre quadrature used to evaluate the
        brightness integral. Default is 50 points.
    cutoff:
        Radial distance from the Sun at which marks the end point of all
        line of sights. Defaults to 5.2 AU which is the distance to Jupiter.
    """

    def __init__(
        self,
        model: str = "dirbe",
        ephemeris: str = "de432s",
        solar_irradiance_model: str = "dirbe",
        extrapolate: bool = False,
        gauss_quad_order: int = 100,
        cutoff: float = DISTANCE_TO_JUPITER,
    ) -> None:

        self.model = model_registry.get_model(model)
        self.ephemeris = ephemeris
        self.solar_irradiance_model = solar_irradiance_model_registry.get_model(
            solar_irradiance_model
        )
        self.extrapolate = extrapolate
        self.cutoff = cutoff
        self.integration_scheme = quadpy.c1.gauss_legendre(gauss_quad_order)

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
    def supported_observers(self) -> list[str]:
        """Returns a list of all supported observers given the ephemeris."""

        return list(solar_system_ephemeris.bodies) + ["semb-l2"]

    @validate_frequency
    @validate_angles
    def get_emission_ang(
        self,
        freq: u.Quantity[u.Hz] | u.Quantity[u.m],
        theta: u.Quantity[u.rad] | u.Quantity[u.deg],
        phi: u.Quantity[u.rad] | u.Quantity[u.deg],
        obs: str = "earth",
        obs_time: Time = Time.now(),
        obs_pos: u.Quantity[u.AU] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated Zodiacal Emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form
        of angles on the sky given by `theta` and `phi`.

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or length.
        theta
            Angular co-latitude coordinate of a point, or a sequence of points,
            on the celestial sphere. Must be in the range [0, π] rad. Units
            must be either radians or degrees.
        phi
            Angular longitude coordinate of a point, or a sequence of points, on
            the celestial sphere. Must be in the range [0, 2π] rad. Units must
            be either radians or degrees.
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
        lonlat
            If True, input angles (`theta`, `phi`) are assumed to be longitude
            and latitude, otherwise, they are co-latitude and longitude.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_in
            Coordinate frame of the input pointing. Assumes 'E' (ecliptic
            coordinates) by default.

        Returns
        -------
        emission
            Simulated Zodiacal emission in units of 'MJy/sr'.
        """

        unique_angles, indicies = np.unique(
            np.asarray([theta, phi]), return_inverse=True, axis=1
        )
        unit_vectors = get_unit_vectors_from_angles(
            coord_in=coord_in,
            theta=unique_angles[0],
            phi=unique_angles[1],
            lonlat=lonlat,
        )

        emission = self._get_emission(
            frequency=freq,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
        )

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    @validate_frequency
    @validate_pixels
    def get_emission_pix(
        self,
        freq: u.Quantity[u.Hz] | u.Quantity[u.m],
        pixels: int | Sequence[int] | NDArray[np.integer],
        nside: int,
        obs: str = "earth",
        obs_time: Time = Time.now(),
        obs_pos: u.Quantity[u.AU] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated Zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with
        resolution given by `nside`.

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or length.
        pixels
            A single, or a sequence of HEALPix pixel indicies representing points
            on the celestial sphere.
        nside
            HEALPix resolution parameter of the pixels.
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
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_in
            Coordinate frame of the input pointing. Assumes 'E' (ecliptic
            coordinates) by default.

        Returns
        -------
        emission
            Simulated Zodiacal emission in units of 'MJy/sr'.
        """

        unique_pixels, indicies = np.unique(pixels, return_inverse=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = self._get_emission(
            frequency=freq,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    @validate_angles
    @validate_frequency
    def get_binned_emission_ang(
        self,
        freq: u.Quantity[u.Hz] | u.Quantity[u.m],
        theta: u.Quantity[u.rad] | u.Quantity[u.deg],
        phi: u.Quantity[u.rad] | u.Quantity[u.deg],
        nside: int,
        obs: str = "earth",
        obs_time: Time = Time.now(),
        obs_pos: u.Quantity[u.AU] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned Zodiacal Emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form
        of angles on the sky given by `theta` and `phi`. The emission is binned
        to a HEALPix grid of resolution given by `nside`

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or length.
        theta
            Angular co-latitude coordinate of a point, or a sequence of points,
            on the celestial sphere. Must be in the range [0, π] rad. Units
            must be either radians or degrees.
        phi
            Angular longitude coordinate of a point, or a sequence of points, on
            the celestial sphere. Must be in the range [0, 2π] rad. Units must
            be either radians or degrees.
        nside
            HEALPix resolution parameter of the pixels.
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
        lonlat
            If True, input angles (`theta`, `phi`) are assumed to be longitude
            and latitude, otherwise, they are co-latitude and longitude.
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_in
            Coordinate frame of the input pointing. Assumes 'E' (ecliptic
            coordinates) by default.

        Returns
        -------
        emission
            Simulated Zodiacal emission in units of 'MJy/sr'.
        """

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

        emission = self._get_emission(
            frequency=freq,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=counts,
            binned=True,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    @validate_frequency
    @validate_pixels
    def get_binned_emission_pix(
        self,
        freq: u.Quantity[u.Hz] | u.Quantity[u.m],
        pixels: int | Sequence[int] | NDArray[np.integer],
        nside: int,
        obs: str = "earth",
        obs_time: Time = Time.now(),
        obs_pos: u.Quantity[u.AU] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned Zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with
        resolution given by `nside`. The emission is binned to a HEALPix grid
        of resolution given by `nside`.

        Parameters
        ----------
        freq
            Frequency or wavelength at which to evaluate the Zodiacal emission.
            Must have units compatible with Hz or length.
        pixels
            A single, or a sequence of HEALPix pixel indicies representing points
            on the celestial sphere.
        nside
            HEALPix resolution parameter of the pixels.
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
        return_comps
            If True, the emission is returned component-wise. Defaults to False.
        coord_in
            Coordinate frame of the input pointing. Assumes 'E' (ecliptic
            coordinates) by default.

        Returns
        -------
        emission
            Simulated Zodiacal emission in units of 'MJy/sr'.
        """

        unique_pixels, counts = np.unique(pixels, return_counts=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = self._get_emission(
            frequency=freq,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=counts,
            binned=True,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    def _get_emission(
        self,
        frequency: u.Quantity[u.GHz],
        obs: str,
        obs_time: Time,
        unit_vectors: NDArray[np.floating],
        indicies: NDArray[np.floating],
        binned: bool = False,
        obs_pos: u.Quantity[u.AU] | None = None,
        pixels: NDArray[np.floating] | None = None,
        nside: int | None = None,
    ) -> NDArray[np.floating]:
        """Computes the component-wise Zodiacal emission."""

        observer_position, earth_position = get_solar_system_positions(
            observer=obs,
            time_of_observation=obs_time,
            observer_position=obs_pos,
        )

        # Model includes scattering so we compute the Solar irradiance.
        if self.model.albedos is not None:
            solar_irradiance = self.solar_irradiance_model.get_solar_irradiance(
                frequency, self.extrapolate
            )
        else:
            solar_irradiance = 0

        start, stop = get_line_of_sights(
            cutoff=self.cutoff,
            observer_position=observer_position,
            unit_vectors=unit_vectors,
        )

        output_shape = (
            self.model.n_components,
            hp.nside2npix(nside) if binned else len(indicies),
        )
        emission = np.zeros(output_shape)

        for idx, (label, component) in enumerate(self.model.components.items()):
            # We get the interpolated/extrapolated model parameters.
            source_parameters = self.model.get_source_parameters(label, frequency)
            emissivity, albedo, phase_coefficients = source_parameters

            # Here we create a partial function that will be passed to the
            # integration scheme. The arrays are reshaped to (d, n, p) where d
            # is the geometrical dimensionality, n is the number of different
            # pointings, and p is the number of integration points of the
            # quadrature.
            emission_integrand = partial(
                get_emission_at_step,
                start=start,
                stop=np.expand_dims(stop, axis=-1),
                X_obs=np.expand_dims(observer_position, axis=-1),
                X_earth=np.expand_dims(earth_position, axis=-1),
                u_los=np.expand_dims(unit_vectors, axis=-1),
                component=component,
                frequency=frequency.value,
                T_0=self.model.T_0,
                delta=self.model.delta,
                emissivity=emissivity,
                albedo=albedo,
                phase_coefficients=phase_coefficients,
                solar_irradiance=solar_irradiance,
            )

            integrated_comp_emission = self.integration_scheme.integrate(
                emission_integrand, [-1, 1]
            )
            # We convert the integral from [-1, 1] back to [start, stop].
            integrated_comp_emission *= 0.5 * (stop - start)

            if binned:
                emission[idx, pixels] = integrated_comp_emission
            else:
                emission[idx] = integrated_comp_emission[indicies]

        if binned:
            # We multiply the binned map by the number of hits.
            emission[:, pixels] *= indicies

        return emission

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model.name!r}, "
            f"ephemeris={self.ephemeris!r}, "
            f"extrapolate={self.extrapolate!r})"
        )

    def __str__(self) -> str:
        return repr(self.model)
