from __future__ import annotations

from functools import partial
from typing import Literal, Sequence, Union

import astropy.units as u
import healpy as hp
import numpy as np
import quadpy
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time
from numpy.typing import NDArray

from ._component import Component
from ._decorators import validate_ang, validate_freq, validate_pixels
from ._ephemeris import get_obs_and_earth_positions
from ._line_of_sight import get_line_of_sight_endpoints
from ._source_functions import (
    SPECIFIC_INTENSITY_UNITS,
    get_blackbody_emission,
    get_dust_grain_temperature,
    get_phase_function,
    get_scattering_angle,
)
from ._unit_vectors import get_unit_vectors_from_ang, get_unit_vectors_from_pixels
from .models import model_registry

DISTANCE_TO_JUPITER = u.Quantity(5.2, u.AU)
DEFAULT_SOLAR_CUTOFF = u.Quantity(5, u.deg)

HEALPixIndicies = Union[int, Sequence[int], NDArray[np.integer]]
SkyAngles = Union[u.Quantity[u.deg], u.Quantity[u.rad]]
FrequencyOrWavelength = Union[u.Quantity[u.Hz], u.Quantity[u.m]]


class Zodipy:
    """Interface for simulating zodiacal emission.

    Sets up the simulation configuration and provides methods for simulating the zodiacal
    emission that a solar system observer is predicted to see given an interplanetary dust
    model.

    Attributes:
        model (str): Name of the interplanetary dust model to use in the simulations.
            Defaults to DIRBE.
        ephemeris (str): Ephemeris used to compute the positions of the observer and the
            Earth. Defaults to 'de432s' which requires downloading (and caching) a ~10MB
            file. For more information on available ephemeridis, please visit
            https://docs.astropy.org/en/stable/coordinates/solarsystem.html
        extrapolate (bool): If True all spectral quantities in the selected model are
            linearly extrapolated to the requested frequency/wavelength. If False, an
            Exception will be raised on requested frequencies/wavelengths outside of the
            valid model range. Default is False.
        gauss_quad_order (int): Order of the Gaussian-Legendre quadrature used to evaluate
            the brightness integral. Default is 50 points.
        cutoff (u.Quantity[u.AU]): Radial distance from the Sun at which all line of sights
            are truncated. Defaults to 5.2 AU which is the distance to Jupiter.
        solar_cutoff (u.Quantity[u.deg]): Cutoff angle from the sun in degrees. The emission
            for all the pointing with angular distance between the sun smaller than
            `solar_cutoff` are set to `np.nan`. This is due to the model singularity of the
            diffuse cloud component at the heliocentric origin. Such a cutoff may be
            useful when using simulated pointing, but actual scanning strategies are
            unlikely to look directly at the sun. This feature is turned of by setting this
            argument to `None`. Defaults to 5 degrees.

    """

    def __init__(
        self,
        model: str = "dirbe",
        ephemeris: str = "de432s",
        extrapolate: bool = False,
        gauss_quad_order: int = 100,
        cutoff: u.Quantity[u.AU] = DISTANCE_TO_JUPITER,
        solar_cutoff: u.Quantity[u.deg] | None = DEFAULT_SOLAR_CUTOFF,
    ) -> None:

        self.model = model_registry.get_model(model)
        self.ephemeris = ephemeris
        self.extrapolate = extrapolate
        self.cutoff = cutoff
        self.solar_cutoff = solar_cutoff.to(u.rad) if solar_cutoff is not None else solar_cutoff
        self.integration_scheme = quadpy.c1.gauss_legendre(gauss_quad_order)

    @property
    def supported_observers(self) -> list[str]:
        """Returns a list of available observers given an ephemeris."""

        return list(solar_system_ephemeris.bodies) + ["semb-l2"]

    @validate_freq
    @validate_ang
    def get_emission_ang(
        self,
        freq: FrequencyOrWavelength,
        theta: SkyAngles,
        phi: SkyAngles,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`.

        Args:
            freq: Frequency or wavelength at which to evaluate the zodiacal emission.
            theta: Angular co-latitude coordinate of a point, or a sequence of points, on
                the celestial sphere. Must be in the range [0, π] rad. Units must be either
                radians or degrees.
            phi: Angular longitude coordinate of a point, or a sequence of points, on the
                celestial sphere. Must be in the range [0, 2π] rad. Units must be either
                radians or degrees.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        unique_angles, indicies = np.unique(
            np.asarray([theta, phi]), return_inverse=True, axis=1
        )
        unit_vectors = get_unit_vectors_from_ang(
            coord_in=coord_in,
            theta=unique_angles[0],
            phi=unique_angles[1],
            lonlat=lonlat,
        )

        emission = self._compute_emission(
            freq=freq,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
        )

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    @validate_freq
    @validate_pixels
    def get_emission_pix(
        self,
        freq: FrequencyOrWavelength,
        pixels: HEALPixIndicies,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated zodiacal emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`.

        Args:
            freq: Frequency or wavelength at which to evaluate the zodiacal emission.
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        unique_pixels, indicies = np.unique(pixels, return_inverse=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = self._compute_emission(
            freq=freq,
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

    @validate_ang
    @validate_freq
    def get_binned_emission_ang(
        self,
        freq: FrequencyOrWavelength,
        theta: SkyAngles,
        phi: SkyAngles,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`. The emission is binned to a HEALPix map with
        resolution given by `nside`.

        Args:
            freq: Frequency or wavelength at which to evaluate the zodiacal emission.
            theta: Angular co-latitude coordinate of a point, or a sequence of points, on
                the celestial sphere. Must be in the range [0, π] rad. Units must be either
                radians or degrees.
            phi: Angular longitude coordinate of a point, or a sequence of points, on the
                celestial sphere. Must be in the range [0, 2π] rad. Units must be either
                radians or degrees.
            nside: HEALPix resolution parameter of the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            lonlat: If True, input angles `theta`, `phi` are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        unique_angles, counts = np.unique(
            np.asarray([theta, phi]), return_counts=True, axis=1
        )
        unique_pixels = hp.ang2pix(nside, *unique_angles, lonlat=lonlat)
        unit_vectors = get_unit_vectors_from_ang(
            coord_in=coord_in,
            theta=unique_angles[0],
            phi=unique_angles[1],
            lonlat=lonlat,
        )

        emission = self._compute_emission(
            freq=freq,
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

    @validate_freq
    @validate_pixels
    def get_binned_emission_pix(
        self,
        freq: FrequencyOrWavelength,
        pixels: HEALPixIndicies,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`. The emission is binned to a HEALPix map with resolution given by
        `nside`.

        Args:
            freq: Frequency or wavelength at which to evaluate the zodiacal emission.
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        unique_pixels, counts = np.unique(pixels, return_counts=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        emission = self._compute_emission(
            freq=freq,
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

    def _compute_emission(
        self,
        freq: u.Quantity[u.GHz],
        obs: str,
        obs_time: Time,
        unit_vectors: NDArray[np.floating],
        indicies: NDArray[np.integer],
        binned: bool = False,
        obs_pos: u.Quantity[u.AU] | None = None,
        pixels: NDArray[np.integer] | None = None,
        nside: int | None = None,
    ) -> NDArray[np.floating]:
        """Computes the component-wise zodiacal emission."""

        observer_position, earth_position = get_obs_and_earth_positions(
            obs=obs, obs_time=obs_time, obs_pos=obs_pos
        )

        if self.solar_cutoff is not None:
            # The observer position is aquired in geocentric coordinates before being
            # rotated to ecliptic coordinates which means that we can find the
            # heliocentric origin by simply taking the negative of the observer position.
            # We set all unit_vectors with a angular distance from the suns smaller than
            # the cutoff to np.nan which will propagate to np.nan/hp.UNSEEN in the emission.
            ang_dist = hp.rotator.angdist(-observer_position, unit_vectors)
            unit_vectors[:, ang_dist < self.solar_cutoff.value] = np.nan

        if self.model.solar_irradiance_model is not None:
            solar_irradiance = (
                self.model.solar_irradiance_model.interpolate_solar_irradiance(
                    freq=freq, albedos=self.model.albedos, extrapolate=self.extrapolate
                )
            )
        else:
            solar_irradiance = 0

        start, stop = get_line_of_sight_endpoints(
            cutoff=self.cutoff.value,
            obs_pos=observer_position,
            unit_vectors=unit_vectors,
        )

        emission = np.zeros(
            (self.model.n_comps, hp.nside2npix(nside) if binned else indicies.size)
        )

        # Preparing quantities for broadcasting
        stop_expanded = np.expand_dims(stop, axis=-1)
        observer_position = np.expand_dims(observer_position, axis=-1)
        earth_position = np.expand_dims(earth_position, axis=-1)
        unit_vectors = np.expand_dims(unit_vectors, axis=-1)

        # For each component, compute the integrated line of sight emission
        for idx, (label, comp) in enumerate(self.model.comps.items()):
            source_parameters = self.model.interpolate_source_parameters(label, freq)
            emissivity, albedo, phase_coefficients = source_parameters

            # Here we create a partial function that will be passed to the
            # integration scheme. The arrays are reshaped to (d, n, p) where d
            # is the geometrical dimensionality, n is the number of different
            # pointings, and p is the number of integration points of the
            # quadrature.
            emission_comp_integrand = partial(
                _compute_comp_emission_at_step,
                start=start,
                stop=stop_expanded,
                X_obs=observer_position,
                X_earth=earth_position,
                u_los=unit_vectors,
                comp=comp,
                freq=freq.value,
                T_0=self.model.T_0,
                delta=self.model.delta,
                emissivity=emissivity,
                albedo=albedo,
                phase_coefficients=phase_coefficients,
                solar_irradiance=solar_irradiance,
            )

            integrated_comp_emission = self.integration_scheme.integrate(
                emission_comp_integrand, [-1, 1]
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

    # def _remove_masked_pointing

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model.name!r}, "
            f"ephemeris={self.ephemeris!r}, "
            f"extrapolate={self.extrapolate!r})"
        )

    def __str__(self) -> str:
        return repr(self.model)


def _compute_comp_emission_at_step(
    r: float | NDArray[np.floating],
    *,
    start: float,
    stop: float | NDArray[np.floating],
    X_obs: NDArray[np.floating],
    X_earth: NDArray[np.floating],
    u_los: NDArray[np.floating],
    comp: Component,
    freq: float,
    T_0: float,
    delta: float,
    emissivity: float,
    albedo: float,
    phase_coefficients: tuple[float, float, float],
    solar_irradiance: float,
) -> NDArray[np.floating]:
    """Returns the zodiacal emission at a step along a line of sight."""

    # Convert the line of sight range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    density = comp.compute_density(X_helio, X_earth=X_earth)
    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = get_blackbody_emission(freq, temperature)

    emission = (1 - albedo) * (emissivity * blackbody_emission)

    if albedo > 0:
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)

        emission += albedo * solar_flux * phase_function

    return emission * density
