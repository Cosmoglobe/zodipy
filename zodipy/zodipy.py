from __future__ import annotations

import multiprocessing
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
        extrapolate (bool): If True all spectral quantities in the selected model are
            linearly extrapolated to the requested frequency/wavelength. If False, an
            Exception will be raised on requested frequencies/wavelengths outside of the
            valid model range. Default is False.
        parallel (bool): If True, input pointing sequences will be split among all
            available cores on the machine, and the emission will be computed in parallel.
            This is useful for large simulations. NOTE: This will require the user to guard
            the executed code in a `if __name__ == "__main__"` block. Default is False.
        n_proc (int): Number of cores to use when parallel computation. Defaults is None,
            which will use all available cores.
        solar_cut (u.Quantity[u.deg]): Cutoff angle from the sun in degrees. The emission
            for all the pointing with angular distance between the sun smaller than
            `solar_cutoff` are set to `np.nan`. This is due to the model singularity of the
            diffuse cloud component at the heliocentric origin. Such a cutoff may be
            useful when using simulated pointing, but actual scanning strategies are
            unlikely to look directly at the sun. This feature is turned of by setting this
            argument to `None`. Defaults to 5 degrees.
        solar_cut_fill_value (float): Fill value for the masked solar cut pointing.
            Defaults to `np.nan`.
        gauss_quad_order (int): Order of the Gaussian-Legendre quadrature used to evaluate
            the brightness integral. Default is 50 points.
        los_dist_cut (u.Quantity[u.AU]): Radial distance from the Sun at which all line of
            sights are truncated. Defaults to 5.2 AU which is the distance to Jupiter.
        ephemeris (str): Ephemeris used to compute the positions of the observer and the
            Earth. Defaults to 'de432s' which requires downloading (and caching) a ~10MB
            file. For more information on available ephemeridis, please visit
            https://docs.astropy.org/en/stable/coordinates/solarsystem.html

    """

    def __init__(
        self,
        model: str = "dirbe",
        extrapolate: bool = False,
        parallel: bool = False,
        n_proc: int | None = None,
        solar_cut: u.Quantity[u.deg] | None = None,
        solar_cut_fill_value: float = np.nan,
        gauss_quad_order: int = 100,
        los_dist_cut: u.Quantity[u.AU] = DISTANCE_TO_JUPITER,
        ephemeris: str = "de432s",
    ) -> None:

        self.model = model_registry.get_model(model)
        self.extrapolate = extrapolate
        self.parallel = parallel
        self.n_proc = n_proc
        self.solar_cut = solar_cut.to(u.rad) if solar_cut is not None else solar_cut
        self.solar_cut_fill_value = solar_cut_fill_value
        self.gauss_quad_order = gauss_quad_order
        self.integration_scheme = quadpy.c1.gauss_legendre(self.gauss_quad_order)
        self.los_dist_cut = los_dist_cut
        self.ephemeris = ephemeris

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

        if self.model.solar_irradiance_model is not None:
            solar_irradiance = (
                self.model.solar_irradiance_model.interpolate_solar_irradiance(
                    freq=freq,
                    albedos=self.model.albedos,
                    extrapolate=self.extrapolate,
                )
            )
        else:
            solar_irradiance = 0

        start, stop = get_line_of_sight_endpoints(
            cutoff=self.los_dist_cut.value,
            obs_pos=observer_position,
            unit_vectors=unit_vectors,
        )

        emission = np.zeros(
            (self.model.n_comps, hp.nside2npix(nside) if binned else indicies.size)
        )

        # Preparing quantities for broadcasting
        observer_position_expanded = np.expand_dims(observer_position, axis=-1)
        earth_position_expanded = np.expand_dims(earth_position, axis=-1)

        emissivities, albedos, phase_coefficients = self.model.interp_source_params(
            freq
        )
        # Distribute pointing to available CPUs and compute the emission in parallel.
        if self.parallel:
            n_proc = multiprocessing.cpu_count() if self.n_proc is None else self.n_proc

            unit_vector_chunks = np.array_split(unit_vectors, n_proc, axis=-1)
            stop_chunks = np.array_split(stop, n_proc, axis=-1)

            with multiprocessing.Pool(processes=n_proc) as pool:
                # Here we create a partial function that will be passed to the
                # integration scheme. The arrays are reshaped to (d, n, p) where d
                # is the geometrical dimensionality, n is the number of different
                # pointings, and p is the number of integration points of the
                # quadrature.
                partial_integrand_chunks = [
                    partial(
                        _compute_emission_at_step,
                        start=start,
                        stop=np.expand_dims(stop, axis=-1),
                        n_quad_points=self.gauss_quad_order,
                        X_obs=observer_position_expanded,
                        X_earth=earth_position_expanded,
                        u_los=np.expand_dims(unit_vectors, axis=-1),
                        comps=list(self.model.comps.values()),
                        freq=freq.value,
                        T_0=self.model.T_0,
                        delta=self.model.delta,
                        emissivities=emissivities,
                        albedos=albedos,
                        phase_coefficients=phase_coefficients,
                        solar_irradiance=solar_irradiance,
                    )
                    for unit_vectors, stop in zip(unit_vector_chunks, stop_chunks)
                ]

                proc_chunks = [
                    pool.apply_async(
                        self.integration_scheme.integrate,
                        args=(partial_integrand, [-1, 1]),
                    )
                    for partial_integrand in partial_integrand_chunks
                ]
                integrated_comp_emission = np.concatenate(
                    [result.get() for result in proc_chunks], axis=1
                )

        # Compute the emission only in the main process.
        else:
            unit_vectors_expanded = np.expand_dims(unit_vectors, axis=-1)
            stop_expanded = np.expand_dims(stop, axis=-1)

            partial_emission_integrand = partial(
                _compute_emission_at_step,
                start=start,
                stop=stop_expanded,
                n_quad_points=self.gauss_quad_order,
                X_obs=observer_position_expanded,
                X_earth=earth_position_expanded,
                u_los=unit_vectors_expanded,
                comps=list(self.model.comps.values()),
                freq=freq.value,
                T_0=self.model.T_0,
                delta=self.model.delta,
                emissivities=emissivities,
                albedos=albedos,
                phase_coefficients=phase_coefficients,
                solar_irradiance=solar_irradiance,
            )

            integrated_comp_emission = self.integration_scheme.integrate(
                partial_emission_integrand, [-1, 1]
            )

        # We convert the integral from [-1, 1] back to [start, stop].
        integrated_comp_emission *= 0.5 * (stop - start)

        if binned:
            emission[:, pixels] = integrated_comp_emission
        else:
            emission = integrated_comp_emission[:, indicies]

        if self.solar_cut is not None:
            # The observer position is aquired in geocentric coordinates before being
            # rotated to ecliptic coordinates. This means that we can find the
            # heliocentric origin in this coordinate system by simply taking the negative
            # of the observer position. The emission corresponding to unit_vectors with a
            # angular distance (between the pointing and the sun) smaller than the
            # specified `solar_cutoff` value is masked with the `solar_cutoff_fill_value`.
            ang_dist = hp.rotator.angdist(-observer_position, unit_vectors)
            solar_mask = ang_dist < self.solar_cut.value
            if binned and pixels is not None:
                emission[:, pixels[solar_mask]] = self.solar_cut_fill_value
            else:
                emission[:, solar_mask[indicies]] = self.solar_cut_fill_value

        return emission

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model={self.model.name!r}, "
            f"ephemeris={self.ephemeris!r}, "
            f"extrapolate={self.extrapolate!r})"
        )

    def __str__(self) -> str:
        return repr(self.model)


def _compute_emission_at_step(
    r: float | NDArray[np.floating],
    *,
    start: float,
    stop: float | NDArray[np.floating],
    n_quad_points: int,
    X_obs: NDArray[np.floating],
    X_earth: NDArray[np.floating],
    u_los: NDArray[np.floating],
    comps: list[Component],
    freq: float,
    T_0: float,
    delta: float,
    emissivities: list[float],
    albedos: list[float],
    phase_coefficients: tuple[float, float, float],
    solar_irradiance: float,
) -> NDArray[np.floating]:
    """Returns the zodiacal emission at a step along all lines of sight."""

    # Convert the line of sight range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)
    blackbody_emission = get_blackbody_emission(freq, temperature)

    emission = np.zeros((len(comps), np.shape(X_helio)[1], n_quad_points))
    density = np.zeros((len(comps), np.shape(X_helio)[1], n_quad_points))

    for idx, (comp, albedo, emissivity) in enumerate(zip(comps, albedos, emissivities)):
        density[idx] = comp.compute_density(X_helio, X_earth=X_earth)
        emission[idx] = (1 - albedo) * (emissivity * blackbody_emission)

    if any(albedo != 0 for albedo in albedos):
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)

        for idx, albedo in enumerate(albedos):
            emission[idx] += albedo * solar_flux * phase_function

    return emission * density
