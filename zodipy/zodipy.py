from __future__ import annotations

import multiprocessing
import platform
from dataclasses import asdict
from functools import partial
from typing import Literal

import astropy.units as u
import healpy as hp
import numpy as np
import numpy.typing as npt
import quadpy
from astropy.coordinates import solar_system_ephemeris
from astropy.time import Time

from ._constants import SPECIFIC_INTENSITY_UNITS
from ._interp import interpolate_source_parameters
from ._ipd_dens_funcs import PartialComputeDensityFunc, construct_density_partials
from ._line_of_sight import get_line_of_sight_endpoints
from ._sky_coords import DISTANCE_TO_JUPITER, get_obs_and_earth_positions
from ._source_funcs import (
    get_bandpass_integrated_blackbody_emission,
    get_blackbody_emission,
    get_dust_grain_temperature,
    get_phase_function,
    get_scattering_angle,
)
from ._types import FrequencyOrWavelength, Pixels, SkyAngles
from ._unit_vectors import get_unit_vectors_from_ang, get_unit_vectors_from_pixels
from ._validators import (
    validate_and_normalize_weights,
    validate_ang,
    validate_frequency_in_model_range,
    validate_pixels,
)
from .ipd_models import model_registry

PLATFORM = platform.system().lower()
SYS_PROC_START_METHOD = "fork" if "windows" not in PLATFORM else None


class Zodipy:
    """Interface for simulating zodiacal emission.

    Sets up the simulation configuration and provides methods for simulating the zodiacal
    emission that a solar system observer is predicted to see given an interplanetary dust
    model.

    Attributes:
        model (str): Name of the interplanetary dust model to use in the simulations.
            Defaults to DIRBE.
        extrapolate (bool): If `True` all spectral quantities in the selected model are
            linearly extrapolated to the requested frequency/wavelength. If `False`, an
            exception is raised on requested frequencies/wavelengths outside of the
            valid model range. Default is `False`.
        parallel (bool): If `True`, input pointing sequences will be split among all
            available cores on the machine, and the emission will be computed in parallel.
            This is useful for large pointing chunks. Default is `False`.
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
        gauss_quad_degree (int): Order of the Gaussian-Legendre quadrature used to evaluate
            the brightness integral. Default is 100 points.
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
        gauss_quad_degree: int = 100,
        los_dist_cut: u.Quantity[u.AU] = DISTANCE_TO_JUPITER,
        ephemeris: str = "de432s",
    ) -> None:

        self.model = model
        self.extrapolate = extrapolate
        self.parallel = parallel
        self.n_proc = n_proc
        self.solar_cut = solar_cut.to(u.rad) if solar_cut is not None else solar_cut
        self.solar_cut_fill_value = solar_cut_fill_value
        self.gauss_quad_degree = gauss_quad_degree
        self.los_dist_cut = los_dist_cut
        self.ephemeris = ephemeris

        self.ipd_model = model_registry.get_model(model)
        self._integration_scheme = quadpy.c1.gauss_legendre(self.gauss_quad_degree)

    @property
    def supported_observers(self) -> list[str]:
        """Returns a list of available observers given an ephemeris."""

        return list(solar_system_ephemeris.bodies) + ["semb-l2"]

    def get_emission_ang(
        self,
        freq: FrequencyOrWavelength,
        theta: SkyAngles,
        phi: SkyAngles,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: u.Quantity[u.MJy / u.sr] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission.
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
            weights: Bandpass weights corresponding the the frequencies in `freq`. The
                weights are assumed to be in units of MJy/sr.
            lonlat: If True, input angles (`theta`, `phi`) are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        theta, phi = validate_ang(theta=theta, phi=phi, lonlat=lonlat)

        unique_angles, indicies = np.unique(
            np.asarray([theta, phi]), return_inverse=True, axis=1
        )
        unit_vectors = get_unit_vectors_from_ang(
            coord_in=coord_in,
            theta=unique_angles[0],
            phi=unique_angles[1],
            lonlat=lonlat,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
            return_comps=return_comps,
        )

    def get_emission_pix(
        self,
        freq: FrequencyOrWavelength,
        pixels: Pixels,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: u.Quantity[u.MJy / u.sr] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated zodiacal emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission.
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The
                weights are assumed to be in units of MJy/sr.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        pixels = validate_pixels(pixels=pixels, nside=nside)

        unique_pixels, indicies = np.unique(pixels, return_inverse=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=indicies,
            pixels=unique_pixels,
            nside=nside,
            return_comps=return_comps,
        )

    def get_binned_emission_ang(
        self,
        freq: FrequencyOrWavelength,
        theta: SkyAngles,
        phi: SkyAngles,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: u.Quantity[u.MJy / u.sr] | None = None,
        lonlat: bool = False,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned zodiacal emission given angles on the sky.

        The pointing, for which to compute the emission, is specified in form of angles on
        the sky given by `theta` and `phi`. The emission is binned to a HEALPix map with
        resolution given by `nside`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission.
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
            weights: Bandpass weights corresponding the the frequencies in `freq`. The
                weights are assumed to be in units of MJy/sr.
            lonlat: If True, input angles `theta`, `phi` are assumed to be longitude and
                latitude, otherwise, they are co-latitude and longitude.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        theta, phi = validate_ang(theta=theta, phi=phi, lonlat=lonlat)

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

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=counts,
            binned=True,
            pixels=unique_pixels,
            nside=nside,
            return_comps=return_comps,
        )

    def get_binned_emission_pix(
        self,
        freq: FrequencyOrWavelength,
        pixels: Pixels,
        nside: int,
        obs_time: Time,
        obs: str = "earth",
        obs_pos: u.Quantity[u.AU] | None = None,
        weights: u.Quantity[u.MJy / u.sr] | None = None,
        return_comps: bool = False,
        coord_in: Literal["E", "G", "C"] = "E",
    ) -> u.Quantity[u.MJy / u.sr]:
        """Returns the simulated binned zodiacal Emission given pixel numbers.

        The pixel numbers represent the pixel indicies on a HEALPix grid with resolution
        given by `nside`. The emission is binned to a HEALPix map with resolution given by
        `nside`.

        Args:
            freq: Delta frequency/wavelength or a sequence of frequencies corresponding to
                a bandpass over which to evaluate the zodiacal emission.
            pixels: HEALPix pixel indicies representing points on the celestial sphere.
            nside: HEALPix resolution parameter of the pixels and the binned map.
            obs_time: Time of observation.
            obs: Name of the Solar System observer. A list of all support observers (for a
                given ephemeridis) is specified in `supported_observers` attribute of the
                `Zodipy` instance. Defaults to 'earth'.
            obs_pos: The heliocentric ecliptic cartesian position of the observer in AU.
                Overrides the `obs` argument. Default is None.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The
                weights are assumed to be in units of MJy/sr.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            coord_in: Coordinate frame of the input pointing. Assumes 'E' (ecliptic
                coordinates) by default.

        Returns:
            emission: Simulated zodiacal emission in units of 'MJy/sr'.

        """

        pixels = validate_pixels(pixels=pixels, nside=nside)

        unique_pixels, counts = np.unique(pixels, return_counts=True)
        unit_vectors = get_unit_vectors_from_pixels(
            coord_in=coord_in,
            pixels=unique_pixels,
            nside=nside,
        )

        return self._compute_emission(
            freq=freq,
            weights=weights,
            obs=obs,
            obs_time=obs_time,
            obs_pos=obs_pos,
            unit_vectors=unit_vectors,
            indicies=counts,
            binned=True,
            pixels=unique_pixels,
            nside=nside,
            return_comps=return_comps,
        )

    def _compute_emission(
        self,
        freq: FrequencyOrWavelength,
        weights: u.Quantity[u.MJy / u.sr] | None,
        obs: str,
        obs_time: Time,
        unit_vectors: npt.NDArray[np.float64],
        indicies: npt.NDArray[np.int64],
        binned: bool = False,
        obs_pos: u.Quantity[u.AU] | None = None,
        pixels: npt.NDArray[np.int64] | None = None,
        nside: int | None = None,
        return_comps: bool = False,
    ) -> u.Quantity[u.MJy / u.sr]:
        """Computes the component-wise zodiacal emission."""

        if not self.extrapolate:
            validate_frequency_in_model_range(freq=freq, model=self.ipd_model)

        normalized_weights = validate_and_normalize_weights(freq=freq, weights=weights)

        interpolated_source_params = interpolate_source_parameters(
            model=self.ipd_model, freq=freq, weights=normalized_weights
        )

        observer_position, earth_position = get_obs_and_earth_positions(
            obs=obs, obs_time=obs_time, obs_pos=obs_pos
        )

        start, stop = get_line_of_sight_endpoints(
            cutoff=self.los_dist_cut.value,
            obs_pos=observer_position,
            unit_vectors=unit_vectors,
        )

        emission = np.zeros(
            (self.ipd_model.n_comps, hp.nside2npix(nside) if binned else indicies.size)
        )

        # Convert to Hz if `freq` is in units of wavelength
        if not freq.unit.is_equivalent(u.Hz):
            freq = freq.to(u.Hz, u.spectral())

            # Flip weights if `freq` is in units of wavelength
            if normalized_weights is not None:
                normalized_weights = np.flip(normalized_weights) / np.trapz(
                    normalized_weights, freq.value
                )

        # Some components require additional non-static parameters to be computed, such as
        # the earth-trailing feature which needs the earth position in addition to the
        # model parameters.
        computed_parameters = {
            "X_earth": np.expand_dims(earth_position, axis=-1),
        }
        # Construct component density functions
        density_partials = construct_density_partials(
            comps=list(self.ipd_model.comps.values()),
            computed_params=computed_parameters,
        )
        # Create partial function with fixed arguments. This partial will again be used
        # as the basis for another partial depnding on wether or not the code is run in
        # parallel.
        partial_common_integrand = partial(
            _get_emission_at_step,
            start=start,
            gauss_quad_degree=self.gauss_quad_degree,
            X_obs=np.expand_dims(observer_position, axis=-1),
            density_partials=density_partials,
            freq=freq.value,
            weights=normalized_weights,
            T_0=self.ipd_model.T_0,
            delta=self.ipd_model.delta,
            **asdict(interpolated_source_params),
        )
        # Distribute pointing to CPUs and compute the emission in parallel.
        if self.parallel:
            n_proc = multiprocessing.cpu_count() if self.n_proc is None else self.n_proc

            unit_vector_chunks = np.array_split(unit_vectors, n_proc, axis=-1)
            stop_chunks = np.array_split(stop, n_proc, axis=-1)

            with multiprocessing.get_context(SYS_PROC_START_METHOD).Pool(
                processes=n_proc
            ) as pool:
                # Create a partial functions that will be passed to the quadpy integration
                # scheme. Arrays are reshaped to (d, n, p) for broadcasting purposes where
                # d is the geometrical dimensionality, n is the number of different
                # pointings, and p is the number of integration points of the quadrature.
                partial_integrand_chunks = [
                    partial(
                        partial_common_integrand,
                        stop=np.expand_dims(stop, axis=-1),
                        u_los=np.expand_dims(unit_vectors, axis=-1),
                    )
                    for unit_vectors, stop in zip(unit_vector_chunks, stop_chunks)
                ]

                proc_chunks = [
                    pool.apply_async(
                        self._integration_scheme.integrate,
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

            partial_integrand = partial(
                partial_common_integrand,
                stop=stop_expanded,
                u_los=unit_vectors_expanded,
            )

            integrated_comp_emission = self._integration_scheme.integrate(
                partial_integrand, [-1, 1]
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

        emission = (emission << SPECIFIC_INTENSITY_UNITS).to(u.MJy / u.sr)

        return emission if return_comps else emission.sum(axis=0)

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for attribute_name, attribute in self.__dict__.items():
            if attribute_name.startswith("_"):
                continue
            repr_str += f"{attribute_name}={attribute!r}, "

        return repr_str[:-2] + ")"


def _get_emission_at_step(
    r: float | npt.NDArray[np.float64],
    start: float,
    stop: float | npt.NDArray[np.float64],
    gauss_quad_degree: int,
    X_obs: npt.NDArray[np.float64],
    u_los: npt.NDArray[np.float64],
    density_partials: tuple[PartialComputeDensityFunc],
    freq: float | npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64] | None,
    T_0: float,
    delta: float,
    emissivities: npt.NDArray[np.float64],
    albedos: npt.NDArray[np.float64],
    phase_coefficients: tuple[float, ...],
    solar_irradiance: float,
) -> npt.NDArray[np.float64]:
    """Returns the zodiacal emission at a step along all lines of sight."""

    # Convert the line of sight range from [-1, 1] to the true ecliptic positions
    R_los = ((stop - start) / 2) * r + (stop + start) / 2

    X_los = R_los * u_los
    X_helio = X_los + X_obs
    R_helio = np.sqrt(X_helio[0] ** 2 + X_helio[1] ** 2 + X_helio[2] ** 2)

    temperature = get_dust_grain_temperature(R_helio, T_0, delta)

    if weights is not None:
        blackbody_emission = get_bandpass_integrated_blackbody_emission(
            freq=freq,
            weights=weights,
            T=temperature,
        )
    else:
        blackbody_emission = get_blackbody_emission(freq=freq, T=temperature)

    emission = np.zeros(
        (len(density_partials), np.shape(X_helio)[1], gauss_quad_degree)
    )
    density = np.zeros_like(emission)
    for idx, (get_density_func, albedo, emissivity) in enumerate(
        zip(density_partials, albedos, emissivities)
    ):
        density[idx] = get_density_func(X_helio)
        emission[idx] = (1 - albedo) * (emissivity * blackbody_emission)

    if any(albedo != 0 for albedo in albedos):
        solar_flux = solar_irradiance / R_helio**2
        scattering_angle = get_scattering_angle(R_los, R_helio, X_los, X_helio)
        phase_function = get_phase_function(scattering_angle, phase_coefficients)

        for idx, albedo in enumerate(albedos):
            emission[idx] += albedo * solar_flux * phase_function

    return emission * density
