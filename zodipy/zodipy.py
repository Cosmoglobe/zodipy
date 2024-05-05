from __future__ import annotations

import functools
import multiprocessing
import multiprocessing.pool
import platform
from typing import cast

import numpy as np
import numpy.typing as npt
from astropy import coordinates as coords
from astropy import time, units
from scipy import integrate

from zodipy.blackbody import tabulate_blackbody_emission
from zodipy.bodies import get_earthpos_xyz, get_obspos_xyz
from zodipy.line_of_sight import get_line_of_sight_range, integrate_gauss_legendre
from zodipy.model_registry import model_registry
from zodipy.number_density import populate_number_density_with_model
from zodipy.unpack_model import get_model_to_dicts_callable
from zodipy.zodiacal_component import ComponentLabel

_platform_start_method = "fork" if "windows" not in platform.system().lower() else None


class Model:
    """Main interface to ZodiPy."""

    def __init__(
        self,
        x: units.Quantity,
        *,
        weights: npt.ArrayLike | None = None,
        name: str = "dirbe",
        gauss_quad_degree: int = 50,
        extrapolate: bool = False,
        ephemeris: str = "builtin",
        n_proc: int = 1,
        pool: multiprocessing.pool.Pool | None = None,
    ) -> None:
        """Initialize the Zodipy interface.

        Args:
            x: Wavelength or frequency. If `x` is a array-like it is assumed to be a bandpass and
                `weights` must be provided.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights are
                assumed to be given in spectral radiance units (Jy/sr).
            name: Interplanetary dust model to use. For a list of available models, see
                https://cosmoglobe.github.io/zodipy/introduction/. Defaults to 'dirbe'.
            gauss_quad_degree: Order of the Gaussian-Legendre quadrature used to evaluate the
                line-of-sight integral in the simulations. Default is 50 points.
            extrapolate: If `True` all spectral quantities in the selected model are extrapolated to
                the requested frequencies or wavelengths. If `False`, an exception is raised on
                requested frequencies/wavelengths outside of the valid model range. Default is
                `False`.
            ephemeris: Ephemeris used in `astropy.coordinates.solar_system_ephemeris` to compute the
                positions of the observer and the Earth. Defaults to 'builtin'. See the
                [Astropy documentation](https://docs.astropy.org/en/stable/coordinates/solarsystem.html)
                for available ephemerides.
            n_proc: Number of cores to use. If `n_proc` is greater than 1, the line-of-sight
                integrals are parallelized using the `multiprocessing` module. Defaults to 1.
            pool: Custom multiprocessing pool to use. If `None`, a new pool is created. Defaults to
                `None`.

        """
        try:
            if not x.isscalar and weights is None:
                msg = "Several wavelengths are provided by no weights."
                raise ValueError(msg)
        except AttributeError as error:
            msg = "The input 'x' must be an astropy Quantity."
            raise TypeError(msg) from error
        if x.isscalar and weights is not None:
            msg = "A single wavelength is provided with weights."
            raise ValueError(msg)

        self._interplanetary_dust_model = model_registry.get_model(name)

        if not extrapolate and not self._interplanetary_dust_model.is_valid_at(x):
            msg = (
                "The requested frequencies are outside the valid range of the model. "
                "If this was intended, set the extrapolate argument to True."
            )
            raise ValueError(msg)

        bandpass_is_provided = weights is not None
        if bandpass_is_provided:
            weights = np.asarray(weights)
            if x.size != weights.size:
                msg = "Number of wavelengths and weights must be the same in the bandpass."
                raise ValueError(msg)
            normalized_weights = weights / integrate.trapezoid(weights, x)
        else:
            normalized_weights = None
        self._b_nu_table = tabulate_blackbody_emission(x, normalized_weights)

        # Interpolate and convert the model parameters to dictionaries which can be used to evaluate
        # the zodiacal light model.
        brightness_callable_dicts = get_model_to_dicts_callable(self._interplanetary_dust_model)(
            x, normalized_weights, self._interplanetary_dust_model
        )
        self._comp_parameters, self._common_parameters = brightness_callable_dicts

        self._ephemeris = ephemeris
        self._gauss_points_and_weights = np.polynomial.legendre.leggauss(gauss_quad_degree)

        if pool is None:
            self._process_pool = multiprocessing.get_context(_platform_start_method).Pool(n_proc)
            self._n_proc = n_proc
        else:
            self._process_pool = pool
            self._n_proc = pool._processes  # type: ignore

    def evaluate(
        self,
        skycoord: coords.SkyCoord,
        *,
        obspos: units.Quantity | str = "earth",
        return_comps: bool = False,
        contains_duplicates: bool = False,
    ) -> units.Quantity[units.MJy / units.sr]:
        """Return the simulated zodiacal light for all observations in a `SkyCoord` object.

        Args:
            skycoord: `astropy.coordinates.SkyCoord` object representing the observations for which
                to simulate the zodiacal light. The `frame` and `obstime` attributes of the
                `SkyCoord` object must be set. The `obstime` attribute should correspond to a single
                observational time for which the zodiacal light is assumed to be stationary.
                Additionally, the frame must be convertible to the `BarycentricMeanEcliptic` frame.
            obspos: The heliocentric ecliptic position of the observer, or a string representing
                an observer in the `astropy.coordinates.solar_system_ephemeris`. This should
                correspond to a single position. Defaults to 'earth'.
            return_comps: If True, the emission is returned component-wise. Defaults to False.
            contains_duplicates: If True, the input coordinates are filtered and only unique
                pointing is used to calculate the emission. The output is then mapped back to the
                original coordinates resulting in the same output shape. Defaults to False.

        Returns:
            emission: Simulated zodiacal light in units of 'MJy/sr'.

        """
        try:
            obstime = cast(time.Time, skycoord.obstime)
        except AttributeError as error:
            msg = "The input coordinates must be an astropy SkyCoord object."
            raise TypeError(msg) from error
        if obstime is None:
            msg = "The `obstime` attribute of the `SkyCoord` object must be set."
            raise ValueError(msg)

        n_coords_in = skycoord.size
        if contains_duplicates:
            _, index, inverse = np.unique(
                cast(
                    list[npt.NDArray[np.float64]],
                    [skycoord.spherical.lon.value, skycoord.spherical.lat.value],
                ),
                return_index=True,
                return_inverse=True,
                axis=1,
            )
            skycoord = cast(coords.SkyCoord, skycoord[index])  # filter out identical coordinates
        n_coords = skycoord.size
        earth_xyz = get_earthpos_xyz(obstime, self._ephemeris)
        obs_xyz = get_obspos_xyz(obstime, obspos, earth_xyz, self._ephemeris)

        skycoord = skycoord.transform_to(coords.BarycentricMeanEcliptic)
        if skycoord.isscalar:
            skycoord_xyz = cast(
                npt.NDArray[np.float64], skycoord.cartesian.xyz.value[:, np.newaxis]
            )
        else:
            skycoord_xyz = cast(npt.NDArray[np.float64], skycoord.cartesian.xyz.value)

        start, stop = get_line_of_sight_range(
            components=self._interplanetary_dust_model.comps.keys(),
            unit_vectors=skycoord_xyz,
            obs_pos=obs_xyz,
        )

        # Return a dict of partial functions corresponding to the number density each zodiacal
        # component in the interplanetary dust model.
        density_partials = populate_number_density_with_model(
            comps=self._interplanetary_dust_model.comps,
            dynamic_params={"X_earth": earth_xyz[:, np.newaxis, np.newaxis]},
        )

        # Partial function of the brightness integral at a step along the line-of-sight prepopulated
        # with shared arguments between zodiacal components.
        common_integrand = functools.partial(
            self._interplanetary_dust_model.brightness_at_step_callable,
            X_obs=obs_xyz[:, np.newaxis, np.newaxis],
            bp_interpolation_table=self._b_nu_table,
            **self._common_parameters,
        )

        distribute_to_cores = n_coords > self._n_proc and self._n_proc > 1
        if distribute_to_cores:
            unit_vector_chunks = np.array_split(skycoord_xyz, self._n_proc, axis=-1)
            integrated_comp_emission = np.zeros((self._interplanetary_dust_model.ncomps, n_coords))
            with self._process_pool as pool:
                for idx, comp_label in enumerate(self._interplanetary_dust_model.comps.keys()):
                    stop_chunks = np.array_split(stop[comp_label], self._n_proc, axis=-1)
                    if start[comp_label].size == 1:
                        start_chunks = [start[comp_label]] * self._n_proc
                    else:
                        start_chunks = np.array_split(start[comp_label], self._n_proc, axis=-1)
                    comp_integrands = [
                        functools.partial(
                            common_integrand,
                            u_los=np.expand_dims(unit_vector, axis=-1),
                            start=np.expand_dims(start, axis=-1),
                            stop=np.expand_dims(stop, axis=-1),
                            get_density_function=density_partials[comp_label],
                            **self._comp_parameters[comp_label],
                        )
                        for unit_vector, start, stop in zip(
                            unit_vector_chunks, start_chunks, stop_chunks
                        )
                    ]

                    proc_chunks = [
                        pool.apply_async(
                            integrate_gauss_legendre,
                            args=(comp_integrand, *self._gauss_points_and_weights),
                        )
                        for comp_integrand in comp_integrands
                    ]

                    integrated_comp_emission[idx] += (
                        np.concatenate([result.get() for result in proc_chunks])
                        * 0.5
                        * (stop[comp_label] - start[comp_label])
                    )

        else:
            integrated_comp_emission = np.zeros((self._interplanetary_dust_model.ncomps, n_coords))
            for idx, comp_label in enumerate(self._interplanetary_dust_model.comps.keys()):
                comp_integrand = functools.partial(
                    common_integrand,
                    u_los=np.expand_dims(skycoord_xyz, axis=-1),
                    start=np.expand_dims(start[comp_label], axis=-1),
                    stop=np.expand_dims(stop[comp_label], axis=-1),
                    get_density_function=density_partials[comp_label],
                    **self._comp_parameters[comp_label],
                )

                integrated_comp_emission[idx] = (
                    integrate_gauss_legendre(comp_integrand, *self._gauss_points_and_weights)
                    * 0.5
                    * (stop[comp_label] - start[comp_label])
                )

        if contains_duplicates:
            emission = np.zeros((self._interplanetary_dust_model.ncomps, n_coords_in))
            emission = integrated_comp_emission[:, inverse] << (units.MJy / units.sr)
        else:
            emission = integrated_comp_emission << (units.MJy / units.sr)

        return emission if return_comps else emission.sum(axis=0)

    def get_parameters(self) -> dict:
        """Return a dictionary containing the interplanetary dust model parameters."""
        return self._interplanetary_dust_model.to_dict()

    def update_parameters(self, parameters: dict) -> None:
        """Update the interplanetary dust model parameters.

        Args:
            parameters: Dictionary of parameters to update. The keys must be the names
                of the parameters as defined in the model. To get the parameters dict
                of an existing model, use `Zodipy("dirbe").get_parameters()`.

        """
        _dict = parameters.copy()
        _dict["comps"] = {}
        for key, value in parameters.items():
            if key == "comps":
                for comp_key, comp_value in value.items():
                    _dict["comps"][ComponentLabel(comp_key)] = type(
                        self._interplanetary_dust_model.comps[ComponentLabel(comp_key)]
                    )(**comp_value)
            elif isinstance(value, dict):
                _dict[key] = {ComponentLabel(k): v for k, v in value.items()}

        self._interplanetary_dust_model = self._interplanetary_dust_model.__class__(**_dict)
