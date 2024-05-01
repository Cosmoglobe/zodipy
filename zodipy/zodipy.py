from __future__ import annotations

import functools
import multiprocessing
import platform
from typing import TYPE_CHECKING, Callable

import numpy as np
from astropy import coordinates as coords
from astropy import units
from scipy import integrate

from zodipy._coords import get_earth_skycoord, get_obs_skycoord
from zodipy._emission import EMISSION_MAPPING
from zodipy._ipd_comps import ComponentLabel
from zodipy._ipd_dens_funcs import construct_density_partials_comps
from zodipy._line_of_sight import get_line_of_sight_range
from zodipy.blackbody import tabulate_bandpass_integrated_bnu, tabulate_center_wavelength_bnu
from zodipy.interpolate import get_model_to_dicts_callable
from zodipy.model_registry import model_registry

if TYPE_CHECKING:
    import numpy.typing as npt

PLATFORM = platform.system().lower()
SYS_PROC_START_METHOD = "fork" if "windows" not in PLATFORM else None


class Model:
    """Main interface to ZodiPy.

    The zodiacal light simulations are configured by specifying a bandpass (`freq`, `weights`)
    or a delta/center frequency (`freq`), and a string representation of a built in interplanetary
    dust model (`model`). See https://cosmoglobe.github.io/zodipy/introduction/ for a list of
    available models.
    """

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
    ) -> None:
        """Initialize the Zodipy interface.

        Args:
            x: Wavelength or frequency. If `x` is a array-like it is assumed to be a bandpass and
                `weights` must be provided.
            weights: Bandpass weights corresponding the the frequencies in `freq`. The weights are
                assumed to be given in spectral radiance units (Jy/sr).
            name: Interplanetary dust model to use. Defaults to 'dirbe' which is the DIRBE model.
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
        """
        if not x.isscalar and weights is None:
            msg = "Several wavelengths are provided by no weights."
            raise ValueError(msg)
        if x.isscalar and weights is not None:
            msg = "A single wavelength is provided with weights."
            raise ValueError(msg)

        self._interplanetary_dust_model = model_registry.get_model(name)
        if not extrapolate and not self._interplanetary_dust_model.is_valid_at(x):
            msg = (
                "The requested frequencies are outside the valid range of the model."
                "If this was intended, set `extrapolate=True`."
            )
            raise ValueError(msg)

        bandpass_is_provided = weights is not None
        if bandpass_is_provided:
            weights = np.asarray(weights)
            if x.size != weights.size:
                msg = "Number of wavelengths and weights must be the same in the bandpass."
                raise ValueError(msg)
            normalized_weights = weights / integrate.trapezoid(weights, x)
            self._b_nu_table = tabulate_bandpass_integrated_bnu(x, normalized_weights)
        else:
            self._b_nu_table = tabulate_center_wavelength_bnu(x)
            normalized_weights = None

        # Interpolate and convert the model parameters to dictionaries which can be used to evaluate
        # the zodiacal light model.
        self._comp_parameters, self._common_parameters = get_model_to_dicts_callable(
            self._interplanetary_dust_model
        )(x, normalized_weights, self._interplanetary_dust_model)

        self._ephemeris = ephemeris
        self._n_proc = n_proc
        self._gauss_points_and_weights = np.polynomial.legendre.leggauss(gauss_quad_degree)

    def evaluate(
        self,
        skycoord: coords.SkyCoord,
        *,
        obspos: units.Quantity | str = "earth",
        return_comps: bool = False,
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

        Returns:
            emission: Simulated zodiacal light in units of 'MJy/sr'.

        """
        (unique_lon, unique_lat), indicies = np.unique(
            np.vstack([skycoord.spherical.lon.value, skycoord.spherical.lat.value]),
            return_inverse=True,
            axis=1,
        )

        if skycoord.obstime is None:
            msg = "The `obstime` attribute of the `SkyCoord` object must be set."
            raise ValueError(msg)

        skycoord = coords.SkyCoord(
            unique_lon,
            unique_lat,
            unit=units.deg,
            frame=skycoord.frame,
            obstime=skycoord.obstime,
        )

        earth_skycoord = get_earth_skycoord(skycoord.obstime, ephemeris=self._ephemeris)
        obs_skycoord = get_obs_skycoord(
            obspos, skycoord.obstime, earth_skycoord, ephemeris=self._ephemeris
        )

        skycoord = skycoord.transform_to(coords.BarycentricMeanEcliptic)

        start, stop = get_line_of_sight_range(
            components=self._interplanetary_dust_model.comps.keys(),
            unit_vectors=skycoord.cartesian.xyz.value,
            obs_pos=obs_skycoord.cartesian.xyz.to_value(units.AU),
        )

        density_partials = construct_density_partials_comps(
            comps=self._interplanetary_dust_model.comps,
            dynamic_params={
                "X_earth": earth_skycoord.cartesian.xyz.to_value(units.AU)[
                    :, np.newaxis, np.newaxis
                ]
            },
        )

        common_integrand = functools.partial(
            EMISSION_MAPPING[type(self._interplanetary_dust_model)],
            X_obs=obs_skycoord.cartesian.xyz.to_value(units.AU)[:, np.newaxis, np.newaxis],
            bp_interpolation_table=self._b_nu_table,
            **self._common_parameters,
        )

        distribute_to_cores = self._n_proc > 1 and skycoord.size > self._n_proc
        if distribute_to_cores:
            unit_vector_chunks = np.array_split(skycoord.cartesian.xyz.value, self._n_proc, axis=-1)
            integrated_comp_emission = np.zeros(
                (self._interplanetary_dust_model.ncomps, skycoord.size)
            )

            with multiprocessing.get_context(SYS_PROC_START_METHOD).Pool(
                processes=self._n_proc
            ) as pool:
                for idx, comp_label in enumerate(self._interplanetary_dust_model.comps.keys()):
                    stop_chunks = np.array_split(stop[comp_label], self._n_proc, axis=-1)
                    if start[comp_label].size == 1:
                        start_chunks = [start[comp_label]] * self._n_proc
                    else:
                        start_chunks = np.array_split(start[comp_label], self._n_proc, axis=-1)
                    comp_integrands = [
                        functools.partial(
                            common_integrand,
                            u_los=np.expand_dims(unit_vectors, axis=-1),
                            start=np.expand_dims(start, axis=-1),
                            stop=np.expand_dims(stop, axis=-1),
                            get_density_function=density_partials[comp_label],
                            **self._comp_parameters[comp_label],
                        )
                        for unit_vectors, start, stop in zip(
                            unit_vector_chunks, start_chunks, stop_chunks
                        )
                    ]

                    proc_chunks = [
                        pool.apply_async(
                            _integrate_gauss_quad,
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
            integrated_comp_emission = np.zeros(
                (self._interplanetary_dust_model.ncomps, skycoord.size)
            )
            unit_vectors_expanded = np.expand_dims(skycoord.cartesian.xyz.value, axis=-1)

            for idx, comp_label in enumerate(self._interplanetary_dust_model.comps.keys()):
                comp_integrand = functools.partial(
                    common_integrand,
                    u_los=unit_vectors_expanded,
                    start=np.expand_dims(start[comp_label], axis=-1),
                    stop=np.expand_dims(stop[comp_label], axis=-1),
                    get_density_function=density_partials[comp_label],
                    **self._comp_parameters[comp_label],
                )

                integrated_comp_emission[idx] = (
                    _integrate_gauss_quad(comp_integrand, *self._gauss_points_and_weights)
                    * 0.5
                    * (stop[comp_label] - start[comp_label])
                )

        emission = np.zeros((self._interplanetary_dust_model.ncomps, indicies.size))
        emission = integrated_comp_emission[:, indicies] << (units.MJy / units.sr)

        return emission if return_comps else emission.sum(axis=0)

    @property
    def supported_observers(self) -> list[str]:
        """Return a list of available observers given an ephemeris."""
        with coords.solar_system_ephemeris.set(self._ephemeris):
            return [*list(coords.solar_system_ephemeris.bodies), "semb-l2"]

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

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        for attribute_name, attribute in self.__dict__.items():
            if attribute_name.startswith("_"):
                continue
            repr_str += f"{attribute_name}={attribute!r}, "

        return repr_str[:-2] + ")"


def _integrate_gauss_quad(
    fn: Callable[[float], npt.NDArray[np.float64]],
    points: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Integrate a function using Gauss-Legendre quadrature."""
    return np.squeeze(sum(fn(x) * w for x, w in zip(points, weights)))
