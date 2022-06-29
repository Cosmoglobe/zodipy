from functools import wraps
from typing import TYPE_CHECKING, Sequence, Union

import astropy.units as u
import healpy as hp
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .zodipy import Zodipy


def validate_freq(function):
    """
    Decorator that validates the user inputed frequency or wavelength. This
    function also converts the 'freq' argument to units of GHz.
    """

    @wraps(function)
    @u.quantity_input
    def wrapper(
        self,
        freq: Union[u.Quantity[u.Hz], u.Quantity[u.m]],
        *args,
        **kwargs,
    ):
        # `u.quantity_input` doesnt work with TYPE_CHECKING string annotations.
        zodipy: "Zodipy" = self

        if not zodipy.extrapolate:
            freq = freq.to(zodipy.model.spectrum.unit, equivalencies=u.spectral())
            spectrum_min = zodipy.model.spectrum.min()
            spectrum_max = zodipy.model.spectrum.max()

            if not (spectrum_min <= freq <= spectrum_max):
                raise ValueError(
                    f"model {zodipy.model.name!r} is only valid in the [{spectrum_min.value},"
                    f" {spectrum_max.value}] {zodipy.model.spectrum.unit} range."
                )

        freq = freq.to(u.GHz, equivalencies=u.spectral())

        return function(self=self, freq=freq, *args, **kwargs)

    return wrapper


def validate_ang(function):
    """Decorator that validates the user inputed angles."""

    @wraps(function)
    @u.quantity_input
    def wrapper(
        *args,
        theta: Union[u.Quantity[u.deg], u.Quantity[u.rad]],
        phi: Union[u.Quantity[u.deg], u.Quantity[u.rad]],
        **kwargs,
    ):
        lonlat = kwargs.get("lonlat")
        theta = theta.to(u.deg) if lonlat is not None else theta.to(u.rad)
        phi = phi.to(u.deg) if lonlat is not None else phi.to(u.rad)

        if theta.isscalar:
            theta = u.Quantity([theta])
        if phi.isscalar:
            phi = u.Quantity([phi])

        return function(theta=theta, phi=phi, *args, **kwargs)

    return wrapper


def validate_pixels(function):
    """Decorator that validates the user inputed pixels."""

    @wraps(function)
    def wrapper(
        *args,
        pixels: Union[int, Sequence[int], NDArray[np.integer]],
        nside: int,
        **kwargs,
    ):

        if (np.max(pixels) > hp.nside2npix(nside)) or (np.min(pixels) < 0):
            raise ValueError("invalid pixel number given nside")

        if np.ndim(pixels) == 0:
            pixels = np.expand_dims(pixels, axis=0)

        return function(pixels=pixels, nside=nside, *args, **kwargs)

    return wrapper
