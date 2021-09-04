from typing import Tuple, Dict, Union

import numpy as np

LOSConfigType = Dict[str, Union[np.ndarray, Tuple[float, int]]]


class LOSFactory:
    """Factory responsible for registering and book-keeping of LOS configs."""

    def __init__(self) -> None:
        self._configs = {}

    def register_config(self, name: str, components: LOSConfigType) -> None:
        """Initializes and stores a LOS."""

        error_msg = (
            "Line-of-sight config must either be an array, or a tuple with "
            "the format (start, stop, n, geom) where geom is either "
            "'linear' or 'log'"
        )
        config = {}
        for key, value in components.items():
            if isinstance(value, np.ndarray):
                config[key] = value
            elif isinstance(value, tuple):
                try:
                    start, stop, n, geom = value
                except ValueError:
                    raise ValueError(error_msg)
                if geom.lower() == "linear":
                    geom = np.linspace
                elif geom.lower() == "log":
                    geom = np.geomspace
                else:
                    raise ValueError(error_msg)
                config[key] = geom(start, stop, n)
            else:
                raise ValueError(error_msg)
        self._configs[name] = config

    def get_config(self, name: str) -> Dict[str, np.ndarray]:
        """Returns a registered config."""

        config = self._configs.get(name)
        if config is None:
            raise ValueError(
                f"Config {name} is not registered. Available configs are "
                f"{list(self._configs.keys())}"
            )
        return config
