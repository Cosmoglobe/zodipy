from typing import Tuple, Dict

import numpy as np


class LOSFactory:
    """Factory responsible for registring and book-keeping a line-of-sight (LOS)."""

    def __init__(self) -> None:
        self._configs = {}

    def register_config(
        self, name: str, components: Dict[str, Tuple[float, int]]
    ) -> None:
        """Initializes and stores a LOS."""

        config = {}
        for key, value in components.items():
            if isinstance(value, np.ndarray):
                config[key] = value
            elif isinstance(value, (tuple, list)):
                try:
                    start, stop, n, geom = value
                except ValueError:
                    raise ValueError(
                        "Line-of-sight config must either be an array, or "
                        "a tuple with the format (start, stop, n, geom)"
                        "where geom is either 'linear' or 'log'"
                    )
                if geom.lower() == "linear":
                    geom = np.linspace
                elif geom.lower() == "log":
                    geom = np.geomspace
                else:
                    raise ValueError("geom must be either 'linear' or 'log'")
                config[key] = geom(start, stop, n)

        self._configs[name] = config

    def get_config(self, name: str) -> np.ndarray:
        """Returns a registered config."""

        config = self._configs.get(name)
        if config is None:
            raise ValueError(
                f"Config {name} is not registered. Available configs are "
                f"{list(self._configs.keys())}"
            )

        return config
