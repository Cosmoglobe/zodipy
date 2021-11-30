class ModelNotFoundError(Exception):
    """Raised if a pre-initialized IPD model is not found."""

class SimulationStrategyNotFoundError(Exception):
    """Raised if a non implemented simulation stategy is selected."""