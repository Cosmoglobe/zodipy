class ModelNotFoundError(Exception):
    """Raised if a pre-initialized IPD model is not found."""

class SimulationStrategyNotFoundError(Exception):
    """Raised if a non implemented simulation stategy is selected."""

class TargetNotSupportedError(Exception):
    """Raised when a target that is not supported is selected."""

class ComponentNotImplemented(Exception):
    """Raised when one attempts to access a non implemented component."""