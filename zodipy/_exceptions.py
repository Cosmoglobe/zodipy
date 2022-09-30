class FrequencyOutOfBoundsError(Exception):
    """Raise when the user inputed frequency is out of bounds of the model."""

    def __init__(self, lower_limit: float, upper_limit: float):
        super().__init__()
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
