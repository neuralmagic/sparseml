from enum import Enum
__all__ = [
    "Verbosity",
           ]

class Verbosity(Enum):
    DEFAULT = 1
    ON_LR_CHANGE = 2
    ON_EPOCH_CHANGE = 3
    ON_LR_OR_EPOCH_CHANGE = 4
    OFF = 0

    @staticmethod
    def convert_int_to_verbosity(value):
        try:
            return value if isinstance(value, Verbosity) else Verbosity(value)
        except ValueError:
            raise ValueError(f"Verbosity levels range from 0-4 got {value}")


