"""
Code related to the Singleton design pattern
"""

__all__ = ["Singleton"]


class Singleton(type):
    """
    A singleton class implementation meant to be added to others
    as a metaclass.

    Ex: class Logger(metaclass=Singleton)
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]
