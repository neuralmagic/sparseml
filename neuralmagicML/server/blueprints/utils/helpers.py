"""
Helper functions and classes for flask blueprints
"""


__all__ = [
    "API_ROOT_PATH",
    "HTTPNotFoundError",
]


API_ROOT_PATH = "/api"


class HTTPNotFoundError(Exception):
    """
    Expected error raised when a 404 should be encountered by the user
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
