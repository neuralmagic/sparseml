"""
Flask blueprint setup for serving UI files for the server application
"""

__all__ = ["API_ROOT_PATH", "HTTPNotFoundError"]


API_ROOT_PATH = "/api"


class HTTPNotFoundError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
