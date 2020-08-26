"""
Base DB model classes for the server
"""

import logging

from peewee import Model
from playhouse.pool import PooledSqliteDatabase


__all__ = ["FileStorage", "database", "storage", "BaseModel"]

_LOGGER = logging.getLogger(__name__)


class FileStorage(object):
    """
    Class for handling local file storage and the path that is located at.
    Used for storing large files that would not be good in the DB
    such as model and data files.
    """
    def __init__(self):
        self._root_path = None

    @property
    def root_path(self) -> str:
        """
        :return: the root path on the local file system for where to store files
        """
        self._validate_setup()

        return self._root_path

    def init(self, root_path: str):
        """
        Initialize the file storage class for a given path

        :param root_path: the root path on the local file system
            for where to store files
        """
        self._root_path = root_path

    def _validate_setup(self):
        if self._root_path is None:
            raise ValueError("root_path is not set, call init first")


database = PooledSqliteDatabase(None)
storage = FileStorage()


class BaseModel(Model):
    """
    Base peewee model all DB models must extend from
    """

    class Meta(object):
        database = database
        storage = storage
