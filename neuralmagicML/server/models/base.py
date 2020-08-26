"""
Base DB model classes for the server
"""

import logging
import datetime

from peewee import Model, TextField, DateTimeField
from playhouse.sqlite_ext import JSONField
from playhouse.pool import PooledSqliteDatabase


__all__ = [
    "FileStorage",
    "database",
    "storage",
    "BaseModel",
    "BaseCreatedModifiedModel",
    "ListObjField",
    "CSVField",
    "CSVIntField",
    "CSVFloatField",
]

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


class BaseCreatedModifiedModel(BaseModel):
    """
    Base peewee model that includes created and modified timestamp functionality
    """

    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)

    def save(self, *args, **kwargs):
        self.modified = datetime.datetime.now()

        return super().save(*args, **kwargs)


class ListObjField(JSONField):
    """
    Field for handling lists of objects in a peewee database
    """

    def db_value(self, value):
        if value:
            value = {"list": value}

        return super().db_value(value)

    def python_value(self, value):
        value = super().python_value(value)

        return value["list"] if value else []


class CSVField(TextField):
    """
    CSV field for handling lists of strings in a peewee database
    """

    def db_value(self, value):
        if value:
            value = ",".join([str(v) for v in value])

        return value

    def python_value(self, value):
        return value.split(",") if value else []


class CSVIntField(CSVField):
    """
    CSV field for handling lists of integers in a peewee database
    """

    def python_value(self, value):
        return [int(v) for v in value.split(",")] if value else []


class CSVFloatField(CSVField):
    """
    CSV field for handling lists of floats in a peewee database
    """

    def python_value(self, value):
        return [float(v) for v in value.split(",")] if value else []
