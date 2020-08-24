import logging

from peewee import Model
from playhouse.pool import PooledSqliteDatabase


__all__ = ["FileStorage", "database", "storage", "BaseModel"]

_LOGGER = logging.getLogger(__name__)


class FileStorage(object):
    def __init__(self):
        self._root_path = None

    @property
    def root_path(self) -> str:
        return self._root_path

    def init(self, root_path: str):
        self._root_path = root_path

    def _validate_setup(self):
        if self.root_path is None:
            raise ValueError("root_path is not set, call init first")


database = PooledSqliteDatabase(None)
storage = FileStorage()


class BaseModel(Model):
    class Meta(object):
        database = database
        storage = storage
