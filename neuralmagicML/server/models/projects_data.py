"""
DB model classes for a project's sample data files
"""

import logging
import os
import uuid
import datetime

from peewee import (
    CharField,
    TextField,
    DateTimeField,
    ForeignKeyField,
)

from neuralmagicML.utils import create_dirs
from neuralmagicML.server.models.projects import (
    BaseProjectModel,
    Project,
    PROJECTS_DIR_NAME,
)
from neuralmagicML.server.models.jobs import Job


__all__ = ["ProjectData", "PROJECTS_DATA_DIR_NAME"]


_LOGGER = logging.getLogger(__name__)
PROJECTS_DATA_DIR_NAME = "data"


class ProjectData(BaseProjectModel):
    """
    DB model for a project's data file.
    A project may have multiple data files stored in the DB.
    """

    data_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    project = ForeignKeyField(Project, backref="data")
    created = DateTimeField(default=datetime.datetime.now)
    file = TextField(null=True, default=None)
    source = TextField(null=True, default=None)
    job = ForeignKeyField(Job, null=True, default=None)

    @property
    def dir_path(self) -> str:
        """
        :return: the local directory path for where the data is stored
        """
        project_id = self.project_id  # type: str

        return os.path.join(
            self._meta.storage.root_path,
            PROJECTS_DIR_NAME,
            project_id,
            PROJECTS_DATA_DIR_NAME,
        )

    @property
    def file_path(self) -> str:
        """
        :return: the local file path to the data file
        """
        file_name = self.file  # type: str

        return os.path.join(self.dir_path, file_name)

    def setup_filesystem(self):
        """
        Setup the local file system so that it can be used with the data
        """
        create_dirs(self.dir_path)

    def validate_filesystem(self):
        """
        Validate that the local file system and expected files are correct and exist
        """
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(
                "project data directory at {} does not exist anymore".format(
                    self.dir_path
                )
            )

        if self.file and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                "project data file at {} does not exist anymore".format(self.file_path)
            )

    def delete_filesystem(self):
        """
        Delete the data file from the local file system
        """
        if self.file:
            os.remove(self.file_path)
