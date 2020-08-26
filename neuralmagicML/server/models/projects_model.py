"""
DB model classes for a project's model file
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
from playhouse.sqlite_ext import JSONField

from neuralmagicML.utils import create_dirs
from neuralmagicML.server.models.projects import (
    BaseProjectModel,
    Project,
    PROJECTS_DIR_NAME,
)
from neuralmagicML.server.models.jobs import Job


__all__ = ["ProjectModel", "PROJECTS_MODEL_DIR_NAME"]


_LOGGER = logging.getLogger(__name__)
PROJECTS_MODEL_DIR_NAME = "model"


class ProjectModel(BaseProjectModel):
    """
    DB model for a project's model file.
    A project must have only one model file stored in the DB.
    """

    model_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    project = ForeignKeyField(Project, unique=True, backref="models")
    created = DateTimeField(default=datetime.datetime.now)
    source = TextField(null=True, default=None)
    job = ForeignKeyField(Job, null=True, default=None)
    file = TextField(null=True, default=None)
    analysis_job = ForeignKeyField(Job, null=True, default=None)
    analysis = JSONField(null=True, default=None)

    @property
    def dir_path(self) -> str:
        """
        :return: the local directory path for where the model file is stored
        """
        project_id = self.project_id  # type: str

        return os.path.join(
            self._meta.storage.root_path,
            PROJECTS_DIR_NAME,
            project_id,
            PROJECTS_MODEL_DIR_NAME,
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
                "project model directory at {} does not exist anymore".format(
                    self.dir_path
                )
            )

        if self.file and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                "project model file at {} does not exist anymore".format(self.file_path)
            )
