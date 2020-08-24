"""
Base classes necessary for running background jobs
"""

from abc import abstractmethod
import logging
import os
import uuid
import datetime
import shutil

from peewee import (
    CharField,
    TextField,
    DateTimeField,
    ForeignKeyField,
    IntegerField,
    FloatField,
)

from neuralmagicML.utils import path_file_size, create_dirs
from neuralmagicML.server.models.base import BaseModel
from neuralmagicML.server.models.jobs import Job


__all__ = ["BaseProjectModel", "Project", "ProjectModel", "ProjectData"]


_LOGGER = logging.getLogger(__name__)
PROJECTS_DIR_NAME = "projects"
PROJECTS_MODEL_DIR_NAME = "model"
PROJECTS_DATA_DIR_NAME = "data"


class BaseProjectModel(BaseModel):
    @abstractmethod
    def setup_filesystem(self):
        raise NotImplementedError()

    @abstractmethod
    def validate_filesystem(self):
        raise NotImplementedError()


class Project(BaseProjectModel):
    project_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    name = TextField(null=True, default="")
    description = TextField(null=True, default="")
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    training_optimizer = TextField(null=True, default=None)
    training_epochs = IntegerField(null=True, default=None)
    training_lr_init = FloatField(null=True, default=None)
    training_lr_final = FloatField(null=True, default=None)

    @property
    def dir_path(self) -> str:
        project_id = self.project_id  # type: str

        return os.path.join(self._meta.storage.root_path, PROJECTS_DIR_NAME, project_id)

    @property
    def dir_size(self) -> int:
        try:
            return path_file_size(self.dir_path)
        except Exception:
            return 0

    def save(self, *args, **kwargs):
        self.modified = datetime.datetime.now()

        return super().save(*args, **kwargs)

    def setup_filesystem(self):
        create_dirs(self.dir_path)

    def validate_filesystem(self):
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(
                "project directory at {} does not exist anymore".format(self.dir_path)
            )

    def delete_filesystem(self):
        shutil.rmtree(self.dir_path)


class ProjectModel(BaseProjectModel):
    model_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    created = DateTimeField(default=datetime.datetime.now)
    file_name = TextField(null=True, default=None)
    file_source = TextField(null=True, default=None)
    analysis_file_name = TextField(null=True, default=None)

    file_source_job = ForeignKeyField(Job, null=True, default=None)
    project = ForeignKeyField(Project, backref="models")

    @property
    def dir_path(self) -> str:
        project_id = self.project_id  # type: str

        return os.path.join(
            self._meta.storage.root_path,
            PROJECTS_DIR_NAME,
            project_id,
            PROJECTS_MODEL_DIR_NAME,
        )

    @property
    def file_path(self) -> str:
        file_name = self.file_name  # type: str

        return os.path.join(self.dir_path, file_name)

    @property
    def analysis_file_path(self) -> str:
        analysis_file_name = self.analysis_file_name  # type: str

        return os.path.join(self.dir_path, analysis_file_name)

    def setup_filesystem(self):
        create_dirs(self.dir_path)

    def validate_filesystem(self):
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(
                "project model directory at {} does not exist anymore".format(
                    self.dir_path
                )
            )

        if self.file_name and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                "project model file at {} does not exist anymore".format(self.file_path)
            )

        if self.analysis_file_name and not os.path.exists(self.analysis_file_path):
            raise FileNotFoundError(
                "project model analysis file at {} does not exist anymore".format(
                    self.analysis_file_path
                )
            )


class ProjectData(BaseProjectModel):
    data_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    created = DateTimeField(default=datetime.datetime.now)
    file_name = TextField(null=True, default=None)
    file_source = TextField(null=True, default=None)

    file_source_job = ForeignKeyField(Job, null=True, default=None)
    project = ForeignKeyField(Project, unique=True, backref="data")

    @property
    def dir_path(self) -> str:
        project_id = self.project_id  # type: str

        return os.path.join(
            self._meta.storage.root_path,
            PROJECTS_DIR_NAME,
            project_id,
            PROJECTS_DATA_DIR_NAME,
        )

    @property
    def file_path(self) -> str:
        file_name = self.file_name  # type: str

        return os.path.join(self.dir_path, file_name)

    def setup_filesystem(self):
        create_dirs(self.dir_path)

    def validate_filesystem(self):
        if not os.path.exists(self.dir_path):
            raise FileNotFoundError(
                "project data directory at {} does not exist anymore".format(
                    self.dir_path
                )
            )

        if self.file_name and not os.path.exists(self.file_path):
            raise FileNotFoundError(
                "project data file at {} does not exist anymore".format(self.file_path)
            )
