"""
DB model classes for jobs
"""

from enum import Enum
import logging
import uuid
import datetime

from peewee import (
    CharField,
    Field,
    TextField,
    UUIDField,
    DateTimeField,
    ForeignKeyField,
)
from playhouse.sqlite_ext import JSONField

from neuralmagicML.server.models.base import BaseModel
from neuralmagicML.server.models.projects import Project


__all__ = ["JobStatus", "JobStatusField", "Job"]


_LOGGER = logging.getLogger(__name__)


class JobStatus(Enum):
    """
    Enumerator class for tracking the status of jobs
    """

    pending = "pending"
    started = "started"
    canceling = "canceling"
    completed = "completed"
    canceled = "canceled"
    error = "error"


class JobStatusField(Field):
    """
    peewee DB field for saving and loading JobStatus from the database
    """

    field_type = "VARCHAR"

    def db_value(self, value: JobStatus):
        return value.name

    def python_value(self, value: str):
        return JobStatus[value]


class Job(BaseModel):
    """
    DB model for a project's job.
    """

    job_id = UUIDField(primary_key=True, default=uuid.uuid4)
    project = ForeignKeyField(Project, backref="jobs")
    created = DateTimeField(default=datetime.datetime.now)
    modified = DateTimeField(default=datetime.datetime.now)
    type_ = CharField()
    worker_args = JSONField(null=True, default=None)
    status = JobStatusField(default=JobStatus.pending)
    progress = JSONField(null=True, default=None)
    error = TextField(null=True)

    def save(self, *args, **kwargs):
        """
        Override for peewee save function to update the modified date
        """
        self.modified = datetime.datetime.now()

        return super().save(*args, **kwargs)
