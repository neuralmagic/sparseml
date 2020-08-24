"""
Base classes necessary for running background jobs
"""

from enum import Enum
import logging
import uuid

from peewee import CharField, Field, TextField, UUIDField
from playhouse.sqlite_ext import JSONField

from neuralmagicML.server.models.base import BaseModel


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
    field_type = "VARCHAR"

    def db_value(self, value: JobStatus):
        return value.name

    def python_value(self, value: str):
        return JobStatus[value]


class Job(BaseModel):
    job_id = UUIDField(primary_key=True, default=uuid.uuid4)
    type_ = CharField()
    worker_args = JSONField(null=True, default=None)
    status = JobStatusField(default=JobStatus.pending)
    progress = JSONField(null=True, default=None)
    error = TextField(null=True)
    result = TextField(null=True)
