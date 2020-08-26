"""
DB model classes for project's profiles such as performance and loss
"""

import logging
import datetime
import uuid

from peewee import (
    CharField,
    TextField,
    ForeignKeyField,
    DateTimeField,
    BooleanField,
    IntegerField,
)
from playhouse.sqlite_ext import JSONField

from neuralmagicML.server.models.base import BaseModel, CSVField
from neuralmagicML.server.models.projects import Project
from neuralmagicML.server.models.jobs import Job


__all__ = ["ProjectLossProfile", "ProjectPerfProfile"]


_LOGGER = logging.getLogger(__name__)


class ProjectLossProfile(BaseModel):
    """
    DB model for a project's loss profile.
    A project may have multiple loss profiles stored in the DB.
    """

    profile_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    project = ForeignKeyField(Project, backref="loss_profiles")
    created = DateTimeField(default=datetime.datetime.now)
    source = TextField(null=True, default=None)
    job = ForeignKeyField(Job, null=True, default=None)
    analysis = JSONField(null=True, default=None)
    name = TextField(null=True, default="")
    pruning_estimations = BooleanField(default=False)
    pruning_estimation_type = TextField(null=True, default=None)
    pruning_structure = TextField(null=True, default=None)
    quantized_estimations = BooleanField(default=False)
    quantized_estimation_type = TextField(null=True, default=None)


class ProjectPerfProfile(BaseModel):
    """
    DB model for a project's performance profile.
    A project may have multiple perf profiles stored in the DB
    """

    profile_id = CharField(primary_key=True, default=lambda: uuid.uuid4().hex)
    project = ForeignKeyField(Project, backref="loss_profiles")
    created = DateTimeField(default=datetime.datetime.now)
    source = TextField(null=True, default=None)
    job = ForeignKeyField(Job, null=True, default=None)
    analysis = JSONField(null=True, default=None)
    name = TextField(null=True, default="")
    batch_size = IntegerField(null=True, default=None)
    core_count = IntegerField(null=True, default=None)
    instruction_sets = CSVField(null=True, default=None)
    pruning_estimations = BooleanField(default=False)
    quantized_estimations = BooleanField(default=False)
